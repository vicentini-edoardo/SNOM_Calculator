import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from snom_calculator import _calculation_cache, _progress
from snom_calculator._spheroid_engine import clear_caches
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters


def test_concurrent_sessions_keep_their_own_progress_callback_and_window():
    barrier = Barrier(2)

    def session(label):
        events = []
        _progress.set_callback(lambda text, fraction: events.append((text, fraction)))
        try:
            with _progress.stage(label, .2, .6):
                barrier.wait(timeout=5)
                _progress.report("response", .5)
                barrier.wait(timeout=5)
        finally:
            _progress.set_callback(None)
        return events

    with ThreadPoolExecutor(2) as pool:
        events = list(pool.map(session, ("first", "second")))
    for label, reported in zip(("first", "second"), events):
        assert len(reported) == 1
        assert reported[0][0] == f"{label}: response"
        assert reported[0][1] == pytest.approx(.4)


@pytest.mark.parametrize("check_n", [True, False])
@pytest.mark.parametrize("mode, workers", [
    ("bulk", 1), ("layer", 1), ("uniaxial", 1), ("layer", 2), ("uniaxial", 2),
    ("zero-layer", 1),
])
def test_slow_calculation_progress_includes_checks_and_stays_ordered(mode, workers, check_n):
    frequency = np.array([900., 910.])
    material = constant_material(6 + .2j, frequency, "sample")
    if mode == "uniaxial":
        material = MaterialSpectrum(
            np.full(2, 4 + .1j), np.full(2, 6 + .2j), label="sample")
    materials = {
        "sample": material, "layer": material, "thickness_nm": 0. if mode == "zero-layer" else 20.,
        "substrate": constant_material(12 + .1j, frequency, "substrate"),
        "reference": constant_material(-5000 + 800j, frequency, "reference"),
    }
    controls = dict(tip=TipParameters(), numerical=NumericalSettings(M=8, N=8, quadQ=16),
                    far_field=FarFieldSettings(enabled=False), workers=workers, preset="Custom",
                    average_gaps=(2., 3., 2), check_n=check_n)
    _calculation_cache._cached_profile.clear()
    clear_caches()
    events = []
    _progress.set_callback(lambda text, fraction: events.append((text, fraction)))
    try:
        result = _calculation_cache._run_calculations(
            "bulk" if mode == "bulk" else "layer", frequency, materials, controls, (2, 3))
        fractions = [fraction for _, fraction in events if fraction is not None]
        assert fractions == sorted(fractions)
        assert fractions[-1] == .9
        texts = [text for text, _ in events]
        assert any("Sample" in text and "Preparing geometry" in text for text in texts)
        assert any("Reference" in text for text in texts)
        assert any("Checking convergence (M)" in text for text in texts)
        assert any("Checking convergence (N)" in text for text in texts) == check_n
        assert ("N_check_max_error" in result[3].metadata) == check_n
        assert any("Solving response" in text and "/" in text for text in texts)
        if mode == "uniaxial" and workers == 1:
            assert any("Calculating coupling" in text and "2/2" in text for text in texts)
            assert any("FIM" in text for text in texts)
        if workers > 1:
            assert any("frequency blocks 2/2" in text for text in texts)
        assert any("Gap 2/2" in text for text in texts)
        assert np.isfinite(result[3].contrast).all()

        # Cached runs still identify sample, reference and convergence stages.
        events.clear()
        _calculation_cache._run_calculations(
            "bulk" if mode == "bulk" else "layer", frequency, materials, controls, (2, 3))
        assert any("Sample" in text for text, _ in events)
        assert any("Reference" in text for text, _ in events)
        assert any("Checking convergence" in text for text, _ in events)
    finally:
        _progress.set_callback(None)
