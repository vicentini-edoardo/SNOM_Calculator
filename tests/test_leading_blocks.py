"""N/2 check reuses leading blocks of the full-N coupling matrices."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest

from snom_calculator import _fim_coupling, _leading_blocks
from snom_calculator._leading_blocks import held, leading_block, reserve, retaining
from snom_calculator._quadrature import demodulation_weights
from snom_calculator._spectra import calculate_spectra, solve_profile
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters

FREQUENCY = np.linspace(900., 1000., 3)


@pytest.mark.parametrize("checked", [True, False])
def test_concurrent_calculations_cannot_replace_another_runs_retained_tables(checked):
    entered, release, attempted, second_entered = (Event() for _ in range(4))

    def first():
        with retaining():
            value, _ = leading_block("table", 2, lambda: (np.eye(2), None))
            entered.set()
            assert release.wait(5)
            np.testing.assert_array_equal(held("table", 2), value)

    def second():
        assert entered.wait(5)
        attempted.set()
        with retaining(checked):
            second_entered.set()
            assert held("table", 2) is None

    with ThreadPoolExecutor(2) as pool:
        one, two = pool.submit(first), pool.submit(second)
        try:
            assert attempted.wait(5)
            assert not second_entered.wait(.1)
        finally:
            release.set()
        one.result(timeout=5)
        two.result(timeout=5)
    assert second_entered.is_set()


def _stack():
    film = MaterialSpectrum(4 + .2j + 0 * FREQUENCY, .75 * (4 + .2j) + 0 * FREQUENCY, "film")
    return ((film, 10.), (film, 30.)), constant_material(11.7 + .05j, FREQUENCY, "substrate")


def test_leading_block_slices_larger_and_respects_scope_and_cap():
    calls = []

    def compute(N):
        calls.append(N)
        return np.arange(N * N, dtype=float).reshape(N, N), "err"

    leading_block("k", 4, lambda: compute(4))
    assert calls == [4]
    with retaining():
        leading_block("k", 8, lambda: compute(8))
        count = len(calls)
        small, extra = leading_block("k", 4, lambda: compute(4))
        assert len(calls) == count and extra == "err"
        np.testing.assert_array_equal(small, np.arange(64.).reshape(8, 8)[:4, :4])
        with patch.object(_leading_blocks, "MAX_BYTES", 0):
            leading_block("big", 8, lambda: compute(8))
            assert "big" not in _leading_blocks._store
    assert _leading_blocks._store is None


def test_reserve_and_held_respect_scope_order_and_cap():
    assert reserve("t", 4, (4, 4)) is None and held("t", 4) is None  # outside retaining()
    with retaining():
        table = reserve("t", 8, (8, 8))
        table[:] = np.arange(64.).reshape(8, 8)
        assert reserve("t", 4, (4, 4)) is None  # already held at a larger order
        np.testing.assert_array_equal(held("t", 4), table[:4, :4])
        assert held("t", 16) is None
        with patch.object(_leading_blocks, "MAX_BYTES", 0):
            assert reserve("big", 8, (8, 8)) is None


def _run_pair(tip, target, name):
    layers, substrate = _stack()
    full = NumericalSettings(M=8, N=16, quadQ=32)
    half = replace(full, N=8)
    weights = demodulation_weights(8, (2, 3))
    gap = tip.sample_gap_nm
    plain, _ = solve_profile(layers, substrate, tip, gap, half, (), "auto", 1)
    with patch.object(target, name, autospec=True, side_effect=getattr(target, name)) as spy:
        with retaining():
            solve_profile(layers, substrate, tip, gap, full, (), "auto", 1)
            full_calls = spy.call_count
            assert full_calls > 0
            reused, _ = solve_profile(layers, substrate, tip, gap, half, (), "auto", 1)
        assert spy.call_count == full_calls, "half-N solve must not recompute couplings"
    np.testing.assert_allclose(weights @ reused, weights @ plain, rtol=full.momentum_rtol * 10)


def test_spheroid_fim_half_n_reuses_full_n_matrices():
    _run_pair(TipParameters(length_nm=600., sample_gap_nm=2.), _fim_coupling.FIMCoupling, "_matrices")


def test_checked_n_error_unchanged_by_reuse():
    layers, substrate = _stack()
    reference = constant_material(11.7 + .05j, FREQUENCY, "reference")
    numerical = NumericalSettings(M=8, N=16, quadQ=32)
    tip = TipParameters(length_nm=600., sample_gap_nm=2.)
    common = dict(frequency_cm1=FREQUENCY, layers=layers, substrate=substrate, reference=reference,
                  tip=tip, numerical=numerical, harmonics=(2, 3),
                  far_field=FarFieldSettings(enabled=False), checked=True)
    with patch.object(_leading_blocks, "MAX_BYTES", 0):  # nothing retained: independent half-N
        expected = calculate_spectra(**common)
    actual = calculate_spectra(**common)
    for harmonic in (2, 3):
        np.testing.assert_allclose(actual[harmonic].metadata["N_check_max_error"],
                                   expected[harmonic].metadata["N_check_max_error"],
                                   rtol=1e-3, atol=1e-9)


def test_streamed_image_tables_keep_half_n_block_for_n_check():
    from snom_calculator import _spheroid_engine

    film = MaterialSpectrum(-.9 + .05j + 0 * FREQUENCY, -.9 + .05j + 0 * FREQUENCY, "film")  # near the pole: exact tables
    common = dict(frequency_cm1=FREQUENCY, layers=((film, 10.),),
                  substrate=constant_material(11.7 + .05j, FREQUENCY, "substrate"),
                  reference=constant_material(11.7 + .05j, FREQUENCY, "reference"),
                  tip=TipParameters(length_nm=600., sample_gap_nm=2.),
                  numerical=NumericalSettings(M=8, N=16, quadQ=32, K=8), harmonics=(2, 3),
                  far_field=FarFieldSettings(enabled=False), checked=True)
    with patch.object(_spheroid_engine, "_EXACT_TABLE_BYTES", 0):  # stream every table set
        with patch.object(_leading_blocks, "MAX_BYTES", 0):
            expected = calculate_spectra(**common)
        _spheroid_engine.clear_caches()
        with patch.object(_spheroid_engine, "_image_tables", wraps=_spheroid_engine._image_tables) as spy:
            actual = calculate_spectra(**common)
        orders = [call.args[3] for call in spy.call_args_list]
    assert 16 in orders and 8 not in orders, "N/2 check must reuse the kept leading blocks"
    for harmonic in (2, 3):
        np.testing.assert_allclose(actual[harmonic].metadata["N_check_max_error"],
                                   expected[harmonic].metadata["N_check_max_error"], rtol=1e-6, atol=1e-12)
