"""Streamlit cache integration: reuse, demodulation and engine equivalence."""

from dataclasses import replace

import numpy as np

import snom_calculator._calculation_cache as cache_module
from snom_calculator.materials import constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters


def test_bulk_far_field_and_reference_changes_reuse_raw_signals(monkeypatch):
    from snom_calculator import _spheroid_engine
    from snom_calculator.bulk import calculate_bulk_harmonics

    frequency = np.array([900., 910.])
    sample = constant_material(6 + .2j, frequency, "sample")
    reference = constant_material(12 + .1j, frequency, "reference")
    other = constant_material(9 + .3j, frequency, "other")
    tip = TipParameters()
    numerical = NumericalSettings(M=8, N=8, quadQ=16)
    harmonics = (2, 3)
    scenarios = (
        # Sample and reference at M, N/2 and refined quadQ; M/2 reuses the full-M heights.
        (reference, FarFieldSettings(enabled=False), 6),
        (reference, FarFieldSettings(enabled=True), 6),
        (other, FarFieldSettings(enabled=True, angle_deg=45), 9),
    )
    expected_results = [calculate_bulk_harmonics(
        frequency, sample=sample, reference=current_reference, tip=tip,
        numerical=numerical, harmonics=harmonics, far_field=far_field)
        for current_reference, far_field, _ in scenarios]
    cache_module._cached_profile.clear()
    calls = []
    original = _spheroid_engine._bulk_profile

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_spheroid_engine, "_bulk_profile", counted)
    for (current_reference, far_field, expected_calls), expected in zip(scenarios, expected_results):
        actual = cache_module._cached_bulk(
            frequency, sample, current_reference, tip, numerical, harmonics, far_field)
        assert len(calls) == expected_calls
        for harmonic in harmonics:
            np.testing.assert_allclose(actual[harmonic].contrast, expected[harmonic].contrast)


def test_layer_far_field_and_reference_changes_reuse_raw_signals(monkeypatch):
    from snom_calculator import _spheroid_engine
    from snom_calculator.layer import calculate_layer_harmonics

    frequency = np.array([900., 910.])
    layer = constant_material(6 + .2j, frequency, "layer")
    substrate = constant_material(12 + .1j, frequency, "substrate")
    reference = constant_material(-5000 + 800j, frequency, "reference")
    other = constant_material(9 + .3j, frequency, "other")
    tip = TipParameters()
    numerical = NumericalSettings(M=8, N=8, quadQ=16, K=2)
    scenarios = (
        # Sample at M and N/2 only: M/2 reuses the full-M heights, and the
        # reference comes from the bulk eigen solver.
        (reference, FarFieldSettings(enabled=False)),
        (reference, FarFieldSettings(enabled=True)),
        (other, FarFieldSettings(enabled=True, angle_deg=45)),
        (reference, FarFieldSettings(enabled=False)),
    )
    expected_results = [calculate_layer_harmonics(
        frequency, layer=layer, substrate=substrate, reference=current_reference,
        thickness_nm=20, tip=tip, numerical=numerical, harmonics=(2, 3), far_field=far_field)
        for current_reference, far_field in scenarios]
    cache_module._cached_profile.clear()
    calls = []
    original = _spheroid_engine._film_profile

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_spheroid_engine, "_film_profile", counted)
    for (current_reference, far_field), expected in zip(scenarios, expected_results):
        actual = cache_module._cached_layer(
            frequency, layer, substrate, current_reference, 20, tip, numerical,
            (2, 3), far_field, 1)
        assert len(calls) == 2
        for harmonic in (2, 3):
            np.testing.assert_allclose(actual[harmonic].sample_harmonic, expected[harmonic].sample_harmonic)
            np.testing.assert_allclose(actual[harmonic].reference_harmonic, expected[harmonic].reference_harmonic)
            np.testing.assert_allclose(actual[harmonic].contrast, expected[harmonic].contrast)


def test_stack_cached_signals_match_direct_anisotropic_far_field():
    from snom_calculator.layer import calculate_stack_harmonics
    from snom_calculator.materials import MaterialSpectrum

    frequency = np.array([900., 910.])
    layers = (
        (MaterialSpectrum(np.full(2, 6 + .2j), np.full(2, 8 + .3j), "film"), 20.),
        (constant_material(12 + .1j, frequency, "middle"), 30.),
    )
    substrate = constant_material(4 + .2j, frequency, "substrate")
    reference = constant_material(-5000 + 800j, frequency, "reference")
    tip = TipParameters()
    numerical = NumericalSettings(M=8, N=8, quadQ=16, K=2)
    far_field = FarFieldSettings(enabled=True)
    actual = cache_module._cached_stack(
        frequency, layers, substrate, reference, tip, numerical, (2, 3), far_field, 1)
    expected = calculate_stack_harmonics(
        frequency, layers=layers, substrate=substrate, reference=reference,
        tip=tip, numerical=numerical, harmonics=(2, 3), far_field=far_field)
    for harmonic in (2, 3):
        np.testing.assert_allclose(actual[harmonic].sample_harmonic, expected[harmonic].sample_harmonic)
        np.testing.assert_allclose(actual[harmonic].reference_harmonic, expected[harmonic].reference_harmonic)
        np.testing.assert_allclose(actual[harmonic].contrast, expected[harmonic].contrast)


def test_layer_m_half_check_reuses_even_full_m_heights():
    frequency = np.array([900., 950., 1000.])
    layers = ((constant_material(-3 + .4j, frequency, "film"), 20.),)
    substrate = constant_material(12 + .1j, frequency, "substrate")
    tip = TipParameters()
    numerical = NumericalSettings(M=12, N=8, quadQ=16, K=4)
    def profile(layers, substrate, settings):
        return cache_module._cached_profile(layers, substrate, tip, tip.sample_gap_nm, settings,
                                            (), "auto", 1)[0]

    np.testing.assert_allclose(profile(layers, substrate, numerical)[::2],
                               profile(layers, substrate, replace(numerical, M=6)), rtol=1e-11)
    reference = constant_material(-5000 + 800j, frequency, "reference")
    np.testing.assert_allclose(profile((), reference, numerical)[::2],
                               profile((), reference, replace(numerical, M=6)), rtol=1e-11)
