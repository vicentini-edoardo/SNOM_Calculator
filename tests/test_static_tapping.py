"""DC demodulation and the single-height static response."""

import numpy as np
import pytest

from _reference import film_profile
from snom_calculator import _spheroid_engine
from snom_calculator.bulk import calculate_bulk_harmonics
from snom_calculator.calculations import calculate_bulk_checked, calculate_stack_checked
from snom_calculator.materials import constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters
from snom_calculator.setup import _apply_setup, _default_setup


def test_bulk_harmonic_zero_is_cycle_average():
    frequency = np.array([900., 905.])
    sample = constant_material(6 + .2j, frequency, "sample")
    reference = constant_material(12 + .1j, frequency, "reference")
    tip = TipParameters()
    numerical = NumericalSettings(M=8, N=8, quadQ=16)
    results = calculate_bulk_harmonics(
        frequency, sample=sample, reference=reference, tip=tip,
        numerical=numerical, harmonics=(0, 3), far_field=FarFieldSettings(enabled=False))
    profile = _spheroid_engine.profile((), sample, tip, 2., numerical)[0]
    expected = (profile[0] / 2 + profile[1:-1].sum(axis=0) + profile[-1] / 2) / 8
    np.testing.assert_allclose(results[0].sample_harmonic, expected, rtol=1e-12)
    assert np.isfinite(results[0].contrast).all()


def test_static_setup_round_trip():
    state = _default_setup()
    state.update(tip_A=0., harmonic_0=True, harmonic_3=False)
    restored = {}
    _apply_setup(restored, state)
    assert restored["tip_A"] == 0
    assert restored["harmonic_0"] is True


def test_static_bulk_solves_one_height():
    tip = TipParameters(25, 600, 0, 2, 2, -1e10 + 0j)
    profile, _ = _spheroid_engine.profile((), constant_material(6 + .2j, np.array([900., 905.]), "s"),
                                          tip, 2., NumericalSettings(M=8, N=8, quadQ=16))
    assert profile.shape == (1, 2)


@pytest.mark.parametrize("method", ["gicm", "fim"])
def test_static_layer_solves_one_height(method):
    profile, signals = film_profile([900., 905.], 4 + .2j, 12 + .1j, radius=30, gap=3, amplitude=0,
                                    M=8, N=8, quadQ=16, K=3, harmonics=(0, 3), method=method)
    assert profile.shape == (1, 2)
    np.testing.assert_allclose(signals[0], profile[0])
    np.testing.assert_array_equal(signals[1], np.zeros(2))


@pytest.mark.parametrize("model", ["bulk", "layer", "multilayer"])
@pytest.mark.parametrize("epsilon", [-1e10 + 0j])
def test_checked_static_calculation(model, epsilon):
    frequency = np.array([900., 905.])
    material = constant_material(6 + .2j, frequency, "sample")
    reference = constant_material(12 + .1j, frequency, "reference")
    tip = TipParameters(amplitude_nm=0, epsilon=epsilon)
    numerical = NumericalSettings(M=8, N=8, quadQ=16, K=3)
    common = dict(tip=tip, numerical=numerical, harmonics=(0,),
                  far_field=FarFieldSettings(enabled=False), check_n=True)
    if model == "bulk":
        result = calculate_bulk_checked(frequency, material, reference, **common)[0]
    else:
        layers = ((material, 20.),)
        if model == "multilayer":
            layers += ((constant_material(8 + .3j, frequency, "second"), 10.),)
        result = calculate_stack_checked(frequency, layers, reference, reference,
                                         workers=1, **common)[0]
    assert np.isfinite(result.contrast).all()
    assert result.metadata["M_check_max_error"] == 0
    assert np.isfinite(result.metadata["N_check_max_error"])


@pytest.mark.parametrize("model", ["bulk", "layer", "multilayer"])
@pytest.mark.parametrize("checked", [False, True])
def test_static_higher_harmonics_have_zero_signals_and_undefined_contrast(model, checked):
    from snom_calculator.layer import calculate_stack_harmonics

    frequency = np.array([900., 905.])
    sample = constant_material(6 + .2j, frequency, "sample")
    reference = constant_material(12 + .1j, frequency, "reference")
    common = dict(tip=TipParameters(amplitude_nm=0),
                  numerical=NumericalSettings(M=8, N=8, quadQ=16, K=3),
                  harmonics=(0, 2, 3), far_field=FarFieldSettings(enabled=False))
    with np.errstate(divide="raise", invalid="raise"):
        if model == "bulk":
            results = (calculate_bulk_checked(frequency, sample, reference, **common) if checked
                       else calculate_bulk_harmonics(frequency, sample=sample, reference=reference, **common))
        else:
            layers = ((sample, 20.),)
            if model == "multilayer":
                layers += ((constant_material(8 + .3j, frequency, "second"), 10.),)
            results = (calculate_stack_checked(frequency, layers, reference, reference, workers=1, **common)
                       if checked else calculate_stack_harmonics(
                           frequency, layers=layers, substrate=reference, reference=reference, workers=1, **common))
    assert np.isfinite(results[0].contrast).all()
    for n in (2, 3):
        np.testing.assert_array_equal(results[n].sample_harmonic, np.zeros(2))
        np.testing.assert_array_equal(results[n].reference_harmonic, np.zeros(2))
        assert np.isnan(results[n].contrast).all()
        assert np.isnan(results[n].amplitude).all()
        assert np.isnan(results[n].phase_rad).all()


@pytest.mark.parametrize("method", ["gicm", "fim"])
def test_layer_harmonic_zero_is_cycle_average_and_static_limit(method):
    common = dict(frequency=[900., 905.], layer=4 + .2j, substrate=12 + .1j, radius=30, gap=3,
                  M=8, N=8, quadQ=16, K=3, harmonics=(0, 3), method=method)
    profile, tapped = film_profile(amplitude=40, **common)
    expected = (profile[0] / 2 + profile[1:-1].sum(axis=0) + profile[-1] / 2) / 8
    np.testing.assert_allclose(tapped[0], expected, rtol=1e-12)
    _, static = film_profile(amplitude=0, **common)
    np.testing.assert_allclose(static[0], profile[0], rtol=1e-10)
    np.testing.assert_array_equal(static[1], np.zeros(2))


def test_static_ui_calculation():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    app.checkbox(key="harmonic_3").uncheck()
    app.checkbox(key="harmonic_0").check().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)
    app.number_input(key="tip_A").set_value(0.).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)
    app.checkbox(key="harmonic_3").check().run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert 0 in app.session_state["results"]
    assert np.isnan(app.session_state["results"][3].contrast).all()


def test_static_postprocessing_preserves_zero_raw_signals():
    from snom_calculator.calculations import run_calculations

    frequency = np.array([900., 905.])
    results = run_calculations(
        "bulk", frequency,
        {"sample": constant_material(6 + .2j, frequency, "sample"),
         "reference": constant_material(12 + .1j, frequency, "reference")},
        {"tip": TipParameters(amplitude_nm=0), "numerical": NumericalSettings(M=8, N=8, quadQ=16),
         "far_field": FarFieldSettings(enabled=False), "preset": "Custom",
         "average_gaps": (2., 4., 2), "resolution_fwhm_cm1": 7.}, (0, 3))
    np.testing.assert_array_equal(results[3].sample_harmonic, np.zeros(2))
    assert np.isnan(results[3].contrast).all()


@pytest.mark.parametrize("M", [1, 7, 30])
def test_static_response_does_not_depend_on_height_resolution(M):
    frequency = np.array([900., 905.])
    material = constant_material(6 + .2j, frequency, "sample")
    results = calculate_bulk_checked(
        frequency, material, material, TipParameters(amplitude_nm=0),
        NumericalSettings(M=M, N=8, quadQ=16), (0, 4), FarFieldSettings(enabled=False))
    np.testing.assert_allclose(results[0].contrast, np.ones(2))
    assert results[4].metadata["M_check_max_error"] == 0


def test_static_unnormalized_higher_harmonic_is_zero():
    frequency = np.array([900., 905.])
    material = constant_material(6 + .2j, frequency, "sample")
    result = calculate_bulk_checked(
        frequency, material, None, TipParameters(amplitude_nm=0),
        NumericalSettings(M=8, N=8, quadQ=16), (3,), FarFieldSettings(enabled=False))[3]
    np.testing.assert_array_equal(result.sample_harmonic, np.zeros(2))
    np.testing.assert_array_equal(result.contrast, np.zeros(2))
