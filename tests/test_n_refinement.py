import contextlib
from dataclasses import replace
import sys

import numpy as np
import pytest

from snom_calculator._spectra import engine, solve_profile
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import NumericalSettings, SpectrumResult, TipParameters


@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("M", [8, 9])
def test_doubled_profile_reuses_old_heights_and_matches_full_solve(film, M, monkeypatch):
    frequency = np.array([900., 910.])
    sample = constant_material(6 + .4j, frequency, "sample")
    layers = ((sample, 20.),) if film else ()
    tip = TipParameters(25., 600., 50., 5., 5., -100 + 10j)
    numerical = NumericalSettings(M=M, N=4, quadQ=16, K=2)
    harmonics = (1, 2)
    coarse, _ = solve_profile(layers, sample, tip, 5., numerical, harmonics, "auto", 1)
    fine = replace(numerical, M=2 * M)
    expected, _ = solve_profile(layers, sample, tip, 5., fine, harmonics, "auto", 1)
    module = engine()
    calls = []
    name = "_surface_trajectory"
    original = getattr(module, name)

    def counted(*args, **kwargs):
        heights = args[7]
        calls.append(len(heights))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    actual, _ = solve_profile(layers, sample, tip, 5., fine, harmonics, "auto", 1,
                              previous=coarse)
    np.testing.assert_array_equal(actual[::2], coarse)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    assert calls and set(calls) == {M}


def _refinement_app():
    import numpy as np
    import streamlit as st
    from snom_calculator._ui_controls import _calculation_action, _render_m_refinement_prompt
    from snom_calculator.materials import constant_material
    from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters

    st.session_state.setdefault("frequency_start", 900.)
    st.session_state.setdefault("frequency_end", 910.)
    frequency = np.array([900., 910.])
    _calculation_action("bulk", frequency, {"sample": constant_material(6 + .4j, frequency, "sample")},
                        {"tip": TipParameters(), "numerical": NumericalSettings(M=8),
                         "preset": st.session_state.get("preset", "Custom"),
                         "far_field": FarFieldSettings(enabled=False)})
    _render_m_refinement_prompt()


@pytest.mark.parametrize("preset, limit", [("Weak", "20%"), ("Strong", "20%"), ("Custom", "20%")])
def test_app_asks_each_time_and_shows_current_error(monkeypatch, preset, limit):
    from streamlit.testing.v1 import AppTest
    from snom_calculator import _ui_controls as ui

    # AppTest replaces __main__; restore it for multiprocessing tests.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(sys.modules["__main__"], "__file__", sys.modules["__main__"].__file__)

    seen = []

    def calculate(model, frequency, materials, controls, harmonics):
        M = controls["numerical"].M
        seen.append((M, controls.get("_reuse_M")))
        return {2: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
            contrast=np.ones(2), permittivities={},
            metadata={"M": M, "M_check_max_error": {8: .4, 16: .25, 32: .004}[M]})}

    monkeypatch.setattr(ui, "_run_calculations", calculate)
    monkeypatch.setattr(ui, "_validate_structure", lambda *args: None)
    monkeypatch.setattr(ui, "_append_results_history", lambda *args: None)
    app = AppTest.from_function(_refinement_app).run()
    app.session_state["preset"] = preset
    app.button[0].click().run()
    assert not app.exception
    assert any("40.00%" in warning.value for warning in app.warning)
    assert any(f"limit {limit}" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Calculate with M = 16").click().run()
    assert not app.exception
    assert seen == [(8, None), (16, 8)]
    assert any("25.00%" in warning.value for warning in app.warning)
    assert any(f"limit {limit}" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Calculate with M = 32").click().run()
    assert not app.exception
    assert seen == [(8, None), (16, 8), (32, 16)]
    app.run()  # AppTest drops a closed dialog one run late
    assert not app.warning


@pytest.mark.parametrize("preset, error, asks", [("Weak", .20, False), ("Weak", .21, True),
                                              ("Strong", .20, False), ("Very strong", .21, True),
                                              ("Automatic", .20, False), ("Automatic", .21, True),
                                              ("Custom", .08, False), ("Custom", .21, True)])
def test_app_m_refinement_threshold_follows_preset(monkeypatch, preset, error, asks):
    from streamlit.testing.v1 import AppTest
    from snom_calculator import _ui_controls as ui

    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(sys.modules["__main__"], "__file__", sys.modules["__main__"].__file__)

    def calculate(model, frequency, materials, controls, harmonics):
        return {2: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
            contrast=np.ones(2), permittivities={}, metadata={"M": 8, "M_check_max_error": error})}

    monkeypatch.setattr(ui, "_run_calculations", calculate)
    monkeypatch.setattr(ui, "_validate_structure", lambda *args: None)
    monkeypatch.setattr(ui, "_append_results_history", lambda *args: None)
    app = AppTest.from_function(_refinement_app).run()
    app.session_state["preset"] = preset
    app.button[0].click().run()
    assert not app.exception
    assert any(button.key == "double_m" for button in app.button) == asks


def test_precision_box_shows_checks():
    from snom_calculator import _ui_controls as ui

    class Slot:
        def container(self):
            return contextlib.nullcontext()

    shown = []
    result = SpectrumResult.from_contrast(
        frequency_cm1=np.array([900., 910.]), sample_harmonic=np.ones(2),
        reference_harmonic=np.ones(2), contrast=np.ones(2), permittivities={},
        metadata={"M_check_max_error": .08, "M_requested": 8, "M_requested_error": .1,
                  "N_check_max_error": .25})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ui.st, "markdown", lambda body, **kwargs: shown.append((body, kwargs)))
        ui._render_checks(Slot(), {2: result})
    body, kwargs = shown[0]
    assert body.startswith("M vs M/2 : 8.00 %")
    assert ":red[N vs N/2 : 25.00 %]" in body
    assert body.endswith(":red[N vs N/2 : 25.00 %]")
    assert "20%" in kwargs["help"]


def test_app_keeps_current_result_when_refinement_declined(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from snom_calculator import _ui_controls as ui

    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(sys.modules["__main__"], "__file__", sys.modules["__main__"].__file__)

    def calculate(model, frequency, materials, controls, harmonics):
        return {2: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
            contrast=np.ones(2), permittivities={}, metadata={"M": 8, "M_check_max_error": .25})}

    monkeypatch.setattr(ui, "_run_calculations", calculate)
    monkeypatch.setattr(ui, "_validate_structure", lambda *args: None)
    monkeypatch.setattr(ui, "_append_results_history", lambda *args: None)
    app = AppTest.from_function(_refinement_app).run()
    app.button[0].click().run()
    next(button for button in app.button if button.label == "Keep current M").click().run()
    assert not app.exception
    assert app.session_state["results"][2].metadata["M"] == 8
    app.run()  # AppTest drops a closed dialog one run late
    assert not app.warning


@pytest.mark.parametrize("M, shape, amplitude", [(15, (8, 2), 50.), (16, (8, 2), 50.),
                                                (16, (9, 1), 50.), (16, (9, 2), 0.)])
def test_profile_rejects_invalid_reuse(M, shape, amplitude):
    frequency = np.array([900., 910.])
    with pytest.raises(ValueError, match="Reused profile"):
        solve_profile((), constant_material(6 + .2j, frequency, "sample"),
                      TipParameters(amplitude_nm=amplitude), 5., NumericalSettings(M=M),
                      (1,), "auto", 1, previous=np.zeros(shape, complex))


@pytest.mark.parametrize("method", ["fim", "real-axis"])
def test_spheroid_momentum_films_reuse_old_heights(method):
    frequency = np.array([900., 910.])
    film = MaterialSpectrum(np.full(2, 6 + .4j), np.full(2, 8 + .4j), "uniaxial")
    substrate = constant_material(12 + .2j, frequency, "substrate")
    tip = TipParameters(25., 600., 50., 5., 5., -100 + 10j)
    numerical = NumericalSettings(M=8, N=4, quadQ=16)
    args = (((film, 20.),), substrate, tip, 5.)
    coarse, _ = solve_profile(*args, numerical, (1, 2), method, 1)
    fine = replace(numerical, M=16)
    expected, _ = solve_profile(*args, fine, (1, 2), method, 1)
    actual, _ = solve_profile(*args, fine, (1, 2), method, 1, previous=coarse)
    np.testing.assert_array_equal(actual[::2], coarse)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-7)


def test_app_doubles_n_after_m_converges(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from snom_calculator import _ui_controls as ui

    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(sys.modules["__main__"], "__file__", sys.modules["__main__"].__file__)

    seen = []

    def calculate(model, frequency, materials, controls, harmonics):
        numerical = controls["numerical"]
        seen.append((numerical.M, numerical.N, numerical.quadQ, controls.get("_reuse_M")))
        return {2: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
            contrast=np.ones(2), permittivities={},
            metadata={"M": numerical.M, "N": numerical.N,
                      "M_check_max_error": {8: .25, 16: .01}[numerical.M],
                      "N_check_max_error": {40: .3, 80: .01}[numerical.N]})}

    monkeypatch.setattr(ui, "_run_calculations", calculate)
    monkeypatch.setattr(ui, "_validate_structure", lambda *args: None)
    monkeypatch.setattr(ui, "_append_results_history", lambda *args: None)
    app = AppTest.from_function(_refinement_app).run()
    app.button[0].click().run()
    next(button for button in app.button if button.label == "Calculate with M = 16").click().run()
    assert not app.exception
    assert any("30.00%" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Calculate with N = 80").click().run()
    assert not app.exception
    assert seen == [(8, 40, 48, None), (16, 40, 48, 8), (16, 80, 120, None)]
    metadata = app.session_state["results"][2].metadata
    assert (metadata["M_requested"], metadata["N_requested"], metadata["N_requested_error"]) == (8, 40, .3)
    app.run()  # AppTest drops a closed dialog one run late
    assert not app.warning
