import base64
import json
import re
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import warnings as warning_module

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import snom_calculator.app as app_module
import snom_calculator._ui_controls as controls_module
import snom_calculator._ui_editors as editors_module
import snom_calculator._ui_structure as structure_module
import snom_calculator._ui_results as results_module
import snom_calculator._ui_state as state_module
import snom_calculator._calculation_cache as cache_module
from snom_calculator._ui_results import (
    _append_results_history,
    _load_experimental_overlay,
    _render_results,
    _sync_selected_history_result,
)
from snom_calculator._ui_state import _calculation_model, _material_from_state
from snom_calculator._ui_structure import _physical_setup
from snom_calculator.setup import _apply_setup, _default_setup, _encode_setup
from snom_calculator.material_library import MaterialLibrary
from snom_calculator.models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters
from snom_calculator.materials import MaterialSpectrum, constant_material, custom_material

# AppTest resolves relative paths against this file, not the working directory.
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _scene_from_app(app):
    # Streamlit 1.65 gives image elements a typed name; older versions use the proto name.
    images = app.get("image") or app.get("imgs")
    return base64.b64decode(images[0].proto.imgs[0].url.partition(",")[2]).decode()


@pytest.mark.parametrize("layer_count", (1, 2))
def test_layer_checks_report_worst_frequency_and_warn(monkeypatch, layer_count):
    frequency = np.array([900., 910.])
    layers = tuple(
        (constant_material(3 + index + .1j, frequency, f"layer {index}"), 20.)
        for index in range(layer_count)
    )
    substrate = constant_material(12 + .1j, frequency, "substrate")
    reference = constant_material(8 + .1j, frequency, "reference")
    # Odd M, so the M/2 check is recomputed rather than read from the full-M heights.
    numerical = NumericalSettings(M=13, N=10, quadQ=16, K=2)
    seen = []

    def profile(settings, h2=1., h3=1.):
        # Only the end heights carry weight: demodulates to h2 for n=2 and h3 for n=3.
        values = np.zeros((settings.M + 1, frequency.size), dtype=complex)
        values[0], values[-1] = settings.M * (h2 + h3), settings.M * (h2 - h3)
        return values

    def fake_profile(current_layers, material, tip, gap, settings, harmonics, method, workers):
        values = profile(settings)
        if current_layers:
            seen.append(("sample", settings.M, settings.N))
            if settings.M == 6:
                values[:, 1] = profile(settings, h2=1.25)[:, 1]
        else:
            seen.append(("reference", settings.M, settings.N))
            if settings.N == 5:
                values[:, 0] = profile(settings, h2=1.30)[:, 0]
        return values, {}

    monkeypatch.setattr(cache_module, "_cached_profile", fake_profile)
    results = cache_module._cached_stack(
        frequency, layers, substrate, reference, TipParameters(), numerical,
        (2, 3), FarFieldSettings(enabled=False), 1,
    )

    assert len(seen) == 6
    assert results[2].metadata["M_check_max_error"] == pytest.approx(.25)
    assert results[2].metadata["N_check_max_error"] == pytest.approx(.30)
    assert results[3].metadata["M_check_max_error"] == pytest.approx(0, abs=1e-12)
    warnings = []
    monkeypatch.setattr(st, "warning", warnings.append)
    monkeypatch.setattr(st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "plotly_chart", lambda *args, **kwargs: None)
    _render_results(results, (2, 3), {})
    assert len(warnings) == 1
    assert "M vs M/2" in warnings[0] and "N vs N/2" in warnings[0]
    assert "Bulk" not in warnings[0]


def test_custom_material_library_is_private_to_each_session(monkeypatch):
    monkeypatch.delenv("SNOM_MATERIAL_LIBRARY_FILE", raising=False)
    definition = {
        "name": "mine",
        "tensor_mode": "isotropic",
        "axes": {
            axis: {"source_type": "weak_lorentz", "eps_inf": 2.0, "oscillators": []}
            for axis in ("xx", "yy", "zz")
        },
    }
    first, second = {}, {}
    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state=first))
    state_module._material_library().save(definition)
    archive = state_module._material_library().archive_json_bytes()
    assert "_custom_materials" not in _encode_setup(first)

    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state=second))
    assert state_module._material_library().names() == []
    state_module._material_library().import_archive(archive)
    state_module._material_library().delete("mine")
    assert state_module._material_library().names() == []

    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state=first))
    assert state_module._material_library().names() == ["mine"]


def test_local_library_loads_at_startup(tmp_path, monkeypatch):
    path = tmp_path / "library.json"
    monkeypatch.setenv("SNOM_MATERIAL_LIBRARY_FILE", str(path))
    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state={}))
    definition = {"name": "local", "tensor_mode": "isotropic", "axes": {
        axis: {"source_type": "weak_lorentz", "eps_inf": 2., "oscillators": []}
        for axis in ("xx", "yy", "zz")}}
    state_module._material_library().save(definition)
    fresh = {}
    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state=fresh))
    state_module._initialize_state()
    assert fresh["_custom_materials"]["local"]["name"] == "local"


def test_unreadable_local_library_falls_back_without_overwriting(tmp_path, monkeypatch):
    path = tmp_path / "library.json"
    path.write_text("{not json")
    monkeypatch.setenv("SNOM_MATERIAL_LIBRARY_FILE", str(path))
    errors = []
    monkeypatch.setattr(state_module, "st", SimpleNamespace(session_state={}, error=errors.append))
    state_module._initialize_state()
    assert errors and "not valid JSON" in errors[0]
    definition = {"name": "session", "tensor_mode": "isotropic", "axes": {
        axis: {"source_type": "weak_lorentz", "eps_inf": 2., "oscillators": []}
        for axis in ("xx", "yy", "zz")}}
    state_module._material_library().save(definition)
    assert state_module._material_library().names() == ["session"]
    assert path.read_text() == "{not json"


def test_visual_stack_selects_multilayer_calculation_model():
    assert _calculation_model([]) == "bulk"
    assert _calculation_model([1]) == "layer"
    assert _calculation_model([1, 2]) == "multilayer"
    assert _calculation_model([1, 2, 3, 4]) == "multilayer"


def test_visual_stack_resolves_all_calculation_materials(monkeypatch):
    state = _default_setup()
    state.update(
        {
            "finite_layer_ids": [1, 2],
            "layer_1_source": "Constant",
            "layer_1_real": 2.0,
            "layer_1_imag": 0.0,
            "layer_1_thickness": 20.0,
            "layer_2_source": "Constant",
            "layer_2_real": 4.0,
            "layer_2_imag": 0.0,
            "layer_2_thickness": 10.0,
        }
    )
    view = SimpleNamespace(
        session_state=state,
        button=lambda *args, **kwargs: False,
        caption=lambda *args, **kwargs: None,
        columns=lambda count, **kwargs: (view,) * (count if isinstance(count, int) else len(count)),
        container=lambda *args, **kwargs: nullcontext(),
        divider=lambda: None,
        html=lambda *args, **kwargs: None,
        image=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
        subheader=lambda *args, **kwargs: None,
        tabs=lambda *args, **kwargs: [nullcontext()],
        checkbox=lambda *args, **kwargs: False,
        number_input=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(structure_module, "st", view)
    monkeypatch.setattr(state_module, "st", view)
    monkeypatch.setattr(structure_module, "_render_quick_controls", lambda: None)

    materials = _physical_setup(np.array([600.0, 601.0]))
    assert [thickness for _, thickness in materials["layers"]] == [20.0, 10.0]
    assert [material.parallel[0].real for material, _ in materials["layers"]] == [2.0, 4.0]


def test_experimental_upload_opens_import_dialog(monkeypatch):
    state = {
        "experimental_overlay_upload": BytesIO(b"not experimental data"),
    }
    monkeypatch.setattr("snom_calculator._ui_results.st", SimpleNamespace(session_state=state))

    _load_experimental_overlay()
    assert state["experimental_import_open"] is True
    state["experimental_overlay_upload"] = None
    _load_experimental_overlay()

    assert state["experimental_import_open"] is False


def test_dismissing_editor_restores_unsaved_settings(monkeypatch):
    state = _default_setup()
    monkeypatch.setattr(st, "session_state", state)

    editors_module._open_editor("tip")
    state["tip_R"] = 31.5
    editors_module._dismiss_editor()
    state_module._initialize_state()

    assert state["tip_R"] == 25.0
    assert state["editor"] is None


def test_dismissing_import_does_not_reopen_it(monkeypatch):
    state = {"experimental_import_open": True}
    monkeypatch.setattr(st, "session_state", state)

    results_module._dismiss_experimental_import()

    assert state["experimental_import_open"] is False


def test_multiple_named_experimental_datasets_appear_on_plots(monkeypatch):
    figures = []
    view = SimpleNamespace(
        subheader=lambda *args, **kwargs: None,
        columns=lambda *args, **kwargs: (nullcontext(), nullcontext()),
        caption=lambda *args, **kwargs: None,
        checkbox=lambda *args, **kwargs: None,
        plotly_chart=lambda figure, **kwargs: figures.append(figure),
    )
    monkeypatch.setattr(results_module, "st", view)
    datasets = {
        "Sample A": {2: {"frequency_cm1": np.array([900, 901]), "amplitude": np.array([1, 2]), "phase_rad": np.array([0.1, 0.2])},
                     3: {"frequency_cm1": np.array([900, 901]), "amplitude": np.array([5, 6]), "phase_rad": None}},
        "Sample B": {2: {"frequency_cm1": np.array([900, 901]), "amplitude": np.array([3, 4]), "phase_rad": None}},
    }

    _render_results({}, (2,), datasets)

    assert [trace.name for trace in figures[0].data] == ["Sample A n = 2", "Sample B n = 2"]
    assert [trace.name for trace in figures[1].data] == ["Sample A n = 2"]


@pytest.mark.parametrize("reverse", [False, True])
def test_experimental_import_can_reverse_descending_rows(reverse):
    app = AppTest.from_string('''
from io import BytesIO
import streamlit as st
from snom_calculator._ui_results import _experimental_import_dialog
uploaded = BytesIO(b"frequency amplitude phase\\n902 3 0.3\\n901 2 0.2\\n900 1 0.1\\n")
uploaded.name = "descending.txt"
st.session_state["experimental_overlay_upload"] = uploaded
_experimental_import_dialog()
''').run()
    assert not app.exception
    checkbox = next(item for item in app.checkbox if item.label == "Reverse row order")
    assert checkbox.value is False
    if reverse:
        checkbox.check().run()
        assert app.dataframe[0].value.iloc[:, 0].tolist() == ["900", "901", "902"]
    app.selectbox(key="experimental_import_first_2").select("2 · amplitude")
    app.selectbox(key="experimental_import_second_2").select("3 · phase").run()
    next(button for button in app.button if button.label == "Load data").click().run()
    assert not app.exception
    if reverse:
        data = app.session_state["experimental_overlays"]["descending"][2]
        np.testing.assert_array_equal(data["frequency_cm1"], [900., 901., 902.])
        np.testing.assert_array_equal(data["amplitude"], [1., 2., 3.])
        np.testing.assert_allclose(data["phase_rad"], [0.1, 0.2, 0.3])
    else:
        assert any("increasing" in error.value for error in app.error)


def test_experimental_import_loads_several_harmonics_from_one_file():
    app = AppTest.from_string('''
from io import BytesIO
import streamlit as st
from snom_calculator._ui_results import _experimental_import_dialog
uploaded = BytesIO(b"frequency O2A O2P O3A O3P\\n900 1 0.1 3 0.3\\n901 2 0.2 4 0.4\\n")
uploaded.name = "harmonics.txt"
st.session_state["experimental_overlay_upload"] = uploaded
_experimental_import_dialog()
''').run()
    load = next(button for button in app.button if button.label == "Load data")
    load.click().run()
    assert any("at least one harmonic" in error.value for error in app.error)
    app.selectbox(key="experimental_import_first_2").select("2 · O2A")
    app.selectbox(key="experimental_import_second_2").select("3 · O2P")
    app.selectbox(key="experimental_import_first_3").select("4 · O3A")
    app.selectbox(key="experimental_import_second_3").select("5 · O3P").run()
    next(button for button in app.button if button.label == "Load data").click().run()
    assert not app.exception
    assert app.session_state["experimental_import_open"] is False
    overlay = app.session_state["experimental_overlays"]["harmonics"]
    assert sorted(overlay) == [2, 3]
    np.testing.assert_array_equal(overlay[2]["amplitude"], [1., 2.])
    np.testing.assert_allclose(overlay[3]["phase_rad"], [0.3, 0.4])


def test_experimental_phase_controls_update_plot_and_reset():
    app = AppTest.from_string('''
import numpy as np
import streamlit as st
from snom_calculator._ui_results import _results_workspace
if "experimental_overlays" not in st.session_state:
    st.session_state["harmonic_2"] = True
    st.session_state["experimental_overlays"] = {
        "Sample": {2: {"frequency_cm1": np.array([900., 901., 902.]),
                       "amplitude": np.ones(3), "phase_rad": np.array([3., -3., -2.])}},
        "Amplitude only": {2: {"frequency_cm1": np.array([900., 901.]),
                               "amplitude": np.ones(2), "phase_rad": None}}}
_results_workspace({})
''').run()
    assert not app.exception
    assert len(app.number_input) == 2
    assert app.checkbox[0].value is True
    app.number_input[0].set_value(7.).run()
    app.number_input[1].set_value(4.).run()
    assert not app.exception

    def plotted_phase():
        figure = json.loads(app.get("plotly_chart")[1].proto.spec)
        values = figure["data"][0]["y"]
        if isinstance(values, dict):
            return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        return np.asarray(values)

    original = np.array([3., -3., -2.])
    np.testing.assert_allclose(plotted_phase(), np.unwrap(original) + 7. + np.array([0., 4., 8.]))
    app.checkbox[0].uncheck().run()
    assert np.all(np.abs(plotted_phase()) <= np.pi)
    next(button for button in app.button if button.label == "Reset").click().run()
    assert app.number_input[0].value == 0.
    assert app.number_input[1].value == 0.
    assert app.checkbox[0].value is True
    np.testing.assert_allclose(plotted_phase(), np.unwrap(original))
    np.testing.assert_array_equal(app.session_state["experimental_overlays"]["Sample"][2]["phase_rad"], original)
    next(button for button in app.button if button.label == "Remove Sample").click().run()
    assert len(app.number_input) == 0


@pytest.mark.parametrize(
    ("M_error", "n_error", "quad_error", "expected"),
    ((0.25, 0.02, 0.02, "M"), (0.02, 0.25, 0.02, "N"),
     (0.02, 0.02, 0.04, "N"), (0.25, 0.25, 0.02, "M and N"),
     (0.25, 0.25, 0.06, "M and N"),
     (0.005, 0.005, 0.005, None)),
)
def test_bulk_precision_warning_for_one_affected_frequency(
    monkeypatch, M_error, n_error, quad_error, expected
):
    warnings = []
    monkeypatch.setattr(st, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "plotly_chart", lambda *args, **kwargs: None)
    frequency = np.array([900., 905.])
    result = SpectrumResult.from_contrast(
        frequency_cm1=frequency,
        sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
        contrast=np.ones(2), permittivities={},
        metadata={"model": "bulk", "M_check_max_error": M_error, "N_check_max_error": n_error,
                  "quadQ_check_max_error": quad_error, "quadQ": 16, "quadQ_check_higher": 24},
    )

    _render_results({3: result}, (3,), {})

    if expected is None:
        assert not warnings
        return
    assert len(warnings) == 1
    assert f"increase {expected}" in warnings[0]
    assert "quad" not in warnings[0]
    for parameter in expected.split(" and "):
        assert f"{parameter} vs {parameter}/2" in warnings[0]


def test_app_renders_scientific_workbench():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    labels = [button.label for button in app.button]

    assert not app.exception
    assert app.title[0].value == "s-SNOM Spectrum Calculator"
    assert "Tip · R 25 nm · L 600 nm · ε = -1e+10 + 0i" in labels
    assert "Add finite layer" in labels
    assert "Substrate · aSiO2 · semi-infinite" in labels
    assert "Reference · ε = -5000 + 800i" in labels
    assert "Calculate spectrum" in labels


def test_do_not_use_reference_checkbox_disables_reference_and_survives_setup():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.checkbox(key="reference_disabled").check().run(timeout=20)
    assert not app.exception
    assert app.button(key="physical_reference").disabled
    setup = json.loads(_encode_setup(app.session_state.filtered_state))
    restored = {}
    _apply_setup(restored, setup)
    assert restored["reference_disabled"] is True
    app.button(key="physical_substrate").click().run(timeout=20)
    assert not app.exception
    next(button for button in app.button if button.label == "Cancel").click().run(timeout=20)
    app.checkbox(key="reference_disabled").uncheck().run(timeout=20)
    assert not app.button(key="physical_reference").disabled
    app.button(key="physical_reference").click().run(timeout=20)
    assert not app.exception


def test_app_exposes_material_library_backup_and_restore_controls():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert any(button.label == "Download material library" for button in app.get("download_button"))
    assert any(uploader.label == "Restore material library" for uploader in app.file_uploader)


def test_precision_controls_are_compactly_contained():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert any(expander.label == "Calculation precision" for expander in app.get("expander"))
    assert [group.options for group in app.button_group if group.label == "Numerical settings"] == [
        ["Automatic", "Presets", "Custom"]
    ]
    assert not app.get("metric")
    assert app.checkbox(key="check_n").value is True
    app.checkbox(key="check_n").uncheck().run(timeout=20)
    assert not app.exception
    assert app.session_state["check_n"] is False
    state = {}
    setup_state = (app.session_state.to_dict() if hasattr(app.session_state, "to_dict")
                   else app.session_state.filtered_state)
    _apply_setup(state, json.loads(_encode_setup(setup_state)))
    assert state["check_n"] is False


def test_automatic_and_preset_precision_follow_each_layer():
    app = AppTest.from_file(APP_PATH)
    app.session_state["frequency_start"], app.session_state["frequency_end"] = 780., 990.
    app.session_state["substrate_source"], app.session_state["substrate_catalog"] = "Catalog", "SiC4H"
    app.session_state["tip_Hs"] = app.session_state["tip_Hs_max"] = 1.
    app.run(timeout=20)
    assert not app.exception
    assert any("Sample: Very strong" in caption.value for caption in app.caption)
    assert "Oscillator strength" not in [field.label for field in app.selectbox]
    automatic = next(caption.value for caption in app.caption if caption.value.startswith("Automatic · M"))

    # A 50 nm weak film on top moves SiC4H 50 nm away from the tip: far fewer heights.
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    app.session_state["layer_1_source"], app.session_state["layer_1_catalog"] = "Catalog", "aSiO2"
    app.session_state["layer_1_thickness"] = 50.
    app.run(timeout=20)
    assert any("Layer 1: Weak" in c.value and "Substrate: Very strong" in c.value for c in app.caption)
    stacked = next(caption.value for caption in app.caption if caption.value.startswith("Automatic · M"))
    assert int(stacked.split("M ")[1].split(" ")[0]) < int(automatic.split("M ")[1].split(" ")[0])

    next(group for group in app.button_group if group.label == "Numerical settings").set_value("Presets").run(timeout=20)
    next(field for field in app.selectbox if field.label == "Oscillator strength").set_value("Weak").run(timeout=20)
    assert not app.exception
    assert any(caption.value.startswith("Weak · M 30") for caption in app.caption)


def test_custom_precision_offers_nested_orders_without_quadrature_control():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(group for group in app.button_group if group.label == "Numerical settings").set_value("Custom").run(timeout=20)

    choices = {field.label: [int(value) for value in field.options] for field in app.selectbox
               if field.label in {"M", "N", "quadQ", "K"}}
    assert choices["M"] == [30, 60, 80, 120, 240, 480]
    assert 40 in choices["N"] and 80 in choices["N"]
    assert "quadQ" not in choices and "K" not in choices

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    app.session_state["layer_1_source"] = "Constant"
    next(group for group in app.button_group if group.label == "Numerical settings").set_value("Custom").run(timeout=20)
    choices = {field.label: [int(value) for value in field.options] for field in app.selectbox
               if field.label in {"M", "N", "quadQ", "K"}}
    assert choices["M"] == [30, 60, 80, 120, 240, 480]
    assert choices["K"] == [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40]
    assert "quadQ" not in choices

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    next(group for group in app.button_group if group.label == "Numerical settings").set_value("Custom").run(timeout=20)
    assert [int(value) for value in next(field for field in app.selectbox if field.label == "M").options] == [30, 60, 80, 120, 240, 480]


@pytest.mark.parametrize("layer_ids", [[1], [1, 2]])
def test_uniaxial_precision_replaces_k_and_switches_back(layer_ids):
    app = AppTest.from_file(APP_PATH)
    app.session_state["finite_layer_ids"] = layer_ids
    app.session_state["precision_mode"] = "Custom"
    for layer_id in layer_ids:
        app.session_state[f"layer_{layer_id}_source"] = "Constant"
        app.session_state[f"layer_{layer_id}_thickness"] = 20.0
        app.session_state[f"layer_{layer_id}_real"] = 4.0
        app.session_state[f"layer_{layer_id}_imag"] = 0.1
    app.session_state["layer_1_source"] = "Catalog"
    app.session_state["layer_1_catalog"] = "hBN"
    app.run(timeout=20)
    assert not app.exception
    assert "K" not in [field.label for field in app.selectbox]
    assert "Momentum tolerance" not in [field.label for field in app.selectbox]
    assert not any("momentum tolerance" in caption.value for caption in app.caption)
    app.session_state["layer_1_source"] = "Constant"
    app.run(timeout=20)
    assert not app.exception
    assert "K" in [field.label for field in app.selectbox]
    assert "Momentum tolerance" not in [field.label for field in app.selectbox]


def test_custom_precision_adjusts_legacy_saved_values():
    app = AppTest.from_file(APP_PATH)
    app.session_state["precision_mode"] = "Custom"
    app.session_state["bulk_custom_M"] = 42
    app.session_state["bulk_custom_N"] = 43
    app.session_state["bulk_custom_quad"] = 44
    app.run(timeout=20)

    values = {field.label: field.value for field in app.selectbox if field.label in {"M", "N", "quadQ"}}
    assert values == {"M": 30, "N": 40}  # quadQ is derived from N, not a control


def test_multiple_visual_layers_enable_calculation():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)

    calculate = next(button for button in app.button if button.label == "Calculate spectrum")
    assert not app.exception
    assert not calculate.disabled


def test_five_layer_visual_stack_stays_calculable():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    for _ in range(5):
        next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    app.number_input(key="frequency_points").set_value(3).run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=40)

    assert not app.exception
    assert not app.error
    assert not next(button for button in app.button if button.label == "Calculate spectrum").disabled
    assert len(app.get("plotly_chart")) == 2
    assert sum(button.label.startswith("Layer ") for button in app.button) >= 5


def test_tip_dialog_updates_left_summary():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)
    next(widget for widget in app.number_input if widget.label == "Radius R (nm)").set_value(31.5).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)

    assert not app.exception
    assert any(button.label.startswith("Tip · R 31.5 nm") for button in app.button)


def test_tip_editor_reopens_with_the_last_saved_values():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)
    next(widget for widget in app.number_input if widget.label == "Radius R (nm)").set_value(31.5).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)

    assert next(widget for widget in app.number_input if widget.label == "Radius R (nm)").value == 31.5

def test_new_material_builds_constant_and_uploaded_table_materials():
    def click(app, label):
        next(button for button in app.button if button.label == label).click().run(timeout=20)

    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.button(key="physical_substrate").click().run(timeout=20)
    source = app.button_group(key="substrate_source")
    assert source.options == ["Standard Catalog", "Constant", "Custom Catalog"]
    source.set_value("Custom library").run(timeout=20)

    click(app, "New material")
    app.text_input(key="custom_material_builder_name").set_value("Flat").run(timeout=20)
    app.selectbox(key="custom_material_builder_kind").set_value("Constant").run(timeout=20)
    app.number_input(key="custom_material_builder_in_plane_eps_real").set_value(-3.).run(timeout=20)
    app.number_input(key="custom_material_builder_in_plane_eps_imag").set_value(2.).run(timeout=20)
    click(app, "Save material")
    assert not app.exception
    flat = app.session_state["_custom_materials"]["Flat"]
    assert flat["axes"]["zz"] == {"source_type": "constant", "eps_real": -3.0, "eps_imag": 2.0}
    np.testing.assert_allclose(custom_material(flat, np.array([900., 1000.])).parallel, -3 + 2j)

    # AppTest keeps closed-dialog text widgets, so build the second material on a fresh page.
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.button(key="physical_substrate").click().run(timeout=20)
    app.button_group(key="substrate_source").set_value("Custom library").run(timeout=20)
    click(app, "New material")
    app.text_input(key="custom_material_builder_name").set_value("Measured").run(timeout=20)
    app.selectbox(key="custom_material_builder_kind").set_value("Upload table").run(timeout=20)
    assert not any(box.label == "Tensor type" for box in app.selectbox)
    table = (b"frequency_cm1,eps_parallel_real,eps_parallel_imag,eps_perpendicular_real,eps_perpendicular_imag\n"
             b"900,2,0.1,3,0.2\n1000,4,0.3,5,0.4\n")
    app.file_uploader(key="custom_material_builder_table_upload").set_value(("t.csv", table, "text/csv")).run(timeout=20)
    click(app, "Save material")
    assert not app.exception
    measured = app.session_state["_custom_materials"]["Measured"]
    assert measured["tensor_mode"] == "anisotropic_diagonal"
    assert measured["axes"]["zz"]["table"]["eps_real"] == [2.0, 4.0]
    assert measured["axes"]["xx"] == measured["axes"]["yy"]
    assert measured["axes"]["xx"]["table"]["eps_imag"] == [0.2, 0.4]



def test_tip_cancel_discards_unsaved_value():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)
    next(widget for widget in app.number_input if widget.label == "Radius R (nm)").set_value(31.5).run(timeout=20)
    next(button for button in app.button if button.label == "Cancel").click().run(timeout=20)

    assert app.session_state["tip_R"] == 25.0
    assert app.session_state["editor"] is None
    assert any(button.label.startswith("Tip · R 25 nm") for button in app.button)


def test_material_cancel_discards_unsaved_source():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label.startswith("Substrate ·")).click().run(timeout=20)
    next(box for box in app.button_group if box.label == "Material source").set_value("Constant").run(timeout=20)
    next(button for button in app.button if button.label == "Cancel").click().run(timeout=20)

    assert app.session_state["substrate_source"] == "Catalog"
    assert app.session_state["editor"] is None


def test_sample_editor_reopens_with_the_last_saved_values():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Substrate ·")).click().run(timeout=20)
    next(box for box in app.button_group if box.label == "Material source").set_value("Constant").run(timeout=20)
    next(widget for widget in app.number_input if widget.label == "Re ε").set_value(7.5).run(timeout=20)
    next(button for button in app.button if button.label == "Save substrate").click().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Substrate ·")).click().run(timeout=20)

    assert next(box for box in app.button_group if box.label == "Material source").value == "Constant"
    assert next(widget for widget in app.number_input if widget.label == "Re ε").value == 7.5


def test_tip_preview_uses_compact_tip_parameters_without_annotations():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Tip ·")).click().run(timeout=20)
    next(field for field in app.number_input if field.label == "Length L (nm)").set_value(900).run(timeout=20)
    app.number_input(key="tip_A").set_value(42).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)

    scene = _scene_from_app(app)
    assert "L = 900 nm" in scene and "TA = 42 nm" in scene
    assert "Near-field" not in scene and "hₛ" not in scene and "snom-pointer" not in scene


def test_plane_wave_controls_far_field_reflection():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.session_state["far_enabled"] = False
    app.run(timeout=20)

    scene = _scene_from_app(app)
    assert 'class="snom-plane-wave"' in scene and 'stroke-opacity=".25"' in scene
    assert not app.toggle

    next(button for button in app.button if button.label == "Far-field reflection").click().run(timeout=20)
    next(toggle for toggle in app.toggle if toggle.label == "Include far-field reflection").set_value(True).run(timeout=20)
    next(field for field in app.number_input if field.label == "Incidence angle (degrees)").set_value(60).run(timeout=20)
    next(button for button in app.button if button.label == "Save far-field settings").click().run(timeout=20)

    scene = _scene_from_app(app)
    assert 'stroke-opacity=".82"' in scene and 'transform="translate(0 -55) rotate(15 210 180)"' in scene
    assert '<path d="M102 212 L202 112"/>' in scene


@pytest.mark.parametrize("angle", (0., 45., 60., 90.))
@pytest.mark.parametrize("layer_count", (0, 3))
def test_far_field_hit_area_follows_wave_angle_and_scene_height(monkeypatch, angle, layer_count):
    state = {**_default_setup(), "far_angle": angle}
    layer_ids = list(range(1, layer_count + 1))
    for layer_id in layer_ids:
        state.update({f"layer_{layer_id}_{key}": value for key, value in
                      {"source": "Constant", "real": 2., "imag": 0., "thickness": 20.}.items()})
    monkeypatch.setattr(state_module.st, "session_state", state)
    css, _ = structure_module._scene_markup(layer_ids, state)
    area = re.search(r"\.st-key-physical_far_field\s*\{([^}]+)\}", css)
    assert area is not None, "Far-field hit area must be generated with the drawing"
    polygon = re.search(r"clip-path:\s*polygon\(([^)]+)\)", area[1])
    assert polygon is not None
    corners = np.array([tuple(map(float, re.findall(r"-?[\d.]+", point)))
                        for point in polygon[1].split(",")])
    corners *= np.array([520., 328. + 24 * layer_count]) / 100
    # Undo the SVG's translation and rotation to measure against the red lines.
    theta = np.deg2rad(angle - 45)
    inverse = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = (corners + [0, 55] - [210, 180]) @ inverse + [210, 180]
    along = corners @ np.array([1., -1.]) / np.sqrt(2)
    across = corners @ np.array([1., 1.]) / np.sqrt(2)
    np.testing.assert_allclose([along.min(), along.max()],
                               [-110 / np.sqrt(2) - 8, 90 / np.sqrt(2) + 8], atol=.06)
    np.testing.assert_allclose([across.min(), across.max()],
                               [230 / np.sqrt(2) - 8, 314 / np.sqrt(2) + 8], atol=.06)


def test_isometric_scene_tracks_layer_and_tip_parameters():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)

    scene = _scene_from_app(app)
    assert "Layer 1 · SiC4H · 20 nm" in scene
    assert "Substrate · aSiO2 · semi-infinite" in scene
    assert "Tip · R = 25 nm · L = 600 nm · TA = 40 nm" in scene
    assert 'class="snom-field-arc"' in scene
    assert any(button.label.startswith("Layer 1 ·") for button in app.button)


@pytest.mark.parametrize(
    ("button_prefix", "save_label"),
    (("Substrate ·", "Save substrate"), ("Reference ·", "Save reference")),
)
def test_schematic_material_buttons_open_the_matching_editor(button_prefix, save_label):
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith(button_prefix)).click().run(timeout=20)

    assert any(button.label == save_label for button in app.button)
    assert len(app.get("plotly_chart")) == (4 if button_prefix.startswith("Substrate") else 2)


def test_catalog_material_preview_appears_after_selecting_catalog_for_reference():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Reference ·")).click().run(timeout=20)
    next(box for box in app.button_group if box.label == "Material source").set_value("Catalog").run(timeout=20)

    assert len(app.get("plotly_chart")) == 4


@pytest.mark.parametrize("prefix,material", [("substrate", "SiC4H"), ("reference", "Si")])
def test_catalog_editor_handles_empty_selection(prefix, material):
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.session_state[f"{prefix}_source"] = "Catalog"
    app.session_state[f"{prefix}_catalog"] = None
    app.session_state["editor"] = prefix
    app.run(timeout=20)

    assert not app.exception
    save = next(button for button in app.button if button.label == f"Save {prefix}")
    assert save.disabled
    assert app.selectbox(key=f"{prefix}_catalog").value is None

    app.selectbox(key=f"{prefix}_catalog").set_value(material).run(timeout=20)
    assert not app.exception
    assert not next(button for button in app.button if button.label == f"Save {prefix}").disabled
    assert len(app.get("plotly_chart")) == 4


@pytest.mark.parametrize("button_prefix", ("Substrate ·", "Reference ·"))
def test_material_editors_offer_custom_library_source(button_prefix):
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith(button_prefix)).click().run(timeout=20)

    source = next(box for box in app.button_group if box.label == "Material source")
    assert "Custom Catalog" in source.options


def test_custom_material_builder_opens_from_reference_editor():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label.startswith("Reference ·")).click().run(timeout=20)
    next(box for box in app.button_group if box.label == "Material source").set_value("Custom library").run(timeout=20)
    next(button for button in app.button if button.label == "New material").click().run(timeout=20)

    assert not app.exception
    assert any(field.label == "Material name" for field in app.text_input)
    assert any(box.label == "Tensor type" for box in app.selectbox)
    assert any(toggle.label == "Add Drude term" for toggle in app.toggle)
    assert len(app.get("plotly_chart")) == 2

    next(toggle for toggle in app.toggle if toggle.label == "Add Drude term").set_value(True).run(timeout=20)

    assert any(field.label == "ωₚ (cm⁻¹)" for field in app.number_input)
    assert not app.exception


def test_custom_material_preview_shows_in_plane_and_normal_axes(monkeypatch):
    plots = []
    column = SimpleNamespace(plotly_chart=lambda figure, **kwargs: plots.append(figure))
    view = SimpleNamespace(
        session_state={"frequency_start": 600.0, "frequency_end": 1200.0, "frequency_points": 10},
        caption=lambda *args, **kwargs: None,
        columns=lambda *args, **kwargs: (column, column),
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(editors_module, "st", view)
    axis = {"source_type": "weak_lorentz", "eps_inf": 2.0, "oscillators": []}

    editors_module._render_custom_material_preview({
        "name": "uniaxial",
        "tensor_mode": "anisotropic_diagonal",
        "axes": {"xx": axis, "yy": axis, "zz": {**axis, "eps_inf": 3.0}},
    })

    assert len(plots) == 2
    assert [trace.name for trace in plots[0].data] == ["ε∥ (in-plane)", "εzz (normal)"]


def test_custom_library_materials_resolve_for_all_calculation_roles(monkeypatch, tmp_path):
    definition = {
        "name": "custom",
        "tensor_mode": "isotropic",
        "axes": {
            axis: {
                "source_type": "weak_lorentz",
                "eps_inf": 2.0,
                "oscillators": [{"strength": 1.0, "w0_cm1": 800.0, "gamma_cm1": 5.0}],
            }
            for axis in ("xx", "yy", "zz")
        },
    }
    library = MaterialLibrary(tmp_path)
    library.save(definition)
    state = _default_setup()
    for prefix in ("substrate", "reference", "layer_1"):
        state[f"{prefix}_source"] = "Custom library"
        state[f"{prefix}_custom"] = "custom"
    view = SimpleNamespace(session_state=state, error=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    monkeypatch.setattr("snom_calculator._ui_state._material_library", lambda: library)
    monkeypatch.setattr("snom_calculator._ui_state.st", view)

    materials = [
        _material_from_state(prefix, np.array([700.0, 800.0]), allow_upload=False)
        for prefix in ("substrate", "reference", "layer_1")
    ]

    assert all(isinstance(material, MaterialSpectrum) for material in materials)
    assert [material.label for material in materials] == ["custom", "custom", "custom"]


def test_custom_library_table_material_warns_when_frequency_exceeds_table(monkeypatch, tmp_path):
    table = {"frequency_cm1": [800.0, 900.0], "eps_real": [2.0, 3.0], "eps_imag": [0.1, 0.2]}
    library = MaterialLibrary(tmp_path)
    library.save({
        "name": "table material",
        "tensor_mode": "isotropic",
        "axes": {axis: {"source_type": "table", "table": table} for axis in ("xx", "yy", "zz")},
    })
    warnings = []
    view = SimpleNamespace(
        session_state={"substrate_source": "Custom library", "substrate_custom": "table material"},
        error=lambda *args, **kwargs: None,
        warning=warnings.append,
    )
    monkeypatch.setattr("snom_calculator._ui_state._material_library", lambda: library)
    monkeypatch.setattr("snom_calculator._ui_state.st", view)

    material = _material_from_state("substrate", np.array([700.0, 1000.0]))

    assert isinstance(material, MaterialSpectrum)
    assert len(warnings) == 3


def test_layer_editor_can_remove_layer_and_restore_bulk_mode():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    layer_button = next(button for button in app.button if button.label.startswith("Layer 1 ·"))
    assert "SiC4H" in layer_button.label
    assert "20 nm" in layer_button.label
    layer_button.click().run(timeout=20)
    next(button for button in app.button if button.label == "Remove layer").click().run(timeout=20)

    assert not app.exception
    assert not any(button.label.startswith("Layer 1 ·") for button in app.button)
    assert not next(button for button in app.button if button.label == "Calculate spectrum").disabled


def test_layer_editor_matches_the_material_and_thickness_shown_in_the_stack():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Layer 1 ·")).click().run(timeout=20)

    assert next(box for box in app.selectbox if box.label == "Catalog material").value == "SiC4H"
    assert next(field for field in app.number_input if field.label == "Layer thickness (nm)").value == 20.0


def test_catalog_editor_shows_material_references():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Layer 1 ·")).click().run(timeout=20)

    assert any(caption.value.startswith("References") for caption in app.caption)


def test_catalog_editor_switches_parameter_set_and_references():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    next(button for button in app.button if button.label.startswith("Layer 1 ·")).click().run(timeout=20)

    def references():
        return next(c.value for c in app.caption if c.value.startswith("References"))

    assert "Klein" in references() and "Tiwald" not in references()

    next(box for box in app.selectbox if box.label == "Parameter set").select("Tiwald1999_sample9").run(timeout=20)

    assert "Tiwald" in references() and "Klein" not in references()
    assert app.session_state["layer_1_variant"] == "Tiwald1999_sample9"


def test_catalog_variant_changes_spectrum_and_label():
    from snom_calculator.materials import catalog_material, catalog_variants

    frequency = np.array([900.0])
    assert "Tiwald1999_sample9" in catalog_variants("SiC4H") and catalog_variants("Si") == ()
    default, variant = catalog_material("SiC4H", frequency), catalog_material("SiC4H", frequency, "Tiwald1999_sample9")
    assert variant.label == "SiC4H (Tiwald1999_sample9)" and not np.allclose(default.parallel, variant.parallel)
    assert [ref["key"] for ref in variant.metadata["references"]] == ["Tiwald1999"]
    with pytest.raises(ValueError):
        catalog_material("SiC4H", frequency, "nope")


def test_oscillator_rows_cover_catalog_and_custom_models():
    from snom_calculator._ui_editors import _oscillator_rows
    from snom_calculator.materials import catalog_material

    definition = catalog_material("SiC4H", np.array([1000.0])).metadata["definition"]
    mode = next(iter(definition["axes"].values()))["modes"][0]
    tolo = _oscillator_rows(definition)
    assert tolo[0]["ω₀ / ω_TO (cm⁻¹)"] == mode["to"] and tolo[0]["ω_LO (cm⁻¹)"] == mode["lo"]
    weak = _oscillator_rows({"axes": {"xx": {"source_type": "weak_lorentz", "eps_inf": 2.0, "oscillators": [
        {"strength": 0.5, "w0_cm1": 800.0, "gamma_cm1": 5.0}]}}})
    assert weak == [{"Axis": "xx", "ε∞": 2.0, "ω₀ / ω_TO (cm⁻¹)": 800.0, "ω_LO (cm⁻¹)": None, "S": 0.5,
                     "γ / γ_TO (cm⁻¹)": 5.0, "γ_LO (cm⁻¹)": None}]
    assert _oscillator_rows({"epsilon": 1 + 0j}) == []


def test_layer_numbers_count_from_the_bottom_and_close_gaps_after_removal():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    for _ in range(3):
        app.button(key="physical_add").click().run(timeout=20)
    labels = [button.label.split(" ·")[0] for button in app.button if (button.key or "").startswith("physical_layer_")]
    assert labels == ["Layer 3", "Layer 2", "Layer 1"]  # top to bottom

    app.button(key="physical_remove_layer_1").click().run(timeout=20)  # the bottom layer
    labels = [button.label.split(" ·")[0] for button in app.button if (button.key or "").startswith("physical_layer_")]
    assert labels == ["Layer 2", "Layer 1"]


def test_custom_builder_preserves_imported_citations(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    references = [{"citation": "Measured optical constants", "doi": "10.example/material"}]
    definition = {"name": "Measured", "tensor_mode": "isotropic", "references": references,
                  "axes": {axis: editors_module._default_axis_definition() for axis in ("xx", "yy", "zz")}}
    editors_module._reset_custom_builder("substrate", None, definition)
    restored, error = editors_module._custom_builder_definition()
    assert error is None
    assert restored["references"] == references


def test_custom_builder_rejects_table_mix_without_touching_state(monkeypatch):
    monkeypatch.setattr(st, "session_state", {editors_module._custom_key("name"): "edited"})
    table = {"source_type": "table", "table": {"frequency_cm1": [900.0, 1000.0]}}
    definition = {"name": "Mixed", "tensor_mode": "anisotropic_diagonal",
                  "axes": {"xx": table, "yy": table, "zz": editors_module._default_axis_definition()}}
    with pytest.raises(ValueError, match="cannot be edited"):
        editors_module._reset_custom_builder("substrate", "Mixed", definition)
    assert st.session_state == {editors_module._custom_key("name"): "edited"}


def test_custom_builder_round_trips_constant_and_lorentz_axes(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    constant = {"source_type": "constant", "eps_real": 12.0, "eps_imag": 0.1}
    definition = editors_module.normalize_snom_material_definition(
        {"name": "Mixed", "tensor_mode": "anisotropic_diagonal",
         "axes": {"xx": constant, "yy": constant, "zz": editors_module._default_axis_definition()}})
    editors_module._reset_custom_builder("substrate", "Mixed", definition)
    assert st.session_state[editors_module._custom_key("in_plane_source")] == "Constant"
    restored, error = editors_module._custom_builder_definition()
    assert error is None
    assert restored == definition


def test_custom_material_users_finds_every_selection(monkeypatch):
    monkeypatch.setattr(st, "session_state", {
        "finite_layer_ids": [4, 2], "substrate_source": "Custom library", "substrate_custom": "Glass",
        "layer_2_source": "Custom library", "layer_2_custom": "Glass",
        "layer_4_source": "Catalog", "layer_4_custom": "Glass",  # stale name, not selected
    })
    assert editors_module._custom_material_users("Glass") == {"substrate": "Substrate", "layer_2": "Layer 1"}


@pytest.mark.parametrize("action", ["rename", "delete"])
def test_shared_material_rename_updates_every_role_and_delete_is_blocked(action):
    app = AppTest.from_file(APP_PATH)
    axis = {"source_type": "constant", "eps_real": 4., "eps_imag": .1}
    state = _default_setup()
    state.update({"_custom_materials": {"Shared": {
        "name": "Shared", "tensor_mode": "isotropic", "axes": {name: axis for name in ("xx", "yy", "zz")}}},
        "finite_layer_ids": [1], "next_layer_id": 2, "layer_1_thickness": 20.})
    for prefix in ("substrate", "reference", "layer_1"):
        state[f"{prefix}_source"] = "Custom library"
        state[f"{prefix}_custom"] = "Shared"
    for key, value in state.items():
        app.session_state[key] = value
    app.run(timeout=20)
    app.button(key="physical_substrate").click().run(timeout=20)
    if action == "delete":
        next(button for button in app.button if button.label == "Delete").click().run(timeout=20)
        assert not app.exception
        assert any("used by Reference, Layer 1" in error.value for error in app.error)
        assert not any(button.label == "Confirm delete" for button in app.button)
        assert "Shared" in app.session_state["_custom_materials"]
        return
    next(button for button in app.button if button.label == "Edit").click().run(timeout=20)
    app.text_input(key="custom_material_builder_name").set_value("Renamed").run(timeout=20)
    next(button for button in app.button if button.label == "Save material").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    assert set(app.session_state["_custom_materials"]) == {"Renamed"}
    for prefix in ("substrate", "reference", "layer_1"):
        assert app.session_state[f"{prefix}_custom"] == "Renamed"


def test_tip_editor_exposes_independent_processing_controls():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert not any(box.key == "average_gaps_enabled" for box in app.checkbox)
    app.button(key="physical_tip").click().run(timeout=20)
    assert not any(expander.label == "Advanced" for expander in app.expander)
    assert not any(box.key == "average_gaps_enabled" for box in app.checkbox)
    assert {field.label for field in app.number_input} >= {
        "Min tip-sample gap (nm)", "Max tip-sample gap (nm)", "Min tip-ref gap (nm)",
        "Max tip-ref gap (nm)", "Gaps",
    }
    assert app.checkbox(key="resolution_enabled").value is False


def test_quick_controls_sync_duplicate_selections_and_tip_dialog():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.number_input(key="quick_value_1").set_value(3.5).run(timeout=20)
    assert app.session_state["tip_Hs"] == 3.5
    app.selectbox(key="quick_parameter_2").set_value("tip_Hs").run(timeout=20)
    assert app.number_input(key="quick_value_2").value == 3.5
    app.number_input(key="quick_value_2").set_value(4.5).run(timeout=20)
    assert app.number_input(key="quick_value_1").value == 4.5
    app.button(key="physical_tip").click().run(timeout=20)
    assert app.number_input(key="tip_Hs").value == 4.5
    assert app.number_input(key="quick_value_1").disabled
    app.number_input(key="tip_Hs_max").set_value(6.0).run(timeout=20)
    app.number_input(key="tip_Hs").set_value(6.0).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)
    assert not app.exception
    assert app.number_input(key="quick_value_1").value == 6.0
    assert app.number_input(key="quick_value_2").value == 6.0


@pytest.mark.parametrize("save", [False, True])
def test_tip_processing_values_persist_or_cancel_together(save):
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.button(key="physical_tip").click().run(timeout=20)
    app.number_input(key="tip_Hs_max").set_value(8.0).run(timeout=20)
    app.number_input(key="tip_Hr_max").set_value(7.0).run(timeout=20)
    label = "Save tip" if save else "Cancel"
    next(button for button in app.button if button.label == label).click().run(timeout=20)
    app.run(timeout=20)
    assert app.session_state["tip_Hr_max"] == (7.0 if save else 2.0)
    assert app.session_state["tip_Hs_max"] == (8.0 if save else 2.0)


def test_quick_thickness_updates_setup_and_removed_layer_resets_selection():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.button(key="physical_add").click().run(timeout=20)
    app.selectbox(key="quick_parameter_1").set_value("layer_1_thickness").run(timeout=20)
    app.number_input(key="quick_value_1").set_value(32.0).run(timeout=20)
    assert app.session_state["layer_1_thickness"] == 32.0
    assert "Layer 1 · PMMA · 32 nm" in _scene_from_app(app)
    app.button(key="physical_remove_layer_1").click().run(timeout=20)
    assert not app.exception
    assert app.selectbox(key="quick_parameter_1").value == "tip_Hs"


def test_quick_custom_edit_updates_shared_material_and_stale_oscillator_resets():
    from copy import deepcopy

    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.button(key="physical_add").click().run(timeout=20)
    axis = {"source_type": "weak_lorentz", "eps_inf": 2.0,
            "oscillators": [{"strength": 1.0, "w0_cm1": 800.0, "gamma_cm1": 5.0}]}
    definition = {"name": "Shared", "tensor_mode": "isotropic",
                  "axes": {name: deepcopy(axis) for name in ("xx", "yy", "zz")}}
    app.session_state["_custom_materials"] = {"Shared": definition}
    for prefix in ("substrate", "layer_1"):
        app.session_state[f"{prefix}_source"] = "Custom library"
        app.session_state[f"{prefix}_custom"] = "Shared"
    app.run(timeout=20)
    selected = "custom:Shared:xx:oscillators:0:w0_cm1"
    app.selectbox(key="quick_parameter_1").set_value(selected).run(timeout=20)
    app.selectbox(key="quick_parameter_2").set_value(selected).run(timeout=20)
    app.number_input(key="quick_value_1").set_value(820.0).run(timeout=20)
    assert not app.exception
    assert app.number_input(key="quick_value_2").value == 820.0
    frequency = np.array([850.0])
    spectrum = state_module._base_material_from_state
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(state_module.st, "session_state", app.session_state.filtered_state)
        assert np.allclose(spectrum("substrate", frequency).perpendicular,
                           spectrum("layer_1", frequency).perpendicular)
    for axis in app.session_state["_custom_materials"]["Shared"]["axes"].values():
        assert axis["oscillators"][0]["w0_cm1"] == 820.0
        axis["oscillators"] = []
    app.run(timeout=20)
    assert not app.exception
    assert app.selectbox(key="quick_parameter_1").value == "tip_Hs"
    assert app.selectbox(key="quick_parameter_2").value == "tip_A"


def test_quick_processing_controls_switch_numeric_types_and_retain_disabled_values():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.selectbox(key="quick_parameter_1").set_value("average_gap_count").run(timeout=20)
    assert app.number_input(key="quick_value_1").disabled
    app.button(key="physical_tip").click().run(timeout=20)
    app.number_input(key="tip_Hs_max").set_value(8.0).run(timeout=20)
    next(button for button in app.button if button.label == "Save tip").click().run(timeout=20)
    assert not app.number_input(key="quick_value_1").disabled
    app.number_input(key="quick_value_1").set_value(6).run(timeout=20)
    assert app.session_state["average_gap_count"] == 6
    assert isinstance(app.session_state["average_gap_count"], int)
    app.selectbox(key="quick_parameter_1").set_value("tip_Hs").run(timeout=20)
    app.number_input(key="quick_value_1").set_value(2.75).run(timeout=20)
    assert not app.exception
    assert app.session_state["tip_Hs"] == 2.75


def test_calculation_history_keeps_runs_and_restores_selected_result(monkeypatch):
    state = {
        "results_history": [],
        "history_seq": 0,
        "selected_history_id": None,
    }
    monkeypatch.setattr("snom_calculator._ui_results.st", SimpleNamespace(session_state=state))
    first = {3: object()}
    second = {3: object()}
    materials = {"sample": SimpleNamespace(label="SiC4H")}
    controls = {"preset": "Weak"}

    _append_results_history("bulk", np.array([600.0, 601.0]), materials, controls, first)
    _append_results_history("bulk", np.array([700.0, 701.0]), materials, controls, second)

    assert [entry["id"] for entry in state["results_history"]] == ["run_1", "run_2"]
    assert state["results"] is second

    state["selected_history_id"] = "run_1"
    _sync_selected_history_result()

    assert state["results"] is first


@pytest.mark.parametrize("valid", (True, False))
def test_loading_setup_resets_results_only_after_valid_restore(monkeypatch, valid):
    old_results = {3: object()}
    history = [{"id": "run_1", "results": old_results}]
    setup = _default_setup()
    setup["frequency_start"] = 1100.0 if valid else 1500.0
    state = {
        **_default_setup(),
        "setup_upload": BytesIO(_encode_setup(setup).encode()),
        "results": old_results,
        "results_history": history,
        "selected_history_id": "run_1",
        "history_seq": 1,
        "m_refinement_request": {"error": .2},
        "m_refinement_next": {"error": .2},
    }
    monkeypatch.setattr(st, "session_state", state)

    state_module._load_setup()

    if valid:
        assert state["frequency_start"] == 1100.0
        assert "results" not in state
        assert state["results_history"] == []
        assert state["selected_history_id"] is None
        assert state["history_seq"] == 0
        assert state["setup_error"] is None
        assert "m_refinement_request" not in state
        assert "m_refinement_next" not in state
    else:
        assert state["frequency_start"] == 1000.0
        assert state["results"] is old_results
        assert state["results_history"] is history
        assert state["selected_history_id"] == "run_1"
        assert state["history_seq"] == 1
        assert state["setup_error"]
        assert state["m_refinement_request"] == {"error": .2}
        assert state["m_refinement_next"] == {"error": .2}


def test_setup_upload_is_applied_once_per_file(monkeypatch):
    # After a main-page st.rerun() the browser resends the same file and on_change fires again.
    setup = _default_setup()
    upload = BytesIO(_encode_setup(setup).encode())
    upload.file_id = "file-1"
    state = {**_default_setup(), "setup_upload": upload}
    monkeypatch.setattr(st, "session_state", state)
    state_module._load_setup()
    state["tip_R"] = 40.0
    state_module._load_setup()
    assert state["tip_R"] == 40.0
    upload.file_id = "file-2"
    state_module._load_setup()
    assert state["tip_R"] == 25.0
    assert "_setup_upload_id" not in json.loads(_encode_setup(state))


def test_loading_setup_keeps_other_custom_materials():
    kept = {"name": "kept"}
    state = {**_default_setup(), "_custom_materials": {"kept": kept}}
    loaded = {"name": "loaded", "tensor_mode": "isotropic", "axes": {
        axis: {"source_type": "weak_lorentz", "eps_inf": 2., "oscillators": []}
        for axis in ("xx", "yy", "zz")}}
    setup = {**_default_setup(), "_custom_materials": {"loaded": loaded}}
    _apply_setup(state, json.loads(_encode_setup(setup)))
    assert state["_custom_materials"]["kept"] is kept
    assert state["_custom_materials"]["loaded"]["name"] == "loaded"


def test_m_refinement_plots_each_result_before_asking_again(monkeypatch):
    events = []
    original_warning = st.warning
    original_workspace = results_module._results_workspace

    def warning(message, *args, **kwargs):
        if message.startswith("Warning! Check M vs M/2 did not pass"):
            events.append("prompt")
        return original_warning(message, *args, **kwargs)

    def workspace(results):
        original_workspace(results)
        if results:
            events.append("plots")

    def calculate(model, frequency, materials, controls, harmonics):
        M = controls["numerical"].M
        return {harmonic: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=np.full(frequency.size, M),
            reference_harmonic=np.ones(frequency.size), contrast=np.full(frequency.size, M),
            permittivities={}, metadata={"model": model, "M": M, "M_check_max_error": .25})
            for harmonic in harmonics}

    monkeypatch.setattr(controls_module, "_run_calculations", calculate)
    monkeypatch.setattr(st, "warning", warning)
    monkeypatch.setattr(app_module, "_results_workspace", workspace)
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=20)
    for iteration in range(2):
        assert not app.exception
        # A dialog button reruns the dialog, then the whole app: check the final run.
        assert events[-2:] == ["plots", "prompt"]
        assert len(app.get("plotly_chart")) == 2
        M = app.session_state["results"][3].metadata["M"]
        np.testing.assert_array_equal(app.session_state["results"][3].amplitude, M)
        assert len(app.session_state["results_history"]) == iteration + 1
        if iteration == 0:
            first_plot = app.get("plotly_chart")[0].proto.spec
            events.clear()
            app.button(key="double_m").click().run(timeout=20)
        else:
            assert app.get("plotly_chart")[0].proto.spec != first_plot


def test_calculation_action_retains_both_runs_and_plots_latest(monkeypatch):
    calls = []

    def calculate(model, frequency, materials, controls, harmonics):
        calls.append(model)
        signal = np.full(frequency.size, len(calls), dtype=complex)
        return {
            harmonic: SpectrumResult.from_contrast(
                frequency_cm1=frequency, sample_harmonic=signal,
                reference_harmonic=np.ones(frequency.size), contrast=signal,
                permittivities={}, metadata={"model": model},
            )
            for harmonic in harmonics
        }

    monkeypatch.setattr(controls_module, "_run_calculations", calculate)
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=20)
    first_plot = json.loads(app.get("plotly_chart")[0].proto.spec)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=20)
    assert json.loads(app.get("plotly_chart")[0].proto.spec) != first_plot

    assert not app.exception
    assert not app.error
    assert calls == ["bulk", "bulk"]
    assert app.session_state["selected_history_id"] == "run_2"
    history = app.session_state["results_history"]
    assert [entry["id"] for entry in history] == ["run_1", "run_2"]
    np.testing.assert_array_equal(history[0]["results"][3].amplitude, 1.0)
    np.testing.assert_array_equal(history[1]["results"][3].amplitude, 2.0)
    np.testing.assert_array_equal(app.session_state["results"][3].amplitude, 2.0)

    latest_plots = [chart.proto.spec for chart in app.get("plotly_chart")]
    next(box for box in app.selectbox if box.label == "Saved spectrum").select("run_1").run(timeout=20)
    assert not app.exception
    assert app.session_state["selected_history_id"] == "run_1"
    np.testing.assert_array_equal(app.session_state["results"][3].amplitude, 1.0)
    assert json.loads(app.get("plotly_chart")[0].proto.spec) == first_plot
    assert [chart.proto.spec for chart in app.get("plotly_chart")] != latest_plots

    app.button(key="physical_add").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["selected_history_id"] == "run_1"
    np.testing.assert_array_equal(app.session_state["results"][3].amplitude, 1.0)
    assert json.loads(app.get("plotly_chart")[0].proto.spec) == first_plot

    next(box for box in app.selectbox if box.label == "Saved spectrum").select("run_2").run(timeout=20)
    assert not app.exception
    assert app.session_state["selected_history_id"] == "run_2"
    np.testing.assert_array_equal(app.session_state["results"][3].amplitude, 2.0)
    assert [chart.proto.spec for chart in app.get("plotly_chart")] == latest_plots
    assert calls == ["bulk", "bulk"]


@pytest.mark.parametrize("action", ("add", "remove_scene", "remove_editor"))
def test_layer_changes_preserve_harmonic_selections(action):
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    if action != "add":
        app.button(key="physical_add").click().run(timeout=20)
    app.checkbox(key="harmonic_1").check()
    app.checkbox(key="harmonic_3").uncheck()
    app.checkbox(key="harmonic_4").check().run(timeout=20)

    if action == "add":
        app.button(key="physical_add").click().run(timeout=20)
    elif action == "remove_scene":
        app.button(key="physical_remove_layer_1").click().run(timeout=20)
    else:
        app.button(key="physical_layer_1").click().run(timeout=20)
        next(button for button in app.button if button.label == "Remove layer").click().run(timeout=20)

    assert not app.exception
    assert app.session_state["finite_layer_ids"] == ([1] if action == "add" else [])
    assert [app.checkbox(key=f"harmonic_{n}").value for n in (1, 2, 3, 4)] == [True, False, False, True]


def test_selector_reveals_precalculated_harmonics_without_rerunning():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(box for box in app.checkbox if box.label == "n = 2").check().run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click().run(timeout=40)
    next(box for box in app.checkbox if box.label == "n = 4").check().run(timeout=20)

    assert not app.exception
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert [[trace["name"] for trace in figure["data"]] for figure in figures] == [
        ["n = 2", "n = 3", "n = 4"],
        ["n = 2", "n = 3", "n = 4"],
    ]


def test_default_bulk_calculation_populates_two_plots_and_export():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Calculate spectrum").click()
    app.run(timeout=40)

    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 2
    assert any(box.label == "Saved spectrum" for box in app.selectbox)
    assert any(button.label == "Export harmonics CSV" for button in app.get("download_button"))


def test_one_layer_workbench_calculates():
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    next(button for button in app.button if button.label == "Add finite layer").click()
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Calculate spectrum").click()
    app.run(timeout=40)

    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 2


def test_history_is_capped_to_the_most_recent_runs(monkeypatch):
    state = {"results_history": [], "history_seq": 0, "selected_history_id": None}
    monkeypatch.setattr("snom_calculator._ui_results.st", SimpleNamespace(session_state=state))
    materials = {"sample": SimpleNamespace(label="SiC4H")}
    for _ in range(results_module.MAX_HISTORY_RUNS + 5):
        _append_results_history("bulk", np.array([600.0, 601.0]), materials, {}, {3: object()})

    ids = [entry["id"] for entry in state["results_history"]]
    assert len(ids) == results_module.MAX_HISTORY_RUNS
    assert ids[-1] == f"run_{results_module.MAX_HISTORY_RUNS + 5}"
    assert state["selected_history_id"] == ids[-1]


def test_invalid_frequency_range_disables_calculation():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.session_state["frequency_start"] = 1200.0
    app.session_state["frequency_end"] = 600.0
    app.run(timeout=20)

    assert next(button for button in app.button if button.label == "Calculate spectrum").disabled
    assert not app.session_state["results_history"]


def test_removing_a_layer_discards_its_state():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    assert "layer_1_thickness" in app.session_state
    next(button for button in app.button if button.label.startswith("Layer 1 ·")).click().run(timeout=20)
    next(button for button in app.button if button.label == "Remove layer").click().run(timeout=20)

    assert "layer_1_thickness" not in app.session_state
    assert "layer_1_catalog" not in app.session_state


def test_legacy_custom_precision_migrates_only_once():
    app = AppTest.from_file(APP_PATH)
    app.session_state["bulk_custom_M"] = 30
    app.session_state["bulk_custom_quad"] = 16
    app.session_state["bulk_custom_N"] = 8
    app.run(timeout=20)
    assert (app.session_state["bulk_custom_M"], app.session_state["bulk_custom_N"]) == (80, 80)
    assert "bulk_custom_quad" not in app.session_state

    app.session_state["bulk_custom_M"] = 30
    app.session_state["bulk_custom_quad"] = 16
    app.session_state["bulk_custom_N"] = 8
    app.run(timeout=20)
    assert app.session_state["bulk_custom_M"] == 30


def test_inverted_gap_range_shows_an_inline_error():
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.session_state["tip_Hs"] = 5.0
    app.session_state["tip_Hs_max"] = 2.0
    app.run(timeout=20)

    assert any("Maximum gap must be at least" in error.value for error in app.error)


def test_automatic_precision_picks_k_from_the_film_and_falls_back_to_fim():
    app = AppTest.from_file(APP_PATH)
    app.session_state["frequency_start"], app.session_state["frequency_end"] = 780., 990.
    app.session_state["substrate_source"], app.session_state["substrate_catalog"] = "Catalog", "Au"
    app.session_state["tip_Hs"] = app.session_state["tip_Hs_max"] = 1.
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Add finite layer").click().run(timeout=20)
    app.session_state["layer_1_source"], app.session_state["layer_1_catalog"] = "Catalog", "aSiO2"
    app.session_state["layer_1_thickness"] = 50.
    app.run(timeout=20)
    assert not app.exception
    assert any(caption.value.startswith("Automatic · M") and caption.value.endswith("· K 4") for caption in app.caption)

    app.session_state["layer_1_source"], app.session_state["layer_1_catalog"] = "Catalog", "SiC3C"
    app.session_state["layer_1_thickness"] = 5.
    app.run(timeout=20)
    assert not app.exception
    assert any(caption.value.startswith("FIM:") for caption in app.caption)
    assert not any(" · K " in caption.value for caption in app.caption)


def test_solver_warnings_are_shown_after_a_calculation(monkeypatch):
    import snom_calculator._ui_controls as controls_module

    def fake_run(*args):
        warning_module.warn("Legendre Q_8 did not converge")
        warning_module.warn("Legendre Q_8 did not converge")
        return {}

    monkeypatch.setattr(controls_module, "_run_calculations", fake_run)
    monkeypatch.setattr(controls_module, "_validate_structure", lambda *args: None)
    monkeypatch.setattr(controls_module, "_append_results_history", lambda *args: None)
    app = AppTest.from_string('''
import numpy as np
import streamlit as st
from snom_calculator._ui_controls import _calculation_action
from snom_calculator.models import TipParameters
st.session_state.setdefault("frequency_start", 900.)
st.session_state.setdefault("frequency_end", 1000.)
_calculation_action("bulk", np.array([900., 1000.]), {}, {"tip": TipParameters()})
''').run()
    next(button for button in app.button if button.label == "Calculate spectrum").click().run()
    assert not app.exception
    assert [warning.value for warning in app.warning] == ["Legendre Q_8 did not converge"]
