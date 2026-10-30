"""Quick edits must change validated setup values, including shared material axes."""

from copy import deepcopy
import json

import numpy as np
import pytest

from snom_calculator.materials import custom_material
from snom_calculator.setup import _default_setup, _encode_setup


def _custom_state(source="weak_lorentz", mode="isotropic"):
    oscillator = ({"strength": 1.0, "w0_cm1": 800.0, "gamma_cm1": 5.0}
                  if source == "weak_lorentz" else
                  {"to_cm1": 800.0, "lo_cm1": 900.0, "gamma_cm1": 5.0})
    axis = {"source_type": source, "eps_inf": 2.0, "oscillators": [oscillator]}
    definition = {"name": "Shared", "tensor_mode": mode,
                  "axes": {name: deepcopy(axis) for name in ("xx", "yy", "zz")},
                  "drude": {name: {"wp_cm1": 100.0, "gamma_cm1": 5.0}
                            for name in ("xx", "yy", "zz")}}
    if mode != "isotropic":
        definition["axes"]["zz"]["eps_inf"] = 3.0
    state = {**_default_setup(), "_custom_materials": {"Shared": definition},
             "substrate_source": "Custom library", "substrate_custom": "Shared",
             "finite_layer_ids": [1], "layer_1_source": "Custom library",
             "layer_1_custom": "Shared", "layer_1_thickness": 20.0}
    return state


@pytest.mark.parametrize("source,field", [("weak_lorentz", "w0_cm1"), ("strong_lorentz", "to_cm1")])
@pytest.mark.parametrize("mode,axes", [("isotropic", ("xx", "yy", "zz")),
                                      ("anisotropic_diagonal", ("xx", "yy"))])
def test_shared_oscillator_edit_preserves_linked_axes_and_changes_spectrum(source, field, mode, axes):
    from snom_calculator._ui_quick import _quick_parameters, _set_quick_parameter

    state = _custom_state(source, mode)
    before = custom_material(state["_custom_materials"]["Shared"], np.array([850.0])).perpendicular.copy()
    parameters = _quick_parameters(state)
    parameter = parameters[f"custom:Shared:xx:oscillators:0:{field}"]
    _set_quick_parameter(state, parameter, 810.0)
    definition = state["_custom_materials"]["Shared"]
    for axis in axes:
        assert definition["axes"][axis]["oscillators"][0][field] == 810.0
    if mode != "isotropic":
        assert definition["axes"]["zz"]["oscillators"][0][field] == 800.0
    after = custom_material(definition, np.array([850.0])).perpendicular
    assert not np.allclose(before, after)
    assert sum(key.endswith(f":{field}") for key in parameters) == (1 if mode == "isotropic" else 2)


def test_invalid_quick_material_edit_is_atomic_and_drude_edits_link_axes():
    from snom_calculator._ui_quick import _quick_parameters, _set_quick_parameter

    state = _custom_state()
    parameters = _quick_parameters(state)
    snapshot = deepcopy(state)
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            _set_quick_parameter(state, parameters["custom:Shared:xx:oscillators:0:w0_cm1"], value)
        assert state == snapshot
    _set_quick_parameter(state, parameters["custom:Shared:xx:drude:wp_cm1"], 150.0)
    assert all(term["wp_cm1"] == 150.0 for term in state["_custom_materials"]["Shared"]["drude"].values())


def test_quick_options_respect_sources():
    from snom_calculator._ui_quick import _quick_parameters

    state = _default_setup()
    parameters = _quick_parameters(state)
    assert "tip_eps_real" in parameters
    assert not any("drude" in key for key in parameters)
    assert parameters["average_gap_count"].disabled
    assert parameters["resolution_fwhm"].disabled
    state["tip_Hs_max"] = 8.0
    assert not _quick_parameters(state)["average_gap_count"].disabled


def test_quick_selector_state_is_not_exported_but_edits_are():
    state = _custom_state()
    state.update(quick_parameter_1="tip_A", quick_value_1=45.0, quick_error_1="error", tip_A=45.0)
    saved = json.loads(_encode_setup(state))
    assert not any(key.startswith("quick_") for key in saved)
    assert saved["tip_A"] == 45.0
    assert "Shared" in saved["material_definitions"]


def test_custom_table_axes_have_no_oscillator_controls_and_inactive_drude_is_disabled():
    from snom_calculator._ui_quick import _quick_parameters, _set_quick_parameter

    state = _custom_state()
    definition = state["_custom_materials"]["Shared"]
    for axis in ("xx", "yy", "zz"):
        definition["axes"][axis] = {"source_type": "table", "table": {
            "frequency_cm1": [800.0, 900.0], "eps_real": [2.0, 2.0], "eps_imag": [0.1, 0.1]}}
        definition["drude"][axis]["wp_cm1"] = 0.0
    parameters = _quick_parameters(state)
    assert not any("oscillators" in key or "eps_inf" in key for key in parameters)
    parameter = parameters["custom:Shared:xx:drude:wp_cm1"]
    assert parameter.disabled
    with pytest.raises(ValueError):
        _set_quick_parameter(state, parameter, 100.0)


def test_quick_edits_reject_out_of_bounds_counts_and_edits_during_dialogs():
    from snom_calculator._ui_quick import _quick_parameters, _set_quick_parameter

    state = _default_setup()
    state["tip_Hs_max"] = 8.0
    parameter = _quick_parameters(state)["average_gap_count"]
    for value in (1, 26, 2.5):
        with pytest.raises(ValueError):
            _set_quick_parameter(state, parameter, value)
    assert state["average_gap_count"] == 4
    state["editor"] = "tip"
    with pytest.raises(ValueError):
        _set_quick_parameter(state, _quick_parameters(state)["tip_R"], 30.0)
    assert state["tip_R"] == 25.0


def test_quick_gap_options_match_tip_editor_order_and_names():
    from snom_calculator._ui_quick import _quick_parameters

    parameters = _quick_parameters(_default_setup())
    gap_keys = ["tip_Hs", "tip_Hs_max", "average_gap_count",
                "tip_Hr", "tip_Hr_max", "tip_reference_gap_count"]
    keys = list(parameters)
    start = keys.index("tip_Hs")
    assert keys[start:start + 6] == gap_keys
    assert [parameters[key].label for key in gap_keys] == [
        "Tip · Min tip-sample gap (nm)", "Tip · Max tip-sample gap (nm)",
        "Tip · Tip-sample gaps", "Tip · Min tip-ref gap (nm)",
        "Tip · Max tip-ref gap (nm)", "Tip · Tip-ref gaps"]


def test_reference_quick_range_enables_count_and_updates_shared_state():
    from snom_calculator._ui_quick import _quick_parameters, _set_quick_parameter

    state = _default_setup()
    assert _quick_parameters(state)["tip_reference_gap_count"].disabled
    _set_quick_parameter(state, _quick_parameters(state)["tip_Hr_max"], 8.0)
    parameters = _quick_parameters(state)
    assert not parameters["tip_reference_gap_count"].disabled
    assert parameters["average_gap_count"].disabled
    _set_quick_parameter(state, parameters["tip_reference_gap_count"], 6)
    assert state["tip_reference_gap_count"] == 6
    _set_quick_parameter(state, parameters["tip_Hr_max"], state["tip_Hr"])
    assert _quick_parameters(state)["tip_reference_gap_count"].disabled
