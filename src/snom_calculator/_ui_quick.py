"""Two numeric shortcuts into the existing setup and session material library."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import math

import streamlit as st

from .material_library import material_library_for_state


@dataclass(frozen=True)
class _QuickParameter:
    label: str
    key: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int = 1.0
    disabled: bool = False
    material: str = ""
    axes: tuple[str, ...] = ()
    path: tuple[str | int, ...] = ()

    def value(self, state: Mapping) -> float | int:
        if not self.material:
            return state[self.key]
        definition = state["_custom_materials"][self.material]
        value = definition["drude"][self.axes[0]] if self.path[0] == "drude" else definition["axes"][self.axes[0]]
        for part in self.path[1:] if self.path[0] == "drude" else self.path:
            value = value[part]
        return float(value)


def _quick_parameters(state: Mapping) -> dict[str, _QuickParameter]:
    parameters: dict[str, _QuickParameter] = {}

    def add(key, label, minimum=None, step=1.0, maximum=None, disabled=False):
        parameters[key] = _QuickParameter(label, key, minimum, maximum, step, disabled)

    add("tip_R", "Tip · Radius R (nm)", 0.1)
    add("tip_L", "Tip · Length L (nm)", 0.0)
    add("tip_A", "Tip · Tapping amplitude (nm)", 0.0)
    for key, label, count_key in (("tip_Hs", "tip-sample", "average_gap_count"),
                                  ("tip_Hr", "tip-ref", "tip_reference_gap_count")):
        add(key, f"Tip · Min {label} gap (nm)", 0.01)
        add(f"{key}_max", f"Tip · Max {label} gap (nm)", 0.01)
        add(count_key, f"Tip · {label.capitalize()} gaps", 2, 1, 25,
            state[key] == state[f"{key}_max"])
    add("tip_eps_real", "Tip · Re ε")
    add("tip_eps_imag", "Tip · Im ε")
    add("resolution_fwhm", "Tip · Gaussian FWHM (cm⁻¹)", 0.01, disabled=not state["resolution_enabled"])
    prefixes = ["substrate"]

    layer_ids = state["finite_layer_ids"]
    for index, layer_id in enumerate(layer_ids):
        prefix = f"layer_{layer_id}"
        prefixes.append(prefix)
        add(f"{prefix}_thickness", f"Layer {len(layer_ids) - index} · Thickness (nm)", 0.0)

    names = dict.fromkeys(state.get(f"{prefix}_custom", "") for prefix in prefixes
                          if state.get(f"{prefix}_source") == "Custom library")
    for name in names:
        definition = state.get("_custom_materials", {}).get(name)
        if definition is None:
            continue
        axes, drude = definition["axes"], definition.get("drude", {})
        groups = (("xx", "yy", "zz"),) if definition["tensor_mode"] == "isotropic" else (("xx", "yy"), ("zz",))
        for group in groups:
            axis = group[0]
            label = f"{name} · ε{'/'.join(group)}"

            def custom(path, field_label, minimum=0.0, step=0.1):
                identifier = f"custom:{name}:{axis}:" + ":".join(map(str, path))
                parameters[identifier] = _QuickParameter(
                    f"{label} · {field_label}", minimum=minimum, step=step,
                    disabled=path[0] == "drude" and drude[axis].get("wp_cm1", 0.0) == 0.0,
                    material=name, axes=group, path=path)

            response = axes[axis]
            if response["source_type"] == "constant":
                custom(("eps_real",), "Re ε", minimum=None)
                custom(("eps_imag",), "Im ε", minimum=None)
            elif response["source_type"] != "table":
                custom(("eps_inf",), "ε∞")
                fields = (("strength", "Strength", 0.0, 0.1), ("w0_cm1", "ω₀ (cm⁻¹)", 0.0, 1.0),
                          ("gamma_cm1", "γ (cm⁻¹)", 0.0, 0.1)) if response["source_type"] == "weak_lorentz" else (
                          ("to_cm1", "TO (cm⁻¹)", 0.0, 1.0), ("lo_cm1", "LO (cm⁻¹)", 0.0, 1.0),
                          ("gamma_cm1", "γ (cm⁻¹)", 0.0, 0.1))
                for index, _ in enumerate(response["oscillators"]):
                    for field, field_label, minimum, step in fields:
                        custom(("oscillators", index, field), f"Oscillator {index + 1} · {field_label}", minimum, step)
            for field, field_label, step in (("wp_cm1", "ωₚ (cm⁻¹)", 10.0), ("gamma_cm1", "γD (cm⁻¹)", 1.0)):
                if field in drude.get(axis, {}):
                    custom(("drude", field), f"Drude {field_label}", step=step)
    return parameters


def _set_quick_parameter(state: MutableMapping, parameter: _QuickParameter, value: float | int) -> None:
    if parameter.disabled or state.get("editor"):
        raise ValueError("This parameter is currently disabled.")
    if (not math.isfinite(value)
            or parameter.minimum is not None and value < parameter.minimum
            or parameter.maximum is not None and value > parameter.maximum
            or isinstance(parameter.step, int) and int(value) != value):
        raise ValueError("Enter a valid value within the parameter limits.")
    if not parameter.material:
        if parameter.key in ("tip_Hs", "tip_Hr") and state[f"{parameter.key}_max"] == state[parameter.key]:
            state[f"{parameter.key}_max"] = float(value)
        state[parameter.key] = int(value) if isinstance(parameter.step, int) else float(value)
        return
    library = material_library_for_state(state)
    definition = library.load(parameter.material)
    for axis in parameter.axes:
        target = definition["drude"][axis] if parameter.path[0] == "drude" else definition["axes"][axis]
        path = parameter.path[1:] if parameter.path[0] == "drude" else parameter.path
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = float(value)
    library.save(definition)


def _quick_value_changed(slot: int) -> None:
    state = st.session_state
    parameter = _quick_parameters(state).get(state[f"quick_parameter_{slot}"])
    try:
        if parameter is None:
            raise ValueError("This parameter is no longer available.")
        _set_quick_parameter(state, parameter, state[f"quick_value_{slot}"])
    except (OSError, ValueError) as exc:
        state[f"quick_error_{slot}"] = str(exc)


def _render_quick_controls() -> None:
    st.markdown("**Quick controls**")
    parameters = _quick_parameters(st.session_state)
    for slot, default in ((1, "tip_Hs"), (2, "tip_A")):
        selector_key, value_key = f"quick_parameter_{slot}", f"quick_value_{slot}"
        if st.session_state.get(selector_key) not in parameters:
            st.session_state[selector_key] = default
        name_col, value_col = st.columns(2)
        selected = name_col.selectbox(f"Parameter {slot}", tuple(parameters), key=selector_key,
                                      format_func=lambda key: parameters[key].label,
                                      disabled=bool(st.session_state.get("editor")),
                                      label_visibility="collapsed")
        parameter = parameters[selected]
        st.session_state[value_key] = parameter.value(st.session_state)
        value_col.number_input(parameter.label, key=value_key, min_value=parameter.minimum,
                               max_value=parameter.maximum, step=parameter.step,
                               format="%d" if isinstance(parameter.step, int) else "%.6g",
                               disabled=parameter.disabled or bool(st.session_state.get("editor")),
                               help=("Set different minimum and maximum gaps to enable averaging."
                                     if selected in ("average_gap_count", "tip_reference_gap_count")
                                     else "Enable this option in the tip or material editor.") if parameter.disabled else None,
                               on_change=_quick_value_changed, args=(slot,),
                               label_visibility="collapsed")
        if error := st.session_state.pop(f"quick_error_{slot}", None):
            st.error(error)
    for key in ("tip_Hs", "tip_Hr"):
        if st.session_state[f"{key}_max"] < st.session_state[key]:
            st.error("Maximum gap must be at least the minimum gap.")
