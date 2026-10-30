"""Tip, far-field and material dialogs, including custom-material previews."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ._ui_save import save_button

from .custom_materials import load_material_definition_json
from .material_library import normalize_snom_material_definition
from .materials import MaterialSpectrum, catalog_material, catalog_names, catalog_variant_label, catalog_variants, custom_material, parse_material_axes
from .setup import _CUSTOM_BUILDER_PREFIX
from ._ui_results import _style_plot
from ._ui_state import _material_library, _persist_setup_state, _remove_layer, _state_drude


def _open_editor(editor: object) -> None:
    state = st.session_state
    for key in tuple(state):
        if str(key).startswith("draft_"):
            state.pop(key)
    state["editor_snapshot"] = {
        key: value for key, value in state.items()
        if str(key).startswith(("tip_", "far_", "substrate_", "reference_", "layer_", "average_gap"))
        and not str(key).endswith("_upload")
    }
    state["editor"] = editor


def _dismiss_editor() -> None:
    state = st.session_state
    state["editor_restore"] = state.pop("editor_snapshot", {})
    state["editor_cleanup"] = True
    state["editor"] = None


def _save_editor() -> None:
    st.session_state.pop("editor_snapshot", None)
    st.session_state["editor"] = None


@st.dialog("Edit tip", on_dismiss=_dismiss_editor)
def _tip_editor() -> None:
    _persist_setup_state()
    row1 = st.columns(3, gap="small")
    row1[0].number_input("Radius R (nm)", min_value=0.1, step=1.0, value=float(st.session_state["tip_R"]), key="tip_R")
    row1[1].number_input("Length L (nm)", min_value=0.0, step=1.0, value=float(st.session_state["tip_L"]), key="tip_L",
                         help="A spheroid requires L > 2R.")
    row1[2].number_input("Tapping amplitude A (nm)", min_value=0.0, step=1.0, value=float(st.session_state["tip_A"]), key="tip_A")
    invalid_gaps = False
    for key, label, count_key in (("tip_Hs", "tip-sample", "average_gap_count"),
                                  ("tip_Hr", "tip-ref", "tip_reference_gap_count")):
        columns = st.columns(3, gap="small")
        columns[0].number_input(f"Min {label} gap (nm)", min_value=0.01, step=1.0, key=key)
        columns[1].number_input(f"Max {label} gap (nm)", min_value=0.01, step=1.0, key=f"{key}_max")
        columns[2].number_input("Gaps", min_value=2, max_value=25, step=1, key=count_key,
                                disabled=st.session_state[key] == st.session_state[f"{key}_max"])
        if st.session_state[f"{key}_max"] < st.session_state[key]:
            st.error(f"Maximum {label} gap must be at least the minimum gap.")
            invalid_gaps = True
    st.caption("Equal min/max uses one gap. Otherwise, complex sample and reference signals "
               "are averaged separately before normalization.")
    _render_constant_tip_epsilon()
    if st.button("Save tip", type="primary", disabled=invalid_gaps):
        _save_editor()
        st.rerun()
    if st.button("Cancel"):
        _dismiss_editor()
        st.rerun()


def _render_constant_tip_epsilon() -> None:
    row = st.columns(2, gap="small")
    row[0].number_input("Tip Re ε", value=float(st.session_state["tip_eps_real"]), format="%.3e", key="tip_eps_real")
    row[1].number_input("Tip Im ε", value=float(st.session_state["tip_eps_imag"]), format="%.3e", key="tip_eps_imag")


@st.dialog("Far-field reflection", on_dismiss=_dismiss_editor)
def _far_field_editor() -> None:
    enabled = st.toggle("Include far-field reflection", key="far_enabled")
    st.number_input(
        "Incidence angle (degrees)",
        min_value=0.1,
        max_value=89.9,
        key="far_angle",
        disabled=not enabled,
    )
    st.number_input("Reflection coefficient ζ", key="far_coefficient", disabled=not enabled)
    if st.button("Save far-field settings", type="primary"):
        _save_editor()
        st.rerun()
    if st.button("Cancel"):
        _dismiss_editor()
        st.rerun()
    st.markdown(
        '<div style="text-align:right; color:#5c6875; font-size:.8rem;">E = E₀(1 + ζrₚ)² sin φ</div>',
        unsafe_allow_html=True,
    )


def _custom_key(name: str) -> str:
    return f"{_CUSTOM_BUILDER_PREFIX}{name}"


def _default_axis_definition() -> dict[str, object]:
    return {
        "source_type": "weak_lorentz",
        "eps_inf": 1.0,
        "oscillators": [{"strength": 1.0, "w0_cm1": 800.0, "gamma_cm1": 5.0}],
    }


def _builder_kind(definition: Mapping[str, Any]) -> str:
    sources = {str(axis["source_type"]) for axis in definition["axes"].values()}
    if sources == {"constant"}:
        return "Constant"
    if sources == {"table"}:
        return "Upload table"
    if sources <= {"weak_lorentz", "strong_lorentz", "constant"}:
        return "Lorentz model"  # a constant axis becomes the Lorentz builder's Constant response
    raise ValueError("Materials mixing table and model axes cannot be edited. Create the material again.")


def _reset_custom_builder(prefix: str, original_name: str | None, definition: Mapping[str, Any]) -> None:
    kind = _builder_kind(definition)  # raises before any builder state is touched
    for key in tuple(st.session_state):
        if key.startswith(_CUSTOM_BUILDER_PREFIX):
            st.session_state.pop(key)
    st.session_state[_custom_key("target")] = (prefix, original_name)
    st.session_state[_custom_key("name")] = str(definition["name"])
    st.session_state[_custom_key("references")] = definition.get("references", [])
    mode = str(definition["tensor_mode"])
    drude = definition.get("drude", {})
    st.session_state[_custom_key("mode")] = "Isotropic" if mode == "isotropic" else "Uniaxial"
    st.session_state[_custom_key("kind")] = kind
    st.session_state[_custom_key("upload_mode")] = st.session_state[_custom_key("mode")]
    for role, axis_name in (("in_plane", "xx"), ("normal", "zz")):
        axis = definition["axes"][axis_name]
        source = str(axis["source_type"])
        st.session_state[_custom_key(f"{role}_source")] = {
            "weak_lorentz": "Weak Lorentz",
            "strong_lorentz": "Strong Lorentz",
            "constant": "Constant",
        }[source] if kind == "Lorentz model" else "Weak Lorentz"
        term = drude.get(axis_name, {}) if isinstance(drude, Mapping) else {}
        wp = float(term.get("wp_cm1", 0.0)) if isinstance(term, Mapping) else 0.0
        st.session_state[_custom_key(f"{role}_drude_enabled")] = wp > 0.0
        st.session_state[_custom_key(f"{role}_drude_wp_cm1")] = wp
        st.session_state[_custom_key(f"{role}_drude_gamma_cm1")] = float(term.get("gamma_cm1", 0.0)) if isinstance(term, Mapping) else 0.0
        if source == "table":
            st.session_state[_custom_key(f"{role}_table")] = axis["table"]
            continue
        if source == "constant":
            st.session_state[_custom_key(f"{role}_eps_real")] = float(axis["eps_real"])
            st.session_state[_custom_key(f"{role}_eps_imag")] = float(axis["eps_imag"])
            continue
        st.session_state[_custom_key(f"{role}_eps_inf")] = float(axis["eps_inf"])
        st.session_state[_custom_key(f"{role}_count")] = len(axis["oscillators"])
        for index, oscillator in enumerate(axis["oscillators"]):
            for field, value in oscillator.items():
                st.session_state[_custom_key(f"{role}_{field}_{index}")] = float(value)


def _axis_definition_from_builder(role: str) -> dict[str, object] | None:
    kind = st.session_state[_custom_key("kind")]
    if kind == "Upload table":
        table = st.session_state.get(_custom_key(f"{role}_table"))
        return {"source_type": "table", "table": table} if table is not None else None
    source = "Constant" if kind == "Constant" else str(st.session_state[_custom_key(f"{role}_source")])
    if source == "Constant":
        return {"source_type": "constant", "eps_real": float(st.session_state[_custom_key(f"{role}_eps_real")]),
                "eps_imag": float(st.session_state[_custom_key(f"{role}_eps_imag")])}
    count = int(st.session_state[_custom_key(f"{role}_count")])
    oscillators: list[dict[str, float]] = []
    fields = ("strength", "w0_cm1", "gamma_cm1") if source == "Weak Lorentz" else ("to_cm1", "lo_cm1", "gamma_cm1")
    for index in range(count):
        oscillators.append(
            {field: float(st.session_state[_custom_key(f"{role}_{field}_{index}")]) for field in fields}
        )
    return {
        "source_type": "weak_lorentz" if source == "Weak Lorentz" else "strong_lorentz",
        "eps_inf": float(st.session_state[_custom_key(f"{role}_eps_inf")]),
        "oscillators": oscillators,
    }


def _drude_term_from_builder(role: str) -> dict[str, float] | None:
    if not st.session_state.get(_custom_key(f"{role}_drude_enabled"), False):
        return None
    return {
        "wp_cm1": float(st.session_state[_custom_key(f"{role}_drude_wp_cm1")]),
        "gamma_cm1": float(st.session_state[_custom_key(f"{role}_drude_gamma_cm1")]),
    }


def _render_drude_controls(prefix: str) -> None:
    if st.toggle("Add Drude term", key=f"{prefix}_drude_enabled"):
        drude = st.columns(2, gap="small")
        drude[0].number_input("ωₚ (cm⁻¹)", min_value=0.0, step=10.0, key=f"{prefix}_drude_wp_cm1")
        drude[1].number_input("γD (cm⁻¹)", min_value=0.0, step=1.0, key=f"{prefix}_drude_gamma_cm1")
        st.latex(r"\varepsilon_D(\omega)=-\frac{\omega_p^2}{\omega^2+i\gamma_D\omega}")


def _render_axis_builder(role: str, label: str) -> None:
    kind = st.session_state[_custom_key("kind")]
    if kind == "Upload table":
        if (table := st.session_state.get(_custom_key(f"{role}_table"))) is not None:
            st.caption(f"{label} table range: {table['frequency_cm1'][0]:g}–{table['frequency_cm1'][-1]:g} cm⁻¹.")
        _render_drude_controls(_custom_key(role))
        return
    source = "Constant" if kind == "Constant" else st.selectbox(
        f"{label} response",
        ("Weak Lorentz", "Strong Lorentz", "Constant"),
        key=_custom_key(f"{role}_source"),
    )
    if source == "Constant":
        columns = st.columns(2, gap="small")
        for column, part, default in zip(columns, ("real", "imag"), (1.0, 0.0)):
            st.session_state.setdefault(_custom_key(f"{role}_eps_{part}"), default)
            column.number_input(f"{'Re' if part == 'real' else 'Im'} {label}", step=0.1,
                                key=_custom_key(f"{role}_eps_{part}"))
        _render_drude_controls(_custom_key(role))
        return
    if source == "Weak Lorentz":
        st.latex(r"\varepsilon(\omega)=\varepsilon_\infty+\sum_j\frac{S_j\omega_{0,j}^2}{\omega_{0,j}^2-\omega^2-i\gamma_j\omega}")
    elif source == "Strong Lorentz":
        st.latex(r"\varepsilon(\omega)=\varepsilon_\infty\prod_j\frac{\omega_{LO,j}^2-\omega^2-i\gamma_j\omega}{\omega_{TO,j}^2-\omega^2-i\gamma_j\omega}")

    st.number_input(f"{label} ε∞", min_value=0.0, step=0.1, key=_custom_key(f"{role}_eps_inf"))
    count = int(st.number_input("Oscillators", min_value=1, max_value=6, step=1, key=_custom_key(f"{role}_count")))
    fields = (("Strength", "strength", 0.1), ("ω₀ (cm⁻¹)", "w0_cm1", 1.0), ("γ (cm⁻¹)", "gamma_cm1", 0.1))
    if source == "Strong Lorentz":
        fields = (("TO (cm⁻¹)", "to_cm1", 1.0), ("LO (cm⁻¹)", "lo_cm1", 1.0), ("γ (cm⁻¹)", "gamma_cm1", 0.1))
    for index in range(count):
        cols = st.columns(3, gap="small")
        for column, (field_label, field, step) in zip(cols, fields):
            state_key = _custom_key(f"{role}_{field}_{index}")
            st.session_state.setdefault(state_key, 1.0 if field != "gamma_cm1" else 5.0)
            column.number_input(field_label, min_value=0.0, step=step, key=state_key)
    _render_drude_controls(_custom_key(role))


def _custom_builder_definition() -> tuple[dict[str, object] | None, str | None]:
    name = str(st.session_state[_custom_key("name")]).strip()
    if not name:
        return None, "Material name cannot be empty."
    in_plane = _axis_definition_from_builder("in_plane")
    if in_plane is None:
        return None, "Upload a material table." if st.session_state[_custom_key("kind")] == "Upload table" \
            else "Provide an in-plane epsilon table."
    isotropic = _builder_mode() == "Isotropic"
    normal = in_plane if isotropic else _axis_definition_from_builder("normal")
    if normal is None:
        return None, "Provide a normal epsilon table."
    definition = {
        "name": name,
        "tensor_mode": "isotropic" if isotropic else "anisotropic_diagonal",
        "axes": {"xx": in_plane, "yy": in_plane, "zz": normal},
    }
    if references := st.session_state.get(_custom_key("references")):
        definition["references"] = references
    in_plane_drude = _drude_term_from_builder("in_plane")
    normal_drude = in_plane_drude if isotropic else _drude_term_from_builder("normal")
    if in_plane_drude is not None or normal_drude is not None:
        definition["drude"] = {
            "xx": in_plane_drude or {},
            "yy": in_plane_drude or {},
            "zz": normal_drude or {},
        }
    try:
        return normalize_snom_material_definition(definition), None
    except ValueError as exc:
        return None, str(exc)


def _builder_mode() -> str:
    # Uploads keep the tensor type read from the file in plain state, since the Tensor type widget is hidden.
    if st.session_state[_custom_key("kind")] == "Upload table":
        return str(st.session_state.get(_custom_key("upload_mode"), st.session_state.get(_custom_key("mode"), "Isotropic")))
    return str(st.session_state[_custom_key("mode")])


def _render_table_upload() -> None:
    uploaded = st.file_uploader(
        "Material table", type=("csv", "txt"), key=_custom_key("table_upload"),
        help="3 columns: frequency_cm1, eps_real, eps_imag. 5: frequency_cm1, eps_parallel_real/imag, eps_perpendicular_real/imag.",
    )
    if uploaded is not None:
        digest = sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get(_custom_key("table_digest")) != digest:
            try:
                uploaded.seek(0)
                mode, tables = parse_material_axes(uploaded)
            except ValueError as exc:
                st.error(str(exc))
            else:
                for role, axis in (("in_plane", "xx"), ("normal", "zz")):
                    st.session_state[_custom_key(f"{role}_table")] = tables[axis]
                st.session_state[_custom_key("upload_mode")] = mode
                st.session_state[_custom_key("table_digest")] = digest
    if st.session_state.get(_custom_key("in_plane_table")) is not None:
        st.caption(f"Tensor type: {_builder_mode()} (from the table columns).")


def _open_custom_builder(prefix: str, original_name: str | None = None) -> None:
    for key in tuple(st.session_state):
        if str(key).startswith(_CUSTOM_BUILDER_PREFIX):
            st.session_state.pop(key)
    st.session_state["editor"] = ("custom_material", prefix, original_name)
    st.rerun()


def _preview_frequency() -> np.ndarray | None:
    start = float(st.session_state.get("frequency_start", 600.0))
    end = float(st.session_state.get("frequency_end", 1200.0))
    if start >= end:
        return None
    return np.linspace(start, end, min(int(st.session_state.get("frequency_points", 301)), 501))


def _render_material_preview(spectrum: MaterialSpectrum, frequency: np.ndarray) -> None:
    traces = [("ε∥ (in-plane)", spectrum.perpendicular)]
    if not spectrum.isotropic:
        traces.append(("εzz (normal)", spectrum.parallel))
    real, imag = go.Figure(), go.Figure()
    for label, values in traces:
        real.add_scatter(x=frequency, y=values.real, name=label)
        imag.add_scatter(x=frequency, y=values.imag, name=label)
    st.caption("Dielectric-function preview over the calculation range")
    columns = st.columns(2, gap="small")
    columns[0].plotly_chart(_style_plot(real, "Re(ε)", "Re(ε)"), width="stretch")
    columns[1].plotly_chart(_style_plot(imag, "Im(ε)", "Im(ε)"), width="stretch")
    definition, references = spectrum.metadata.get("definition", {}), spectrum.metadata.get("references", [])
    if references or _oscillator_rows(definition):
        with st.expander("Model parameters and references"):
            _render_oscillators(definition)
            _render_references(references)


def _oscillator_rows(definition: Mapping[str, Any]) -> list[dict[str, object]]:
    """One row per Lorentz/TO-LO oscillator of a catalog or custom definition; other axis types give none."""
    axes = definition.get("axes")
    if not isinstance(axes, Mapping):  # constant tip epsilon, sheets, monoclinic
        return []
    rows: list[dict[str, object]] = []
    for axis, spec in axes.items():
        kind = spec.get("source_type", spec.get("type"))
        if kind in {"lorentz", "weak_lorentz"}:
            terms = [(o.get("w0", o.get("w0_cm1")), None, o["strength"], o["gamma"] if "gamma" in o else o["gamma_cm1"], None)
                     for o in spec["oscillators"]]
        elif kind in {"tolo", "strong_lorentz"}:
            terms = [(o.get("to", o.get("to_cm1")), o.get("lo", o.get("lo_cm1")), None,
                      o.get("gamma_to", o.get("gamma", o.get("gamma_cm1"))), o.get("gamma_lo", o.get("gamma", o.get("gamma_cm1"))))
                     for o in spec.get("modes", spec.get("oscillators", []))]
        else:
            continue
        for w, lo, strength, g, gl in terms:
            row = {"Axis": axis, "ε∞": spec["eps_inf"], "ω₀ / ω_TO (cm⁻¹)": w, "ω_LO (cm⁻¹)": lo, "S": strength,
                   "γ / γ_TO (cm⁻¹)": g, "γ_LO (cm⁻¹)": gl}
            twin = next((r for r in rows if {k: v for k, v in r.items() if k != "Axis"}
                         == {k: v for k, v in row.items() if k != "Axis"}), None)
            if twin:  # identical on another axis (uniaxial xx = yy)
                twin["Axis"] += f", {axis}"
            else:
                rows.append(row)
    return rows


def _render_oscillators(definition: Mapping[str, Any]) -> None:
    if rows := _oscillator_rows(definition):
        st.caption("Oscillator parameters")
        st.dataframe(rows, hide_index=True, width="stretch", height=140)


def _render_references(references) -> None:
    lines = []
    for reference in references:
        text = re.sub(r"([\\`*_$\[\]<>])", r"\\\1", str(reference.get("citation") or reference.get("key", "")))
        if doi := reference.get("doi"):
            text += f" [doi:{doi}](https://doi.org/{doi})"
        if url := reference.get("url"):
            text += f" [link]({url})"
        lines.append(f"- {text}")
    if lines:
        st.caption("References\n" + "\n".join(lines))


def _render_custom_material_preview(definition: Mapping[str, object], prefix: str | None = None) -> None:
    frequency = _preview_frequency()
    if frequency is None:
        return
    try:
        spectrum = custom_material(dict(definition), frequency)
    except ValueError as exc:
        st.error(f"Preview unavailable: {exc}")
        return
    if prefix is not None:
        spectrum = _state_drude(prefix, spectrum, frequency)
    _render_material_preview(spectrum, frequency)


@st.dialog("Custom material", width="large", on_dismiss=_dismiss_editor)
def _custom_material_editor(prefix: str, original_name: str | None) -> None:
    _persist_setup_state()
    library = _material_library()
    marker = (prefix, original_name)
    if st.session_state.get(_custom_key("target")) != marker:
        definition = library.load(original_name) if original_name else {
            "name": "",
            "tensor_mode": "isotropic",
            "axes": {axis: _default_axis_definition() for axis in ("xx", "yy", "zz")},
        }
        try:
            _reset_custom_builder(prefix, original_name, definition)
        except ValueError as exc:
            st.error(str(exc))
            return

    uploaded = st.file_uploader("Import material JSON", type="json", key=_custom_key("import"))
    if uploaded is not None:
        digest = sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get(_custom_key("import_digest")) != digest:
            try:
                imported = normalize_snom_material_definition(load_material_definition_json(uploaded.getvalue()))
                _reset_custom_builder(prefix, original_name, imported)
                st.session_state[_custom_key("import_digest")] = digest
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.text_input("Material name", key=_custom_key("name"))
    kind = st.selectbox("Material type", ("Constant", "Lorentz model", "Upload table"), key=_custom_key("kind"))
    if kind == "Upload table":
        _render_table_upload()
    else:
        st.selectbox("Tensor type", ["Isotropic", "Uniaxial"], key=_custom_key("mode"))
    with st.container(border=True):
        _render_axis_builder("in_plane", "ε")
    if _builder_mode() != "Isotropic":
        with st.container(border=True):
            _render_axis_builder("normal", "εzz")


    definition, error = _custom_builder_definition()
    if error:
        st.error(error)
    elif definition is not None:
        _render_custom_material_preview(definition)
        save_button(
            st,
            "Export material JSON", data=json.dumps(definition, indent=2, sort_keys=True).encode(),
            file_name=f"{definition['name']}.json", mime="application/json",
        )
    actions = st.columns(2, gap="small")
    if actions[0].button("Cancel", width="stretch"):
        _dismiss_editor()
        st.rerun()
    if actions[1].button("Save material", type="primary", width="stretch", disabled=definition is None):
        assert definition is not None
        new_name = str(definition["name"])
        if new_name in library.names() and new_name != original_name:
            st.error(f"A material named '{new_name}' already exists.")
            return
        try:
            library.save(definition)
            if original_name and original_name != new_name:
                library.delete(original_name)
        except (OSError, ValueError) as exc:
            st.error(f"Could not save material: {exc}")
            return
        if original_name and original_name != new_name:  # follow the rename everywhere it is used
            for user in _custom_material_users(original_name):
                st.session_state[f"{user}_custom"] = new_name
        st.session_state[f"{prefix}_source"] = "Custom library"
        st.session_state[f"{prefix}_custom"] = new_name
        _save_editor()
        st.rerun()


def _custom_material_users(name: str) -> dict[str, str]:
    """Prefixes whose selected Custom Catalog material is ``name``, with display labels."""
    state = st.session_state
    layer_ids = state.get("finite_layer_ids", [])
    roles = {"substrate": "Substrate", "reference": "Reference"}
    roles |= {f"layer_{layer_id}": f"Layer {len(layer_ids) - index}" for index, layer_id in enumerate(layer_ids)}
    return {prefix: label for prefix, label in roles.items()
            if state.get(f"{prefix}_source") == "Custom library" and state.get(f"{prefix}_custom") == name}


def _render_custom_library_selection(prefix: str) -> None:
    library = _material_library()
    names = library.names()
    if not names:
        st.caption("No saved custom materials yet.")
    else:
        selected = str(st.session_state.get(f"{prefix}_custom", ""))
        pick, edit, delete = st.columns([4, 1, 1], gap="small", vertical_alignment="bottom")
        pick.selectbox(
            "Library material", names,
            index=names.index(selected) if selected in names else 0,
            key=f"{prefix}_custom",
        )
        selected = str(st.session_state[f"{prefix}_custom"])
        if edit.button("Edit", width="stretch"):
            _open_custom_builder(prefix, selected)
        if delete.button("Delete", width="stretch"):
            others = [label for user, label in _custom_material_users(selected).items() if user != prefix]
            if others:
                st.error(f"'{selected}' is used by {', '.join(others)}. Choose another material there first.")
            else:
                st.session_state[f"{prefix}_delete_custom_confirm"] = selected
    if st.button("New material"):
        _open_custom_builder(prefix)
    if names:
        try:
            _render_custom_material_preview(library.load(selected), prefix)
        except ValueError as exc:
            st.error(str(exc))
    pending_delete = st.session_state.get(f"{prefix}_delete_custom_confirm")
    if isinstance(pending_delete, str):
        st.warning(f"Delete '{pending_delete}' from the material library?")
        confirm, cancel = st.columns(2, gap="small")
        if confirm.button("Confirm delete", type="primary"):
            try:
                _material_library().delete(pending_delete)
            except (OSError, ValueError) as exc:
                st.error(f"Could not delete material: {exc}")
                return
            # Apply before widgets are built on the full rerun, just as when
            # restoring an editor snapshot on cancel.
            st.session_state.setdefault("editor_restore", {})[f"{prefix}_custom"] = ""
            st.session_state.pop(f"{prefix}_delete_custom_confirm", None)
            st.rerun()
        if cancel.button("Keep material"):
            st.session_state.pop(f"{prefix}_delete_custom_confirm", None)
            st.rerun()


SOURCE_LABELS = {"Catalog": "Standard Catalog", "Custom library": "Custom Catalog", "Upload CSV/TXT": "Uploaded table"}


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def _select_variant(prefix: str, name: str | None) -> str:
    """Parameter-set picker of a catalog material; "" is the default set. Hidden when there are no alternatives."""
    options = ("", *catalog_variants(name or ""))
    if st.session_state.get(f"{prefix}_variant", "") not in options:
        st.session_state[f"{prefix}_variant"] = ""
    if len(options) == 1:
        return ""
    return st.selectbox("Parameter set", options, key=f"{prefix}_variant", format_func=lambda v: catalog_variant_label(name, v))


@st.dialog("Edit material", on_dismiss=_dismiss_editor)
def _material_editor(
    prefix: str,
    label: str,
    *,
    allow_upload: bool = True,
    layer_id: int | None = None,
) -> None:
    _persist_setup_state()
    selected_source = str(st.session_state[f"{prefix}_source"])
    # Uploads now go through New material; an uploaded table from an older setup stays selectable.
    choices = ("Catalog", "Constant", "Custom library") + (
        ("Upload CSV/TXT",) if allow_upload and selected_source == "Upload CSV/TXT" else ())
    if selected_source not in choices:
        st.session_state[f"{prefix}_source"] = choices[0]
    source = st.segmented_control("Material source", choices, required=True, key=f"{prefix}_source",
                                  format_func=_source_label)
    if source == "Catalog":
        names = catalog_names()
        selected_catalog = str(st.session_state[f"{prefix}_catalog"])
        selected_catalog = st.selectbox("Catalog material", names, index=names.index(selected_catalog) if selected_catalog in names else 0, key=f"{prefix}_catalog")
        variant = _select_variant(prefix, selected_catalog)
        frequency = _preview_frequency()
        if selected_catalog is None:
            st.info("Select a catalog material to preview it.")
        elif frequency is not None:
            spectrum = catalog_material(selected_catalog, frequency, variant)
            _render_material_preview(_state_drude(prefix, spectrum, frequency), frequency)
    elif source == "Constant":
        real_col, imag_col = st.columns(2, gap="small")
        real_col.number_input("Re ε", value=float(st.session_state[f"{prefix}_real"]), key=f"{prefix}_real")
        imag_col.number_input("Im ε", value=float(st.session_state[f"{prefix}_imag"]), key=f"{prefix}_imag")
    elif source == "Custom library":
        _render_custom_library_selection(prefix)
    else:
        if uploaded := st.session_state.get(f"{prefix}_upload"):
            st.caption(f"Current file: {uploaded.name}")
        st.file_uploader(
            "Upload CSV/TXT",
            type=("csv", "txt"),
            key=f"draft_{prefix}_upload",
            help="3 columns: frequency_cm1, eps_real, eps_imag. 5: frequency_cm1, eps_parallel_real/imag, eps_perpendicular_real/imag.",
        )
    _render_drude_controls(prefix)

    if layer_id is not None:
        st.number_input("Layer thickness (nm)", min_value=0.0, step=1.0, value=float(st.session_state[f"{prefix}_thickness"]), key=f"{prefix}_thickness")
    if st.button(f"Save {label}", type="primary", disabled=source == "Catalog" and selected_catalog is None):
        if source == "Upload CSV/TXT":
            if uploaded := st.session_state.get(f"draft_{prefix}_upload"):
                st.session_state[f"{prefix}_upload"] = uploaded
        _save_editor()
        st.rerun()
    if st.button("Cancel"):
        _dismiss_editor()
        st.rerun()
    if layer_id is not None and st.button("Remove layer"):
        _remove_layer(layer_id)
        _save_editor()
        st.rerun()
