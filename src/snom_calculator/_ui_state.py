"""Streamlit session state, setup restoration and domain parameter resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import streamlit as st

from .custom_materials import material_frequency_range_warnings
from .material_library import SessionMaterialLibrary, material_library_for_state
from .materials import (
    MaterialSpectrum, catalog_material, constant_material, custom_material,
    parse_material_table, with_drude,
)
from .models import FarFieldSettings, TipParameters
from .setup import (
    CALCULATED_HARMONICS, _CUSTOM_BUILDER_PREFIX,
    _apply_setup, _decode_setup, _default_setup,
)


def _material_library() -> SessionMaterialLibrary:
    return material_library_for_state(st.session_state)


def _calculation_model(layer_ids: Sequence[int]) -> str:
    return "bulk" if not layer_ids else "layer" if len(layer_ids) == 1 else "multilayer"


def _initialize_state() -> None:
    _material_library()
    if error := st.session_state.get("_local_material_library_error"):
        st.error(f"Could not load the local material library ({error}). The file is left unchanged and "
                 "materials saved now stay in this session only. Fix or move the file, then restart the app.")
    if "tip_Hs_max" not in st.session_state and st.session_state.get("average_gaps_enabled", False):
        st.session_state["tip_Hs"] = st.session_state["average_gap_min"]
        st.session_state["tip_Hs_max"] = st.session_state["average_gap_max"]
    for key in ("tip_Hs", "tip_Hr"):
        if key in st.session_state and f"{key}_max" not in st.session_state:
            st.session_state[f"{key}_max"] = st.session_state[key]
    for key, value in _default_setup().items():
        st.session_state.setdefault(key, value)
    if not st.session_state.get("_custom_defaults_migrated"):
        # One-time upgrade of the old default custom bulk precision.
        st.session_state["_custom_defaults_migrated"] = True
        if tuple(st.session_state.get(key) for key in (
            "bulk_custom_M", "bulk_custom_quad", "bulk_custom_N",
        )) == (30, 16, 8):
            st.session_state["bulk_custom_M"] = 80
            st.session_state["bulk_custom_N"] = 80
    st.session_state.pop("bulk_custom_quad", None)
    st.session_state.pop("layer_custom_quad", None)
    for key, value in st.session_state.pop("editor_restore", {}).items():
        st.session_state[key] = value
    if st.session_state.pop("editor_cleanup", False):
        for key in tuple(st.session_state):
            if str(key).startswith((_CUSTOM_BUILDER_PREFIX, "draft_")) or str(key).endswith("_delete_custom_confirm"):
                st.session_state.pop(key)
    st.session_state.setdefault("results_history", [])
    st.session_state.setdefault("selected_history_id", None)
    st.session_state.setdefault("history_seq", 0)
    _persist_setup_state()


def _persist_setup_state() -> None:
    # Re-set widget-bound keys so Streamlit's stale-widget cleanup cannot drop them.
    # Dialog reruns are fragment runs that skip _initialize_state, so dialogs call this too.
    for key in tuple(st.session_state):
        if key == "reference_disabled":
            # This checkbox remains on the main page during dialog reruns.
            continue
        if key == "selected_history_id" or str(key).startswith((
            "tip_", "far_", "substrate_", "reference_", "layer_", "harmonic_",
            "average_gap",
        )):
            st.session_state[key] = st.session_state[key]


def _add_layer() -> None:
    layer_id = int(st.session_state["next_layer_id"])
    st.session_state["next_layer_id"] = layer_id + 1
    st.session_state["finite_layer_ids"] = [
        layer_id,
        *st.session_state["finite_layer_ids"],
    ]
    defaults = {
        f"layer_{layer_id}_source": "Catalog",
        f"layer_{layer_id}_catalog": "PMMA",
        f"layer_{layer_id}_variant": "",
        f"layer_{layer_id}_real": 12.0,
        f"layer_{layer_id}_imag": 0.0,
        f"layer_{layer_id}_custom": "",
        f"layer_{layer_id}_thickness": 20.0,
        f"layer_{layer_id}_drude_enabled": False,
        f"layer_{layer_id}_drude_wp_cm1": 0.0,
        f"layer_{layer_id}_drude_gamma_cm1": 0.0,
    }

    for key, value in defaults.items():
        st.session_state[key] = value


def _layer_number(layer_id: int, layer_ids: Sequence[int]) -> int:
    """Displayed layer number: ids run top to bottom and the bottom layer is 1."""
    return len(layer_ids) - list(layer_ids).index(layer_id)


def _remove_layer(layer_id: int) -> None:
    st.session_state["finite_layer_ids"] = [
        item for item in st.session_state["finite_layer_ids"] if item != layer_id
    ]
    for key in tuple(st.session_state):
        if str(key).startswith((f"layer_{layer_id}_", f"draft_layer_{layer_id}_")):
            st.session_state.pop(key)


def _material_summary(prefix: str, *, thickness: float | None = None) -> str:
    source = "Constant" if prefix == "tip" else st.session_state.get(f"{prefix}_source", "Constant")
    if source == "Catalog":
        summary = str(st.session_state[f"{prefix}_catalog"])
        if variant := st.session_state.get(f"{prefix}_variant"):
            summary += f" ({variant})"
    elif source == "Constant":
        epsilon_prefix = "tip_eps" if prefix == "tip" else prefix
        summary = (
            f"ε = {float(st.session_state[f'{epsilon_prefix}_real']):g}"
            f" + {float(st.session_state[f'{epsilon_prefix}_imag']):g}i"
        )
    elif source == "Custom library":
        summary = str(st.session_state.get(f"{prefix}_custom", "Custom material"))
    else:
        uploaded = st.session_state.get(f"{prefix}_upload")
        summary = getattr(uploaded, "name", "Upload material")
    if _drude_from_state(prefix)[0]:
        summary += " + Drude"
    return f"{summary} · {thickness:g} nm" if thickness is not None else summary


def _drude_from_state(prefix: str) -> tuple[float, float]:
    state = st.session_state
    if not state.get(f"{prefix}_drude_enabled", False):
        return 0.0, 0.0
    return float(state.get(f"{prefix}_drude_wp_cm1", 0.0)), float(state.get(f"{prefix}_drude_gamma_cm1", 0.0))


def _state_drude(prefix: str, spectrum: MaterialSpectrum, frequency: np.ndarray) -> MaterialSpectrum:
    return with_drude(spectrum, frequency, *_drude_from_state(prefix))


def _tip_material_from_state(frequency: np.ndarray | None = None) -> MaterialSpectrum | None:
    state = st.session_state
    if frequency is None:
        frequency = np.linspace(state["frequency_start"], state["frequency_end"], int(state["frequency_points"]))
    try:
        epsilon = complex(float(state["tip_eps_real"]), float(state["tip_eps_imag"]))
        values = np.full(frequency.shape, epsilon, dtype=complex)
        return MaterialSpectrum(values, values, "Tip", metadata={
            "source": "Constant", "definition": {"epsilon": epsilon}, "references": [],
        })
    except ValueError as exc:
        st.error(str(exc))
        return None


def _tip_from_state(
    frequency: np.ndarray | None = None, *, spectrum: MaterialSpectrum | None = None,
) -> TipParameters | None:
    state = st.session_state
    if spectrum is None:
        spectrum = _tip_material_from_state(frequency)
    if spectrum is None:
        return None
    return TipParameters(
        radius_nm=float(state["tip_R"]),
        length_nm=float(state["tip_L"]),
        amplitude_nm=float(state["tip_A"]),
        sample_gap_nm=float(state["tip_Hs"]),
        reference_gap_nm=float(state["tip_Hr"]),
        epsilon=complex(spectrum.parallel[0]),
    )


def _material_from_state(
    prefix: str, frequency: np.ndarray, *, allow_upload: bool = True
) -> MaterialSpectrum | None:
    spectrum = _base_material_from_state(prefix, frequency, allow_upload=allow_upload)
    return None if spectrum is None else _state_drude(prefix, spectrum, frequency)


def _base_material_from_state(
    prefix: str, frequency: np.ndarray, *, allow_upload: bool = True
) -> MaterialSpectrum | None:
    source = st.session_state.get(f"{prefix}_source")
    try:
        if source == "Catalog":
            return catalog_material(str(st.session_state[f"{prefix}_catalog"]), frequency,
                                    str(st.session_state.get(f"{prefix}_variant", "")))
        if source == "Constant":
            return constant_material(
                complex(
                    float(st.session_state[f"{prefix}_real"]),
                    float(st.session_state[f"{prefix}_imag"]),
                ),
                frequency,
                f"{prefix} constant",
            )
        if source == "Upload CSV/TXT" and allow_upload:
            uploaded = st.session_state.get(f"{prefix}_upload")
            if uploaded is None:
                return None
            uploaded.seek(0)
            return parse_material_table(uploaded, frequency, uploaded.name)
        if source == "Custom library":
            name = str(st.session_state.get(f"{prefix}_custom", "")).strip()
            if not name:
                return None
            definition = _material_library().load(name)
            warnings = material_frequency_range_warnings(
                [name], {name: definition}, float(frequency[0]), float(frequency[-1])
            )
            for warning in warnings:
                st.warning(warning)
            return custom_material(definition, frequency)
    except ValueError as exc:
        st.error(str(exc))
    return None


def _far_field_from_state() -> FarFieldSettings:
    return FarFieldSettings(
        bool(st.session_state["far_enabled"]),
        float(st.session_state["far_angle"]),
        float(st.session_state["far_coefficient"]),
    )


def _selected_harmonics(state: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(
        harmonic
        for harmonic in CALCULATED_HARMONICS
        if state.get(f"harmonic_{harmonic}", False)
    )


def _new_upload(key: str) -> object | None:
    # A main-page st.rerun() drops the uploader's server state while the browser keeps the file,
    # so its on_change fires again on the next interaction; handle each upload only once.
    uploaded = st.session_state.get(key)
    file_id = getattr(uploaded, "file_id", None)
    if uploaded is None or file_id is not None and st.session_state.get(f"_{key}_id") == file_id:
        return None
    st.session_state[f"_{key}_id"] = file_id
    return uploaded


def _load_setup() -> None:
    uploaded = _new_upload("setup_upload")
    if uploaded is None:
        return
    try:
        _apply_setup(st.session_state, _decode_setup(uploaded.getvalue()))
    except ValueError as exc:
        st.session_state["setup_error"] = str(exc)
    else:
        st.session_state.pop("results", None)
        st.session_state.pop("m_refinement_request", None)
        st.session_state.pop("m_refinement_next", None)
        st.session_state["results_history"] = []
        st.session_state["selected_history_id"] = None
        st.session_state["history_seq"] = 0
        st.session_state["setup_error"] = None


def _load_material_library() -> None:
    uploaded = _new_upload("material_library_upload")
    if uploaded is None:
        return
    try:
        names = _material_library().import_archive(uploaded.getvalue())
    except (OSError, ValueError) as exc:
        st.session_state["material_library_error"] = str(exc)
        st.session_state.pop("material_library_message", None)
    else:
        st.session_state["material_library_error"] = None
        st.session_state["material_library_message"] = f"Restored {len(names)} material(s): {', '.join(names) or 'none'}."
