"""Spectrum plots, saved-run history, experimental imports and export controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ._ui_save import save_button

from .io import adjust_experimental_phase, combined_results_csv, map_experiment_columns, read_experiment_columns
from ._spectra import M_ERROR_LIMIT
from .models import SpectrumResult
from .setup import CALCULATED_HARMONICS, _encode_setup
from ._ui_state import (
    _load_material_library, _load_setup, _material_library, _selected_harmonics,
)


MAX_HISTORY_RUNS = 20


def _history_summary(
    model: str,
    frequency: np.ndarray,
    materials: Mapping[str, object],
    controls: Mapping[str, object],
) -> str:
    frequency = np.asarray(frequency, dtype=float)
    if model == "bulk":
        labels = [getattr(materials.get("sample"), "label", "sample")]
    elif model == "multilayer":
        labels = [f"{material.label} ({thickness:g} nm)" for material, thickness in materials["layers"]]
        labels.append(getattr(materials.get("substrate"), "label", "substrate"))
    else:
        labels = [getattr(materials.get("layer"), "label", "layer"),
                  getattr(materials.get("substrate"), "label", "substrate")]
    if model == "layer":
        labels.append(f"{float(materials['thickness_nm']):g} nm")
    return (
        f"{model.title()} | {' on '.join(str(label) for label in labels)} | "
        f"{float(frequency[0]):g}–{float(frequency[-1]):g} cm⁻¹ · {frequency.size} pts | "
        f"{controls.get('preset', 'Custom')}"
    )


def _history_label(entry: Mapping[str, object]) -> str:
    timestamp = str(entry.get("timestamp_utc", ""))
    timestamp = timestamp.replace("T", " ").replace("+00:00", "Z")[:19]
    # Streamlit serializes formatted labels; the run ID keeps identical runs distinct.
    return f"{entry['id']} | {timestamp} | {entry.get('summary', 'Saved spectrum')}"


def _append_results_history(
    model: str,
    frequency: np.ndarray,
    materials: Mapping[str, object],
    controls: Mapping[str, object],
    results: Mapping[int, SpectrumResult],
) -> None:
    sequence = int(st.session_state.get("history_seq", 0)) + 1
    entry = {
        "id": f"run_{sequence}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": _history_summary(model, frequency, materials, controls),
        "results": results,
    }
    st.session_state["history_seq"] = sequence
    history = st.session_state.setdefault("results_history", [])
    history.append(entry)
    del history[:-MAX_HISTORY_RUNS]
    st.session_state["selected_history_id"] = entry["id"]
    st.session_state["results"] = results


def _sync_selected_history_result() -> None:
    history = list(st.session_state.get("results_history", []))
    if not history:
        st.session_state["results"] = {}
        st.session_state["selected_history_id"] = None
        return
    options = [str(entry["id"]) for entry in history]
    selected = str(st.session_state.get("selected_history_id") or "")
    if selected not in options:
        selected = options[-1]
        st.session_state["selected_history_id"] = selected
    entry = next(entry for entry in history if str(entry["id"]) == selected)
    st.session_state["results"] = entry["results"]


def _render_history_selector() -> None:
    history = list(st.session_state.get("results_history", []))
    if not history:
        return
    options = [str(entry["id"]) for entry in history]
    selected = str(st.session_state.get("selected_history_id") or options[-1])
    if selected not in options:
        selected = options[-1]
    # Bind the selection so its callback sees the new run before plots render.
    st.session_state["selected_history_id"] = selected
    st.selectbox(
        "Saved spectrum",
        options,
        key="selected_history_id",
        format_func=lambda option: _history_label(
            next(entry for entry in history if str(entry["id"]) == str(option))
        ),
        on_change=_sync_selected_history_result,
    )
    _sync_selected_history_result()


def _render_results(
    results: Mapping[int, SpectrumResult],
    harmonics: Sequence[int],
    experimental: Mapping[str, Mapping[int, dict[str, np.ndarray | None]]],
    phase_adjustments: Mapping[str, tuple[float, float, bool]] | None = None,
) -> None:
    st.subheader("Spectrum", anchor=False)
    flagged = {}
    limit = M_ERROR_LIMIT
    # quadQ follows N (quadQ >= 3N/2), so a failed quadQ check is reported as an N check.
    for key, parameter, threshold in (("M_check_max_error", "M", limit), ("N_check_max_error", "N", limit),
                                      ("quadQ_check_max_error", "N", 0.03)):
        worst = max(
            (results[harmonic].metadata.get(key, 0.0)
             for harmonic in harmonics if harmonic in results),
            default=0.0,
        )
        if worst > threshold:
            flagged[parameter] = max(worst, flagged.get(parameter, 0.0))
    flagged = list(flagged.items())
    if flagged:
        checked_result = next((results[h] for h in harmonics if h in results), None)
        failed = "; ".join(
            "check " + (f"{parameter} {checked_result.metadata[parameter]} vs "
                        f"{checked_result.metadata[f'{parameter}_check_higher']}"
                        if f"{parameter}_check_higher" in checked_result.metadata
                        else f"{parameter} vs {parameter}/2") + f" did not pass ({worst:.2%})"
            for parameter, worst in flagged
        )
        parameters = [parameter for parameter, _ in flagged]
        if len(parameters) > 2:
            parameter_names = ", ".join(parameters[:-1]) + ", and " + parameters[-1]
        else:
            parameter_names = " and ".join(parameters)
        model = str(checked_result.metadata.get("model", "bulk")).capitalize()
        st.warning(f"Warning! {model}: {failed}. Suggest to increase {parameter_names}.")
    amplitude = go.Figure()
    phase = go.Figure()
    for harmonic in harmonics:
        result = results.get(harmonic)
        if result is None:
            continue
        amplitude.add_scatter(
            x=result.frequency_cm1,
            y=result.amplitude,
            name=f"n = {harmonic}",
            line={"width": 2.4},
        )
        phase.add_scatter(
            x=result.frequency_cm1,
            y=result.phase_rad,
            name=f"n = {harmonic}",
            line={"width": 2.4},
        )
    for name, overlay in experimental.items():
        for harmonic in harmonics:
            data = overlay.get(harmonic)
            if data is None:
                continue
            trace_name = f"{name} n = {harmonic}"
            amplitude.add_scatter(
                x=data["frequency_cm1"],
                y=data["amplitude"],
                name=trace_name,
                mode="markers",
            )
            if data["phase_rad"] is None:
                continue
            displayed_phase = data["phase_rad"]
            if phase_adjustments is not None:
                offset, tilt, unwrap = phase_adjustments.get(name, (0.0, 0.0, True))
                displayed_phase = adjust_experimental_phase(
                    data["frequency_cm1"], displayed_phase, offset, tilt, unwrap=unwrap,
                )
            phase.add_scatter(
                x=data["frequency_cm1"],
                y=displayed_phase,
                name=trace_name,
                mode="markers",
            )
    if not amplitude.data:
        amplitude.add_annotation(
            text="Select a harmonic and calculate a spectrum.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        phase.add_annotation(
            text="Select a harmonic and calculate a spectrum.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    unnormalized = any(result.metadata.get("normalization") == "none" for result in results.values())
    st.plotly_chart(_style_plot(
        amplitude, "Unnormalized amplitude" if unnormalized else "Normalized amplitude",
        "|sₙ|" if unnormalized else "|σₙ|",
    ), width="stretch")
    st.plotly_chart(_style_plot(phase, "Phase", "Phase (rad)"), width="stretch")


def _load_experimental_overlay() -> None:
    for key in tuple(st.session_state):
        if key.startswith("experimental_import_"):
            st.session_state.pop(key)
    st.session_state["experimental_import_open"] = st.session_state.get("experimental_overlay_upload") is not None


def _dismiss_experimental_import() -> None:
    st.session_state["experimental_import_open"] = False


@st.dialog("Load text data", width="large", on_dismiss=_dismiss_experimental_import)
def _experimental_import_dialog() -> None:
    uploaded = st.session_state.get("experimental_overlay_upload")
    if uploaded is None:
        return
    st.caption(uploaded.name)
    try:
        uploaded.seek(0)
        headers, rows = read_experiment_columns(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    columns = [f"{index + 1} · {header}" for index, header in enumerate(headers)]
    reverse_rows = st.checkbox(
        "Reverse row order", key="experimental_import_reverse_rows",
        help="Enable for descending wavenumbers. Reverses all signal columns together with the frequency column.",
    )
    if reverse_rows:
        rows = rows[::-1]
    st.dataframe([dict(zip(columns, row)) for row in rows[:12]], hide_index=True, width="stretch")
    st.caption(f"Preview: first {min(len(rows), 12)} of {len(rows)} rows")
    name = st.text_input("Variable name", value=uploaded.name.rsplit(".", 1)[0], key="experimental_import_name")
    frequency = st.selectbox("Frequency (cm⁻¹)", columns, key="experimental_import_frequency")
    mode = st.radio("Signal columns", ("Amplitude + phase", "Real + imaginary", "Complex"), horizontal=True, key="experimental_import_mode")
    first_label = "Amplitude" if mode == "Amplitude + phase" else "Real" if mode == "Real + imaginary" else "Complex value"
    second_label = "Phase (rad)" if mode == "Amplitude + phase" else "Imaginary"
    st.caption("Assign columns to the harmonics in this file. Unassigned harmonics are skipped.")
    assigned = {}
    for harmonic in CALCULATED_HARMONICS:
        label_column, first_column, second_column = st.columns([1, 3, 3], gap="small", vertical_alignment="bottom")
        label_column.markdown(f"n = {harmonic}")
        first = first_column.selectbox(first_label, ["None"] + columns, key=f"experimental_import_first_{harmonic}")
        second = None if mode == "Complex" else second_column.selectbox(
            second_label, ["None"] + columns, key=f"experimental_import_second_{harmonic}")
        if first != "None":
            assigned[harmonic] = (first, second)
    if st.button("Load data", type="primary"):
        name = name.strip()
        if not name:
            st.error("Enter a variable name.")
            return
        if name in st.session_state.get("experimental_overlays", {}):
            st.error("That variable name is already in use.")
            return
        if not assigned:
            st.error("Assign a column to at least one harmonic.")
            return
        data = {}
        for harmonic, (first, second) in assigned.items():
            if mode == "Real + imaginary" and second not in columns:
                st.error(f"n = {harmonic}: choose the imaginary column.")
                return
            selected = (columns.index(first),) + ((columns.index(second),) if second in columns else ())
            try:
                data[harmonic] = map_experiment_columns(rows, columns.index(frequency), mode, selected)
            except ValueError as exc:
                st.error(f"n = {harmonic}: {exc}")
                return
        st.session_state.setdefault("experimental_overlays", {})[name] = data
        st.session_state["experimental_import_open"] = False
        st.rerun()
    if st.button("Cancel"):
        _dismiss_experimental_import()
        st.rerun()


def _reset_experimental_phase(name: str) -> None:
    for field in ("offset", "tilt", "unwrap"):
        st.session_state.pop(f"experimental_phase_{name}_{field}", None)


def _remove_experimental_overlay(name: str) -> None:
    st.session_state["experimental_overlays"].pop(name, None)
    _reset_experimental_phase(name)


@st.fragment
def _render_result_plots(
    results: Mapping[int, SpectrumResult],
    harmonics: Sequence[int],
    experimental: Mapping[str, Mapping[int, dict[str, np.ndarray | None]]],
    phase_adjustments: Mapping[str, tuple[float, float, bool]],
) -> None:
    """Render spectra and experimental overlays."""
    _render_results(results, harmonics, experimental, phase_adjustments)


def _results_workspace(results: Mapping[int, SpectrumResult]) -> None:
    results = st.session_state.get("results", results)
    harmonics = _selected_harmonics(st.session_state)
    experimental = st.session_state.get("experimental_overlays", {})
    experimental = experimental if isinstance(experimental, dict) else {}
    phase_adjustments = {
        name: (
            st.session_state.get(f"experimental_phase_{name}_offset", 0.0),
            st.session_state.get(f"experimental_phase_{name}_tilt", 0.0),
            st.session_state.get(f"experimental_phase_{name}_unwrap", True),
        )
        for name, overlay in experimental.items()
        if any(data["phase_rad"] is not None for data in overlay.values())
    }
    _render_result_plots(results, harmonics, experimental, phase_adjustments)
    if phase_adjustments:
        st.markdown("**Experimental phase adjustments**")
        for name in phase_adjustments:
            st.text(name)
            offset_column, tilt_column, unwrap_column, reset_column = st.columns(
                [2, 2, 1.5, 1], gap="small", vertical_alignment="bottom",
            )
            offset_column.number_input(
                "Offset (rad)", value=0.0, step=0.1, format="%.4f",
                key=f"experimental_phase_{name}_offset",
            )
            tilt_column.number_input(
                "Tilt (rad per cm⁻¹)", value=0.0, step=0.0001, format="%.6f",
                help="Adds a linear phase rotation relative to the first frequency.",
                key=f"experimental_phase_{name}_tilt",
            )
            unwrap_column.checkbox(
                "Unwrap phase", value=True, key=f"experimental_phase_{name}_unwrap",
                help="Show continuous phase; turn off to wrap the rotated phase to ±π.",
            )
            reset_column.button(
                "Reset", key=f"experimental_phase_{name}_reset",
                on_click=_reset_experimental_phase, args=(name,),
            )
    _render_history_selector()
    load_column, export_column = st.columns([2, 1], gap="small", vertical_alignment="center")
    load_column.file_uploader(
        "Load text data",
        type=("csv", "txt"),
        key="experimental_overlay_upload",
        help="Preview CSV/TXT data and choose frequency and signal columns.",
        on_change=_load_experimental_overlay,
    )
    for name in tuple(st.session_state.get("experimental_overlays", {})):
        load_column.button(
            f"Remove {name}", key=f"remove_experiment_{name}",
            on_click=_remove_experimental_overlay, args=(name,),
        )
    if results:
        save_button(
            export_column,
            "Export harmonics CSV",
            data=combined_results_csv(results),
            file_name="snom-harmonics.csv",
            mime="text/csv",
            key="export_harmonics",
            width="stretch",
        )


def _render_harmonic_controls() -> None:
    st.markdown("**Harmonics**")
    columns = st.columns(len(CALCULATED_HARMONICS), gap="small")
    for column, harmonic in zip(columns, CALCULATED_HARMONICS):
        column.checkbox(f"n = {harmonic}", key=f"harmonic_{harmonic}")
    if st.session_state["tip_A"] == 0 and not st.session_state.get("reference_disabled", False):
        st.caption("At A = 0, only n = 0 has a defined contrast. Higher harmonics have zero raw signals and undefined contrast (NaN).")


def _results_controls() -> None:
    st.divider()
    st.markdown("**Load and save**")
    load_column, save_column = st.columns(2, gap="small")
    load_column.file_uploader(
        "Load setup",
        type="json",
        key="setup_upload",
        on_change=_load_setup,
        label_visibility="collapsed",
    )
    if error := st.session_state.get("setup_error"):
        load_column.error(str(error))
    save_button(
        save_column,
        "Save setup",
        data=_encode_setup(st.session_state),
        file_name="snom-setup.json",
        mime="application/json",
        width="stretch",
    )
    st.caption("Setup downloads include uploaded material tables and custom definitions.")
    with st.expander("Custom material library"):
        library = _material_library()
        if hasattr(library, "path"):
            st.caption(f"Materials are saved automatically to {library.path} and loaded at startup.")
        else:
            st.caption("Materials are private to this session. Download the library to keep it after refreshing or closing the tab.")
        save_button(
            st,
            "Download material library",
            data=_material_library().archive_json_bytes(),
            file_name="snom-material-library.json",
            mime="application/json",
        )
        st.file_uploader(
            "Restore material library",
            type="json",
            key="material_library_upload",
            on_change=_load_material_library,
            help="Replaces definitions with the same name in the library; other materials stay intact.",
        )
        if error := st.session_state.get("material_library_error"):
            st.error(str(error))
        if message := st.session_state.get("material_library_message"):
            st.success(str(message))
    st.divider()
    st.markdown("**References**")
    st.markdown(
        "1. Voronin, K. V., Herrero León, I., Hillenbrand, R., & Nikitin, A. Y. "
        "‘Quantitative Analytical Spheroid Model for Scattering-Type Scanning Near-Field "
        "Optical Spectroscopy’. *Adv. Opt. Mater.* **13**, e01539 (2025). "
        "[doi:10.1002/adom.202501539](https://doi.org/10.1002/adom.202501539)"
    )
    st.markdown(
        "2. Voronin, K. V., Vicentini, E., Mester, L., Hillenbrand, R., & Nikitin, A. Y. "
        "‘Quantitative Analytical Model for Scattering-type Scanning Near-field "
        "Optical Spectroscopy of Layered Samples’. *arXiv* (2026). "
        "[doi:10.48550/arXiv.2611.01057](https://doi.org/10.48550/arXiv.2611.01057)"
    )
    st.markdown(
        "3. The source code is available at [SNOM_Calculator on GitHub]"
        "(https://github.com/vicentini-edoardo/SNOM_Calculator)."
    )
    if st.session_state.get("experimental_import_open"):
        _experimental_import_dialog()


def _style_plot(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 16}},
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title=y_title,
        margin={"l": 50, "r": 10, "t": 45, "b": 40},
        height=260,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,.14)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,.14)")
    return figure
