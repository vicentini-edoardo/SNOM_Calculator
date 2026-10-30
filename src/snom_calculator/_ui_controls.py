"""Frequency and precision widgets plus calculation progress and error handling."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import streamlit as st

try:  # private Streamlit API; progress updates degrade gracefully without it
    from streamlit.runtime.scriptrunner_utils.script_run_context import in_cached_function
except ImportError:  # pragma: no cover - depends on the installed Streamlit
    in_cached_function = None

from ._calculation_cache import _run_calculations
from ._preset_defaults import PAD_CM1, PRESETS, automatic_preset, material_kind, max_beta, stack_k, stack_nm
from ._progress import report, set_callback, stage
from ._spectra import M_ERROR_LIMIT
from .calculations import validate_structure as _validate_structure
from .models import NumericalSettings, max_multipoles
from .setup import CALCULATED_HARMONICS
from ._ui_results import _append_results_history
from ._ui_state import (_far_field_from_state, _layer_number, _material_from_state, _tip_from_state,
                        _tip_material_from_state)

MODES = ("Automatic", "Presets", "Custom")


def _frequency_toolbar() -> np.ndarray:
    with st.container(border=True):
        start_col, end_col, points_col = st.columns(3, gap="small")
        start = start_col.number_input("Start (cm⁻¹)", min_value=1.0, step=10.0, key="frequency_start")
        end = end_col.number_input("End (cm⁻¹)", min_value=2.0, step=10.0, key="frequency_end")
        points = points_col.number_input("Points", min_value=2, max_value=1000, step=1, key="frequency_points")
        flag_col, fwhm_col = st.columns(2, gap="small", vertical_alignment="center")
        resolution = flag_col.checkbox("Spectrometer resolution", key="resolution_enabled")
        fwhm_col.number_input("Gaussian FWHM (cm⁻¹)", min_value=0.01, step=1.0, key="resolution_fwhm",
                              disabled=not resolution, label_visibility="collapsed")
    if start >= end:
        st.error("Start frequency must be smaller than end frequency.")
        return np.linspace(start, start + 1, int(points))
    return np.linspace(start, end, int(points))


def _precision_controls(model: str | None, materials: Mapping[str, object], frequency: np.ndarray | None = None) -> dict[str, object]:
    active_model = "layer" if model == "multilayer" else model or "bulk"
    layers = materials.get("layers", ()) if model == "multilayer" else (
        ((materials.get("layer"), materials.get("thickness_nm", 0)),) if model == "layer" else ()
    )
    uniaxial = any(
        material is not None and depth > 0 and np.any(material.perpendicular != material.parallel)
        for material, depth in layers
    )
    tip_material = _tip_material_from_state(frequency)
    tip = None if tip_material is None else _tip_from_state(frequency, spectrum=tip_material)
    with st.expander("Calculation precision"):
        fim = bool(layers) and st.checkbox(
            "FIM (Fourier integral method) for all layers", key="force_fim",
            help="Integrate the film reflection directly in momentum space, as for uniaxial layers, instead of "
                 "fitting K image charges to isotropic layers. Slower; use it to check the GICM (generalised image charge method) result.",
        )
        uniaxial = uniaxial or fim
        mode = st.segmented_control(
            "Numerical settings", MODES, key="precision_mode",
            help="Automatic picks each material's oscillator strength, the tip's included, from its permittivity. Presets applies the "
                 "strength you choose to every material. In both, M and N follow from the material type, tip, bulk "
                 "or film, and each material's distance from the tip. Custom sets every parameter by hand.",
        ) or "Automatic"
        preset, nm, K = "Custom", None, 20
        if mode != "Custom":
            strength = None if mode == "Automatic" else st.selectbox(
                "Oscillator strength", PRESETS, key="preset",
                help="Strength of the materials' oscillators: Weak for dielectrics and broad bands, Very strong for "
                     "sharp phonon polaritons such as SiC.")
            entries = _stack_entries(model, materials, frequency)
            if mode == "Automatic" and tip_material is not None:  # a resonant tip needs its own strength
                gap = min(float(st.session_state["tip_Hs"]), float(st.session_state["tip_Hr"]))
                entries.append(("Tip", tip_material, tip_material, False, gap))  # constant: no padding needed
            picks = [(name, strength or automatic_preset(padded), material_kind(material), film, distance, padded)
                     for name, material, padded, film, distance in entries]
            nm = stack_nm([pick[1:5] for pick in picks])
            if mode == "Automatic":
                st.caption(" · ".join(f"{name}: {pick} (|β| {max_beta(padded):.3g})"
                                      for name, pick, _, _, _, padded in picks))
            preset = mode if mode == "Automatic" else strength
            if nm is None:
                st.warning(f"No {preset.lower()} default converges at these distances; using Custom settings.")
                preset = "Custom"
            elif active_model == "layer" and not uniaxial:
                K = stack_k([(pick[1], depth) for pick, (_, depth) in zip(
                    (pick for pick in picks if pick[3]), (layer for layer in layers if layer[1] > 0 and layer[0] is not None))])
                if K is None:  # a thin or strongly resonant film would need more than MAX_K image charges
                    fim = uniaxial = True
                    st.caption("FIM: this film needs more image charges than GICM supports.")
        numerical = _numerical_preset(preset, active_model, uniaxial=uniaxial, nm=nm, K=K or 20)
        check_n = st.checkbox(
            "Check convergence in N", key="check_n",
            help="Compare with half as many multipoles (N). The GICM image-charge "
                 "count K is not checked. "
                 "Disabling it speeds up the calculation, but then you must check "
                 "convergence yourself. The M check stays active.",
        )

        values = f"M {numerical.M} · N {numerical.N}"
        if active_model == "layer" and not uniaxial:
            values += f" · K {numerical.K}"
        st.caption(f"{preset} · {values}")
        slot = st.session_state["_check_slot"] = st.empty()
        _render_checks(slot, st.session_state.get("results", {}))
    return {
        "numerical": numerical,
        "check_n": check_n,
        "tip": tip,
        "tip_material": tip_material,
        "far_field": _far_field_from_state(),
        "preset": preset,
        "method": "fim" if fim else "auto",
        "workers": 1,
        "average_gaps": (
            float(st.session_state["tip_Hs"]), float(st.session_state["tip_Hs_max"]),
            int(st.session_state["average_gap_count"]),
        ) if st.session_state["tip_Hs"] != st.session_state["tip_Hs_max"] else None,
        "average_reference_gaps": (
            float(st.session_state["tip_Hr"]), float(st.session_state["tip_Hr_max"]),
            int(st.session_state["tip_reference_gap_count"]),
        ) if st.session_state["tip_Hr"] != st.session_state["tip_Hr_max"] else None,
        "resolution_fwhm_cm1": (
            float(st.session_state["resolution_fwhm"])
            if st.session_state["resolution_enabled"] else None
        ),
    }


def _render_checks(slot, results: Mapping[int, object]) -> None:
    """M vs M/2 and N vs N/2 errors of the latest results, in the precision box."""
    lines = []
    for parameter in ("M", "N"):
        checked = [result.metadata for result in results.values() if f"{parameter}_check_max_error" in result.metadata]
        if not checked:
            continue
        error = max(metadata[f"{parameter}_check_max_error"] for metadata in checked)
        text = f"{parameter} vs {parameter}/2 : {error:.2%}".replace("%", " %")
        text = f":red[{text}]" if error > M_ERROR_LIMIT else text
        lines.append(text)
    with slot.container():
        if lines:
            st.markdown("  \n".join(lines), help=(
                f"Change of the signal when M or N is halved. Safe region: below {M_ERROR_LIMIT:.0%}. "
                f"Above it a warning appears and a higher value is suggested."))


def _stack_entries(model: str | None, materials: Mapping[str, object], frequency: np.ndarray | None) -> list[tuple]:
    """(name, material, material on the window +- PAD_CM1, film, distance from the tip in nm) for every material:
    the top layer at the gap, a buried layer or the substrate at the gap plus the thickness above it,
    and the reference at its own gap."""
    gap, reference_gap = float(st.session_state["tip_Hs"]), float(st.session_state["tip_Hr"])
    if model == "bulk":
        roles = [("Sample", "substrate", materials.get("sample"), False, gap)]
    else:
        ids = st.session_state["finite_layer_ids"]
        layers = materials.get("layers", ()) if model == "multilayer" else (
            (materials.get("layer"), materials.get("thickness_nm", 0.)),)
        roles, depth = [], gap
        for layer_id, (material, thickness) in zip(ids, layers):
            if thickness > 0:
                roles.append((f"Layer {_layer_number(layer_id, ids)}", f"layer_{layer_id}", material, True, depth))
            depth += thickness
        roles.append(("Substrate", "substrate", materials.get("substrate"), False, depth))
    if materials.get("reference") is not None:
        roles.append(("Reference", "reference", materials["reference"], False, reference_gap))
    return [(name, material, _padded(prefix, frequency) or material, film, distance)
            for name, prefix, material, film, distance in roles if material is not None]


def _padded(prefix: str, frequency: np.ndarray | None):
    """The material on the measurement window widened by PAD_CM1. Only catalog and constant materials:
    an uploaded table or library model may stop at the window (None: the window itself is used)."""
    if frequency is None or len(frequency) == 0 or st.session_state.get(f"{prefix}_source") not in ("Catalog", "Constant"):
        return None
    start, end = max(1., float(frequency[0]) - PAD_CM1), float(frequency[-1]) + PAD_CM1
    grid = np.linspace(start, end, int(end - start) + 2)
    return _material_from_state(prefix, grid, allow_upload=False)


def _numerical_preset(
    preset: str,
    active_model: str,
    *, uniaxial: bool = False,
    nm: tuple[int, int] | None = None, K: int = 20,
) -> NumericalSettings:
    if nm is not None:  # only M and N vary with the oscillator presets; the rest is the study's fixed settings
        M, N = nm
        return (
            NumericalSettings(M=M, N=N, quadQ=max(112, 3 * N // 2), K=10)
            if active_model == "bulk"
            else NumericalSettings(M=M, N=N, quadQ=max(112, 3 * N // 2), K=K,
                                   momentum_rtol=1e-6 if uniaxial else 1e-5)
        )
    return _custom_numerical(active_model, uniaxial=uniaxial)


def _calculation_action(
    model: str,
    frequency: np.ndarray,
    materials: dict[str, object],
    controls: Mapping[str, object],
) -> None:
    valid_range = st.session_state["frequency_start"] < st.session_state["frequency_end"]
    valid_tip = controls.get("tip") is not None
    if not valid_tip:
        st.info("Enter a valid constant tip permittivity before calculating.")
    run_clicked = st.button(
        "Calculate spectrum",
        type="primary",
        width="stretch",
        disabled=not valid_range or not valid_tip,
        help=None if valid_range else "Start frequency must be smaller than end frequency.",
    )
    refinement = st.session_state.pop("m_refinement_next", None)
    if (run_clicked and valid_range and valid_tip) or refinement is not None:
        if refinement is None:
            controls = {**controls, "_profiles": {}}
            st.session_state.pop("m_refinement_request", None)
        else:
            model, frequency, materials, controls = refinement["inputs"]
            parameter, numerical = refinement["parameter"], controls["numerical"]
            value = getattr(numerical, parameter)
            controls = {key: item for key, item in controls.items() if key != "_reuse_M"}
            controls = {**controls,
                        "numerical": (replace(numerical, M=2 * value) if parameter == "M" else
                                      replace(numerical, N=2 * value, quadQ=max(numerical.quadQ, 3 * value))),
                        **({"_reuse_M": value} if parameter == "M" else {}),
                        f"_{parameter}_requested": controls.get(f"_{parameter}_requested", value),
                        f"_{parameter}_requested_error": controls.get(f"_{parameter}_requested_error",
                                                                      refinement["error"])}
        bar = st.progress(0.0, text="Validating inputs")
        shown = [0.0]

        def update(text: str, fraction: float | None) -> None:
            shown[0] = max(shown[0], shown[0] if fraction is None else fraction)
            # not recorded by st.cache_data when called from inside a cached solver
            token = in_cached_function.set(False) if in_cached_function is not None else None
            try:
                bar.progress(min(shown[0], 1.0), text=text)
            finally:
                if token is not None:
                    in_cached_function.reset(token)

        set_callback(update)
        try:
            _validate_structure(model, materials)
            with stage("", 0.0, 1.0), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                results = _run_calculations(
                    model, frequency, materials, controls, CALCULATED_HARMONICS,
                )
            # Solver accuracy warnings (Legendre Q, SciPy quad) would otherwise reach only the console.
            for message in dict.fromkeys(str(item.message) for item in caught if issubclass(item.category, UserWarning)):
                st.warning(message)
            for result in results.values():
                result.metadata.update({key[1:]: value for key, value in controls.items()
                                        if key.startswith(("_M_requested", "_N_requested"))})
            st.session_state["results"] = results
            if slot := st.session_state.get("_check_slot"):
                _render_checks(slot, results)
            # M first, then N (only when its check ran), like the convergence flags.
            for parameter in ("M", "N"):
                error = max((result.metadata.get(f"{parameter}_check_max_error", 0.)
                             for result in results.values()), default=0.)
                if error > M_ERROR_LIMIT:
                    st.session_state["m_refinement_request"] = {
                        "inputs": (model, frequency.copy(), dict(materials), dict(controls)),
                        "parameter": parameter, "error": error}
                    break
            else:
                st.session_state.pop("m_refinement_request", None)
            report("Saving results", 0.95)
            _append_results_history(
                model,
                frequency,
                materials,
                controls,
                st.session_state["results"],
            )
            report("Calculation complete", 1.0)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            bar.empty()
            st.error(str(exc))
        finally:
            set_callback(None)


def _keep_current_m() -> None:
    st.session_state.pop("m_refinement_request", None)


@st.dialog("Convergence", on_dismiss=_keep_current_m)
def _m_refinement_dialog(pending: Mapping[str, object]) -> None:
    parameter = pending["parameter"]
    value = getattr(pending["inputs"][3]["numerical"], parameter)
    limit = M_ERROR_LIMIT
    noun = "points" if parameter == "M" else "multipoles"
    if parameter == "N" and 2 * value > (n_max := max_multipoles(pending["inputs"][3]["tip"])):
        st.warning(f"Warning! Check N vs N/2 did not pass ({pending['error']:.2%}, limit {limit:.0%}, N = {value}). "
                   f"N cannot be doubled: this tip shape allows N ≤ {n_max}. Check the result with care.")
        if st.button("Keep current N", key="keep_n"):
            _keep_current_m()
            st.rerun()
        return
    st.warning(f"Warning! Check {parameter} vs {parameter}/2 did not pass ({pending['error']:.2%}, limit {limit:.0%}, "
               f"{parameter} = {value}). Suggest to increase {parameter}: calculate with {2 * value} {noun}?")
    if st.button(f"Calculate with {parameter} = {2 * value}", key=f"double_{parameter.lower()}", type="primary"):
        # Carry the doubled values into the Custom controls (snapped to their options on the next render).
        prefix = "layer" if pending["inputs"][0] == "multilayer" else pending["inputs"][0] or "bulk"
        st.session_state[f"{prefix}_custom_{parameter}"] = 2 * value
        st.session_state["m_refinement_next"] = pending
        st.rerun()
    if st.button(f"Keep current {parameter}", key=f"keep_{parameter.lower()}"):
        _keep_current_m()
        st.rerun()


def _render_m_refinement_prompt() -> None:
    """Offer the next doubling after the current spectra have been plotted."""
    pending = st.session_state.get("m_refinement_request")
    if pending is not None:
        _m_refinement_dialog(pending)


def _custom_numerical(model: str, *, uniaxial: bool = False) -> NumericalSettings:
    def choice(column, label: str, key: str, options: tuple[int, ...]) -> int:
        current = st.session_state.get(key, options[0])
        if current not in options:
            try:
                current = int(current)
            except (TypeError, ValueError):
                current = options[0]
            st.session_state[key] = min(options, key=lambda value: (abs(value - current), value))
        descriptions = {
            "M": "Tip-height sampling resolution over half a tapping cycle. Increase to better resolve higher harmonics and sharp responses near closest approach.",
            "N": "Number of multipoles describing the tip response. More multipoles resolve tightly confined fields but increase computation time.",
            "K": "Number of image charges used to approximate isotropic layer reflection. Increase to check convergence, especially for thin layers and near resonances.",
        }
        return int(column.selectbox(label, options, key=key, help=descriptions[label]))

    columns = st.columns(3, gap="small")
    M = choice(columns[0], "M", f"{model}_custom_M", (30, 60, 80, 120, 240, 480))
    N = choice(columns[1], "N", f"{model}_custom_N",
               (8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 120, 160, 200, 240, 320, 400))
    quadQ = max(112, 3 * N // 2)  # same rule as the presets
    K = (
        choice(columns[2], "K", "layer_custom_K", (2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40))
        if model == "layer" and not uniaxial
        else 10
    )
    # Direct integration is insensitive to the tolerance, so it is not a user control.
    return NumericalSettings(M, N, quadQ, K, 1e-6 if uniaxial else 1e-5)
