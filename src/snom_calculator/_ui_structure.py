"""Workbench styling, physical-scene hit areas and visual stack controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

import numpy as np
import streamlit as st

from ._ui_quick import _render_quick_controls

from ._ui_editors import (
    _custom_material_editor, _far_field_editor, _material_editor, _open_editor,
    _tip_editor,
)
from ._ui_state import (
    _add_layer, _calculation_model, _layer_number, _material_from_state, _material_summary,
    _remove_layer,
)


def _render_compact_header() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px;}
        [data-testid="stMetric"] {background: transparent;}
        .snom-kicker {color:#2f5f8f; font-size:.72rem; font-weight:700;
            letter-spacing:.12em; text-transform:uppercase; margin-bottom:.15rem;}
        .snom-rule {border-top:1px solid color-mix(in srgb, currentColor 18%, transparent);
            margin:.35rem 0 1rem;}
        .snom-stack-label {color:#5c6875; font-size:.72rem; font-weight:700;
            letter-spacing:.08em; text-transform:uppercase; margin:0 0 .35rem;}
        .st-key-physical_scene {position:relative;}
        .st-key-physical_scene [data-testid="stImage"] img {display:block; height:auto; width:100%;}
        .st-key-physical_scene .st-key-physical_tip,
        .st-key-physical_scene [class*="st-key-physical_layer_"],
        .st-key-physical_scene .st-key-physical_substrate,
        .st-key-physical_scene .st-key-physical_far_field {position:absolute; z-index:2;}
        .st-key-physical_scene .st-key-physical_tip {left:38%; top:5%; width:60%; height:51%;}
        .st-key-physical_scene .st-key-physical_tip button,
        .st-key-physical_scene [class*="st-key-physical_layer_"] button,
        .st-key-physical_scene .st-key-physical_substrate button,
        .st-key-physical_scene .st-key-physical_far_field button {
            background:transparent; border:1px solid transparent; border-radius:6px;
            color:transparent; cursor:pointer; height:100%; inset:0; min-height:0;
            padding:0; position:absolute; transition:background .18s ease, box-shadow .18s ease, transform .18s ease; width:100%;}
        .st-key-physical_scene button p {font-size:0;}
        .st-key-physical_scene button:hover {background:#2f5f8f08; border-color:#2f5f8f;
            box-shadow:0 3px 12px #2f5f8f20; transform:translateY(-2px);}
        .st-key-physical_scene .st-key-physical_far_field button:hover,
        .st-key-physical_scene .st-key-physical_far_field button:focus-visible {
            background:#d7664930; border-color:transparent; box-shadow:none; transform:none;}
        .st-key-physical_scene button:focus-visible {outline:3px solid #2f5f8f; outline-offset:2px;}
        .st-key-physical_reference button {background:#edf2f7; border:1px solid #c8d2de;
            color:#172534; justify-content:flex-start;
            line-height:1.25; padding:.55rem .7rem; text-align:left; transition:background .12s ease;}
        .st-key-physical_reference button:hover {background:#e1ebf4;}
        .st-key-physical_reference button {min-height:3.5rem;}
        .st-key-physical_scene .st-key-physical_add {position:absolute; z-index:3; width:max-content;}
        .st-key-physical_add button {background:#f8fafc; border:1px solid #2f5f8f;
            border-radius:6px; color:#172534; font-size:.8rem; font-weight:600;
            height:2rem; min-height:2rem; min-width:6.2rem; padding:0 .65rem; white-space:nowrap;
            width:max-content !important;}
        .st-key-physical_add button p {font-size:0; line-height:1; margin:0;}
        .st-key-physical_add button::before {content:"+  Add layer"; color:#172534; font-size:.8rem;
            font-weight:600; line-height:1;}
        .st-key-physical_add button::first-letter {color:#2f5f8f;}
        .st-key-physical_add button:hover {background:#edf2f7; border-color:#244b70;}
        .st-key-physical_scene [class*="st-key-physical_remove_layer_"] {position:absolute; z-index:4; width:2rem;}
        .st-key-physical_scene [class*="st-key-physical_remove_layer_"] button {
            background:#fff; border:1px solid #9aa8b4; border-radius:50%; color:#7f3025;
            height:2rem; min-height:2rem; padding:0; width:2rem;}
        .st-key-physical_scene [class*="st-key-physical_remove_layer_"] button p {font-size:1rem; line-height:1; margin:0;}
        .st-key-physical_scene [class*="st-key-physical_remove_layer_"] button:hover {background:#fff1ee; border-color:#c35d4c;}
        .st-key-setup_upload [data-testid="stFileUploaderDropzone"] {
            background:transparent; border:0; min-height:0; padding:0;}
        .st-key-setup_upload [data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
        .st-key-setup_upload [data-testid="stFileUploaderDropzone"] button {
            min-height:2.6rem; width:100%;}
        .st-key-setup_upload [data-testid="stFileUploaderDropzone"] button p {font-size:0;}
        .st-key-setup_upload [data-testid="stFileUploaderDropzone"] button p::after {
            content:"Load setup"; font-size:.875rem;}
        [data-testid="stPlotlyChart"] {margin-bottom:.35rem;}
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {animation-duration:.01ms !important; transition-duration:.01ms !important;}
        }
        </style>
        <div class="snom-kicker">Spheroid model · Python engine</div>
        """,
        unsafe_allow_html=True,
    )
    st.title("s-SNOM Spectrum Calculator", anchor=False)
    st.caption(
        "Normalized near-field spectra for bulk and layered samples."
    )
    st.markdown('<div class="snom-rule"></div>', unsafe_allow_html=True)


def _scene_markup(layer_ids: Sequence[int], state: Mapping[str, object]) -> tuple[str, str]:
    """Return hit-area CSS and a scalable SVG for the current physical setup."""
    count = len(layer_ids)
    height = 328 + 24 * count
    far_opacity = ".82" if state["far_enabled"] else ".25"
    far_rotation = float(state["far_angle"]) - 45
    far_lines = np.array([[(102, 212), (202, 112)], [(88, 198), (188, 98)],
                          [(74, 184), (174, 84)], [(60, 170), (160, 70)]])
    far_paths = "".join(f'<path d="M{x1} {y1} L{x2} {y2}"/>' for (x1, y1), (x2, y2) in far_lines)
    # Pad the outer rays by 8 SVG units, then apply the drawing's transform.
    far_corners = far_lines[[-1, -1, 0, 0], [0, 1, 1, 0]].astype(float)
    far_corners += np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]) * (8 * np.sqrt(2))
    theta = np.deg2rad(far_rotation)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    far_corners = (far_corners - [210, 180]) @ rotation.T + [210, 125]
    far_polygon = ", ".join(f"{x / 520 * 100:.3f}% {y / height * 100:.3f}%" for x, y in far_corners)
    tip_shape = (
        '<ellipse aria-label="Spheroidal tip" cx="255" cy="124" rx="27" ry="65" fill="url(#snom-metal)" stroke="#617787" stroke-width="1.4"/>'
        '<path d="M248 69 C236 85 234 139 246 173" fill="none" stroke="#fff" stroke-opacity=".65" stroke-width="2"/>'
    )
    tip_material = _material_summary("tip")
    tip_material_short = escape(tip_material if len(tip_material) <= 24 else tip_material[:23] + "…")
    tip_material = escape(tip_material)
    tip_size = f"L = {float(state['tip_L']):g} nm"
    layers = []
    hit_areas = [f'''.st-key-physical_scene .st-key-physical_far_field {{
        left:0; top:0; width:100%; height:100%; clip-path:polygon({far_polygon});}}''']
    for index, layer_id in enumerate(layer_ids):
        top = 240 + 24 * index
        label = escape(_material_summary(f"layer_{layer_id}", thickness=float(state[f"layer_{layer_id}_thickness"])))
        layers.append(f'''
            <g class="snom-sample-shape">
                <path d="M40 {top-38} L200 {top} L200 {top+24} L40 {top-14}Z" fill="#a9c5d8" stroke="#7299b5"/>
                <path d="M200 {top} H475 V{top+24} H200Z" fill="url(#snom-layer)" stroke="#7299b5"/>
                <text x="209" y="{top+17}" class="snom-face-label">Layer {count - index} · {label}</text>
            </g>''')
        hit_areas.append(f'''.st-key-physical_scene .st-key-physical_layer_{layer_id} {{
            left:{200 / 520 * 100:.2f}%; top:{top / height * 100:.2f}%;
            width:{(475 - 200) / 520 * 100:.2f}%; height:{24 / height * 100:.2f}%;}}''')
        hit_areas.append(f'''.st-key-physical_scene .st-key-physical_remove_layer_{layer_id} {{
            left:97%; top:{(top + 12) / height * 100:.2f}%; transform:translate(-50%, -50%);}}''')
    substrate_top = 240 + 24 * count
    substrate_label = escape(_material_summary("substrate"))
    top_fill = "url(#snom-layer-top)" if count else "#9fb7c8"
    hit_areas.append(f'''.st-key-physical_scene .st-key-physical_substrate {{
        left:{200 / 520 * 100:.2f}%; top:{substrate_top / height * 100:.2f}%;
        width:{(475 - 200) / 520 * 100:.2f}%; height:{63 / height * 100:.2f}%;}}''')
    hit_areas.append(f'''.st-key-physical_scene .st-key-physical_add {{
        right:1%; top:{216 / height * 100:.2f}%; transform:translateY(-50%);}}''')
    svg = f'''<svg viewBox="0 0 520 {height}" role="img" aria-label="Isometric AFM tip above a layered sample. Select the tip, layer, or substrate to edit.">
            <defs>
                <linearGradient id="snom-metal" x2="1" y2=".15"><stop stop-color="#8a9ba7"/><stop offset=".46" stop-color="#edf2f5"/><stop offset="1" stop-color="#687d8d"/></linearGradient>
                <linearGradient id="snom-layer" x2="0" y2="1"><stop stop-color="#d9ecf7"/><stop offset="1" stop-color="#a8c8dd"/></linearGradient>
                <linearGradient id="snom-layer-top" x2="0" y2="1"><stop stop-color="#e9f5fb" stop-opacity=".9"/><stop offset="1" stop-color="#b6d4e7" stop-opacity=".9"/></linearGradient>
                <linearGradient id="snom-substrate" x2="0" y2="1"><stop stop-color="#46596a"/><stop offset="1" stop-color="#273644"/></linearGradient>
                <radialGradient id="snom-glow"><stop stop-color="#ed754e" stop-opacity=".72"/><stop offset=".45" stop-color="#ef9a72" stop-opacity=".28"/><stop offset="1" stop-color="#ef9a72" stop-opacity="0"/></radialGradient>
                <filter id="snom-shadow"><feGaussianBlur stdDeviation="8"/></filter>
            </defs>
            <ellipse cx="265" cy="{substrate_top+70}" rx="196" ry="13" fill="#4b6176" opacity=".14" filter="url(#snom-shadow)"/>
            <g class="snom-sample-shape" stroke-linejoin="round">
                <path d="M40 {substrate_top-38} L200 {substrate_top} V{substrate_top+63} L40 {substrate_top+25}Z" fill="#344653" stroke="#253645"/>
                <path d="M200 {substrate_top} H475 V{substrate_top+63} H200Z" fill="url(#snom-substrate)" stroke="#253645"/>
                <text x="209" y="{substrate_top+38}" class="snom-substrate-label">Substrate · {substrate_label} · semi-infinite</text>
            </g>
            {''.join(layers)}
            <g class="snom-sample-shape" stroke-linejoin="round">
                <path d="M40 202 H310 L475 240 H200Z" fill="{top_fill}" stroke="#749ab6" stroke-width="1.5"/>
                <path d="M40 202 L200 240" fill="none" stroke="#fff" stroke-opacity=".7"/>
            </g>
            <circle cx="255" cy="205" r="49" fill="url(#snom-glow)"/>
            <g class="snom-field-arc" fill="none" stroke="#d76649" stroke-opacity=".78" stroke-width="1.4" stroke-linecap="round">
                <path d="M225 193 Q255 230 286 193"/>
                <path d="M213 195 Q255 246 300 195"/>
                <path d="M200 198 Q255 261 314 198"/>
            </g>
            <g class="snom-plane-wave" transform="translate(0 -55) rotate({far_rotation:g} 210 180)" fill="none" stroke="#d76649" stroke-opacity="{far_opacity}" stroke-width="2" stroke-linecap="round">
                {far_paths}
            </g>
            <g class="snom-tip-shape" stroke-linejoin="round">
                {tip_shape}
            </g>
            <g aria-label="Tip · R = {float(state['tip_R']):g} nm · L = {float(state['tip_L']):g} nm · TA = {float(state['tip_A']):g} nm · {tip_material}">
                <rect x="319" y="52" width="190" height="124" rx="7" fill="#294a68"/>
                <text x="333" y="77" class="snom-tip-title">Tip</text>
                <text x="333" y="98" class="snom-tip-material">{tip_material_short}<title>{tip_material}</title></text>
                <text x="333" y="118" class="snom-tip-data">R = {float(state['tip_R']):g} nm</text>
                <text x="333" y="139" class="snom-tip-data">{tip_size}</text>
                <text x="333" y="160" class="snom-tip-data">TA = {float(state['tip_A']):g} nm</text>
            </g>
            <style>
                .snom-tip-title{{font:600 18px 'IBM Plex Sans',sans-serif;fill:#fff;}}
                .snom-tip-data{{font:500 15px 'IBM Plex Sans',sans-serif;fill:#fff;}}
                .snom-tip-material{{font:500 14px 'IBM Plex Sans',sans-serif;fill:#fff;}}
                .snom-face-label{{font:600 14px 'IBM Plex Sans',sans-serif;fill:#173650;}}
                .snom-substrate-label{{font:500 14px 'IBM Plex Sans',sans-serif;fill:#f2f6f8;}}
            </style>
        </svg>'''
    return "".join(hit_areas), svg


def _physical_setup(frequency: np.ndarray) -> dict[str, object]:
    st.subheader("Physical setup", anchor=False)
    st.caption("Air · semi-infinite superstrate")
    st.markdown('<p class="snom-stack-label">Tip and sample</p>', unsafe_allow_html=True)
    layer_ids = st.session_state["finite_layer_ids"]
    model = _calculation_model(layer_ids)
    with st.container(key="physical_scene"):
        hit_css, svg = _scene_markup(layer_ids, st.session_state)
        st.html(f"<style>{hit_css}</style>")
        st.image(svg, width="stretch")
        tip_label = (
            f"Tip · R {float(st.session_state['tip_R']):g} nm · "
            f"L {float(st.session_state['tip_L']):g} nm"
        )
        tip_material = _material_summary("tip")
        tip_label += f" · {tip_material}"
        if st.button(tip_label, key="physical_tip", help=f"Edit AFM tip · {tip_material}"):
            _open_editor("tip")
        if st.button(
            "Far-field reflection",
            key="physical_far_field",
            help="Configure far-field reflection and incidence angle",
        ):
            _open_editor("far_field")

        for layer_id in layer_ids:
            prefix = f"layer_{layer_id}"
            summary = _material_summary(prefix, thickness=float(st.session_state[f"{prefix}_thickness"]))
            number = _layer_number(layer_id, layer_ids)
            if st.button(f"Layer {number} · {summary}", key=f"physical_layer_{layer_id}", help=f"Edit layer {number}"):
                _open_editor(("layer", layer_id))
            if st.button("−", key=f"physical_remove_layer_{layer_id}", help=f"Remove layer {number}"):
                _remove_layer(layer_id)
                st.rerun()

        if st.button(
            f"Substrate · {_material_summary('substrate')} · semi-infinite",
            key="physical_substrate",
            help="Edit substrate",
        ):
            _open_editor("substrate")
        if st.button("Add finite layer", key="physical_add", help="Add a finite layer"):
            _add_layer()
            st.rerun()

    _render_quick_controls()

    st.divider()
    st.markdown("**Normalization reference**")
    if st.button(
        f"Reference · {_material_summary('reference')}",
        key="physical_reference",
        width="stretch",
        disabled=st.session_state["reference_disabled"],
    ):
        _open_editor("reference")
    reference_disabled = st.checkbox(
        "Do not use reference", key="reference_disabled",
        help="Show the sample amplitude and phase without reference normalization.",
    )

    editor = st.session_state.get("editor")
    if editor == "tip":
        _tip_editor()
    elif editor == "far_field":
        _far_field_editor()
    elif editor == "substrate":
        _material_editor("substrate", "substrate")
    elif editor == "reference":
        _material_editor("reference", "reference", allow_upload=False)
    elif isinstance(editor, tuple) and len(editor) == 3 and editor[0] == "custom_material":
        _, prefix, original_name = editor
        _custom_material_editor(str(prefix), original_name if isinstance(original_name, str) else None)
    elif isinstance(editor, tuple) and len(editor) == 2 and editor[0] == "layer":
        layer_id = editor[1]
        if layer_id in layer_ids:
            _material_editor(f"layer_{layer_id}", f"layer {layer_id}", layer_id=layer_id)

    reference = (None if reference_disabled else
                 _material_from_state("reference", frequency, allow_upload=False))
    reference_settings = {"reference": reference, "reference_disabled": reference_disabled}
    if model == "bulk":
        return {
            "sample": _material_from_state("substrate", frequency),
            **reference_settings,
        }
    if model == "layer":
        prefix = f"layer_{layer_ids[0]}"
        return {
            "layer": _material_from_state(prefix, frequency),
            "substrate": _material_from_state("substrate", frequency),
            **reference_settings,
            "thickness_nm": float(st.session_state[f"{prefix}_thickness"]),
        }
    if model == "multilayer":
        return {
            "layers": tuple(
                (_material_from_state(f"layer_{layer_id}", frequency),
                 float(st.session_state[f"layer_{layer_id}_thickness"]))
                for layer_id in layer_ids
            ),
            "substrate": _material_from_state("substrate", frequency),
            **reference_settings,
        }
    return {}
