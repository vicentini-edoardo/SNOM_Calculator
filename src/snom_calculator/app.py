"""Streamlit entry point and three-column scientific-workbench layout."""

from __future__ import annotations

import streamlit as st

from ._ui_controls import (
    _calculation_action, _frequency_toolbar, _precision_controls, _render_m_refinement_prompt,
)
from ._ui_results import _render_harmonic_controls, _results_controls, _results_workspace
from ._ui_state import _calculation_model, _initialize_state
from ._ui_structure import _physical_setup, _render_compact_header


def run_app() -> None:
    st.set_page_config(
        page_title="s-SNOM Spectrum Calculator",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _initialize_state()
    _render_compact_header()

    left, center, right = st.columns([1, 2, 1], gap="medium", vertical_alignment="top")
    with right:
        frequency = _frequency_toolbar()
    with left:
        materials = _physical_setup(frequency)
    model = _calculation_model(st.session_state["finite_layer_ids"])
    with right:
        controls = _precision_controls(model, materials, frequency)
        _render_harmonic_controls()

        _calculation_action(model, frequency, materials, controls)
        _results_controls()
    with center:
        _results_workspace(st.session_state.get("results", {}))
    with right:
        _render_m_refinement_prompt()
