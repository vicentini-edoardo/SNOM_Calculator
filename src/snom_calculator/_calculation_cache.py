"""Streamlit cache boundary; profile mathematics lives in the engines.

Only material/geometry/numerical inputs determine cached profiles. Far-field
factors, checks and processing remain outside the cache; sample profiles do not
depend on the reference choice. Worker count is excluded from the cache key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
import pickle

import numpy as np
import streamlit as st

from . import calculations
from ._spectra import _merge, solve_profile
from .materials import MaterialSpectrum
from .models import NumericalSettings, SpectrumResult, TipParameters


@st.cache_data(show_spinner=False, max_entries=64,
               hash_funcs={complex: lambda value: (value.real, value.imag)})
def _cached_profile(
    layers: tuple[tuple[MaterialSpectrum, float], ...],
    substrate: MaterialSpectrum,
    tip: TipParameters,
    gap: float,
    numerical: NumericalSettings,
    harmonics: tuple[int, ...],
    method: str,
    _workers: int,  # leading underscore: not part of the cache key
    _previous=None,  # an optimization, not part of the physical inputs
    fallback: bool = True,
) -> tuple[np.ndarray, dict]:
    if _previous is None:
        return solve_profile(layers, substrate, tip, gap, numerical, harmonics, method, _workers, fallback=fallback)
    values, info = solve_profile(layers, substrate, tip, gap, numerical, harmonics, method, _workers,
                                 previous=_previous[0], fallback=fallback)
    _merge(info, _previous[1])
    return values, info


def _cached_bulk(frequency, sample, reference, tip, numerical, harmonics, far_field,
                 check_n: bool = True, *, solve=None, **gaps) -> dict[int, SpectrumResult]:
    return calculations.calculate_bulk_checked(
        frequency, sample, reference, tip, numerical, harmonics, far_field, check_n,
        solve=_cached_profile if solve is None else solve, **gaps)


def _cached_layer(frequency, layer, substrate, reference, thickness_nm, tip, numerical, harmonics,
                  far_field, workers, check_n: bool = True, *, solve=None, method: str = "auto",
                  **gaps) -> dict[int, SpectrumResult]:
    return calculations.calculate_layer_checked(
        frequency, layer, substrate, reference, thickness_nm, tip, numerical, harmonics,
        far_field, workers, check_n, solve=_cached_profile if solve is None else solve, method=method, **gaps)


def _cached_stack(frequency, layers, substrate, reference, tip, numerical, harmonics, far_field,
                  workers, check_n: bool = True, *, solve=None, method: str = "auto",
                  **gaps) -> dict[int, SpectrumResult]:
    return calculations.calculate_stack_checked(
        frequency, layers, substrate, reference, tip, numerical, harmonics, far_field,
        workers, check_n, solve=_cached_profile if solve is None else solve, method=method, **gaps)


def _reusing_solver(controls: Mapping[str, object], profiles: dict, current_profiles: dict):
    """A ``solve_profile`` stand-in that keeps and reuses tapping profiles between runs.

    ``profiles`` maps pickled profile inputs to the ``(values, info)`` kept from the last run.
    Profiles at the requested ``controls["numerical"]`` are recorded in ``current_profiles``. When
    ``controls["_reuse_M"]`` is set, the matching profile at that smaller M is passed on as
    ``_previous`` so only the new midpoint heights are solved; a missing one is an error.
    Convergence-check profiles (other numerical settings) are neither reused nor kept.
    """
    def solve(layers, substrate, tip, gap, numerical, harmonics, method, workers, **options):
        inputs = (layers, substrate, tip, gap, numerical, harmonics, method)
        previous = None
        if controls.get("_reuse_M") is not None and numerical == controls["numerical"]:
            smaller = (*inputs[:4], replace(numerical, M=controls["_reuse_M"]), *inputs[5:])
            previous = profiles.get(pickle.dumps(smaller))
            if previous is None:
                raise ValueError("Previous tapping profile is unavailable; calculate the spectrum again.")
        values = _cached_profile(*inputs, workers, **({} if previous is None else {"_previous": previous}), **options)
        if numerical == controls["numerical"]:
            current_profiles[pickle.dumps(inputs)] = values
        return values
    return solve


def _run_calculations(
    model: str,
    frequency: np.ndarray,
    materials: Mapping[str, object],
    controls: Mapping[str, object],
    harmonics: Sequence[int],
) -> dict[int, SpectrumResult]:
    profiles = controls.get("_profiles")
    current_profiles = {}
    solve = _cached_profile if profiles is None else _reusing_solver(controls, profiles, current_profiles)
    results = calculations.run_calculations(
        model, frequency, materials, controls, harmonics,
        solve_bulk=partial(_cached_bulk, solve=solve), solve_layer=partial(_cached_layer, solve=solve),
        solve_stack=partial(_cached_stack, solve=solve),
    )
    if profiles is not None:
        profiles.clear()
        profiles.update(current_profiles)
    return results
