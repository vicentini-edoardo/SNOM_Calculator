"""Public finite-layer and multilayer spectra for spheroid tips."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._spectra import calculate_spectra
from .materials import MaterialSpectrum
from .models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters


def calculate_layer_harmonics(
    frequency_cm1: np.ndarray,
    *,
    layer: MaterialSpectrum,
    substrate: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    thickness_nm: float,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: Sequence[int],
    far_field: FarFieldSettings = FarFieldSettings(),
    workers: int = 1,
    method: str = "auto",
) -> dict[int, SpectrumResult]:
    """One film on a substrate; zero thickness is the bulk substrate."""
    return calculate_stack_harmonics(
        frequency_cm1, layers=((layer, thickness_nm),), substrate=substrate, reference=reference,
        tip=tip, numerical=numerical, harmonics=harmonics, far_field=far_field,
        workers=workers, method=method)


def calculate_stack_harmonics(
    frequency_cm1: np.ndarray,
    *,
    layers: Sequence[tuple[MaterialSpectrum, float]],
    substrate: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: Sequence[int],
    far_field: FarFieldSettings = FarFieldSettings(),
    workers: int = 1,
    method: str = "auto",
) -> dict[int, SpectrumResult]:
    """Films ordered from the tip toward the substrate, without convergence checks.

    Zero-thickness layers are dropped and identical neighbours merged; an empty
    stack is the bulk substrate. ``workers`` (1-4) split the frequency grid over
    processes. ``method`` selects the spheroid film coupling: ``gicm`` (image
    charges, isotropic films), ``fim`` (Fourier integral method) or ``real-axis``; ``auto`` chooses.
    """
    return calculate_spectra(frequency_cm1, layers, substrate, reference, tip, numerical, harmonics,
                             far_field, workers=workers, method=method)
