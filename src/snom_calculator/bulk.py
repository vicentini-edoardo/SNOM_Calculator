"""Public bulk-sample spectra for spheroid tips, isotropic or uniaxial samples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._spectra import calculate_spectra
from .materials import MaterialSpectrum
from .models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters


def calculate_bulk_harmonics(
    frequency_cm1: np.ndarray,
    *,
    sample: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: Sequence[int],
    far_field: FarFieldSettings = FarFieldSettings(),
) -> dict[int, SpectrumResult]:
    """Normalized spectra {harmonic: SpectrumResult}, without convergence checks.

    A missing reference leaves the signal unnormalized.
    """
    return calculate_spectra(frequency_cm1, (), sample, reference, tip, numerical, harmonics, far_field)
