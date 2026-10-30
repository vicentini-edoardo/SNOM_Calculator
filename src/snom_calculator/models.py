"""Small shared contracts for both s-SNOM engines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TipParameters:
    radius_nm: float = 25.0
    length_nm: float = 600.0
    amplitude_nm: float = 40.0
    sample_gap_nm: float = 2.0
    reference_gap_nm: float = 2.0
    epsilon: complex = -1e10 + 0j

    def __post_init__(self) -> None:
        values = np.asarray(self.epsilon, dtype=complex)
        if values.ndim:
            raise ValueError("Tip permittivity must be scalar.")
        object.__setattr__(self, "epsilon", complex(values))


@dataclass(frozen=True)
class NumericalSettings:
    M: int = 30
    N: int = 40
    quadQ: int = 48
    K: int = 10
    momentum_rtol: float = 1e-5
    # Above M = 2 height_nodes - 2, solve at log-height nodes and interpolate (_quadrature); 0 never.
    height_nodes: int = 33


@dataclass(frozen=True)
class FarFieldSettings:
    enabled: bool = True
    angle_deg: float = 60.0
    coefficient: float = 1.0


@dataclass(frozen=True)
class SpectrumResult:
    frequency_cm1: np.ndarray
    sample_harmonic: np.ndarray
    reference_harmonic: np.ndarray
    contrast: np.ndarray
    amplitude: np.ndarray
    phase_rad: np.ndarray
    permittivities: Mapping[str, np.ndarray]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_contrast(
        cls,
        *,
        frequency_cm1: np.ndarray,
        sample_harmonic: np.ndarray,
        reference_harmonic: np.ndarray,
        contrast: np.ndarray,
        permittivities: Mapping[str, np.ndarray],
        metadata: Mapping[str, object],
    ) -> "SpectrumResult":
        phase = np.unwrap(np.angle(contrast))
        if phase.size and phase[0] < 0:
            phase += 2 * np.pi
        return cls(
            frequency_cm1=np.asarray(frequency_cm1, dtype=float),
            sample_harmonic=np.asarray(sample_harmonic, dtype=complex),
            reference_harmonic=np.asarray(reference_harmonic, dtype=complex),
            contrast=np.asarray(contrast, dtype=complex),
            amplitude=np.abs(contrast),
            phase_rad=phase,
            permittivities=permittivities,
            metadata=metadata,
        )


def max_multipoles(tip: TipParameters) -> int:
    """Largest usable N for this tip shape (requires L > 2R)."""
    # The tip denominators scale like Q_N/P_N(a/c) ~ exp(-2 N acosh(a/c)) and
    # underflow past this bound; blunter tips (larger R/L) reach it at lower N.
    return int(350 / math.acosh(1 / math.sqrt(1 - 2 * tip.radius_nm / tip.length_nm)))


def validate_inputs(
    frequency_cm1: np.ndarray,
    *,
    harmonic: int,
    tip: TipParameters,
    numerical: NumericalSettings,
    thickness_nm: float | None = None,
    harmonics: Sequence[int] | None = None,
) -> None:
    frequency = np.asarray(frequency_cm1, dtype=float)
    if frequency.ndim != 1 or frequency.size < 2 or not np.isfinite(frequency).all():
        raise ValueError("Frequency must be a finite one-dimensional grid with at least two points.")
    if not np.all(np.diff(frequency) > 0):
        raise ValueError("Frequency values must be strictly increasing.")
    if not np.isfinite([tip.radius_nm, tip.length_nm, tip.amplitude_nm,
                       tip.sample_gap_nm, tip.reference_gap_nm]).all() or not np.isfinite(tip.epsilon).all():
        raise ValueError("Tip parameters must be finite.")
    if tip.radius_nm <= 0 or tip.length_nm <= 2 * tip.radius_nm:
        raise ValueError("L must be greater than 2R so the spheroid geometry remains valid.")
    if tip.amplitude_nm < 0 or tip.sample_gap_nm <= 0 or tip.reference_gap_nm <= 0:
        raise ValueError("Amplitude must be non-negative and sample/reference gaps must be greater than zero.")
    if any(isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
           or order < 0 for order in (harmonic, *(harmonics or ()))):
        raise ValueError("Harmonic must be a non-negative integer.")
    if any(isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
           or order < 1 for order in (numerical.M, numerical.N, numerical.quadQ, numerical.K)):
        raise ValueError("Numerical settings must be positive integers.")
    if (isinstance(numerical.height_nodes, (bool, np.bool_)) or not isinstance(numerical.height_nodes, (int, np.integer))
            or numerical.height_nodes in (1, 2) or numerical.height_nodes < 0):
        raise ValueError("height_nodes must be 0 or an integer of at least 3.")
    if tip.amplitude_nm != 0 and numerical.M <= harmonic:
        raise ValueError("M must be greater than the selected harmonic.")
    if numerical.quadQ < numerical.N:
        raise ValueError("quadQ must be at least N.")
    n_max = max_multipoles(tip)
    if numerical.N > n_max:
        raise ValueError(f"N = {numerical.N} is too large for this tip shape (R/L); use N ≤ {n_max}.")
    if not np.isfinite(numerical.momentum_rtol) or not 0 < numerical.momentum_rtol < 1:
        raise ValueError("Momentum tolerance must be finite and between 0 and 1.")
    if thickness_nm is not None and (not np.isfinite(thickness_nm) or thickness_nm < 0):
        raise ValueError("Layer thickness must be finite and non-negative.")
