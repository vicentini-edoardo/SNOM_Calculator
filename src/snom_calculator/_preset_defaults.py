"""Automatic M and N for the Weak / Strong / Very strong oscillator presets.

Fitted offline. For a group of (oscillator strength, material type, bulk or film), fitted for the spheroid tip:
M = max(M0, cM / (g/MARGIN)**q) and N = max(N0, cN / (g/MARGIN)**p), rounded up to the doubling
ladders below, with g the smallest tip-sample distance in nm.

Automatic picks the preset per material from max |beta|, beta = (eps - 1) / (eps + 1), over the
measurement window +- PAD_CM1.
A stack takes each layer's preset at its own distance from the tip and keeps the largest M and N.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .materials import effective_permittivity

PRESETS = ("Weak", "Strong", "Very strong")
M_LEVELS = (15, 30, 60, 120, 240, 480, 960)
N_LEVELS = (10, 20, 40, 80, 160, 320)
MARGIN = 1.25  # safety factor on the gap; the fit is tight at the training gaps

# (strength, material type, model) -> (M0, cM, q, N0, cN, p, fitted from gap in nm / MARGIN; 0 = no limit)
_FORMULA = {
    ("Strong", "isotropic", "bulk"): (30, 68.3852, 0.75, 10, 69.282, 0.5, 0.0),
    ("Strong", "isotropic", "film"): (30, 68.3852, 0.75, 80, 50, 1.0, 0.0),
    ("Strong", "uniaxial", "bulk"): (15, 162.648, 0.75, 10, 69.282, 0.5, 0.45),
    ("Strong", "uniaxial", "film"): (30, 144, 1.0, 10, 89.4427, 0.5, 0.45),
    ("Very strong", "isotropic", "bulk"): (15, 131.862, 0.75, 80, 50, 1.0, 0.0),
    ("Very strong", "isotropic", "film"): (15, 162.648, 0.75, 10, 160, 1.0, 0.6),
    ("Very strong", "uniaxial", "bulk"): (15, 268.328, 0.5, 10, 89.4427, 0.5, 0.45),
    ("Very strong", "uniaxial", "film"): (15, 325.297, 0.75, 10, 240, 1.0, 1.5),
    ("Weak", "isotropic", "bulk"): (30, 8.21584, 0.5, 20, 30.9839, 0.5, 0.0),
    ("Weak", "isotropic", "film"): (30, 8.21584, 0.5, 10, 43.8178, 0.5, 0.0),
    ("Weak", "uniaxial", "bulk"): (30, 8.21584, 0.5, 20, 12.2474, 0.5, 0.0),
    ("Weak", "uniaxial", "film"): (30, 8.21584, 0.5, 20, 12.2474, 0.5, 0.0),}


def _ceil(value: float, ladder: tuple[int, ...]) -> int | None:
    i = int(np.searchsorted(ladder, value - 1e-9))
    return ladder[i] if i < len(ladder) else None


def _formula_nm(preset: str, kind: str, film: bool, gap_nm: float) -> tuple[int, int] | None:
    M0, cM, q, N0, cN, p, floor = _FORMULA[preset, kind, "film" if film else "bulk"]
    g = gap_nm / MARGIN
    if g < floor - 1e-9:
        return None
    M, N = _ceil(max(M0, cM / g ** q), M_LEVELS), _ceil(max(N0, cN / g ** p), N_LEVELS)
    return None if M is None or N is None else (M, N)


def preset_nm(preset: str, kind: str, film: bool, gap_nm: float) -> tuple[int, int] | None:
    """(M, N) for the preset, or None where no ladder value converged (the app falls back to Custom).

    """
    nm = _formula_nm(preset, kind, film, gap_nm)
    return nm


# =============================================================================
# Image charges K for isotropic films
# =============================================================================

MAX_K = 16  # a film that needs more image charges uses FIM instead (K-free; the app has no K check)
# Preset -> ((film thickness in nm, K), ...): cheapest K within 3 % of the FIM truth in every training config of
# an offline study (11 materials, gaps 0.3-10 nm, Au and Si substrates), raised to the
# next row's K so each row also covers the thicknesses up to the next one. None: no K <= MAX_K reached 3 %.
# The app takes the row of the largest thickness <= the film's, since thinner is mostly harder.
_K_TABLE = {
    "Weak": ((2, 10), (5, 8), (10, 6), (30, 4)),
    "Strong": ((2, None), (10, 12), (30, 16), (300, 12)),
    "Very strong": ((2, None),),  # 30-300 nm stalls at 6 % for K = 16 (the fit of thick SiC3C plateaus)
}


def film_k(preset: str, thickness_nm: float) -> int | None:
    """Image charges for an isotropic film of this strength and thickness, or None for FIM:
    thinner than the study's thinnest film, or needing more than MAX_K."""
    rows = [row for row in _K_TABLE[preset] if row[0] <= thickness_nm + 1e-9]
    K = rows[-1][1] if rows else None
    return None if K is None or K > MAX_K else K


# Same selection for stacks (top film on 100 nm of another isotropic film on Au), by the strongest film's preset and
# the thinnest film. Strong and Very strong stacks miss 3 % at K = 16 (a polariton pole of r(k) near real k), so FIM.
_STACK_K_TABLE = {
    "Weak": ((2, 16), (5, 12), (10, 10), (30, 6)),
    "Strong": ((2, None),),
    "Very strong": ((2, None),),
}


def stack_k(films: Iterable[tuple[str, float]]) -> int | None:
    """K for isotropic (preset, thickness in nm) films; None (FIM) when GICM misses 3 % or needs more than MAX_K."""
    films = list(films)
    if len(films) < 2:
        return film_k(*films[0]) if films else None
    # Limit: studied stacks are two films with a 100 nm under-layer; extend the study before trusting thicker stacks.
    preset = max((preset for preset, _ in films), key=PRESETS.index)
    rows = [row for row in _STACK_K_TABLE[preset] if row[0] <= min(depth for _, depth in films) + 1e-9]
    K = rows[-1][1] if rows else None
    return None if K is None or K > MAX_K else K


# =============================================================================
# Automatic: preset from the permittivity
# =============================================================================

PAD_CM1 = 50.  # eps is read on the measurement window widened by this much on each side
BETA_CUTS = {  # Weak below the first value, Very strong from the second
    "isotropic": (5., 20.),
    "uniaxial": (5., 40.),
}


def material_kind(material) -> str:
    """Formula group of an isotropic or uniaxial material."""
    return "isotropic" if material.isotropic else "uniaxial"


def max_beta(material) -> float:
    """Largest |(eps - 1) / (eps + 1)| over the grid, for the effective eps that sets the quasi-static response:
    sqrt(eps_e eps_o) for a uniaxial crystal."""
    eps = effective_permittivity(material)
    return float(np.max(np.abs((eps - 1) / (eps + 1))))


def automatic_preset(material) -> str:
    """Weak, Strong or Very strong from max |beta| on the grid the material was evaluated on."""
    low, high = BETA_CUTS[material_kind(material)]
    beta = max_beta(material)
    return PRESETS[int(beta >= low) + int(beta >= high)]


def stack_nm(entries: Iterable[tuple[str, str, bool, float]]) -> tuple[int, int] | None:
    """Largest M and largest N over (preset, material type, film, distance from the tip in nm) entries:
    the top layer at the gap, a buried layer at the gap plus the thickness above it. None when any entry
    has no default at its distance."""
    picks = [preset_nm(preset, kind, film, distance) for preset, kind, film, distance in entries]
    if not picks or None in picks:
        return None
    return max(M for M, _ in picks), max(N for _, N in picks)
