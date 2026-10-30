"""Shared Legendre functions, Gauss-Legendre quadrature and the tapping rule.

Geometry, GICM fits and momentum integration use these same routines.
Every engine solves the tip response at ``profile_grid`` heights and callers
demodulate with ``profile_weights``: the M + 1 ``tapping_heights`` for small M,
or Chebyshev-Lobatto nodes in log height, interpolated to the tapping heights,
when M + 1 exceeds 2 ``NumericalSettings.height_nodes`` - 1 (see ``interpolated``).
Averaged gaps can share one node profile over their joint range (``gap_span``).
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple, Union

import numpy as np

ArrayOrScalar = Union[complex, float, np.ndarray]


def gauss_legendre_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Golub-Welsch nodes and weights for Gauss-Legendre quadrature on [-1, 1]."""
    beta = np.arange(1, n) / np.sqrt(4.0 * np.arange(1, n) ** 2 - 1.0)
    jacobi = np.diag(beta, 1) + np.diag(beta, -1)
    eigval, eigvec = np.linalg.eigh(jacobi)
    order = np.argsort(eigval)
    x = eigval[order]
    w = 2.0 * eigvec[0, order] ** 2
    return x, w


def _points(x: ArrayOrScalar) -> np.ndarray:
    """Flat float points, or complex ones for analytically continued arguments."""
    x = np.atleast_1d(np.asarray(x))
    return x.astype(np.result_type(x, float)).ravel()


def legendre_P_all(n: int, x: ArrayOrScalar) -> np.ndarray:
    """P[degree, point] = P_degree(x) for degree = 0..n, via upward recurrence."""
    x = _points(x)
    P = np.zeros((n + 1, x.size), dtype=x.dtype)
    P[0, :] = 1.0
    if n >= 1:
        P[1, :] = x
    for k in range(1, n):
        P[k + 1, :] = ((2 * k + 1) / (k + 1)) * x * P[k, :] - (k / (k + 1)) * P[k - 1, :]
    return P


def _continued_fraction_Hn(n: int, x: np.ndarray) -> np.ndarray:
    """Vectorized continued fraction used by the stable Legendre-Q evaluation."""
    eps_tol = 1e-15
    tiny = 1e-30
    max_iter = 10000

    x = _points(x)
    Hn = np.full(x.shape, tiny, dtype=x.dtype)
    Cj1 = Hn.copy()
    Dj1 = np.zeros_like(x)

    aj = 1.0
    bj = (2.0 + 1.0 / n) * x
    Dj = bj + aj * Dj1
    Dj[np.abs(Dj) < tiny] = tiny
    Cj = bj + aj / Cj1
    Cj[np.abs(Cj) < tiny] = tiny
    Dj = 1.0 / Dj
    delta = Cj * Dj
    Hn = Hn * delta
    Dj1, Cj1 = Dj, Cj

    converged = np.abs(delta - 1.0) < eps_tol
    j = 0
    while (not converged.all()) and j < max_iter:
        aj = -(1.0 + 1.0 / (n + j))
        bj = (2.0 + 1.0 / (n + j + 1)) * x
        Dj = bj + aj * Dj1
        Dj[np.abs(Dj) < tiny] = tiny
        Cj = bj + aj / Cj1
        Cj[np.abs(Cj) < tiny] = tiny
        Dj = 1.0 / Dj
        delta = Cj * Dj
        # Converged entries stop updating, so a value never depends on which
        # other points share the batch (and how many iterations they need).
        Hn = np.where(converged, Hn, Hn * delta)

        converged |= np.abs(delta - 1.0) < eps_tol
        Dj1, Cj1 = Dj, Cj
        j += 1

    if not converged.all():
        warnings.warn(f'Legendre Q_{n} did not converge at every point; check the result '
                      '(very small gaps or sharp tips).', stacklevel=3)
    return Hn


def legendre_Q_all(n: int, x: ArrayOrScalar) -> np.ndarray:
    """Stable Legendre functions Q_0..Q_n for real x > 1, or complex x off [-1, 1].

    One continued fraction seeds Q_n/Q_(n-1). The degree recurrence gives
    the lower ratios downward, then Q_0 fixes the scale. This avoids both
    an unstable upward recurrence and evaluating overflowing P_n values.
    Complex x uses the principal log1p, whose cut is Q_0's cut [-1, 1].
    """
    x = _points(x)
    Q = np.zeros((n + 1, x.size), dtype=x.dtype)
    Q[0, :] = 0.5 * np.log1p(2.0 / (x - 1.0))
    if n >= 1:
        Q[1, :] = x * Q[0, :] - 1.0
    if n <= 1:
        return Q

    Q[n] = _continued_fraction_Hn(n, x)
    for degree in range(n, 1, -1):
        Q[degree - 1] = (degree - 1) / ((2 * degree - 1) * x - degree * Q[degree])
    for degree in range(1, n + 1):
        Q[degree] *= Q[degree - 1]
    return Q


def tapping_heights(gap: float, amplitude: float, M: int) -> np.ndarray:
    """Apex heights (nm) at M+1 phases psi_k = pi k/M over half a tapping cycle.

    H(psi) = gap + amplitude - amplitude cos(psi); M = 0 is the static tip at ``gap``.
    """
    top = amplitude + gap
    if M == 0:
        return np.array([top - amplitude])
    return top - amplitude * np.cos(np.pi * np.arange(M + 1) / M)


def demodulation_weights(M: int, harmonics: Sequence[int]) -> np.ndarray:
    """(harmonic, M+1) trapezoidal cosine-transform weights on ``tapping_heights``.

    The response is even in tapping phase, so the full-period transform reduces
    to M+1 heights: endpoints have half weight, the last one with its complex
    phase. M = 0 keeps only the static (n = 0) signal.
    """
    if M == 0:
        return np.array([[1. if n == 0 else 0.] for n in harmonics], dtype=complex)
    k = np.arange(M + 1)
    weights = np.empty((len(harmonics), M + 1), dtype=complex)
    for row, n in zip(weights, harmonics):
        row[:] = (np.full(M + 1, 1 / M) if n == 0
                  else np.cos(np.pi * k * n / M).astype(complex) / M)
        row[0] = 1.0 / (2 * M)
        row[M] = np.exp(-1j * np.pi * M * n / M) / (2 * M)
    return weights


# The dipole is analytic in log height above the sample, so a Chebyshev interpolant in log h
# replaces solving every tapping height. ``NumericalSettings.height_nodes`` Lobatto nodes are
# doubled (nested, fewer than M + 1 and at most MAX_HEIGHT_NODES) while the Chebyshev tail of the
# profile is between HEIGHT_TAIL_LIMIT and HEIGHT_TAIL_ESCALATE; a tail still above the limit
# falls back to every tapping height. On 52 lab cases (0.2-5 nm gaps, amplitudes 10-100 nm, every
# engine) the pointwise interpolation error was about twice that tail. Near-resonant coupling at
# 0.2-0.5 nm gaps (large amplitudes, blunt tips) left tails of 1e-3 to 0.2: those fall back.
# Interpolation starts above 2 * height_nodes - 1 tapping heights, so the first doubling never
# solves more heights than the tapping grid.
MAX_HEIGHT_NODES = 129
HEIGHT_TAIL_LIMIT = 1e-8
HEIGHT_TAIL_ESCALATE = 1e-3


def interpolated(amplitude: float, M: int, nodes: int) -> bool:
    """True when the profile is solved at ``nodes`` log-height nodes instead of the M + 1 tapping heights."""
    return nodes > 0 and amplitude > 0 and M + 1 > 2 * nodes - 1


def _log_position(gap: float, amplitude: float, heights: np.ndarray) -> np.ndarray:
    """Heights mapped to [-1, 1] in log height over the tapping range."""
    lo, hi = np.log(gap), np.log(gap + 2 * amplitude)
    return (2 * np.log(heights) - lo - hi) / (hi - lo)


def height_nodes(gap: float, amplitude: float, count: int) -> np.ndarray:
    """``count`` Chebyshev-Lobatto nodes in log height, ascending from ``gap`` to ``gap + 2 amplitude``."""
    lo, hi = np.log(gap), np.log(gap + 2 * amplitude)
    nodes = np.exp((lo + hi) / 2 - (hi - lo) / 2 * np.cos(np.pi * np.arange(count) / (count - 1)))
    nodes[0], nodes[-1] = gap, gap + 2 * amplitude
    return nodes


def interpolation_matrix(gap: float, amplitude: float, heights: np.ndarray, count: int) -> np.ndarray:
    """(heights, count) barycentric Chebyshev-Lobatto weights from ``height_nodes`` to ``heights``."""
    x = -np.cos(np.pi * np.arange(count) / (count - 1))
    signs = (-1.) ** np.arange(count)
    signs[[0, -1]] *= .5
    diff = _log_position(gap, amplitude, np.asarray(heights, dtype=float))[:, None] - x[None, :]
    hit = np.isclose(diff, 0, rtol=0, atol=1e-14)
    with np.errstate(divide='ignore'):
        matrix = signs / diff
    on_node = hit.any(1)
    matrix[on_node] = hit[on_node]
    return matrix / matrix.sum(1, keepdims=True)


def interpolation_tail(values: np.ndarray) -> float:
    """Largest of the last three Chebyshev coefficients over the largest, worst column of a node profile."""
    n = values.shape[0]
    k = np.arange(n)
    transform = np.cos(np.pi * np.outer(k, k) / (n - 1)) * 2 / (n - 1)
    transform[:, [0, -1]] *= .5
    coefficients = transform @ values
    coefficients[[0, -1]] *= .5
    peak = np.max(abs(coefficients), axis=0)
    return float(np.max(np.max(abs(coefficients[-3:]), axis=0) / np.where(peak > 0, peak, 1.)))


def profile_weights(gap: float, amplitude: float, M: int, harmonics: Sequence[int], nodes: int = 0) -> np.ndarray:
    """(harmonic, heights) weights that demodulate with M phases a profile on the M + 1 tapping
    heights, or on ``nodes`` log-height nodes when nonzero."""
    if amplitude == 0:
        return demodulation_weights(0, harmonics)
    weights = demodulation_weights(M, harmonics)
    if not nodes:
        return weights
    return weights @ interpolation_matrix(gap, amplitude, tapping_heights(gap, amplitude, M), nodes)


# Averaged gaps share one log-height profile from the smallest gap to the top of the largest gap's
# cycle: the tapping range of a tip at ``gap_span``. It is solved at SPAN_M, which always interpolates
# and lets the nodes double up to MAX_HEIGHT_NODES, so the profile does not depend on M.
SPAN_M = 2 * MAX_HEIGHT_NODES


def gap_span(gaps: Sequence[float], amplitude: float) -> Tuple[float, float]:
    """(gap, amplitude) of the tapping range that covers every gap's cycle."""
    return min(gaps), amplitude + (max(gaps) - min(gaps)) / 2


def spanned(gaps: Sequence[float], amplitude: float, M: int, nodes: int) -> bool:
    """True when one span profile solves fewer heights than every gap's M + 1 tapping heights."""
    return len(gaps) > 1 and interpolated(amplitude, len(gaps) * (M + 1) - 1, nodes)


def span_weights(gaps: Sequence[float], amplitude: float, M: int, harmonics: Sequence[int], nodes: int) -> np.ndarray:
    """(gap x harmonic, nodes) weights that demodulate each gap with M phases from a ``gap_span`` node profile."""
    gap, top = gap_span(gaps, amplitude)
    weights = demodulation_weights(M, harmonics)
    return np.vstack([weights @ interpolation_matrix(gap, top, tapping_heights(g, amplitude, M), nodes)
                      for g in gaps])


def profile_grid(gap, amplitude, M, harmonics, previous=None, nodes=0):
    """Heights, demodulation weights and reused signal for a full or doubled grid.

    Without ``previous`` (a profile on the M/2 grid, shape (M/2 + 1, frequencies)) this is
    the full M + 1 heights and weights with a reused signal of 0. With it, only the
    midpoint heights and their weight columns are returned, and ``reused`` is the
    demodulated contribution of the known even heights, shape (harmonics, frequencies);
    callers add it to the demodulated midpoint signal. When ``interpolated``, the grid is
    ``nodes`` log-height nodes, and ``previous`` (n nodes) asks for the n - 1 new nodes
    of the nested 2n - 1 set.
    """
    if interpolated(amplitude, M, nodes):
        count = nodes if previous is None else 2 * previous.shape[0] - 1
        heights = height_nodes(gap, amplitude, count)
        weights = profile_weights(gap, amplitude, M, harmonics, count)
    else:
        heights = tapping_heights(gap, amplitude, M)
        weights = demodulation_weights(M, harmonics)
        if previous is not None and (amplitude == 0 or M <= 0 or M % 2 or previous.ndim != 2
                                     or previous.shape[0] != M // 2 + 1):
            raise ValueError("Reused profile must have M/2 + 1 heights on a doubled tapping grid.")
    if previous is None:
        return heights, weights, 0.
    return heights[1::2], weights[:, 1::2], weights[:, ::2] @ previous


def complete_profile(values, previous=None):
    """Interleave newly solved midpoint heights with the previous profile."""
    if previous is None:
        return values
    output = np.empty((2 * values.shape[0] + 1, values.shape[1]), dtype=complex)
    output[::2], output[1::2] = previous, values
    return output
