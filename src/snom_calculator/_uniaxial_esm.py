"""Exact quasi-static reflection integrated against the spheroid momentum basis."""

from functools import lru_cache

import numba
import numpy as np
from numpy.polynomial import legendre

from ._quadrature import gauss_legendre_rule, legendre_P_all
from ._progress import report


def _gamma(perpendicular: complex, parallel: complex) -> complex:
    """Passive root of epsilon_perp/epsilon_parallel (exp(-i omega t))."""
    if parallel == 0 or perpendicular == 0:
        raise ValueError("Tensor permittivity components must be nonzero.")
    if perpendicular.imag < 0 or parallel.imag < 0:
        raise ValueError("Passive media require Im(epsilon) >= 0.")
    # Limiting absorption selects the outgoing branch at an exact lossless pole.
    scale = max(1.0, abs(perpendicular), abs(parallel))
    delta = 1e-14 * scale
    p = perpendicular + (1j * delta if perpendicular.imag == 0 else 0)
    z = parallel + (1j * delta if parallel.imag == 0 else 0)
    root = complex(np.sqrt(p / z))
    return root if root.real >= 0 else -root


def reflection(k: np.ndarray, layers: tuple[tuple[complex, complex, float], ...],
               substrate: complex, ambient: complex = 1) -> np.ndarray:
    """Charge-potential r_e for aligned uniaxial layers, top to bottom.

    Layer tuples are (epsilon_perp, epsilon_parallel, thickness_nm). The
    isotropic half-space convention is r_e=(ambient-epsilon)/(ambient+epsilon).
    """
    k = np.asarray(k, dtype=float)
    return np.broadcast_to(_reflections(k, layers, substrate, ambient)[2][0], k.shape)


def _reflections(k: np.ndarray, layers, substrate: complex, ambient: complex):
    """(gamma per layer, interface Fresnel terms, generalized reflections).

    Interface i is the top of medium i (layers top to bottom, then the
    substrate); ``reflections[i]`` is seen from above at interface i.
    """
    if np.any(k < 0) or not np.isfinite(k).all():
        raise ValueError("Momentum must be finite and nonnegative.")
    if any(not np.isfinite(d) or d < 0 for _, _, d in layers):
        raise ValueError("Layer thickness must be finite and nonnegative.")
    gamma = [_gamma(complex(p), complex(z)) for p, z, _ in layers]
    admittance = [complex(ambient),
                  *(complex(z) * g for (_, z, _), g in zip(layers, gamma)),
                  complex(substrate)]
    fresnel = [(admittance[index] - admittance[index + 1]) / (admittance[index] + admittance[index + 1])
               for index in range(len(admittance) - 1)]
    r = fresnel[-1]
    reflections = [r]
    for index in range(len(layers) - 1, -1, -1):
        a = fresnel[index]
        t = np.exp(-2 * k * gamma[index] * layers[index][2])
        r = (a + r * t) / (1 + a * r * t)
        reflections.append(r)
    return gamma, fresnel, reflections[::-1]


def _pole_breaks(layers, substrate, limit, max_width=None):
    """Resolve the single-film Fabry-Perot poles nearest the real k axis.

    Poles with half-width at least ``max_width`` (default limit/8) are skipped.
    """
    if len(layers) != 1:
        return []
    perpendicular, parallel, depth = layers[0]
    if depth == 0:
        return []
    gamma = _gamma(complex(perpendicular), complex(parallel))
    admittance = parallel * gamma
    a = (1 - admittance) / (1 + admittance)
    b = (admittance - substrate) / (admittance + substrate)
    if a * b == 0:
        return []
    count = int(np.ceil(limit * abs(gamma) * depth / np.pi)) + 8
    points = []
    for n in range(-count, count + 1):
        pole = (np.log(-a * b) - 2j * np.pi * n) / (2 * gamma * depth)
        center, width = pole.real, max(abs(pole.imag), 1e-8)
        if 0 < center < limit and width < (limit / 8 if max_width is None else max_width):
            points.extend(center + width * offset for offset in (-16, -4, -1, 0, 1, 4, 16))
    return [point for point in points if 0 < point < limit]


@numba.njit(cache=True)
def _scaled_spherical_i(x: np.ndarray, N: int) -> np.ndarray:
    """exp(-x) i_n(x) for n=1..N by Miller downward recurrence; zero at x=0.

    Equals sqrt(pi/(2x)) ive(n+1/2, x) to ~1e-14, about 100x faster than ive.
    """
    out = np.zeros((x.size, N))
    for q in range(x.size):
        xq = x[q]
        if xq <= 0:
            continue
        above, current = 0.0, 1e-30
        for n in range(N + 40 + int(xq + 4 * np.sqrt(xq)), 0, -1):
            above, current = current, above + (2 * n + 1) / xq * current
            if 1 <= n - 1 <= N:
                out[q, n - 2] = current
            if current > 1e200:
                above *= 1e-200
                current *= 1e-200
                out[q] *= 1e-200
        # current now holds the unnormalized i_0; exp(-x) i_0(x) = (1-exp(-2x))/(2x).
        out[q] *= -np.expm1(-2 * xq) / (2 * xq) / current
    return out


@lru_cache(maxsize=1)
def _gauss_kronrod_rule(n: int = 15):
    """Kronrod 2n+1 nodes/weights plus embedded Gauss n weights on [-1, 1].

    Kronrod nodes add the roots of the Stieltjes polynomial, orthogonal
    to degree <= n under the weight P_n; weights solve Legendre moments.
    """
    xg, wg = gauss_legendre_rule(n)
    t, tw = gauss_legendre_rule(3 * n)
    P = np.array([legendre.legval(t, np.eye(n + 2)[j]) for j in range(n + 2)])
    A = (P[n] * tw * P[: n + 1]) @ P.T
    c = np.r_[np.linalg.lstsq(A[:, :n + 1], -A[:, n + 1], rcond=None)[0], 1]
    x = np.sort(np.r_[xg, legendre.legroots(c).real])
    V = np.array([legendre.legval(x, np.eye(2 * n + 1)[j]) for j in range(2 * n + 1)])
    wk = np.linalg.solve(V, np.r_[2, np.zeros(2 * n)])
    return x, wk, np.searchsorted(x, xg), wg


@lru_cache(maxsize=8)
def _geometry(a: float, c: float, N: int):
    """Cached row factors for the analytic spheroid surface projection."""
    return legendre_P_all(N, a / c)[1:, 0]


def _momentum_basis(momentum, a: float, c: float, N: int) -> np.ndarray:
    """Analytic surface projection; only the momentum integral remains.

    B_ij(k) = 2c P_i(a/c) exp(-2k(a-c)) [exp(-kc)i_i(kc)] [exp(-kc)i_j(kc)].
    """
    k = np.asarray(momentum, dtype=float)
    scaled_i = _scaled_spherical_i(k * c, N)
    projected = (2 * c * _geometry(a, c, N)[None, :]
                 * np.exp(-2 * k * (a - c))[:, None] * scaled_i)
    # Panel nodes rarely repeat; cache only the O(N) geometry factors above.
    return projected[:, :, None] * scaled_i[:, None, :]


def coupling_matrices(heights: np.ndarray,
                      layers: tuple[tuple[complex, complex, float], ...],
                      substrate: complex, *, a: float, c: float, N: int,
                      rtol: float = 1e-5) -> np.ndarray:
    """Adaptive panel integration of r_e(k) times the spheroidal kernel, shape (height, N, N).

    The tail beyond the initial cutoff is integrated in doubling passes until it
    falls below ``rtol`` of the integral.
    """
    heights = np.atleast_1d(np.asarray(heights, dtype=float))
    if np.any(heights <= 0) or rtol <= 0:
        raise ValueError("Heights and rtol must be positive.")
    gap = 2 * min(heights) + a - c
    limit = -np.log(rtol * 1e-3) / gap
    gamma = [_gamma(complex(p), complex(z)) for p, z, _ in layers]
    oscillation = max((abs(g.imag) * d for g, (_, _, d) in zip(gamma, layers)), default=0)
    width = min(limit, np.pi / (4 * oscillation)) if oscillation else limit
    nodes, kronrod, gauss, weights = _gauss_kronrod_rule()
    # Kronrod minus embedded Gauss weights: the error estimate is one matmul.
    difference = kronrod.copy()
    difference[gauss] -= weights

    # Values stay real with rows [Re; Im] per height: real BLAS, no complex temporaries.
    def peak(stacked):
        return float(np.max(np.hypot(*stacked.reshape(2, -1))))

    def estimate(lo, hi):
        """Kronrod value and its distance from the embedded Gauss value."""
        ks = (lo + hi) / 2 + (hi - lo) / 2 * nodes
        basis = _momentum_basis(ks, a, c, N)
        basis = basis.reshape(ks.size, N * N)
        factors = (hi - lo) / 2 * reflection(ks, layers, substrate)[None, :] * np.exp(
            -2 * heights[:, None] * ks[None, :])
        factors = np.concatenate([factors.real, factors.imag])
        error = peak(factors * difference @ basis)
        return factors * kronrod @ basis, error

    # Panels add into ``out`` in place so recursion never holds more than
    # one (2 * heights, N * N) partial result at a time.
    def panel(lo, hi, out, depth=0):
        fine, error = estimate(lo, hi)
        # Error is measured against the whole integral: a panel-local scale
        # never converges where the integrand cancels (e.g. near ENZ).
        if error <= rtol * scale * (hi - lo) / width:
            out += fine
            return
        del fine
        if depth >= 16:
            raise ArithmeticError("Momentum quadrature failed to converge; increase loss or rtol.")
        mid = (lo + hi) / 2
        panel(lo, mid, out, depth + 1)
        panel(mid, hi, out, depth + 1)

    output = np.zeros((2 * heights.size, N * N))
    edges = np.unique(np.r_[np.linspace(0, limit, max(2, int(np.ceil(limit / width)) + 1)),
                            _pole_breaks(layers, substrate, limit)])
    # First pass sums every coarse panel to fix the global scale, keeping only
    # errors; rejected panels are recomputed, subtracted, and refined.
    errors = []
    total = len(edges) - 1
    report(f"Integrating momentum · panels 0/{total}", 0.0)
    for completed, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), 1):
        fine, error = estimate(lo, hi)
        output += fine
        errors.append(error)
        if completed % 10 == 0 or completed == total:
            report(f"Integrating momentum · panels {completed}/{total}", 0.4 * completed / total)
    del fine
    scale = max(peak(output), 1e-12)
    report(f"Refining momentum integral · panels 0/{total}", 0.4)
    for completed, (lo, hi, error) in enumerate(zip(edges[:-1], edges[1:], errors), 1):
        if error > rtol * scale * (hi - lo) / width:
            output -= estimate(lo, hi)[0]
            panel(lo, mid := (lo + hi) / 2, output, 1)
            panel(mid, hi, output, 1)
        if completed % 10 == 0 or completed == total:
            report(f"Refining momentum integral · panels {completed}/{total}",
                   0.4 + 0.4 * completed / total)
    tail = np.empty_like(output)
    for attempt in range(5):
        report(f"Checking momentum tail · pass {attempt + 1}/5", 0.8)
        tail[:] = 0
        starts = np.arange(limit, 2 * limit, width)
        for completed, (lo, hi) in enumerate(zip(starts, np.minimum(starts + width, 2 * limit)), 1):
            panel(lo, hi, tail)
            if completed % 10 == 0 or completed == len(starts):
                report(f"Checking momentum tail · pass {attempt + 1}/5 · panels {completed}/{len(starts)}")
        if peak(tail) <= rtol * max(peak(output), 1e-12):
            break
        output += tail
        limit *= 2
    else:
        raise ArithmeticError("Momentum tail did not converge; increase loss or rtol.")
    report("Momentum integration complete", 1.0)
    return (output[:heights.size] + 1j * output[heights.size:]).reshape(heights.size, N, N)
