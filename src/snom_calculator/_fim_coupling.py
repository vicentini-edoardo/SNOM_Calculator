"""FIM (Fourier integral method) coupling matrices on a rotated momentum contour,
compressed to a kernel skeleton.

W_h(omega) = int_0^inf r(k; omega) K_h(k) dk with the frequency-independent kernel
K_h,ij(k) = c exp(-2 k H_h) A_i(k) S_j(k), entire in k. r(k) is meromorphic; its poles
(the stack's guided modes) lie just off the real axis, which makes real-axis quadrature
cost grow with thickness. When the sector between the real axis and the ray
k = kappa exp(i s phi) holds no pole, Cauchy's theorem moves the integral onto the ray,
where one fixed rule integrates every frequency. The kernel has low numerical rank on
that rule, so an interpolative decomposition reduces it to a few skeleton momenta:
W_h = A_S^T diag(c mu exp(-2 k_S H_h)) S_S, mu = P (w * r).

Each frequency is accepted only if the argument principle finds no pole in the sector
and the Kronrod-minus-Gauss difference passes the gate; otherwise smaller angles and
the other side are tried, then a fixed real-axis skeleton (finer geometric panels; no
sector, so no pole count), then the adaptive real-axis integrator. The fixed real-axis
rule rescues stacks with poles on both sides of the axis (forward and backward modes),
where no rotated ray is pole free. See docs/numerical-models.md.
"""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

import numba
import numpy as np
import scipy.linalg as sla

from . import _uniaxial_esm
from ._leading_blocks import leading_block
from ._uniaxial_esm import _gamma, _gauss_kronrod_rule, _geometry

ANGLES_DEG = (25.0, 15.0, 5.0)
AXIS_RATIO = 1.3  # panel growth of the fixed real-axis skeleton (spheroid rays use 2.0)
ID_TOL = 1e-14
MAX_GATE = 1e-8


@numba.njit(cache=True)
def _scaled_spherical_i_complex(z, N):
    """exp(-z) i_n(z), n = 1..N, for complex z with Re z >= 0 (Miller recurrence)."""
    out = np.zeros((z.size, N), dtype=np.complex128)
    for q in range(z.size):
        zq = z[q]
        az = abs(zq)
        if az == 0:
            continue
        above = 0j
        current = 1e-30 + 0j
        for n in range(N + 40 + int(az + 4 * np.sqrt(az)), 0, -1):
            above, current = current, above + (2 * n + 1) / zq * current
            if 1 <= n - 1 <= N:
                out[q, n - 2] = current
            if abs(current) > 1e200:
                above *= 1e-200
                current *= 1e-200
                for j in range(N):
                    out[q, j] *= 1e-200
        norm = -np.expm1(-2 * zq) / (2 * zq) / current
        for j in range(N):
            out[q, j] *= norm
    return out


def _kernel_factors(k: np.ndarray, a: float, c: float, N: int):
    """A (Q, N) and S (Q, N) with kernel_ij(k) = c A_i(k) S_j(k), for complex k.

    The analytic spheroid projection of ``_uniaxial_esm._momentum_basis``:
    A_i = 2 P_i(a/c) exp(-2k(a-c)) exp(-kc) i_i(kc), S_j = exp(-kc) i_j(kc).
    """
    scaled = _scaled_spherical_i_complex(np.ascontiguousarray(k * c), N)
    return 2 * _geometry(a, c, N)[None, :] * np.exp(-2 * k * (a - c))[:, None] * scaled, scaled


def _decay_rate(a: float, c: float, h_min: float, phi: float) -> float:
    """Decay rate of |kernel| along k = kappa exp(i phi), since |exp(-z) i_n(z)| <= 1.

    The kernel decays as exp(-2(h_min + a - c) Re k); like the real-axis cutoff in
    ``coupling_matrices`` this keeps the margin of the slower rate 2 h_min + a - c.
    """
    return (2 * h_min + a - c) * np.cos(phi)


# ---------------------------------------------------------------------------
# Stack reflection and its entire denominator at complex momentum
# ---------------------------------------------------------------------------

def _stack_parts(layers, substrate: complex):
    if any(not np.isfinite(d) or d < 0 for _, _, d in layers):
        raise ValueError("Layer thickness must be finite and nonnegative.")
    gamma = [_gamma(complex(p), complex(z)) for p, z, _ in layers]
    admittance = [1 + 0j, *(complex(z) * g for (_, z, _), g in zip(layers, gamma)),
                  complex(substrate)]
    rho = [(admittance[i] - admittance[i + 1]) / (admittance[i] + admittance[i + 1])
           for i in range(len(admittance) - 1)]
    return gamma, rho, [float(d) for _, _, d in layers]


def _reflection(k: np.ndarray, parts) -> np.ndarray:
    gamma, rho, depths = parts
    r = np.full(np.shape(k), rho[-1], dtype=complex)
    for j in range(len(depths) - 1, -1, -1):
        exponent = -2 * gamma[j] * depths[j] * k
        # Off the real axis a layer's exponential may grow; divide through by it then.
        grows = exponent.real > 0
        t = np.exp(np.where(grows, -exponent, exponent))
        r = np.where(grows, (rho[j] * t + r) / (t + rho[j] * r),
                     (rho[j] + r * t) / (1 + rho[j] * r * t))
    return r


def _denominator(k: np.ndarray, parts) -> np.ndarray:
    """D(k) with r = M/D, rescaled by real positive factors only (arg D is exact)."""
    gamma, rho, depths = parts
    k = np.asarray(k, dtype=complex)
    num = np.full(k.shape, rho[-1], dtype=complex)
    den = np.ones(k.shape, dtype=complex)
    for j in range(len(depths) - 1, -1, -1):
        exponent = -2 * gamma[j] * depths[j] * k
        grow = np.maximum(exponent.real, 0)
        t = np.exp(exponent - grow)
        shrink = np.exp(-grow)
        num, den = rho[j] * den * shrink + num * t, den * shrink + rho[j] * num * t
        norm = np.maximum(np.maximum(np.abs(num), np.abs(den)), 1e-300)
        num, den = num / norm, den / norm
    return den


def _phase_change(path: np.ndarray, parts, max_level: int = 52, tol: float = np.pi / 4) -> float:
    z = np.asarray(path, dtype=complex)
    D = _denominator(z, parts)
    for _ in range(max_level):
        step = np.angle(D[1:] / D[:-1])
        if not np.all(np.isfinite(step)):
            raise ArithmeticError("denominator vanishes on the contour")
        bad = np.nonzero(np.abs(step) > tol)[0]
        if bad.size == 0:
            return float(np.sum(step))
        mid = (z[bad] + z[bad + 1]) / 2
        z = np.insert(z, bad + 1, mid)
        D = np.insert(D, bad + 1, _denominator(mid, parts))
    raise ArithmeticError("phase tracking did not resolve")


def _sector_pole_count(parts, sign: int, phi: float, k_radius: float) -> int:
    """Poles of r in {0 < sign arg k < phi, |k| < k_radius}, by the argument principle."""
    gamma, _, depths = parts
    optical = max(sum(abs(g) * d for g, d in zip(gamma, depths)), 1e-12)
    h = min(k_radius / 64, np.pi / (8 * optical))
    real = np.r_[0.0, np.geomspace(1e-7, h, 16)[:-1],
                 np.linspace(h, k_radius, int(np.ceil(k_radius / h)))]
    arc = k_radius * np.exp(1j * sign * np.linspace(
        0, phi, max(65, int(np.ceil(k_radius * phi / h)) + 1)))
    total = (_phase_change(real, parts) + _phase_change(arc, parts)
             - _phase_change(real * np.exp(1j * sign * phi), parts))
    return int(round(sign * total / (2 * np.pi)))


def _film_pole_count(parts, sign: int, phi: float, k_radius: float) -> int | None:
    """Sector pole count of a single film from k_n = (ln(-ab) - 2 pi i n) / (2 gamma d).

    Returns None when a pole sits on a sector edge (e.g. a lossless film's poles on
    the real axis), where the argument principle decides instead.
    """
    (gamma,), (top, bottom), (depth,) = parts
    product = top * bottom
    if product == 0:
        return 0
    log = np.log(-product)
    scale = 2 * gamma * depth
    span = int(np.ceil((abs(scale) * k_radius + abs(log)) / (2 * np.pi))) + 1
    poles = (log - 2j * np.pi * np.arange(-span, span + 1)) / scale
    poles = poles[np.abs(poles) < k_radius * (1 + 1e-9)]
    angle = sign * np.angle(poles)
    edge = 1e-9
    if np.any((np.abs(angle) < edge) | (np.abs(angle - phi) < edge)
              | (np.abs(np.abs(poles) - k_radius) < edge * k_radius)):
        return None
    return int(np.sum((angle > 0) & (angle < phi)))


# ---------------------------------------------------------------------------
# One ray: fixed rule + skeleton
# ---------------------------------------------------------------------------

class _RaySkeleton:
    """Fixed Gauss-Kronrod rule on one ray and its interpolative decomposition."""

    def __init__(self, heights: np.ndarray, a: float, c: float, N: int,
                 sign: int, angle_deg: float, *, ratio: float = 2.0, kappa_min: float = 1e-5,
                 id_tol: float = ID_TOL, sketch: int = 300, seed: int = 0,
                 decay_margin: float = 40.0):
        self.phi = np.deg2rad(angle_deg)
        rate = _decay_rate(a, c, float(heights.min()), self.phi)
        self.k_radius = decay_margin / rate
        x, wk, gauss_index, wg = _gauss_kronrod_rule(15)
        wg_full = np.zeros_like(wk)
        wg_full[gauss_index] = wg
        edges = [0.0]
        edge = kappa_min
        while edge < self.k_radius:
            edges.append(edge)
            edge *= ratio
        edges.append(self.k_radius)
        edges = np.array(edges)
        half, mid = np.diff(edges) / 2, (edges[1:] + edges[:-1]) / 2
        rotation = np.exp(1j * sign * self.phi)
        self.k = (mid[:, None] + half[:, None] * x[None]).ravel() * rotation
        self.weights = (half[:, None] * wk[None]).ravel() * rotation
        self.difference = (half[:, None] * (wk - wg_full)[None]).ravel() * rotation
        A, S = _kernel_factors(self.k, a, c, N)
        E = np.exp(-2 * heights[:, None] * self.k[None, :])

        # Equilibrated, weighted kernel sketched with Khatri-Rao rows u (x) v (x) w.
        rng = np.random.default_rng(seed)
        def gaussian(*shape):
            return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        u, v, w = gaussian(sketch, heights.size), gaussian(sketch, N), gaussian(sketch, N)
        size = np.abs(self.weights)
        row_a = 1 / np.maximum(np.max(np.abs(A), 0), 1e-300)
        row_s = 1 / np.maximum(np.max(np.abs(S), 0), 1e-300)
        Y = (u @ E) * (v @ (A * row_a).T) * (w @ (S * row_s).T) * size[None, :]
        _, R, pivots = sla.qr(Y, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(R))
        rank = int(np.sum(diagonal > id_tol * diagonal[0]))
        T = sla.solve_triangular(R[:rank, :rank], R[:rank, rank:])
        P = np.zeros((rank, self.k.size), dtype=complex)
        P[:, pivots[:rank]] = np.eye(rank)
        P[:, pivots[rank:]] = T
        skeleton = pivots[:rank]
        self.P = P * size[skeleton][:, None] / size[None, :]
        self.rank = rank
        self.A_T = np.ascontiguousarray((c * A[skeleton]).T)   # (N, rank)
        self.S_S = np.ascontiguousarray(S[skeleton])            # (rank, N)
        self.E_S = np.ascontiguousarray(E[:, skeleton])         # (H, rank)

    def assemble(self, mu: np.ndarray, rows: slice = slice(None)) -> np.ndarray:
        """mu (rank,) -> W (H_rows, N, N)."""
        coefficients = self.E_S[rows] * mu[None, :]                 # (H, rank)
        return (self.A_T[None] * coefficients[:, None, :]) @ self.S_S


class FIMCoupling:
    """Coupling matrices for one tip geometry; rays are built lazily and reused."""

    def __init__(self, heights: np.ndarray, a: float, c: float, N: int):
        self.heights = np.asarray(heights, dtype=float)
        self.a, self.c, self.N = a, c, N
        self._rays: dict[tuple[int, float], _RaySkeleton] = {}
        self.stats: dict[str, int] = {}
        self._lock = Lock()  # solver threads share rays and stats

    def ray(self, sign: int, angle: float) -> _RaySkeleton:
        key = (sign, angle)
        with self._lock:
            if key not in self._rays:
                self._rays[key] = _RaySkeleton(self.heights, self.a, self.c, self.N, sign, angle,
                                               ratio=AXIS_RATIO if angle == 0 else 2.0)
            return self._rays[key]

    @staticmethod
    def _candidates(parts):
        gamma, _, depths = parts
        # Forward modes (Im gamma > 0, e.g. hBN type II) put poles above the axis.
        first = -1 if sum(g.imag * d for g, d in zip(gamma, depths)) >= 0 else 1
        return [(sign, angle) for sign in (first, -first) for angle in ANGLES_DEG] + [(1, 0.0)]

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    def matrices(self, layers, substrate: complex, rtol: float = 1e-5) -> np.ndarray:
        """W (H, N, N); falls back to the real-axis integrator at ``rtol``."""
        if rtol <= 0:
            raise ValueError("rtol must be positive.")
        layers = tuple((complex(p), complex(z), float(d)) for p, z, d in layers if d > 0)
        key = (self.heights.tobytes(), self.a, self.c, layers, complex(substrate), rtol)
        return leading_block(key, self.N, lambda: (self._matrices(layers, substrate, rtol), None))[0]

    def _matrices(self, layers, substrate: complex, rtol: float) -> np.ndarray:
        """Coupling matrices at order ``self.N`` for a normalized stack."""
        parts = _stack_parts(layers, complex(substrate))
        gate = min(MAX_GATE, 1e-3 * rtol)
        for sign, angle in self._candidates(parts):
            ray = self.ray(sign, angle)
            try:
                count = (0 if angle == 0 else
                         _film_pole_count(parts, sign, ray.phi, ray.k_radius)
                         if len(layers) == 1 else None)
                if count is None:
                    count = _sector_pole_count(parts, sign, ray.phi, ray.k_radius)
                if count:
                    continue
            except ArithmeticError:
                continue
            r = _reflection(ray.k, parts)
            if not np.all(np.isfinite(r)):
                continue
            mu = ray.P @ (ray.weights * r)
            # Kronrod-Gauss difference at the lowest height, whose kernel reaches farthest in k.
            low = int(np.argmin(self.heights))
            rows = slice(low, low + 1)
            value = np.max(np.abs(ray.assemble(mu, rows)))
            error = np.max(np.abs(ray.assemble(ray.P @ (ray.difference * r), rows)))
            if np.isfinite(error) and error <= gate * value:
                self._count(f"ray {'+' if sign > 0 else '-'}{angle:g}" if angle else "real axis")
                return ray.assemble(mu)
        self._count("fallback")
        return _uniaxial_esm.coupling_matrices(self.heights, layers, complex(substrate), a=self.a,
                                               c=self.c, N=self.N, rtol=rtol)


@lru_cache(maxsize=4)
def _cached_fim(heights: tuple[float, ...], a: float, c: float, N: int) -> FIMCoupling:
    return FIMCoupling(np.array(heights), a, c, N)


def fim_coupling(heights: np.ndarray, *, a: float, c: float, N: int) -> FIMCoupling:
    """Cached solver per tip geometry (rays and skeletons are kept between calls)."""
    return _cached_fim(tuple(float(h) for h in np.asarray(heights).ravel()),
                           float(a), float(c), int(N))
