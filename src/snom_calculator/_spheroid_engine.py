"""Prolate-spheroid tip above a bulk sample or isotropic/uniaxial finite layers.

``profile`` returns the vertical tip dipole at every tapping height for unit
external drive. A bulk sample is one eigen-decomposition per geometry, after
which every frequency is a resolvent sum, with a constant scalar tip permittivity.
Isotropic films use the generalised image charge
method (``gicm``), a least-squares image-charge expansion; uniaxial films
integrate the exact reflection with the checked FIM (Fourier integral method on a rotated momentum contour) and an
adaptive real-axis fallback.

Lengths are in nm and momenta in nm^-1. Inside the geometry formulas ``a`` is
the spheroid semi-major axis (L/2) and ``c = sqrt(a (a - R))`` its focal distance.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from math import gcd
from multiprocessing import parent_process
from typing import Sequence

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import lstsq

from ._leading_blocks import held, reserve
from ._progress import report, stage
from ._quadrature import (complete_profile, gauss_legendre_rule, legendre_P_all, legendre_Q_all,
                          profile_grid, tapping_heights)
from .materials import MaterialSpectrum, effective_permittivity
from .models import NumericalSettings, TipParameters

# Rank cutoff of the GICM fit: the exponential basis is nearly
# dependent above K~15, so dependent directions are dropped rather than fitted
# with huge alternating charges.
_LSTSQ_COND = 1e-12
# Tip heights per block: bounds the exact-table and solve temporaries.
_HEIGHT_BLOCK = 2
# Frequencies per batched solve, before the memory cap in _chunk_size.
_FREQUENCY_CHUNK = 4

# Parallel jobs: one frequency block per worker reuses tables within the block.
FREQUENCIES_PER_JOB = None
METHODS = ("auto", "gicm", "fim", "real-axis")
# Default 4 (performance cores of the target Macs); SNOM_THREADS lowers it on small cloud hosts.
_THREADS = max(1, int(os.environ["SNOM_THREADS"])) if os.environ.get("SNOM_THREADS", "").strip().isdigit() else 4
_POOL: ThreadPoolExecutor | None = None


def _thread_map(function, items):
    """map over a shared thread pool, results in order; serial inside worker processes.

    The batched LAPACK calls release the GIL, and threads share every table, so each
    thread only adds its chunk temporaries (a few MB). Worker processes already split
    the cores, so they stay serial.
    """
    global _POOL
    if _THREADS == 1 or parent_process() is not None:
        return map(function, items)
    if _POOL is None:
        _POOL = ThreadPoolExecutor(_THREADS)
    return _POOL.map(function, items)


# =============================================================================
# Surface and image integral tables
# =============================================================================

def _integral_tables(px: np.ndarray, x: np.ndarray, w: np.ndarray, N: int,
                     a: float, c: float, heights: np.ndarray,
                     image_shifts: np.ndarray) -> np.ndarray:
    """T[p, i, j] = int P_i(x) P_j(y(x)) Q_j(eta(x)) dx, i, j = 1..N, for each pair p.

    Fixed Gauss-Legendre rule (x, w); (eta, y) are the spheroidal coordinates of
    the image point ``image_shifts[p]`` below the surface for an apex at
    ``heights[p]``. All pairs share one batched Legendre evaluation: the
    continued fraction in ``legendre_Q_all`` costs about the same for many points
    as for one.
    """
    quad_q = x.size
    out = np.empty((heights.size, N, N))
    # Chunks bound the (N, pairs * quad_q) Legendre temporaries to ~10 MB each.
    step = max(1, 1_000_000 // (N * quad_q))
    for lo in range(0, heights.size, step):
        h, shift = heights[lo:lo + step], image_shifts[lo:lo + step]
        pairs = h.size
        s = 2 * a / c + 2 * h[:, None] / c + shift[:, None] / c - a / c * x[None, :]
        expr = 1 + s ** 2 + (a ** 2 / c ** 2 - 1) * (1 - x[None, :] ** 2)
        rad = np.sqrt(expr ** 2 - 4 * s ** 2)
        eta = np.sqrt((expr + rad) / 2)
        y = s / eta
        py = legendre_P_all(N, y.ravel())[1:N + 1, :].reshape(N, pairs, quad_q)
        qe = legendre_Q_all(N, eta.ravel())[1:N + 1, :].reshape(N, pairs, quad_q)
        py *= qe
        py *= w[None, None, :]
        out[lo:lo + step] = np.einsum('iq,jpq->pij', px, py)
    return out


def _surface_tables(px: np.ndarray, x: np.ndarray, w: np.ndarray, N: int,
                    a: float, c: float, heights: np.ndarray) -> np.ndarray:
    """Surface-reflection tables (heights, N, N): image shift zero."""
    return _integral_tables(px, x, w, N, a, c, heights, np.zeros_like(heights))


def _image_tables(px: np.ndarray, x: np.ndarray, w: np.ndarray, N: int,
                  a: float, c: float, heights: np.ndarray,
                  image_shifts: Sequence[float]) -> np.ndarray:
    """Buried-image tables (K, heights, N, N), every (shift, height) pair in one batch."""
    count = len(image_shifts)
    tables = _integral_tables(px, x, w, N, a, c, np.tile(heights, count),
                              np.repeat(image_shifts, heights.size))
    return tables.reshape(count, heights.size, N, N)


def _extend_surface_tables(a: float, c: float, quad_q: int, heights: np.ndarray,
                           previous: np.ndarray | None, N: int) -> np.ndarray:
    """Surface tables of order N, filling only rows and columns missing from ``previous``."""
    old_n = 0 if previous is None else previous.shape[-1]
    if old_n >= N:
        return previous[:, :N, :N]
    x, w = gauss_legendre_rule(quad_q)
    if old_n == 0:
        return _surface_tables(legendre_P_all(N, x)[1:], x, w, N, a, c, heights)
    s = 2 * a / c + 2 * heights[:, None] / c - a / c * x[None, :]
    expr = 1 + s * s + (a * a / (c * c) - 1) * (1 - x[None, :] ** 2)
    rad = np.sqrt(expr * expr - 4 * s * s)
    eta = np.sqrt((expr + rad) / 2)
    y = s / eta
    py = legendre_P_all(N, y.ravel())[1:].reshape(N, heights.size, quad_q)
    qe = legendre_Q_all(N, eta.ravel())[1:].reshape(N, heights.size, quad_q)
    px = legendre_P_all(N, x)[1:]
    weighted = py * qe * w[None, None, :]
    result = np.empty((heights.size, N, N))
    result[:, :old_n, :old_n] = previous
    result[:, :old_n, old_n:] = np.einsum('iq,jpq->pij', px[:old_n], weighted[old_n:])
    result[:, old_n:, :] = np.einsum('iq,jpq->pij', px[old_n:], weighted)
    return result


class _HeightTables:
    """Surface tables of one tip shape and Gauss rule, interpolated in log height.

    Barycentric interpolation on Chebyshev nodes in log height is validated
    against exact tables at 1e-14; when no node count passes, exact tables are
    computed on request. Tapping trajectories are cached per phase.
    """

    def __init__(self, radius: float, a: float, quad_q: int):
        self.a = a
        self.c = float(np.sqrt(a * (a - radius)))
        self.quad_q = quad_q
        self.bounds = None
        self.nodes = None
        self.values = None
        self.interpolator = None
        self.validation_error = 0.0
        self.samples = OrderedDict()
        self.stats = dict(shared_hits=0, extended_blocks=0, direct_fallbacks=0)

    def prepare(self, heights: np.ndarray, N: int) -> None:
        lo, hi = float(np.min(heights)), float(np.max(heights))
        if not np.isfinite(heights).all() or lo <= 0:
            raise ValueError('Separation must be finite and positive.')
        if N > self.quad_q:
            raise ValueError('quadQ must be at least N.')
        if (self.bounds is not None and self.bounds[0] <= lo
                and hi <= self.bounds[1] and self.values.shape[-1] >= N):
            return
        if self.bounds is not None:
            lo, hi = min(lo, self.bounds[0]), max(hi, self.bounds[1])
            N = max(N, self.values.shape[-1])
        if hi == lo:
            hi = lo * (1 + 1e-3)
        old_nodes, old_values = self.nodes, self.values
        for attempt, count in enumerate((24, 48, 96)):
            report(f"Preparing geometry · interpolation grid with {count} nodes", attempt / 3)
            angles = np.pi * np.arange(count) / (count - 1)
            nodes = np.exp((np.log(lo) + np.log(hi)) / 2
                           + (np.log(hi) - np.log(lo)) / 2 * np.cos(angles))[::-1]
            nodes[0], nodes[-1] = lo, hi
            if old_nodes is not None and count == len(old_nodes) and np.array_equal(nodes, old_nodes):
                values = _extend_surface_tables(self.a, self.c, self.quad_q, nodes, old_values, N)
                if N > old_values.shape[-1]:
                    self.stats['extended_blocks'] += len(nodes)
            else:
                values = _extend_surface_tables(self.a, self.c, self.quad_q, nodes, None, N)
            interpolator = BarycentricInterpolator(np.log(nodes), values, axis=0)
            report(f"Preparing geometry · validating {count}-node grid", (attempt + 0.5) / 3)
            checks = np.exp(np.linspace(np.log(lo), np.log(hi), 9))
            checks[0], checks[-1] = lo, hi
            exact = _extend_surface_tables(self.a, self.c, self.quad_q, checks, None, N)
            error = np.max(abs(interpolator(np.log(checks)) - exact))
            self.validation_error = float(error)
            if error <= 1e-14 * max(np.max(abs(exact)), 1e-12):
                self.interpolator = interpolator
                break
        else:
            self.interpolator = None
        self.bounds, self.nodes, self.values = (lo, hi), nodes, values
        self.values.setflags(write=False)

    def tables(self, heights: np.ndarray, N: int) -> np.ndarray:
        heights = np.asarray(heights, dtype=float)
        self.prepare(heights, N)
        if self.interpolator is None:
            self.stats['direct_fallbacks'] += heights.size
            return _extend_surface_tables(self.a, self.c, self.quad_q, heights, None, N)
        return np.asarray(self.interpolator(np.log(heights)))[:, :N, :N]

    def image_weights(self, heights: np.ndarray, shifts: Sequence[float], N: int) -> np.ndarray:
        """Barycentric weights (K, heights, nodes) at every image height ``heights + shift/2``.

        Contracting them with the node tables gives the interpolated image tables
        without materializing a (K, heights, N, N) array.
        """
        z = heights[None, :] + np.asarray(shifts, dtype=float)[:, None] / 2
        self.prepare(z, N)
        log_nodes = np.log(self.nodes)
        signs = np.ones(log_nodes.size)
        signs[1::2] = -1
        signs[[0, -1]] *= .5  # Chebyshev nodes of the second kind
        diff = np.log(z)[..., None] - log_nodes
        hit = diff == 0
        with np.errstate(divide='ignore'):
            weights = signs / diff
        on_node = hit.any(-1)
        weights[on_node] = hit[on_node]
        return weights / weights.sum(-1, keepdims=True)

    def trajectory(self, gap: float, amplitude: float, M: int,
                   shifts: Sequence[float], N: int) -> np.ndarray:
        """Tables (shifts, M+1, N, N) along the tapping trajectory."""
        if not shifts:
            return np.empty((0, M + 1, N, N))
        heights = tapping_heights(gap, amplitude, M)
        requested = np.concatenate([heights + shift / 2 for shift in shifts])
        with stage("", 0.0, 0.7):
            self.prepare(requested, N)
        keys = [(gap, amplitude, j // gcd(j, M) if M else 0, M // gcd(j, M) if M else 1, shift)
                for shift in shifts for j in range(M + 1)]
        missing = [index for index, key in enumerate(keys) if key not in self.samples]
        missing_set = set(missing)
        stored_n = max(N, self.values.shape[-1])
        max_entries = min(1024, max(1, (32 * 1024 * 1024) // (stored_n * stored_n * 8)))
        # A working set larger than the cache would only thrash it, so skip storing.
        store = len(keys) <= max_entries
        result = np.empty((len(keys), N, N))
        for lo in range(0, len(missing), 256):  # bounded interpolation temporaries
            report(f"Preparing geometry · trajectory tables {lo}/{len(missing)}", 0.7 + 0.3 * lo / len(missing))
            chunk = missing[lo:lo + 256]
            values = self.tables(requested[chunk], N)
            result[chunk] = values[:, :N, :N]
            if store:
                for index, value in zip(chunk, values):
                    # copy: a view would pin the whole batch behind one cache entry
                    value = value.copy()
                    value.setflags(write=False)
                    self.samples[keys[index]] = (requested[index], value)
        groups = {}
        for index, key in enumerate(keys):
            if index in missing_set:
                continue
            height, value = self.samples[key]
            if value.shape[-1] < N:
                groups.setdefault(value.shape[-1], {})[key] = (height, value)
        for old_n, group in groups.items():
            entries = list(group.items())
            extended = _extend_surface_tables(
                self.a, self.c, self.quad_q,
                np.array([item[1][0] for item in entries]),
                np.stack([item[1][1] for item in entries]), N)
            for (key, (height, _)), value in zip(entries, extended):
                value.setflags(write=False)
                self.samples[key] = (height, value)
            self.stats['extended_blocks'] += len(entries)
        for index, key in enumerate(keys):
            if index in missing_set:
                continue
            result[index] = self.samples[key][1][:N, :N]
            self.stats['shared_hits'] += 1
            self.samples.move_to_end(key)
        while len(self.samples) > max_entries:
            self.samples.popitem(last=False)
        return result.reshape(len(shifts), M + 1, N, N)


@lru_cache(maxsize=4)
def _height_tables(radius: float, a: float, quad_q: int) -> _HeightTables:
    return _HeightTables(radius, a, quad_q)


# One stored set of exact image tables, reused by reruns (new materials or
# frequencies) and by the N/2 check, whose tables are the top-left block.
# Note: single slot under a byte cap; larger sets stream per height block.
_EXACT_TABLE_BYTES = 128 << 20
_exact_tables: dict[tuple, np.ndarray] = {}


def _exact_image_tables(a: float, c: float, quad_q: int, heights: np.ndarray,
                        shifts: Sequence[float], N: int) -> np.ndarray | None:
    """(K, heights, N, N) frequency-independent image tables, or None above the cap.

    Tables the full solve kept for the N check (see ``_image_charge_profile``) come first.
    """
    key = (a, c, quad_q, heights.tobytes(), tuple(shifts))
    kept = held(("N/2 image tables", *key), N)
    if kept is not None:
        return kept
    if len(shifts) * heights.size * N * N * 8 > _EXACT_TABLE_BYTES:
        return None
    tables = _exact_tables.get(key)
    if tables is None or tables.shape[-1] < N:
        x, w = gauss_legendre_rule(quad_q)
        tables = _image_tables(legendre_P_all(N, x)[1:], x, w, N, a, c, heights, shifts)
        tables.setflags(write=False)
        _exact_tables.clear()
        _exact_tables[key] = tables
    return tables


def clear_caches() -> None:
    _tip.cache_clear()
    _height_tables.cache_clear()
    _tip_axis_legendre.cache_clear()
    _bulk_modes.cache_clear()
    _exact_tables.clear()
    from ._uniaxial_esm import _geometry
    _geometry.cache_clear()
    from ._fim_coupling import _cached_fim
    _cached_fim.cache_clear()


# =============================================================================
# Tip boundary constants
# =============================================================================

@lru_cache(maxsize=16)
def _tip_axis_legendre(N: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """P_l(a/c), Q_l(a/c), l = 0..N: tip shape and order, not tip epsilon."""
    p = legendre_P_all(N, alpha)[:, 0]
    q = legendre_Q_all(N, alpha)[:, 0]
    p.setflags(write=False)
    q.setflags(write=False)
    return p, q


@dataclass(frozen=True)
class _Tip:
    """Boundary constants of one spheroid shape, tip permittivity and order N."""

    epsilon: complex
    a: float
    c: float
    axis_p: np.ndarray        # P_l(a/c), l = 1..N
    susceptibility: np.ndarray
    denominator: np.ndarray   # axis_p * susceptibility
    row_scale: np.ndarray     # 1 / denominator
    col_scale: np.ndarray     # (2l + 1) / 2
    common: complex           # degree-1 constant shared by source and dipole offset
    source_per_field: complex
    dipole_offset_per_field: complex


@lru_cache(maxsize=16)
def _tip(radius: float, length: float, epsilon: complex, N: int) -> _Tip:
    a = length / 2
    c = float(np.sqrt(a * (a - radius)))
    alpha = a / c
    p, q = _tip_axis_legendre(N, alpha)
    susceptibility = epsilon * q[1:] / p[1:] - (alpha * q[1:] - q[:N]) / (alpha * p[1:] - p[:N])
    denominator = p[1:] * susceptibility
    common = epsilon * q[1] - alpha * q[0] + a * a / (a * a - c * c)
    return _Tip(
        epsilon, a, c, p[1:], susceptibility, denominator, 1 / denominator, (2 * np.arange(1, N + 1) + 1) / 2, common,
        (epsilon - 1) ** 2 * c / (4 * common), (epsilon - 1) * a * c * c / (3 * common),
    )


def _surface_trajectory(radius: float, length: float, gap: float, amplitude: float,
                        M: int, N: int, quad_q: int, heights: np.ndarray) -> np.ndarray:
    """Surface tables (len(heights), N, N) at the tapping ``heights``."""
    if quad_q < N:
        raise ValueError('quadQ must be at least N.')
    tables = _height_tables(radius, length / 2, quad_q)
    if len(heights) == M + 1:  # the full grid: its per-phase table cache is shared across calls
        return tables.trajectory(gap, amplitude, M, (0.,), N)[0]
    return tables.tables(heights, N)  # M/2 midpoints of a doubled grid, never cached by phase


def _solve_coupled(couplings: np.ndarray, tip: _Tip) -> np.ndarray:
    """Dipoles (frequencies, heights) from reflected coupling matrices W (F, H, N, N)."""
    n_freq, n_heights, N, _ = couplings.shape
    eye = np.eye(N, dtype=complex)
    systems = couplings.transpose(0, 1, 3, 2) * ((tip.epsilon - 1) * (tip.col_scale * tip.row_scale))
    systems += eye
    rhs = tip.source_per_field * tip.row_scale[0] * couplings[:, :, 0]
    solution = np.linalg.solve(systems.reshape(-1, N, N), rhs.reshape(-1, N, 1))
    return (2 * tip.a * tip.c * solution[:, 0, 0].reshape(n_freq, n_heights)
            - tip.dipole_offset_per_field)


def _chunk_size(M: int, N: int) -> int:
    """~4 complex (M+1, N, N) temporaries per frequency; keep a chunk under ~256 MB."""
    return max(1, min(_FREQUENCY_CHUNK, (256 << 20) // ((M + 1) * N * N * 64)))


# =============================================================================
# Bulk sample
# =============================================================================

@lru_cache(maxsize=32)
def _bulk_modes(radius: float, length: float, amplitude: float, gap: float,
                epsilon: complex, M: int, N: int, quad_q: int, heights: tuple[float, ...]):
    """Eigenvalues and modal numerators (heights, N) of the bulk boundary system.

    With W = -beta J the system is I - beta M, so the dipole at each height is
    2ac beta sum_k n_k / (1 - beta lambda_k) minus the dipole offset.
    """
    tip = _tip(radius, length, epsilon, N)
    heights = np.asarray(heights)
    report(f"Preparing geometry · integral tables for {len(heights)} heights", 0.0)
    with stage("", 0.0, 0.3):
        table = _surface_trajectory(radius, length, gap, amplitude, M, N, quad_q, heights)
    source = (-(epsilon - 1) ** 2 * tip.c / 4 * table[:, 0]
              / (tip.axis_p * tip.common * tip.susceptibility))
    system = (np.swapaxes(table, 1, 2) / tip.denominator[:, None]) * ((epsilon - 1) * tip.col_scale)
    eigenvalues = np.empty((len(heights), N), dtype=complex)
    numerators = np.empty_like(eigenvalues)
    report(f"Preparing geometry · eigen-decomposition · heights 0/{len(heights)}", 0.3)

    def decompose(index):
        values, vectors = np.linalg.eig(system[index])
        eigenvalues[index] = values
        numerators[index] = vectors[0, :] * np.linalg.solve(vectors, source[index])

    for index, _ in enumerate(_thread_map(decompose, range(len(heights)))):
        if (index + 1) % 10 == 0 or index + 1 == len(heights):
            report(f"Preparing geometry · eigen-decomposition · heights {index + 1}/{len(heights)}",
                   0.3 + 0.7 * (index + 1) / len(heights))
    return eigenvalues, numerators, tip


def _bulk_profile(beta: np.ndarray, tip: TipParameters, gap: float,
                  numerical: NumericalSettings, M: int, heights: np.ndarray) -> np.ndarray:
    with stage("", 0.0, 0.85):
        eigenvalues, numerators, constants = _bulk_modes(
            tip.radius_nm, tip.length_nm, tip.amplitude_nm, gap, tip.epsilon,
            M, numerical.N, numerical.quadQ, tuple(heights))
    report(f"Solving response · frequencies 0/{beta.size}", 0.85)
    # Blocks of frequencies keep the (block, M+1, N) temporaries near 8 MB.
    step = max(1, (8 << 20) // (eigenvalues.size * 16))
    resolvent = np.empty((beta.size, eigenvalues.shape[0]), dtype=complex)
    for lo in range(0, beta.size, step):
        b = beta[lo:lo + step]
        resolvent[lo:lo + step] = b[:, None] * np.sum(
            numerators[None] / (1 - b[:, None, None] * eigenvalues[None]), axis=2)
    return (2 * constants.a * constants.c * resolvent - constants.dipole_offset_per_field).T


# =============================================================================
# Finite layers
# =============================================================================

def _stack_reflection(epsilon: Sequence[np.ndarray], thicknesses: Sequence[float],
                      k: np.ndarray) -> np.ndarray:
    """Electrostatic reflection of air / finite layers / substrate, air to substrate."""
    beta = [(right - left) / (right + left) for left, right in zip(epsilon[:-1], epsilon[1:])]
    reflection = beta[-1]
    for index in range(len(thicknesses) - 1, -1, -1):
        decay = np.exp(-2 * k * thicknesses[index])
        reflection = (beta[index] + reflection * decay) / (1 + beta[index] * reflection * decay)
    return reflection


def _film_charges(epsilon_layer: np.ndarray, epsilon_substrate: np.ndarray, K: int,
                  n_k: int = 1000) -> np.ndarray:
    """Image charges (F, K) at depths 2 d l fitting one film's r(k) on k d in [0, 10]."""
    beta1 = (epsilon_layer - 1) / (epsilon_layer + 1)
    beta2 = (epsilon_substrate - epsilon_layer) / (epsilon_substrate + epsilon_layer)
    t = np.linspace(0, 10, n_k)
    decay = np.exp(-2 * t)
    target = beta1[:, None] - (beta1[:, None] + beta2[:, None] * decay) / (
        1 + beta1[:, None] * beta2[:, None] * decay)
    dt = t[1] - t[0]
    z = np.exp(-2 * np.arange(1, K + 1) * dt)
    design = z[None] ** np.arange(n_k)[:, None]
    return lstsq(design.astype(complex), target.T, lapack_driver='gelsy', cond=_LSTSQ_COND)[0].T


def _stack_charges(layers: Sequence[np.ndarray], epsilon_substrate: np.ndarray,
                   thicknesses: Sequence[float], image_shifts: Sequence[float]) -> np.ndarray:
    """Image charges (F, K) at ``image_shifts`` fitting a stack's r(k)."""
    k = np.linspace(0, 10 / min(thicknesses), 1000)
    epsilon = (1., *(layer[:, None] for layer in layers), epsilon_substrate[:, None])
    reflection = _stack_reflection(epsilon, thicknesses, k[None, :])
    beta_top = (layers[0] - 1) / (layers[0] + 1)
    target = beta_top[:, None] - reflection
    design = np.exp(-k[:, None] * np.asarray(image_shifts)[None, :])
    return lstsq(design.astype(complex), target.T, lapack_driver='gelsy', cond=_LSTSQ_COND)[0].T


def _image_charges(epsilon_layers, epsilon_substrate, thicknesses, K):
    """Image shifts (K,) and charges (F, K) of a film or stack."""
    if len(thicknesses) > 1:
        step = min(thicknesses)
        # Note: uniform image depths are approximate; use path-specific depths if K does not converge.
        shifts = tuple(2 * thicknesses[0] + 2 * step * np.arange(K))
        return shifts, _stack_charges(epsilon_layers, epsilon_substrate, thicknesses, shifts)
    return tuple(2 * thicknesses[0] * np.arange(1, K + 1)), _film_charges(epsilon_layers[0], epsilon_substrate, K)


def _complex_matmul(weights: np.ndarray, table: np.ndarray) -> np.ndarray:
    """weights @ table without casting a real ``table`` to complex (a full copy)."""
    if np.iscomplexobj(table):
        return weights @ table
    out = np.empty((weights.shape[0], table.shape[1]), dtype=complex)
    np.matmul(weights.real, table, out=out.real)
    np.matmul(weights.imag, table, out=out.imag)
    return out


def _image_charge_profile(epsilon_top: np.ndarray, charges: np.ndarray, shifts: Sequence[float],
                          tip: _Tip, surface: np.ndarray, heights: np.ndarray,
                          tables: _HeightTables, exact_mask: np.ndarray) -> np.ndarray:
    """GICM solve (frequencies, heights): height blocks outside, frequencies inside.

    Frequencies outside ``exact_mask`` contract charges with the interpolation
    nodes of ``tables``. Masked frequencies use cached exact tables under the
    byte cap, or exact tables streamed one height block at a time above it.
    """
    n_freq, K = charges.shape
    N = surface.shape[-1]
    beta1 = (epsilon_top - 1) / (epsilon_top + 1)
    if not exact_mask.all():
        image_weights = tables.image_weights(heights, shifts, N)
        node_tables = np.ascontiguousarray(tables.values[:, :N, :N]).reshape(tables.values.shape[0], N * N)
    exact = kept_half = None
    if exact_mask.any():
        exact = _exact_image_tables(tip.a, tip.c, tables.quad_q, heights, shifts, N)
        if exact is None:
            x, w = gauss_legendre_rule(tables.quad_q)
            px = legendre_P_all(N, x)[1:]
            # Streamed tables: keep their N/2 leading blocks for the N check.
            kept_half = reserve(("N/2 image tables", tip.a, tip.c, tables.quad_q, heights.tobytes(), tuple(shifts)),
                                N // 2, (K, heights.size, N // 2, N // 2))
    scale = tip.col_scale * tip.row_scale
    eye = np.eye(N, dtype=complex)
    chunk = _chunk_size(heights.size - 1, N)
    profiles = np.empty((n_freq, heights.size), dtype=complex)
    report(f"Solving response · heights 0/{heights.size}", 0.15)
    for low in range(0, heights.size, _HEIGHT_BLOCK):
        high = min(low + _HEIGHT_BLOCK, heights.size)
        block = high - low
        if exact is not None:
            exact_block = exact[:, low:high, :N, :N]
        elif exact_mask.any():
            exact_block = _image_tables(px, x, w, N, tip.a, tip.c, heights[low:high], shifts)
        if kept_half is not None:
            kept_half[:, low:high] = exact_block[..., :N // 2, :N // 2]
        if exact_mask.any():
            exact_block = exact_block.reshape(K, -1)
        def solve_chunk(start):
            stop = min(start + chunk, n_freq)
            size = stop - start
            local = exact_mask[start:stop]
            mixed = np.empty((size, block * N * N), dtype=complex)
            if not local.all():
                effective = np.einsum("fk,khn->fhn", charges[start:stop][~local], image_weights[:, low:high])
                mixed[~local] = _complex_matmul(
                    effective.reshape(-1, effective.shape[-1]), node_tables).reshape(-1, block * N * N)
            if local.any():
                mixed[local] = _complex_matmul(charges[start:stop][local], exact_block)
            W = mixed.reshape(size, block, N, N)
            W -= beta1[start:stop, None, None, None] * surface[None, low:high]
            systems = W.transpose(0, 1, 3, 2) * ((tip.epsilon - 1) * scale)
            systems += eye
            rhs = tip.source_per_field * tip.row_scale[0] * W[:, :, 0]
            del W, mixed
            solution = np.linalg.solve(systems.reshape(-1, N, N), rhs.reshape(-1, N, 1))
            profiles[start:stop, low:high] = (
                2 * tip.a * tip.c * solution[:, 0, 0].reshape(size, block) - tip.dipole_offset_per_field)

        for _ in _thread_map(solve_chunk, range(0, n_freq, chunk)):
            pass
        report(f"Solving response · heights {high}/{heights.size}", 0.15 + 0.7 * high / heights.size)
    return profiles


def _gicm_profile(epsilon_layers, epsilon_substrate, thicknesses, tip, constants, gap, numerical, M, heights):
    report("Calculating coupling · GICM", 0.05)
    N, quad_q = numerical.N, numerical.quadQ
    shifts, charges = _image_charges(epsilon_layers, epsilon_substrate, thicknesses, numerical.K)
    tables = _height_tables(tip.radius_nm, tip.length_nm / 2, quad_q)
    with stage("", 0.05, 0.1):
        report("Preparing geometry · image tables", 0.0)
        tables.prepare(np.array([gap + min(shifts) / 2, gap + 2 * tip.amplitude_nm + max(shifts) / 2]), N)
        report("Preparing geometry · image tables complete", 1.0)
    # Note: charge amplification is a coupling-error screen; compare final harmonics for a signal guarantee.
    if tables.interpolator is None:
        exact_mask = np.ones(charges.shape[0], dtype=bool)
    else:
        exact_mask = (tables.validation_error * np.sum(abs(charges), axis=1)
                      > 1e-12 * max(np.max(abs(tables.values)), 1e-12))
    with stage("", 0.1, 0.15):
        report("Preparing geometry · layer tables", 0.0)
        surface = _surface_trajectory(tip.radius_nm, tip.length_nm, gap, tip.amplitude_nm, M, N, quad_q, heights)
        report("Preparing geometry · layer tables complete", 1.0)
    return _image_charge_profile(epsilon_layers[0], charges, shifts, constants, surface,
                                 heights, tables, exact_mask)


def _momentum_profile(axes, epsilon_substrate, thicknesses, tip, constants, gap, numerical, M, method, heights):
    from . import _uniaxial_esm
    from ._fim_coupling import fim_coupling

    if method == "fim":
        report("Calculating coupling · FIM", 0.05)
        fim = fim_coupling(heights, a=constants.a, c=constants.c, N=numerical.N)

        def couplings_at(layers, substrate):
            return fim.matrices(layers, substrate, rtol=numerical.momentum_rtol)
    else:
        def couplings_at(layers, substrate):
            return _uniaxial_esm.coupling_matrices(heights, layers, substrate, a=constants.a,
                                                   c=constants.c, N=numerical.N,
                                                   rtol=numerical.momentum_rtol)
    n_freq = epsilon_substrate.size
    # Chunks hold every height: split the serial chunk over the threads to keep the same peak memory.
    chunk = max(1, _chunk_size(len(heights) - 1, numerical.N) // _THREADS)
    profiles = np.empty((n_freq, len(heights)), dtype=complex)

    def solve_chunk(start):
        stop = min(start + chunk, n_freq)
        couplings = np.empty((stop - start, len(heights), numerical.N, numerical.N), dtype=complex)
        for local, index in enumerate(range(start, stop)):
            layers = tuple((complex(in_plane[index]), complex(normal[index]), depth)
                           for (in_plane, normal), depth in zip(axes, thicknesses))
            couplings[local] = couplings_at(layers, complex(epsilon_substrate[index]))
        profiles[start:stop] = _solve_coupled(couplings, constants)
        return stop

    for stop in _thread_map(solve_chunk, range(0, n_freq, chunk)):
        report(f"Calculating coupling · frequencies {stop}/{n_freq}", 0.05 + 0.8 * stop / n_freq)
    return profiles


def resolve_method(layers, method):
    """Film coupling for ``auto``: gicm for isotropic stacks, FIM otherwise."""
    if not layers:
        return method
    anisotropic = any(np.any(material.perpendicular != material.parallel) for material, _ in layers)
    if method == "auto":
        method = "fim" if anisotropic else "gicm"
    if method == "gicm" and anisotropic:
        raise ValueError("GICM is available only for isotropic finite layers.")
    return method


def _film_profile(layers, substrate, tip, gap, numerical, M, method, heights):
    epsilon_layers = tuple(effective_permittivity(material) for material, _ in layers)
    if any(np.any(epsilon == 0) for epsilon in epsilon_layers):
        raise ValueError("Finite-layer effective permittivity must be nonzero.")
    axes = tuple((material.perpendicular, material.parallel) for material, _ in layers)
    thicknesses = tuple(float(depth) for _, depth in layers)
    epsilon_substrate = effective_permittivity(substrate)
    method = resolve_method(layers, method)

    constants = _tip(tip.radius_nm, tip.length_nm, tip.epsilon, numerical.N)
    if method == "gicm":
        return _gicm_profile(epsilon_layers, epsilon_substrate, thicknesses, tip, constants,
                             gap, numerical, M, heights).T
    return _momentum_profile(axes, epsilon_substrate, thicknesses, tip, constants, gap,
                             numerical, M, method, heights).T


# =============================================================================
# Engine entry point
# =============================================================================

def profile(layers: Sequence[tuple[MaterialSpectrum, float]], substrate: MaterialSpectrum,
            tip: TipParameters, gap: float, numerical: NumericalSettings,
            harmonics: Sequence[int] = (), *, method: str = "auto",
            previous: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Vertical dipoles (tapping heights, frequencies) for unit drive, and diagnostics.

    ``layers`` are positive-thickness films, top to bottom. ``method`` selects
    the film coupling: GICM (``gicm``, isotropic only), the checked
    FIM (``fim``) or the adaptive ``real-axis`` integral; ``auto`` uses
    gicm for isotropic stacks and FIM otherwise.

    ``previous`` is the profile of the same inputs at M/2 (shape (M/2 + 1,
    frequencies)): only the M/2 midpoint heights of the doubled grid are
    solved and interleaved with it. The returned profile is always the full grid: the M + 1
    tapping heights, or the ``profile_grid`` log-height nodes when M + 1 exceeds them.
    """
    if method not in METHODS:
        raise ValueError(f"Spheroid films support methods {', '.join(METHODS)}.")
    M = 0 if tip.amplitude_nm == 0 else numerical.M
    heights, _, _ = profile_grid(gap, tip.amplitude_nm, M, (), previous, numerical.height_nodes)
    # FIM and real-axis couplings cost per height group more than the smaller solves save.
    tabled = not layers or resolve_method(layers, method) == "gicm"
    orders = height_orders(heights, gap, numerical.N) if tabled else np.full(heights.shape, numerical.N)
    if orders.min() < numerical.N:
        # One interpolation grid over every height: per-order subsets would each extend and rebuild it.
        _height_tables(tip.radius_nm, tip.length_nm / 2, numerical.quadQ).prepare(heights, numerical.N)
    values = np.empty((heights.size, substrate.parallel.size), dtype=complex)
    # Each order solves the leading blocks of the order-N tables and boundary constants.
    for order in np.unique(orders)[::-1]:
        ids = orders == order
        local = numerical if order == numerical.N else replace(numerical, N=int(order))
        if not layers:
            epsilon = effective_permittivity(substrate)
            values[ids] = _bulk_profile((epsilon - 1) / (epsilon + 1), tip, gap, local, M, heights[ids])
        else:
            values[ids] = _film_profile(tuple(layers), substrate, tip, gap, local, M, method, heights[ids])
    return complete_profile(values, previous), {}


# Multipole order per height for N >= ADAPTIVE_N_MIN: N at the closest approach,
# 8 ceil(1.5 N (gap / h)^0.4 / 8) above, at least 32. On 15 lab cases (0.2-5 nm gaps, R 10-50 nm,
# A 10-100 nm, N 40-120) this added at most 0.2x the uniform-N truncation error for 36-64 % of the
# solve cost at N >= 80; below N = 64 the saving is too small to pay for the extra table groups.
ADAPTIVE_N = True
ADAPTIVE_N_MIN = 64


def height_orders(heights: np.ndarray, gap: float, N: int) -> np.ndarray:
    """Multipole order for each height; the gap is the lowest tapping height."""
    if not ADAPTIVE_N or N < ADAPTIVE_N_MIN:
        return np.full(heights.shape, N)
    orders = 8 * np.ceil(1.5 * N * (gap / np.asarray(heights, dtype=float)) ** .4 / 8)
    return np.clip(orders, 32, N).astype(int)
