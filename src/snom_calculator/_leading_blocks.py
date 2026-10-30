"""Per-run reuse of coupling matrices and tables by the convergence checks.

Coupling matrices at order N/2 are the leading blocks of those at order N (same
geometry, stack and tolerance), so the check can slice instead of re-integrating.
The spheroid GICM solve also keeps the N/2 leading blocks of streamed
image tables here (``reserve`` and ``held``).
Retention is on only inside ``retaining()``, bounded by ``MAX_BYTES``, and
per-process: worker processes see no store and compute normally.
Calculation scopes are serialized because the geometry caches are also shared.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import Lock, RLock

import numpy as np

MAX_BYTES = 256 << 20
_store: dict | None = None
_bytes = 0
_lock = Lock()
_run_lock = RLock()


@contextmanager
def retaining(enabled: bool = True):
    """Context that keeps the largest-N matrix per key until it exits."""
    with _run_lock:
        with _retaining() if enabled else nullcontext():
            yield


@contextmanager
def _retaining():
    global _store, _bytes
    saved, _store, _bytes = (_store, _bytes), {}, 0
    try:
        yield
    finally:
        _store, _bytes = saved


def _square(matrix, N):
    return matrix[..., :N, :N]


def leading_block(key, N: int, compute, cut=_square):
    """``compute()`` -> (array of order N, extra); reuse or ``cut`` a stored larger-N result.

    ``key`` must hold everything but ``N`` that the array depends on. The default
    ``cut`` takes the leading N x N block of (..., N, N) matrices.
    """
    global _bytes
    if _store is None:
        return compute()
    held = _store.get(key)
    if held is not None and held[0] >= N:
        order, matrix, extra = held
        return (matrix if order == N else cut(matrix, N)), extra
    matrix, extra = compute()
    with _lock:  # solver threads share the store
        size = matrix.nbytes - (held[1].nbytes if held else 0)
        if _bytes + size <= MAX_BYTES:
            _store[key] = (N, matrix, extra)
            _bytes += size
    return matrix, extra


def held(key, N: int, cut=_square):
    """The stored array for ``key`` cut to order N, or None (also outside ``retaining()``)."""
    entry = None if _store is None else _store.get(key)
    if entry is None or entry[1] is None or entry[0] < N:
        return None
    return entry[1] if entry[0] == N else cut(entry[1], N)


def reserve(key, N: int, shape, dtype=float):
    """Empty array stored under ``key`` at order N for the caller to fill, or None.

    None outside ``retaining()``, when ``key`` already holds order N or more, or
    when the array would exceed ``MAX_BYTES``.
    """
    global _bytes
    if _store is None:
        return None
    with _lock:
        entry = _store.get(key)
        if entry is not None and entry[0] >= N:
            return None
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize - (entry[1].nbytes if entry else 0)
        if _bytes + size > MAX_BYTES:
            return None
        array = np.empty(shape, dtype)
        _store[key] = (N, array, None)
        _bytes += size
    return array
