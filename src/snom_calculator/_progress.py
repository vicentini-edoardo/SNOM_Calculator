"""Optional progress reporting; the app installs a callback, the solvers call report()."""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident

_callback = ContextVar("progress_callback", default=(None, (0, 0)))
_window = ContextVar("progress_window", default=(0.0, 1.0, ""))


def set_callback(callback: Callable[[str, float | None], None] | None) -> None:
    _callback.set((callback, (os.getpid(), get_ident())))


@contextmanager
def stage(label: str, low: float, high: float):
    """Map fractions reported inside the block onto [low, high] of the enclosing window, with a label prefix."""
    previous = _window.get()
    span = previous[1] - previous[0]
    token = _window.set((previous[0] + span * low, previous[0] + span * high,
                         f"{previous[2]} {label}".strip()))
    try:
        yield
    finally:
        _window.reset(token)


def report(text: str, fraction: float | None = None) -> None:
    # Worker processes may inherit the callback and solver threads may call in: only the caller's thread touches the UI
    callback, owner = _callback.get()
    if callback is None or (os.getpid(), get_ident()) != owner:
        return
    low, high, label = _window.get()
    callback(f"{label}: {text}" if label else text,
             None if fraction is None else low + (high - low) * fraction)
