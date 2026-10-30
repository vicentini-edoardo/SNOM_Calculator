"""Shared spectrum pipeline: engine profiles to checked, normalized spectra.

Every engine module exposes ``profile(layers, substrate, tip, gap, numerical,
harmonics, *, method)``: the vertical tip dipole at each tapping height for
unit drive, shape (heights, frequencies), plus scalar diagnostics. This module
simplifies the stack, runs the spheroid engine, splits frequencies over worker processes, demodulates, runs the
convergence checks, applies the far-field factor and normalizes to the
reference. Signals have shape (harmonic, frequency).
"""

from __future__ import annotations

import atexit
from importlib import import_module
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from numbers import Real

import numpy as np

from ._far_field import far_field_factor
from ._leading_blocks import retaining
from ._progress import report, stage
from ._quadrature import (HEIGHT_TAIL_ESCALATE, HEIGHT_TAIL_LIMIT, MAX_HEIGHT_NODES, SPAN_M, gap_span,
                          interpolated, interpolation_tail, profile_weights, span_weights, spanned)
from .materials import MaterialSpectrum, effective_permittivity
from .models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters, validate_inputs

# Film and stack signals keep the historical drive amplitude 4 pi eps0; bulk uses 1.
E0 = 4 * np.pi * 8.85418782e-12
# Frequency-subset size above which the film N check uses selected points.
CHECK_POINTS = 12
# Largest M or N convergence error accepted before refinement is offered, all precision presets.
M_ERROR_LIMIT = 0.20

Layers = tuple[tuple[MaterialSpectrum, float], ...]


# =============================================================================
# Structure and engine selection
# =============================================================================


def simplify_stack(layers: Sequence[tuple[MaterialSpectrum, float]], substrate: MaterialSpectrum) -> Layers:
    """Drop zero-thickness layers, merge identical neighbours and a last layer equal to the substrate.

    Each step is optically exact: zero thickness has no interface and identical
    neighbouring tensors have no interface between them.
    """
    stack = []
    for material, depth in layers:
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("Layer thickness must be finite and non-negative.")
        if depth == 0:
            continue
        if stack and np.array_equal(stack[-1][0].tensor, material.tensor):
            stack[-1] = (material, stack[-1][1] + float(depth))
        else:
            stack.append((material, float(depth)))
    if stack and np.array_equal(stack[-1][0].tensor, substrate.tensor):
        stack.pop()
    return tuple(stack)


def engine():
    """The spheroid engine module.

    Imported on first use: the engines pull in numba kernels and SciPy integrators.
    """
    return import_module("._spheroid_engine", __package__)


# =============================================================================
# Frequency-block parallelism
# =============================================================================

_EXECUTORS: dict[int, ProcessPoolExecutor] = {}


def _executor(workers: int) -> ProcessPoolExecutor:
    if workers not in _EXECUTORS:
        _EXECUTORS[workers] = ProcessPoolExecutor(max_workers=workers)
    return _EXECUTORS[workers]


@atexit.register
def _shutdown_executors() -> None:
    for executor in _EXECUTORS.values():
        executor.shutdown(wait=False, cancel_futures=True)
    _EXECUTORS.clear()


def _subset(ids, layers, substrate, tip):
    """The structure and tip restricted to frequency indices ``ids`` (all for None)."""
    if ids is None:
        return layers, substrate, tip
    return tuple((material.take(ids), depth) for material, depth in layers), substrate.take(ids), tip


def _merge(diagnostics: dict, info: dict) -> None:
    """Largest value of every real scalar diagnostic."""
    for key, value in info.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            diagnostics[key] = max(diagnostics.get(key, value), value)


def solve_profile(layers: Layers, substrate: MaterialSpectrum, tip: TipParameters, gap: float,
                  numerical: NumericalSettings, harmonics: tuple[int, ...], method: str,
                  workers: int, *, previous: np.ndarray | None = None,
                  fallback: bool = True) -> tuple[np.ndarray, dict]:
    """Engine profile (heights, F) and diagnostics, split over up to ``workers`` processes.

    Each engine sets FREQUENCIES_PER_JOB: None for one block per worker (geometry
    reused within a block), or a job size for engines whose cost varies by frequency.
    ``previous`` reuses a profile at M/2 with otherwise identical inputs, solving
    only the new midpoint heights on the doubled grid. Interpolated grids do not depend
    on M, so an interpolated M/2 profile is returned as is (without diagnostics) and a
    tapping-height one extends on the tapping grid. A fresh interpolated profile doubles
    its nodes while their Chebyshev tail exceeds ``HEIGHT_TAIL_LIMIT`` (up to
    ``HEIGHT_TAIL_ESCALATE``) and falls back to every tapping height when it stays above;
    ``fallback=False`` returns the node profile instead (its tail is in the diagnostics).
    """
    count = substrate.parallel.size
    if previous is not None and (previous.ndim != 2 or previous.shape[1] != count):
        raise ValueError("Reused profile must match the frequency grid.")
    tapping = replace(numerical, height_nodes=0)
    if not interpolated(tip.amplitude_nm, numerical.M, numerical.height_nodes):
        return _engine_profile(layers, substrate, tip, gap, numerical, harmonics, method, workers, previous)
    if previous is not None:
        if previous.shape[0] != numerical.M // 2 + 1:
            return previous, {}
        # The M/2 profile is on tapping heights (small M, or a fallback): keep that grid.
        return _engine_profile(layers, substrate, tip, gap, tapping, harmonics, method, workers, previous)
    values, diagnostics = _engine_profile(layers, substrate, tip, gap, numerical, harmonics, method, workers, None)
    tail = interpolation_tail(values)
    while (HEIGHT_TAIL_LIMIT < tail <= HEIGHT_TAIL_ESCALATE
           and 2 * values.shape[0] - 1 < min(MAX_HEIGHT_NODES + 1, numerical.M + 1)):
        report(f"Refining tip heights · {2 * values.shape[0] - 1} nodes", None)
        values, info = _engine_profile(layers, substrate, tip, gap, numerical, harmonics, method, workers, values)
        _merge(diagnostics, info)
        tail = interpolation_tail(values)
    nodes = values.shape[0]
    if tail > HEIGHT_TAIL_LIMIT and fallback:
        # Not smooth in log height (a near tip-sample resonance at some height): solve every tapping height.
        report("Solving every tapping height", None)
        values, info = _engine_profile(layers, substrate, tip, gap, tapping, harmonics, method, workers, None)
        _merge(diagnostics, info)
        nodes = 0
    # Sample and reference merge by maximum: the flag shows a fallback the node count would hide.
    _merge(diagnostics, {"height_nodes": nodes, "height_tapping_fallback": int(nodes == 0),
                         "height_interpolation_tail": tail})
    return values, diagnostics


def _engine_profile(layers, substrate, tip, gap, numerical, harmonics, method, workers, previous):
    module = engine()
    count = substrate.parallel.size
    reuse = {} if previous is None else {"previous": previous}
    workers = min(workers, count)
    # A scalar bulk sample shares one eigen-decomposition across frequencies: keep it serial.
    if workers == 1 or not layers:
        return module.profile(layers, substrate, tip, gap, numerical, harmonics, method=method, **reuse)
    jobs = workers if module.FREQUENCIES_PER_JOB is None else -(-count // module.FREQUENCIES_PER_JOB)
    blocks = np.array_split(np.arange(count), jobs)
    # Choose the film method on the full grid so every block uses the same one.
    method = getattr(module, "resolve_method", lambda layers, method: method)(layers, method)
    try:
        futures = [_executor(workers).submit(module.profile, *_subset(ids, layers, substrate, tip),
                                             gap, numerical, harmonics, method=method,
                                             **({} if previous is None else {"previous": previous[:, ids]}))
                   for ids in blocks]
        report(f"Solving response · frequency blocks 0/{len(futures)}", 0.0)
        for done, _ in enumerate(as_completed(futures), 1):
            report(f"Solving response · frequency blocks {done}/{len(futures)}", done / len(futures))
        parts = [future.result() for future in futures]
    except (BrokenProcessPool, OSError, RuntimeError) as exc:
        raise ValueError("Parallel calculation failed; retry with CPU workers set to 1.") from exc
    diagnostics = {}
    for _, info in parts:
        _merge(diagnostics, info)
    return np.concatenate([values for values, _ in parts], axis=1), diagnostics


# =============================================================================
# Normalization and convergence reporting
# =============================================================================

def check_reference_signal(reference: np.ndarray) -> None:
    """Reject a normalization signal strictly below the 1e-18 floor."""
    if np.any(np.abs(reference) < 1e-18):
        raise ValueError("The selected reference produced a zero normalization signal.")


def normalized_contrast(sample: np.ndarray, reference: np.ndarray, harmonics: tuple[int, ...],
                        *, static: bool = False) -> np.ndarray:
    """Static higher harmonics are exactly 0/0; their contrast is undefined."""
    if not static:
        check_reference_signal(reference)
        return sample / reference
    defined = np.asarray(harmonics) == 0
    check_reference_signal(reference[defined])
    contrast = np.full(sample.shape, complex(np.nan, np.nan), dtype=complex)
    np.divide(sample, reference, out=contrast, where=defined[:, None])
    return contrast


def max_signal_error(sample: np.ndarray, reference: np.ndarray,
                     smaller_sample: np.ndarray, smaller_reference: np.ndarray) -> np.ndarray:
    """Worst sample/reference relative change per harmonic across frequencies.

    The 1% row-peak floor avoids flagging spectral nulls; 1e-18 is the absolute
    floor. These are reporting conventions, not solver convergence criteria.
    """
    def relative_error(full: np.ndarray, smaller: np.ndarray) -> np.ndarray:
        floor = np.maximum(1e-18, 1e-2 * np.max(np.abs(full), axis=1, keepdims=True))
        return np.abs(full - smaller) / np.maximum(np.abs(full), floor)

    return np.maximum(relative_error(sample, smaller_sample),
                      relative_error(reference, smaller_reference)).max(axis=1)


def check_frequencies(samples: np.ndarray) -> np.ndarray | None:
    """Indices for the frequency-subset checks, or None to use every frequency.

    ``samples`` holds one sample signal per harmonic: both ends, an even grid,
    and each harmonic's signal peak and steepest step.
    """
    size = len(samples[0])
    if size <= CHECK_POINTS:
        return None
    ids = {0, size - 1, *np.linspace(0, size - 1, 5).astype(int)}
    for sample in samples:
        amplitude = abs(sample)
        ids |= {int(np.argmax(amplitude)), int(np.argmax(abs(np.diff(amplitude))))}
    return np.array(sorted(ids))


def _refinements(numerical: NumericalSettings, film: bool):
    """(parameter, compared value, focus) beyond the M and N halving checks.

    ``focus`` holds frequency indices solved on top of the common check subset (none
    for quadQ, whose error follows the signal peaks). quadQ sets the spheroid surface tables of the scalar bulk
    engine. Films have none: the automatic K table (``_preset_defaults.film_k``) is
    validated against FIM, and K vs K/2 overstated the error of K by up to 30x.
    """
    if film:
        return []
    return [("quadQ", numerical.quadQ + max(8, numerical.N // 2), np.array([], dtype=int))]


def _halved_m_error(profiles, numerical, signals, sample, reference, sample_profile, reference_profile, tapping):
    """Worst sample/reference change between M and M/2 heights per harmonic.

    Log-height node profiles are demodulated again with M/2 phases and an even M
    reuses its own even-indexed tapping heights, so neither costs a solve. ``tapping``
    is True when either profile is on tapping heights.
    """
    half = numerical.M // 2
    if numerical.M % 2 and tapping:
        return max_signal_error(sample, reference, *signals(profiles(replace(numerical, M=half)), half))
    return max_signal_error(sample, reference, *signals((sample_profile, reference_profile), half, numerical.M))


def _even_heights(profile, columns):
    """Even-indexed tapping heights at ``columns`` of a profile or of each gap's profiles."""
    if isinstance(profile, list):
        return [part[::2, columns] for part in profile]
    return None if profile is None else profile[::2, columns]


def _convergence_checks(profiles, numerical, harmonics, check_n, refinements, static,
                        sample, reference, sample_profile, reference_profile, film=False,
                        signals=None, tapping=(True,)) -> dict:
    """Metadata: errors against M/2, each refinement, then N/2 (a frequency subset for films).

    ``signals(values, M, grid_M=M)`` demodulates (sample, reference) profiles solved at
    ``grid_M`` with M phases (M = grid_M / 2 takes every other tapping height). Signal rows
    are gap-major blocks of ``harmonics``; each error keeps the worst gap. ``tapping`` flags
    the profiles on tapping heights (a list of per-gap profiles counts when all of them are).
    """
    M = numerical.M
    checks = {}
    if static:
        checks["M_check_max_error"] = np.zeros(len(harmonics))
    elif M // 2 > max(harmonics):
        with stage("Checking convergence (M)", 0.6, 0.7):
            report("Reusing full-M tip heights", 1.0)
            checks["M_check_max_error"] = _halved_m_error(
                profiles, numerical, signals, sample, reference, sample_profile, reference_profile, any(tapping))
    # Refinements compare at the even (M/2) heights, which the full profile already holds:
    # both sides share heights, so the difference is the refined parameter alone.
    # Log-height node profiles have no M/2 subset: refinements solve them at full M.
    halved_m = not static and all(tapping) and M % 2 == 0 and M // 2 > max(harmonics)
    for index, (parameter, higher, focus) in enumerate(refinements):
        span = 0.1 / len(refinements)
        picked = None if focus is None else check_frequencies(sample)
        if picked is not None:
            picked = np.union1d(picked, focus)
        columns = slice(None) if picked is None else picked
        settings = replace(numerical, **{parameter: higher}, **({"M": M // 2} if halved_m else {}))
        with stage(f"Checking convergence ({parameter})", 0.7 + index * span, 0.7 + (index + 1) * span):
            values = profiles(settings, picked)
        base = (signals((_even_heights(sample_profile, columns), _even_heights(reference_profile, columns)), M // 2)
                if halved_m else (sample[:, columns], reference[:, columns]))
        checks[f"{parameter}_check_max_error"] = max_signal_error(*base, *signals(values, settings.M))
        checks[f"{parameter}_check_higher"] = higher
        if picked is not None:
            checks[f"{parameter}_check_points"] = len(picked)
    if check_n and numerical.N // 2 > 0:
        picked = check_frequencies(sample) if film else None
        halved_n = replace(numerical, N=numerical.N // 2)
        with stage("Checking convergence (N)", 0.8, 1.0):
            if film and picked is not None:
                # The N error of a film peaks at resonances the common subset can miss: scan every
                # 12th frequency, then fill in around the three worst.
                # Note: stride 12 and +-11 matched the full-grid error on 26 film cases; a peak narrower than the stride could slip past.
                picked = np.union1d(picked, np.arange(0, sample.shape[1], 12))
                halved = signals(profiles(halved_n, picked), M)
                error = (abs(sample[:, picked] - halved[0]) / abs(sample[:, picked])).max(axis=0)
                around = (picked[np.argsort(error)[-3:], None] + np.arange(-11, 12)).ravel()
                extra = np.setdiff1d(np.clip(around, 0, sample.shape[1] - 1), picked)
                if extra.size:
                    more = signals(profiles(halved_n, extra), M)
                    halved = tuple(np.concatenate(pair, axis=1) for pair in zip(halved, more))
                    picked = np.concatenate([picked, extra])
            else:
                halved = signals(profiles(halved_n, picked), M)
        columns = slice(None) if picked is None else picked
        checks["N_check_max_error"] = max_signal_error(sample[:, columns], reference[:, columns], *halved)
        if picked is not None:
            checks["N_check_points"] = len(picked)
    return {key: value.reshape(-1, len(harmonics)).max(axis=0) if np.ndim(value) else value
            for key, value in checks.items()}


# =============================================================================
# Spectra
# =============================================================================

def _validate(frequency, layers, substrate, reference, tip, numerical, harmonics, workers):
    if not harmonics:
        raise ValueError("Select at least one harmonic.")
    validate_inputs(frequency, harmonic=max(harmonics), harmonics=harmonics, tip=tip, numerical=numerical)
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    if any(material.parallel.size != frequency.size
           for material in (substrate, reference, *(material for material, _ in layers))
           if material is not None):
        raise ValueError("Material spectra must match the frequency grid.")


def _metadata(tip, numerical, layers, reference, workers, count):
    film = bool(layers)
    metadata = {"model": "bulk" if not film else "layer" if len(layers) == 1 else "multilayer",
                "M": numerical.M, "N": numerical.N,
                "normalization": "none" if reference is None else "reference",
                "workers": min(workers, count)}
    metadata["quadQ"] = numerical.quadQ
    if film:
        metadata["momentum_rtol"] = numerical.momentum_rtol
        metadata["K"] = numerical.K
    if len(layers) == 1:
        metadata["thickness_nm"] = layers[0][1]
    elif film:
        metadata["thicknesses_nm"] = tuple(depth for _, depth in layers)
    return metadata


def _permittivities(layers, substrate):
    """Displayed sample permittivities using each engine's effective value."""
    names = [f"Layer · {material.label}" if len(layers) == 1 else f"Layer {len(layers) - index} · {material.label}"
             for index, (material, _) in enumerate(layers)]
    names.append(f"{'Substrate' if layers else 'Bulk'} · {substrate.label}")
    out = {name: effective_permittivity(material) for name, (material, _) in zip(names, layers)}
    out[names[-1]] = effective_permittivity(substrate)
    return out


def calculate_spectra(
    frequency_cm1: np.ndarray,
    layers: Sequence[tuple[MaterialSpectrum, float]],
    substrate: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: Sequence[int],
    far_field: FarFieldSettings = FarFieldSettings(),
    *,
    workers: int = 1,
    method: str = "auto",
    checked: bool = False,
    check_n: bool = True,
    solve: Callable = solve_profile,
    sample_gaps: Sequence[float] | None = None,
    reference_gaps: Sequence[float] | None = None,
) -> dict[int, SpectrumResult]:
    """Normalized spectra of a top-to-bottom stack (empty for bulk) on a substrate.

    ``checked`` adds M, N and engine-specific refinement errors to the metadata;
    ``check_n=False`` keeps only the M check (N and quadQ are skipped).
    ``solve`` replaces ``solve_profile`` (same signature), e.g. with a cached version.
    ``sample_gaps`` and ``reference_gaps`` replace the tip's gaps with several, whose complex
    signals are averaged; the checks report the worst gap. When ``spanned``, all gaps of a side
    share one log-height profile over ``gap_span``; otherwise, or when that profile is not smooth,
    each gap is solved on its own tapping heights.
    """
    frequency = np.asarray(frequency_cm1, dtype=float)
    harmonics = tuple(harmonics)
    layers = tuple(layers)
    _validate(frequency, layers, substrate, reference, tip, numerical, harmonics, workers)
    gaps = {"sample": (tip.sample_gap_nm,) if sample_gaps is None else tuple(map(float, sample_gaps)),
            "reference": (tip.reference_gap_nm,) if reference_gaps is None or reference is None
            else tuple(map(float, reference_gaps))}
    if not all(gap > 0 and np.isfinite(gap) for side in gaps.values() for gap in side) or not all(gaps.values()):
        raise ValueError("Averaged gaps must be finite and greater than zero.")
    blocks = max(map(len, gaps.values()))
    layers = simplify_stack(layers, substrate)
    static = tip.amplitude_nm == 0
    diagnostics = {}

    def solve_side(side, stack, base, local, settings, film_method):
        """One side's profile: (heights, F) for one gap or a span, else a list of each gap's profile."""
        # Gaps enter explicitly; equal tip gaps keep sample and reference cache keys independent.
        def at(gap, amplitude, current, **options):
            return solve(stack, base, replace(local, sample_gap_nm=gap, reference_gap_nm=gap, amplitude_nm=amplitude),
                         gap, current, (), film_method, workers, **options)
        side_gaps = gaps[side]
        if len(side_gaps) == 1:
            return at(side_gaps[0], local.amplitude_nm, settings)
        if spanned(side_gaps, local.amplitude_nm, settings.M, settings.height_nodes):
            report("Solving every gap at shared heights", None)
            values, info = at(*gap_span(side_gaps, local.amplitude_nm), replace(settings, M=SPAN_M), fallback=False)
            if info["height_interpolation_tail"] <= HEIGHT_TAIL_LIMIT:
                return values, info
        # Each gap alone: the larger gaps are often smooth where the span (its smallest gap) is not.
        parts, info = [], {}
        for index, gap in enumerate(side_gaps):
            with stage(f"Gap {index + 1}/{len(side_gaps)}", index / len(side_gaps), (index + 1) / len(side_gaps)):
                values, more = at(gap, local.amplitude_nm, settings)
            _merge(info, more)
            parts.append(values)
        return parts, info

    def profiles(settings, ids=None):
        """Sample and reference profiles for every frequency, or the subset ``ids``."""
        stack, base, local = _subset(ids, layers, substrate, tip)
        with stage("Sample", 0.0, 0.9):
            report("Preparing geometry", 0.0)
            values, info = solve_side("sample", stack, base, local, settings, method)
            report("Response complete", 1.0)
        if ids is None and settings is numerical:
            _merge(diagnostics, info)
        if reference is None:
            ref = None
        else:
            with stage("Reference", 0.9, 1.0):
                report("Preparing geometry", 0.0)
                ref, info = solve_side("reference", (), _subset(ids, (), reference, tip)[1], local, settings, "auto")
                report("Response complete", 1.0)
            if ids is None and settings is numerical:
                _merge(diagnostics, info)
        if layers:
            values = [part * E0 for part in values] if isinstance(values, list) else values * E0
            ref = None if ref is None else [part * E0 for part in ref] if isinstance(ref, list) else ref * E0
        return values, ref

    # The N/2 check slices the full-N coupling matrices kept during this call.
    with retaining(checked and check_n):
        with stage("", 0.0, 0.6 if checked else 1.0):
            sample_profile, reference_profile = profiles(numerical)
        def signals(values, M, grid_M=None):
            """(sample, reference) signals with M phases from profiles solved at ``grid_M`` (default M).

            A single-gap profile that is not the grid_M + 1 tapping heights holds log-height nodes.
            Rows are ``blocks`` gap-major blocks of harmonics; a side with fewer gaps repeats them.
            """
            def demodulate(gap, profile):
                solved = M if grid_M is None else grid_M
                if static or profile.shape[0] != solved + 1:
                    return profile_weights(gap, tip.amplitude_nm, M, harmonics, profile.shape[0]) @ profile
                return profile_weights(gap, tip.amplitude_nm, M, harmonics) @ profile[::solved // M]

            def side(name, profile):
                side_gaps = gaps[name]
                if isinstance(profile, list):
                    rows = np.vstack([demodulate(gap, part) for gap, part in zip(side_gaps, profile)])
                elif len(side_gaps) > 1:
                    rows = span_weights(side_gaps, tip.amplitude_nm, M, harmonics, profile.shape[0]) @ profile
                else:
                    rows = demodulate(side_gaps[0], profile)
                ids = np.arange(blocks) % len(side_gaps)
                return rows.reshape(len(side_gaps), len(harmonics), -1)[ids].reshape(blocks * len(harmonics), -1)
            sample = side("sample", values[0])
            return sample, np.ones_like(sample) if values[1] is None else side("reference", values[1])

        def on_tapping(name, profile):
            if isinstance(profile, list):
                return all(part.shape[0] == numerical.M + 1 for part in profile)
            return len(gaps[name]) == 1 and profile.shape[0] == numerical.M + 1

        sample, reference_signal = signals((sample_profile, reference_profile), numerical.M)
        checks = {}
        if checked:
            checks = _convergence_checks(
                profiles, numerical, harmonics, check_n,
                _refinements(numerical, bool(layers)) if check_n else [], static,
                sample, reference_signal, sample_profile, reference_profile, bool(layers),
                signals=signals, tapping=tuple(on_tapping(name, profile) for name, profile in (
                    ("sample", sample_profile), ("reference", reference_profile)) if profile is not None))
    def average(rows, name):
        return rows.reshape(blocks, len(harmonics), -1)[:len(gaps[name])].mean(axis=0)
    sample, reference_signal = average(sample, "sample"), average(reference_signal, "reference")
    sample = sample * far_field_factor(frequency, layers, substrate, far_field)[None]
    if reference is not None:
        reference_signal = reference_signal * far_field_factor(frequency, (), reference, far_field)[None]
    if not np.isfinite(sample).all() or not np.isfinite(reference_signal).all():
        raise ValueError("Response contains non-finite values; refine numerical settings or material loss.")
    contrast = normalized_contrast(sample, reference_signal, harmonics, static=static and reference is not None)
    metadata = {**_metadata(tip, numerical, layers, reference, workers, frequency.size), **diagnostics}
    permittivities = _permittivities(layers, substrate)
    results = {}
    for index, harmonic in enumerate(harmonics):
        results[harmonic] = SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=sample[index], reference_harmonic=reference_signal[index],
            contrast=contrast[index], permittivities=permittivities,
            metadata={**metadata, "harmonic": harmonic,
                      **{key: float(value[index]) if np.ndim(value) else value
                         for key, value in checks.items()}})
    return results
