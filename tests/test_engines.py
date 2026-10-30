from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import snom_calculator._spectra as spectra
import snom_calculator._spheroid_engine as engine
from snom_calculator._far_field import far_field_factor
from snom_calculator._quadrature import (
    _continued_fraction_Hn, demodulation_weights, legendre_P_all, legendre_Q_all, tapping_heights,
)
from snom_calculator.bulk import calculate_bulk_harmonics
from snom_calculator.calculations import calculate_bulk_checked
from snom_calculator.layer import calculate_layer_harmonics
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters


def _material(value: complex, size: int, label: str) -> MaterialSpectrum:
    values = np.full(size, value, dtype=complex)
    return MaterialSpectrum(values, values, label=label)


def test_layer_passes_momentum_tolerance_to_integration(monkeypatch):
    from snom_calculator._fim_coupling import FIMCoupling

    tolerances = []
    original = FIMCoupling.matrices

    def capture(self, layers, substrate, rtol=1e-5):
        tolerances.append(rtol)
        return original(self, layers, substrate, rtol=rtol)

    monkeypatch.setattr(FIMCoupling, "matrices", capture)
    frequency = np.array([900., 905.])
    result = calculate_layer_harmonics(
        frequency,
        layer=MaterialSpectrum(np.full(2, 4 + .1j), np.full(2, 6 + .2j), label="film"),
        substrate=_material(12 + .1j, 2, "substrate"),
        reference=_material(-5000 + 800j, 2, "reference"),
        thickness_nm=20,
        tip=TipParameters(),
        numerical=NumericalSettings(M=8, N=8, quadQ=16, momentum_rtol=1e-6),
        harmonics=(3,),
    )[3]
    assert tolerances == [1e-6, 1e-6]
    assert result.metadata["momentum_rtol"] == 1e-6
    assert np.isfinite(result.contrast).all()


def test_bulk_returns_finite_normalized_spectrum():
    frequency = np.linspace(900, 905, 3)
    result = calculate_bulk_harmonics(
        frequency,
        sample=_material(6 + 0.2j, 3, "sample"),
        reference=_material(-5000 + 800j, 3, "reference"),
        tip=TipParameters(radius_nm=25, length_nm=1200, amplitude_nm=60),
        numerical=NumericalSettings(M=8, N=8, quadQ=32),
        harmonics=(3,),
    )[3]
    assert result.contrast.shape == frequency.shape
    assert np.isfinite(result.contrast).all()


def test_bulk_demodulates_selected_harmonics_from_two_shared_profiles(monkeypatch):
    calls = []
    original = engine._bulk_profile

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_bulk_profile", counted)
    results = calculate_bulk_harmonics(
        np.linspace(900, 905, 3),
        sample=_material(6 + 0.2j, 3, "sample"),
        reference=_material(-5000 + 800j, 3, "reference"),
        tip=TipParameters(radius_nm=25, length_nm=1200, amplitude_nm=60),
        numerical=NumericalSettings(M=8, N=8, quadQ=32),
        harmonics=(2, 3),
    )
    assert tuple(results) == (2, 3)
    assert len(calls) == 2  # sample and reference


def test_bulk_checks_match_independent_smaller_and_refined_solves():
    frequency = np.array([900., 905.])
    sample = _material(6 + .2j, 2, "sample")
    reference = _material(-5000 + 800j, 2, "reference")
    tip = TipParameters(25., 600., 40., 2., 2., -1e10 + 0j)
    numerical = NumericalSettings(M=8, N=8, quadQ=16)
    off = FarFieldSettings(enabled=False)
    harmonics = (2, 3)

    def solve(settings):
        results = calculate_bulk_harmonics(frequency, sample=sample, reference=reference, tip=tip,
                                           numerical=settings, harmonics=harmonics, far_field=off)
        return (np.stack([results[h].sample_harmonic for h in harmonics]),
                np.stack([results[h].reference_harmonic for h in harmonics]))

    checked = calculate_bulk_checked(frequency, sample, reference, tip, numerical, harmonics, off)
    full = solve(numerical)
    for parameter, value in (("M", 4), ("N", 4), ("quadQ", 24)):
        # quadQ is compared at the M/2 heights on both sides.
        base, smaller = (solve(replace(numerical, M=4)), replace(numerical, M=4)) if parameter == "quadQ" \
            else (full, numerical)
        expected = spectra.max_signal_error(*base, *solve(replace(smaller, **{parameter: value})))
        actual = [checked[h].metadata[f"{parameter}_check_max_error"] for h in harmonics]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-14)
    assert checked[3].metadata["quadQ_check_higher"] == 24
    unchecked_m = calculate_bulk_checked(frequency, sample, reference, tip, numerical, harmonics, off,
                                         check_n=False)
    assert "N_check_max_error" not in unchecked_m[3].metadata


def test_layer_demodulates_selected_harmonics_in_one_engine_call(monkeypatch):
    calls = []
    original = engine._film_profile

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_film_profile", counted)
    results = calculate_layer_harmonics(
        np.linspace(900, 905, 3),
        layer=_material(6 + 0.2j, 3, "layer"),
        substrate=_material(12 + 0j, 3, "substrate"),
        reference=_material(-5000 + 800j, 3, "reference"),
        thickness_nm=20,
        tip=TipParameters(radius_nm=30, length_nm=600, amplitude_nm=50),
        numerical=NumericalSettings(M=8, N=8, quadQ=16, K=2),
        harmonics=(2, 3),
    )
    assert tuple(results) == (2, 3)
    assert len(calls) == 1


def test_bulk_geometry_cache_ignores_material_and_harmonic():
    frequency = np.linspace(900, 905, 3)
    common = dict(reference=_material(-5000 + 800j, 3, "reference"),
                  tip=TipParameters(radius_nm=25, length_nm=1200, amplitude_nm=60),
                  numerical=NumericalSettings(M=8, N=8, quadQ=32))
    engine._bulk_modes.cache_clear()
    calculate_bulk_harmonics(frequency, sample=_material(6 + 0.2j, 3, "first"), harmonics=(2,), **common)
    calculate_bulk_harmonics(frequency, sample=_material(8 + 0.3j, 3, "second"), harmonics=(3,), **common)
    # Sample and reference share one geometry: same gap, M and N.
    info = engine._bulk_modes.cache_info()
    assert info.misses == 1
    assert info.hits == 3


def test_bulk_profile_vectorizes_frequency():
    tip = TipParameters(25.0, 1200.0, 60.0, 5.0, 5.0, -1e10 + 0j)
    numerical = NumericalSettings(M=8, N=8, quadQ=32)
    beta = np.array([0.4 + 0.1j, 0.7 + 0.2j])
    heights = tapping_heights(5.0, tip.amplitude_nm, 8)
    vectorized = engine._bulk_profile(beta, tip, 5.0, numerical, 8, heights)
    scalar = np.concatenate([engine._bulk_profile(beta[i:i + 1], tip, 5.0, numerical, 8, heights)
                             for i in range(2)], axis=1)
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-13, atol=1e-13)


def test_height_tables_are_shared_by_materials_and_thicknesses():
    tip = TipParameters(30, 600, 50, 3, 3, -1e10)
    numerical = NumericalSettings(M=8, N=8, quadQ=16, K=2)
    engine.clear_caches()
    for frequency, layer, substrate, depth in (
            (np.linspace(900, 905, 3), 6 + .2j, 12 + 0j, 20.),
            (np.linspace(910, 920, 4), 8 + .3j, 10 + .1j, 20.),
            (np.linspace(910, 920, 4), 8 + .3j, 10 + .1j, 25.)):
        size = frequency.size
        engine.profile(((_material(layer, size, "layer"), depth),), _material(substrate, size, "substrate"),
                       tip, 3., numerical)
    info = engine._height_tables.cache_info()
    assert info.misses == 1
    assert info.hits >= 2


def test_film_solve_respects_frequency_chunks(monkeypatch):
    solve_sizes = []
    original_solve = engine.np.linalg.solve

    def recording_solve(matrices, right_hand_side):
        solve_sizes.append(matrices.shape[0])
        return original_solve(matrices, right_hand_side)

    monkeypatch.setattr(engine.np.linalg, "solve", recording_solve)
    engine.profile(((_material(6 + 0.2j, 9, "layer"), 20.),), _material(12 + 0j, 9, "substrate"),
                   TipParameters(30, 600, 50, 3, 3, -1e10), 3., NumericalSettings(M=8, N=8, quadQ=16, K=2))
    assert max(solve_sizes) <= engine._FREQUENCY_CHUNK * (8 + 1)


def test_zero_thickness_layer_reduces_to_bulk_substrate_response():
    frequency = np.linspace(900, 905, 3)
    common = dict(
        frequency_cm1=frequency,
        reference=_material(12 + 0j, 3, "reference"),
        tip=TipParameters(30, 600, 50, 3, 3, -1e10 + 0j),
        numerical=NumericalSettings(M=8, N=8, quadQ=16, K=2),
        harmonics=(3,),
        substrate=_material(12 + 0j, 3, "substrate"),
        thickness_nm=0,
    )
    result = calculate_layer_harmonics(layer=_material(-2 + 0.2j, 3, "layer"), **common)[3]
    ignored_layer = calculate_layer_harmonics(layer=_material(7 + 0.5j, 3, "other layer"), **common)[3]
    np.testing.assert_allclose(result.contrast, ignored_layer.contrast, rtol=1e-12, atol=1e-12)


def test_parallel_layer_matches_sequential_and_preserves_frequency_order():
    frequency = np.linspace(900, 912, 8)
    common = dict(
        frequency_cm1=frequency,
        layer=_material(6 + 0.2j, frequency.size, "layer"),
        substrate=_material(12 + 0j, frequency.size, "substrate"),
        reference=_material(12 + 0j, frequency.size, "reference"),
        thickness_nm=20,
        tip=TipParameters(30, 600, 50, 3, 3, -1e10 + 0j),
        numerical=NumericalSettings(M=8, N=8, quadQ=16, K=2),
        harmonics=(3,),
    )
    sequential = calculate_layer_harmonics(workers=1, **common)[3]
    parallel = calculate_layer_harmonics(workers=2, **common)[3]
    np.testing.assert_array_equal(parallel.frequency_cm1, frequency)
    np.testing.assert_allclose(parallel.sample_harmonic, sequential.sample_harmonic, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(parallel.reference_harmonic, sequential.reference_harmonic, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(parallel.contrast, sequential.contrast, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel.phase_rad, sequential.phase_rad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("workers", [0, 5, True, 1.5])
def test_layer_rejects_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="workers must be an integer from 1 to 4"):
        calculate_layer_harmonics(
            np.linspace(900, 905, 3),
            layer=_material(6 + 0.2j, 3, "layer"),
            substrate=_material(12 + 0j, 3, "substrate"),
            reference=_material(12 + 0j, 3, "reference"),
            thickness_nm=20,
            tip=TipParameters(radius_nm=30, length_nm=600, amplitude_nm=50),
            numerical=NumericalSettings(M=8, N=8, quadQ=16, K=2),
            harmonics=(3,),
            workers=workers,
        )


def test_zero_thickness_bypasses_pool_and_matches_bulk(monkeypatch):
    frequency = np.linspace(900, 905, 3)
    common = dict(reference=_material(-5000 + 800j, 3, "reference"),
                  tip=TipParameters(radius_nm=30, length_nm=600, amplitude_nm=50),
                  numerical=NumericalSettings(M=8, N=8, quadQ=16, K=2), harmonics=(3,))
    substrate = _material(12 + 0j, 3, "substrate")
    monkeypatch.setattr(spectra, "_executor",
                        lambda workers: (_ for _ in ()).throw(AssertionError("pool used")))
    layer_result = calculate_layer_harmonics(frequency, layer=_material(6 + 0.2j, 3, "ignored layer"),
                                             substrate=substrate, thickness_nm=0, workers=4, **common)[3]
    bulk_result = calculate_bulk_harmonics(frequency, sample=substrate, **common)[3]
    np.testing.assert_allclose(layer_result.sample_harmonic, bulk_result.sample_harmonic)
    np.testing.assert_allclose(layer_result.reference_harmonic, bulk_result.reference_harmonic)
    np.testing.assert_allclose(layer_result.contrast, bulk_result.contrast)


def test_bulk_eigen_solver_matches_direct_linear_solves():
    radius, length, amplitude, gap = 25.0, 1200.0, 60.0, 5.0
    tip_epsilon, harmonic, M, N, quadQ = -1e10 + 0j, 3, 8, 8, 32
    beta = (6 + 0.2j - 1) / (6 + 0.2j + 1)
    tip = TipParameters(radius, length, amplitude, gap, gap, tip_epsilon)
    eigen = (demodulation_weights(M, (harmonic,)) @ engine._bulk_profile(
        np.array([beta]), tip, gap, NumericalSettings(M=M, N=N, quadQ=quadQ), M,
        tapping_heights(gap, amplitude, M)))[0, 0]

    a = length / 2
    c = np.sqrt(a * (a - radius))
    axis = a / c
    nodes, quadrature_weights = leggauss(quadQ)
    p_nodes = legendre_P_all(N, nodes)
    p_axis = legendre_P_all(N, np.array([axis]))[:, 0]
    q_axis = legendre_Q_all(N, np.array([axis]))[:, 0]
    den = np.array([tip_epsilon * q_axis[i] / p_axis[i]
                    - (axis * q_axis[i] - q_axis[i - 1]) / (axis * p_axis[i] - p_axis[i - 1])
                    for i in range(1, N + 1)])
    den1 = tip_epsilon * q_axis[1] - axis * q_axis[0] + a * a / (a * a - c * c)
    profile = []
    for k in range(M + 1):
        distance = gap + amplitude - amplitude * np.cos(np.pi * k / M)
        shifted = 2 * a / c + 2 * distance / c - axis * nodes
        base = 1 + shifted**2 + (a * a / c**2 - 1) * (1 - nodes**2)
        radial = np.sqrt((base + np.sqrt(np.maximum(base**2 - 4 * shifted**2, 0))) / 2)
        j = (p_nodes[1:] * quadrature_weights) @ (
            legendre_P_all(N, shifted / radial)[1:] * legendre_Q_all(N, radial)[1:]).T
        source = -beta * (tip_epsilon - 1) ** 2 * c / 4 * j[0] / (p_axis[1:] * den1 * den)
        system = np.eye(N) - beta * (j.T / (p_axis[1:] * den)[:, None]) * (
            (tip_epsilon - 1) * (2 * np.arange(1, N + 1) + 1) / 2)[None, :]
        coefficients = np.linalg.solve(system, source)
        profile.append(2 * a * c * coefficients[0] - (tip_epsilon - 1) * a * c**2 / (3 * den1))
    direct = np.sum(np.asarray(profile) * eigen_weights(harmonic, M))
    np.testing.assert_allclose(eigen, direct, rtol=1e-11, atol=1e-8)


def eigen_weights(harmonic: int, M: int) -> np.ndarray:
    weights = np.cos(np.pi * np.arange(M + 1) * harmonic / M) / M
    weights[[0, -1]] = (np.exp(-1j * np.pi * np.array([0, M]) * harmonic / M) / (2 * M)).real
    return weights


def test_legendre_q_stays_finite_where_p_overflows():
    x = np.array([1.5, 50.0, 1e3])
    Q = legendre_Q_all(500, x)
    assert np.isfinite(Q).all() and (Q >= 0).all()
    # Below overflow the result is unchanged: Q_n = H_n * Q_{n-1}.
    n = 60
    np.testing.assert_allclose(_continued_fraction_Hn(n, x[:1]) * Q[n - 1, :1], Q[n, :1], rtol=1e-12)


def test_bulk_high_multipole_order_is_finite_and_blunt_tip_limit_is_reported():
    frequency = np.array([900.0, 950.0])
    epsilon = np.array([-10 + 1j, -5 + 0.5j])
    common = dict(
        frequency_cm1=frequency,
        sample=MaterialSpectrum(epsilon, epsilon, "sample"),
        reference=MaterialSpectrum(np.full(2, -5000 + 800j), np.full(2, -5000 + 800j), "Au"),
        harmonics=(3,),
    )
    result = calculate_bulk_harmonics(tip=TipParameters(25, 1200, 60, 5, 5, -1e10),
                                      numerical=NumericalSettings(M=20, N=400, quadQ=640), **common)[3]
    assert np.isfinite(result.contrast).all()
    with pytest.raises(ValueError, match="too large for this tip shape"):
        calculate_bulk_harmonics(tip=TipParameters(100, 300, 60, 1, 1, -1e10),
                                 numerical=NumericalSettings(M=20, N=400, quadQ=640), **common)


@pytest.mark.parametrize("harmonics", [(1.5, 3), (True, 3), (-1, 3)])
def test_bulk_rejects_invalid_lower_harmonics(harmonics):
    frequency = np.array([900., 901.])
    material = constant_material(12, frequency, "Si")
    with pytest.raises(ValueError, match="Harmonic"):
        calculate_bulk_harmonics(frequency, sample=material, reference=material,
                                 tip=TipParameters(), numerical=NumericalSettings(), harmonics=harmonics)


def test_far_field_removable_singularity_at_zero_normal_wavevector():
    # At 30 degrees sin(phi)^2 = .25, so kz in this medium is exactly zero and r_p = 1.
    frequency = np.array([900., 901.])
    angle = 30.
    with np.errstate(all="raise"):
        field = far_field_factor(frequency, (), constant_material(np.sin(np.deg2rad(angle)) ** 2,
                                                                  frequency, "critical"),
                                 FarFieldSettings(angle_deg=angle))
    np.testing.assert_allclose(field, [2, 2])


def test_bulk_profile_matches_dense_per_height_solve():
    # Bare-surface solves use the eigen resolvent; check against dense boundary solves.
    epsilon = np.array([-4000 + 700j, 6 + 0.2j, -20 + 5j])
    radius, length, amplitude, gap, tip_epsilon, M, N, quadQ = 25.0, 600.0, 50.0, 5.0, -1e10, 10, 12, 32
    tip = TipParameters(radius, length, amplitude, gap, gap, tip_epsilon)
    beta = (epsilon - 1) / (epsilon + 1)
    heights = tapping_heights(gap, amplitude, M)
    eigen = engine._bulk_profile(beta, tip, gap, NumericalSettings(M=M, N=N, quadQ=quadQ), M, heights)
    table = engine._surface_trajectory(radius, length, gap, amplitude, M, N, quadQ, heights)
    dense = engine._solve_coupled(-beta[:, None, None, None] * table[None],
                                  engine._tip(radius, length, complex(tip_epsilon), N))
    np.testing.assert_allclose(eigen, dense.T, rtol=1e-11)


def test_tip_axis_tables_are_reused_across_permittivity_and_gap(monkeypatch):
    from snom_calculator import _spheroid_engine as engine

    engine.clear_caches()
    original = engine.legendre_Q_all
    calls = []

    def counted(order, axis):
        if np.ndim(axis) == 0:
            calls.append((order, axis))
        return original(order, axis)

    monkeypatch.setattr(engine, "legendre_Q_all", counted)
    for epsilon in (-20 + 2j, -30 + 3j, 12 + .1j):
        engine._tip(25., 600., epsilon, 8)
    assert len(calls) == 1
    engine._tip(26., 600., -20 + 2j, 8)
    engine._tip(25., 600., -20 + 2j, 9)
    assert len(calls) == 3
