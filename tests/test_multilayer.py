"""Multilayer stack checks."""

import numpy as np
import pytest

from snom_calculator._far_field import far_field_factor, reflection_matrix
from snom_calculator._spheroid_engine import clear_caches
from snom_calculator.layer import calculate_stack_harmonics, calculate_layer_harmonics
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters
from snom_calculator._uniaxial_esm import reflection
from snom_calculator._fim_coupling import _cached_fim


def _settings(frequency):
    return dict(
        frequency_cm1=frequency,
        substrate=constant_material(12, frequency, "Si"),
        reference=constant_material(12, frequency, "reference"),
        tip=TipParameters(30, 600, 50, 2, 2, -1e10),
        numerical=NumericalSettings(M=8, N=8, quadQ=16, K=8),
        harmonics=(2, 3),
    )


def test_uniaxial_film_uses_normal_decay_with_geometric_mean_admittance():
    k = np.array([0.05])
    beta_top = 5 / 7
    beta_bottom = 1 / 3
    for q_over_k in (2 / 3, 3 / 2):
        decay = np.exp(-2 * k * q_over_k * 20)
        expected = (beta_top + beta_bottom * decay) / (1 + beta_top * beta_bottom * decay)
        perpendicular, parallel = (4, 9) if q_over_k == 2 / 3 else (9, 4)
        actual = -reflection(k, ((perpendicular, parallel, 20),), 12)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_uniaxial_film_far_field_uses_extraordinary_kz():
    om = np.array([950.0])
    phi = np.pi / 3
    sine2 = np.sin(phi) ** 2
    kz_air = np.cos(phi)
    kz_film = np.sqrt(4 - 4 / 9 * sine2)
    kz_sub = np.sqrt(12 - sine2)
    rp_top = (4 / kz_film - 1 / kz_air) / (4 / kz_film + 1 / kz_air)
    rp_bottom = (12 / kz_sub - 4 / kz_film) / (12 / kz_sub + 4 / kz_film)
    phase = np.exp(2j * kz_film * 20 * 2 * np.pi * om * 1e-7)
    rp = (rp_top + rp_bottom * phase) / (1 + rp_top * rp_bottom * phase)
    film = MaterialSpectrum(np.array([9]), np.array([4]), "film")  # epsilon_zz 9, epsilon_xx 4
    actual = far_field_factor(om, ((film, 20),), constant_material(12, om, "Si"), FarFieldSettings(angle_deg=60.))
    np.testing.assert_allclose(actual, (1 + rp) ** 2 * np.sin(phi), rtol=1e-13)


@pytest.mark.parametrize("layers", ((), 20.))
def test_hyperbolic_far_field_takes_the_outgoing_maxwell_branch(layers):
    om = np.array([950.0, 951.0])
    material = MaterialSpectrum(np.full(2, 1 + 0.01j), np.full(2, -1 + 0.01j), "film")
    stack = () if layers == () else ((material, layers),)
    substrate = material if layers == () else constant_material(12, om, "Si")
    settings = FarFieldSettings(enabled=True)
    maxwell = (1 + reflection_matrix(om, stack, substrate, settings)[:, 0, 0]) ** 2 * np.sin(np.pi / 3)
    np.testing.assert_allclose(far_field_factor(om, stack, substrate, settings), maxwell, rtol=1e-12)


def test_one_layer_stack_preserves_existing_engine():
    frequency = np.linspace(950, 952, 3)
    layer = constant_material(3 + 0.2j, frequency, "film")
    common = _settings(frequency)
    old = calculate_layer_harmonics(layer=layer, thickness_nm=10, **common)
    new = calculate_stack_harmonics(layers=((layer, 10),), **common)

    for harmonic in (2, 3):
        np.testing.assert_allclose(new[harmonic].contrast, old[harmonic].contrast, rtol=1e-11, atol=1e-11)


def test_same_admittance_uniaxial_films_have_different_finite_layer_signal():
    frequency = np.array([950.0, 951.0])
    common = _settings(frequency)
    common["numerical"] = NumericalSettings(M=8, N=8, quadQ=16, K=12)
    films = (
        MaterialSpectrum(np.full(2, 9), np.full(2, 4), "slow decay"),
        MaterialSpectrum(np.full(2, 4), np.full(2, 9), "fast decay"),
    )
    signals = [
        calculate_layer_harmonics(layer=film, thickness_nm=20, **common)[3].sample_harmonic
        for film in films
    ]
    assert np.max(abs(signals[0] - signals[1])) > 1e-8


def test_uniaxial_film_differs_from_scalar_geometric_mean_at_same_depth():
    frequency = np.array([950.0, 951.0])
    anisotropic = MaterialSpectrum(np.full(2, 10), np.full(2, 0.1), "film")
    scalar = constant_material(1, frequency, "equivalent film")
    common = _settings(frequency)
    anisotropic_signal = calculate_layer_harmonics(
        layer=anisotropic, thickness_nm=100, **common
    )[3].sample_harmonic
    equivalent_signal = calculate_layer_harmonics(
        layer=scalar, thickness_nm=100, **common
    )[3].sample_harmonic
    assert np.max(abs(anisotropic_signal - equivalent_signal)) > 1e-8


def test_uniaxial_result_at_one_frequency_is_independent_of_other_frequencies():
    frequency = np.array([950.0, 951.0])
    dispersive = MaterialSpectrum(np.array([100, 1]), np.array([0.01, 1]), "film")
    scalar = constant_material(1, frequency, "same at second frequency")
    common = _settings(frequency)
    mixed = calculate_layer_harmonics(
        layer=dispersive, thickness_nm=100, **common
    )[3].sample_harmonic
    alone = calculate_layer_harmonics(
        layer=scalar, thickness_nm=100, **common
    )[3].sample_harmonic
    np.testing.assert_allclose(mixed[1], alone[1], rtol=1e-10)
    parallel = calculate_layer_harmonics(
        layer=dispersive, thickness_nm=100, workers=2, **common
    )[3].sample_harmonic
    np.testing.assert_allclose(parallel, mixed, rtol=1e-10)


def test_dispersive_uniaxial_film_reuses_momentum_geometry():
    frequency = np.array([950.0, 951.0, 952.0])
    film = MaterialSpectrum(
        np.array([9.0, 10.0, 11.0]), np.array([4.0, 4.5, 5.0]), "dispersive film"
    )
    clear_caches()
    common = _settings(frequency)
    calculate_layer_harmonics(layer=film, thickness_nm=20, **common)
    assert _cached_fim.cache_info().misses == 1
    calculate_layer_harmonics(layer=film, thickness_nm=20, **common)
    assert _cached_fim.cache_info().misses == 1
    assert _cached_fim.cache_info().hits >= 1


def test_multilayer_fit_is_independent_of_other_frequencies():
    frequency = np.array([950.0, 951.0])
    dispersive = MaterialSpectrum(np.array([100, 1]), np.array([0.01, 1]), "top")
    scalar = constant_material(1, frequency, "same at second frequency")
    buried = constant_material(4, frequency, "buried")
    common = _settings(frequency)
    mixed = calculate_stack_harmonics(
        layers=((dispersive, 100), (buried, 20)), **common
    )[3].sample_harmonic
    alone = calculate_stack_harmonics(
        layers=((scalar, 100), (buried, 20)), **common
    )[3].sample_harmonic
    np.testing.assert_allclose(mixed[1], alone[1], rtol=1e-10)


def test_zero_effective_permittivity_is_rejected_before_fitting():
    frequency = np.array([950.0, 951.0])
    film = MaterialSpectrum(np.zeros(2), np.full(2, 4), "zero normal")
    with pytest.raises(ValueError, match="effective permittivity"):
        calculate_layer_harmonics(layer=film, thickness_nm=20, **_settings(frequency))


def test_adjacent_layers_with_same_admittance_remain_distinct_tensors():
    frequency = np.array([950.0, 951.0])
    top = MaterialSpectrum(np.full(2, 9), np.full(2, 4), "top")
    bottom = MaterialSpectrum(np.full(2, 4), np.full(2, 9), "bottom")
    result = calculate_stack_harmonics(
        layers=((top, 10), (bottom, 20)), **_settings(frequency)
    )[3]
    assert result.metadata["model"] == "multilayer"
    assert result.metadata["thicknesses_nm"] == (10, 20)


def test_buried_uniaxial_layer_differs_from_scalar_geometric_mean():
    frequency = np.array([950.0, 951.0])
    top = constant_material(2, frequency, "top")
    anisotropic = MaterialSpectrum(np.full(2, 9), np.full(2, 4), "buried")
    isotropic = constant_material(6, frequency, "scalarized buried")
    common = _settings(frequency)
    anisotropic_signal = calculate_stack_harmonics(
        layers=((top, 10), (anisotropic, 20)), **common
    )[3].sample_harmonic
    scalar_signal = calculate_stack_harmonics(
        layers=((top, 10), (isotropic, 20)), **common
    )[3].sample_harmonic
    assert np.max(abs(anisotropic_signal - scalar_signal)) > 1e-8


def test_thin_buried_layer_matching_substrate_preserves_top_film():
    frequency = np.array([950.0, 951.0])
    top = constant_material(2, frequency, "top")
    common = _settings(frequency)
    stack = calculate_stack_harmonics(
        layers=((top, 100), (common["substrate"], 1)), **common
    )[3].sample_harmonic
    single = calculate_layer_harmonics(
        layer=top, thickness_nm=100, **common
    )[3].sample_harmonic
    np.testing.assert_allclose(stack, single, rtol=1e-8)


def test_layer_reference_near_field_uses_scalar_admittance():
    frequency = np.array([950.0, 951.0])
    reference = MaterialSpectrum(np.array([9, 4]), np.array([4, 9]), "reference")
    common = _settings(frequency)
    common["reference"] = reference
    common["far_field"] = FarFieldSettings(enabled=False)
    film = constant_material(6, frequency, "film")
    signal = calculate_layer_harmonics(layer=film, thickness_nm=20, **common)[3].reference_harmonic
    np.testing.assert_allclose(signal[0], signal[1], rtol=1e-12)


def test_equal_adjacent_layers_collapse_and_five_layers_remain_finite():
    frequency = np.linspace(950, 952, 3)
    common = _settings(frequency)
    film = constant_material(3 + 0.2j, frequency, "film")
    five = calculate_stack_harmonics(layers=((film, 4), (film, 6), (film, 5), (film, 3), (film, 2)), **common)
    one = calculate_layer_harmonics(layer=film, thickness_nm=20, **common)

    for harmonic in (2, 3):
        assert np.isfinite(five[harmonic].contrast).all()
        np.testing.assert_allclose(five[harmonic].contrast, one[harmonic].contrast, rtol=2e-3, atol=2e-3)


def test_two_layer_limit_matches_one_layer_when_bottom_matches_substrate():
    frequency = np.linspace(950, 952, 3)
    common = _settings(frequency)
    film = constant_material(3 + 0.2j, frequency, "film")
    bottom = constant_material(12, frequency, "Si")
    stack = calculate_stack_harmonics(layers=((film, 10), (bottom, 20)), **common)
    one = calculate_layer_harmonics(layer=film, thickness_nm=10, **common)

    for harmonic in (2, 3):
        np.testing.assert_allclose(stack[harmonic].contrast, one[harmonic].contrast, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("count", (4, 5))
def test_distinct_deep_stacks_are_finite_and_stable_as_k_increases(count):
    frequency = np.linspace(950, 952, 3)
    common = _settings(frequency)
    layers = tuple(
        (constant_material(2 + index + 0.1j, frequency, f"film {index}"), 8 + index * 2)
        for index in range(count)
    )
    common["numerical"] = NumericalSettings(M=8, N=8, quadQ=16, K=12)
    lower = calculate_stack_harmonics(layers=layers, **common)
    common["numerical"] = NumericalSettings(M=8, N=8, quadQ=16, K=16)
    higher = calculate_stack_harmonics(layers=layers, **common)

    for harmonic in (2, 3):
        assert np.isfinite(higher[harmonic].contrast).all()
        np.testing.assert_allclose(higher[harmonic].contrast, lower[harmonic].contrast, rtol=0.03, atol=0.03)


def test_four_layer_parallel_result_matches_single_worker():
    frequency = np.linspace(950, 956, 6)
    layers = list(
        (constant_material(2 + index + 0.1j, frequency, f"film {index}"), 8 + index * 2)
        for index in range(4)
    )
    layers[1] = (MaterialSpectrum(
        np.linspace(4, 5, frequency.size) + 0.2j,
        np.linspace(2, 3, frequency.size) + 0.1j,
        "uniaxial film",
    ), layers[1][1])
    common = _settings(frequency)
    sequential = calculate_stack_harmonics(layers=layers, workers=1, **common)
    parallel = calculate_stack_harmonics(layers=layers, workers=2, **common)

    for harmonic in (2, 3):
        np.testing.assert_allclose(parallel[harmonic].contrast, sequential[harmonic].contrast, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("depth", (-1, np.nan))
def test_stack_rejects_invalid_thickness(depth):
    frequency = np.array([950.0, 951.0])
    film = constant_material(3, frequency, "film")
    with pytest.raises(ValueError, match="thickness"):
        calculate_stack_harmonics(layers=((film, depth),), **_settings(frequency))


def test_spheroid_film_far_field_uses_coefficient_and_substrate_axes():
    # Every engine shares one far-field factor: the reflection coefficient applies
    # to films too, and a uniaxial substrate keeps its axes (not its effective permittivity).
    frequency = np.array([950.0, 951.0])
    common = _settings(frequency)
    common["substrate"] = MaterialSpectrum(np.full(2, 7 + .2j), np.full(2, 12 + .3j), "uniaxial Si")
    film = constant_material(3 + .2j, frequency, "film")
    near = calculate_layer_harmonics(layer=film, thickness_nm=20, far_field=FarFieldSettings(enabled=False),
                                     **common)[3]
    settings = FarFieldSettings(angle_deg=55., coefficient=.8)
    far = calculate_layer_harmonics(layer=film, thickness_nm=20, far_field=settings, **common)[3]
    maxwell = reflection_matrix(frequency, ((film, 20),), common["substrate"], settings)[:, 0, 0]
    factor = (1 + .8 * maxwell) ** 2 * np.sin(np.radians(55.))
    np.testing.assert_allclose(far.sample_harmonic, near.sample_harmonic * factor, rtol=1e-12)


@pytest.mark.parametrize("coefficient", (.8, 1.3))
def test_spheroid_stack_and_uniaxial_reference_share_the_far_field_factor(coefficient):
    # Two-film GICM stack and a uniaxial reference: both use (1 + zeta r_pp)^2 sin(theta)
    # with their own tensor axes, independently computed by Maxwell scattering.
    frequency = np.array([950.0, 951.0])
    common = _settings(frequency)
    common["reference"] = MaterialSpectrum(np.full(2, 9 + .1j), np.full(2, 4 + .2j), "uniaxial reference")
    layers = ((constant_material(2 + .1j, frequency, "top"), 10), (constant_material(4 + .2j, frequency, "middle"), 20))
    near = calculate_stack_harmonics(layers=layers, far_field=FarFieldSettings(enabled=False), **common)[3]
    settings = FarFieldSettings(angle_deg=50., coefficient=coefficient)
    far = calculate_stack_harmonics(layers=layers, far_field=settings, **common)[3]

    def factor(stack, substrate):
        rpp = reflection_matrix(frequency, stack, substrate, settings)[:, 0, 0]
        return (1 + coefficient * rpp) ** 2 * np.sin(np.radians(50.))

    np.testing.assert_allclose(far.sample_harmonic, near.sample_harmonic * factor(layers, common["substrate"]),
                               rtol=1e-12)
    np.testing.assert_allclose(far.reference_harmonic,
                               near.reference_harmonic * factor((), common["reference"]), rtol=1e-12)
