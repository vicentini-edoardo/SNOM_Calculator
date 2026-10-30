"""Maxwell tensor scattering: Fresnel recovery and stable film limits."""

from types import SimpleNamespace

import numpy as np
import pytest

from snom_calculator.materials import MaterialSpectrum
from snom_calculator.models import FarFieldSettings


FREQUENCY = np.array([950., 1100.])


def material(x, y=None, z=None, twist=0., tilt=0.):
    y, z = (x if y is None else y), (x if z is None else z)
    a, b = np.radians([twist, tilt])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rotation = rz @ ry
    tensor = rotation @ np.diag([x, y, z]) @ rotation.T
    return SimpleNamespace(tensor=np.broadcast_to(tensor, (2, 3, 3)).copy())


def settings(angle=60., enabled=True, coefficient=1.):
    return SimpleNamespace(enabled=enabled, angle_deg=angle, coefficient=coefficient)


def fields(layers, substrate, options=None):
    from snom_calculator._far_field import far_field_factor
    return far_field_factor(FREQUENCY, layers, substrate, options or settings())


@pytest.mark.parametrize("x,y,z", [(4+.2j, 4+.2j, 4+.2j), (4+.2j, 7+.3j, 9+.4j),
                                  (-2+.1j, 3+.1j, 1+.1j)])
def test_bulk_recovers_extraordinary_fresnel(x, y, z):
    phi = np.pi / 3
    kz = np.sqrt(x - x / z * np.sin(phi)**2)
    kz = kz if kz.imag >= 0 else -kz
    rp = (x*np.cos(phi)-kz)/(x*np.cos(phi)+kz)
    np.testing.assert_allclose(fields((), material(x, y, z)), (1+rp)**2*np.sin(phi), rtol=2e-12)


@pytest.mark.parametrize("depths", [(20.,), (10., 40.)])
def test_maxwell_stack_recovers_closed_form_uniaxial_fields(depths):
    from snom_calculator._far_field import reflection_matrix
    axes = [(4+.1j, 9+.2j), (7+.3j, 2+.1j)][:len(depths)]
    films = [(material(x, x, z), d) for (x, z), d in zip(axes, depths)]
    maxwell = reflection_matrix(FREQUENCY, films, material(12+.1j), settings())[:, 0, 0]
    np.testing.assert_allclose(fields(films, material(12+.1j)), (1+maxwell)**2*np.sin(np.pi/3), rtol=2e-12)


def test_zero_and_thick_rotated_film_limits_are_stable():
    film = material(-2+.4j, 5+.2j, 9+.3j, 31, 47)
    substrate = material(12+.1j)
    np.testing.assert_allclose(fields([(film, 0.)], substrate), fields((), substrate), rtol=2e-12)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(fields([(film, 1e9)], substrate), fields((), film), rtol=2e-12)


def test_rotated_stack_preserves_split_layer_and_couples_p_to_s():
    from snom_calculator._far_field import reflection_matrix
    film = material(-2+.3j, 4+.2j, 7+.1j, 38, 29)
    substrate = material(3+.1j, -4+.2j, 8+.3j, 61, 16)
    full = fields([(film, 400.)], substrate)
    split = fields([(film, 150.), (film, 250.)], substrate)
    assert np.isfinite(full).all()
    np.testing.assert_allclose(full, split, rtol=2e-12)
    reflection = reflection_matrix(FREQUENCY, [(film, 400.)], substrate, settings())
    assert np.max(np.abs(reflection[:, 1, 0])) > .01


def test_disabled_and_coefficient_conventions():
    np.testing.assert_array_equal(fields((), material(4), settings(enabled=False)), np.ones(2))
    np.testing.assert_allclose(fields((), material(4), settings(coefficient=0.)), np.sin(np.pi/3))


def test_public_material_and_settings_contract():
    from snom_calculator._far_field import far_field_factor
    sample = MaterialSpectrum(np.full(2, 9+.2j), np.full(2, 4+.1j), "uniaxial")
    expected = fields((), material(4+.1j, 4+.1j, 9+.2j))
    np.testing.assert_allclose(far_field_factor(FREQUENCY, (), sample, FarFieldSettings()), expected)


def test_normal_incidence_rotated_biaxial_reflection_is_analytic():
    from snom_calculator._far_field import reflection_matrix
    angle = np.radians(37.)
    rx = (np.sqrt(3+.1j)-1)/(np.sqrt(3+.1j)+1)
    ry = (np.sqrt(8+.2j)-1)/(np.sqrt(8+.2j)+1)
    expected = np.array([[rx*np.cos(angle)**2+ry*np.sin(angle)**2,
                          (rx-ry)*np.sin(angle)*np.cos(angle)],
                         [(ry-rx)*np.sin(angle)*np.cos(angle),
                          -rx*np.sin(angle)**2-ry*np.cos(angle)**2]])
    actual = reflection_matrix(FREQUENCY, (), material(3+.1j, 8+.2j, 5+.3j, 37.), settings(angle=0.))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2e-12)


def test_tilt_in_incidence_plane_matches_analytic_extraordinary_impedance():
    sample = material(4+.1j, 7+.2j, 9+.3j, tilt=31.)
    e = sample.tensor[0]
    # For xz tilt, q solves ezz*q² + 2*exz*s*q + exx*s²-det(e_xz)=0.
    sine, cosine = np.sin(np.pi/3), np.cos(np.pi/3)
    determinant = e[0, 0]*e[2, 2]-e[0, 2]**2
    q = (-e[0, 2]*sine+np.sqrt(determinant*(e[2, 2]-sine**2)))/e[2, 2]
    impedance = (q*e[2, 2]+sine*e[0, 2])/determinant
    rp = (cosine-impedance)/(cosine+impedance)
    np.testing.assert_allclose(fields((), sample), (1+rp)**2*sine, rtol=2e-12)


@pytest.mark.parametrize("x,z", [(4., .75), (0., 3.)])
def test_exact_critical_or_zero_inplane_film_is_finite(x, z):
    depth = 300.
    kz_sub = np.sqrt(12-.75)
    impedance = kz_sub/12
    k0 = 2*np.pi*FREQUENCY*1e-7
    # Maxwell's triangular p block at q=0 integrates exactly across the film.
    impedance = (impedance-1j*(1-.75/z)*depth*k0)/(1-1j*x*impedance*depth*k0)
    rp = (.5-impedance)/(.5+impedance)
    actual = fields([(material(x, 5., z), depth)], material(12.))
    np.testing.assert_allclose(actual, (1+rp)**2*np.sin(np.pi/3), rtol=2e-12)


@pytest.mark.parametrize("frequency,depth", [(np.array([0., 1000.]), 1.), (FREQUENCY, -1.),
                                             (FREQUENCY, float("nan"))])
def test_invalid_frequency_and_depth_rejected(frequency, depth):
    from snom_calculator._far_field import far_field_factor
    with pytest.raises(ValueError):
        far_field_factor(frequency, [(material(4.), depth)], material(12.), settings())


def test_singular_normal_tensor_is_explicitly_rejected():
    with pytest.raises(ValueError, match="epsilon_zz"):
        fields((), material(3., 4., 0., twist=31.))


def test_closed_form_falls_back_to_maxwell_at_exact_zero_in_plane_permittivity():
    from snom_calculator._far_field import far_field_factor, reflection_matrix
    film = material(0., 0., 3.)
    expected = (1 + reflection_matrix(FREQUENCY, [(film, 300.)], material(12.), settings())[:, 0, 0])**2
    actual = far_field_factor(FREQUENCY, [(film, 300.)], material(12.), settings())
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected * np.sin(np.pi/3), rtol=1e-12)
