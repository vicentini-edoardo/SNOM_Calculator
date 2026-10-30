"""Exact electrostatic response and spheroidal momentum coupling."""

import numpy as np
from numpy.testing import assert_allclose

from _reference import film_profile, integral_table_for_shift
from snom_calculator._quadrature import gauss_legendre_rule, legendre_P_all
from snom_calculator._uniaxial_esm import reflection, coupling_matrices, _gamma, _scaled_spherical_i, _gauss_kronrod_rule
import pytest
from scipy.special import ive, j0


def test_reflection_limits_and_isotropic_negative_film():
    k = np.array([0.0, 0.02, 0.1])
    film, substrate, d = -3 + 0.2j, 11.7, 20.0
    a = (1 - film) / (1 + film)
    b = (film - substrate) / (film + substrate)
    expected = (a + b * np.exp(-2 * k * d)) / (1 + a * b * np.exp(-2 * k * d))
    assert_allclose(reflection(k, ((film, film, d),), substrate), expected)
    assert_allclose(reflection(k, ((film, film, 0),), substrate), (1 - substrate) / (1 + substrate))
    assert_allclose(reflection(k, ((film, film, 30), (film, film, 40)), substrate),
                    reflection(k, ((film, film, 70),), substrate))
    assert_allclose(reflection(k, ((film, film, 1e5),), substrate)[1:], a, rtol=1e-12)


def test_uniaxial_film_uses_gamma_in_propagation():
    k = np.array([0.02, 0.08])
    values = [reflection(k, ((ep, ez, 20),), 11.7)
              for ep, ez in ((9, 4), (4, 9))]
    assert np.max(abs(values[0] - values[1])) > 0.01


def test_constant_reflection_momentum_integral_matches_image_table():
    a, radius, h, N = 300.0, 25.0, 2.0, 5
    c = np.sqrt(a * (a - radius))
    x, w = gauss_legendre_rule(32)
    px = legendre_P_all(N, x)[1:]
    expected = integral_table_for_shift(px, x, w, N, a, c, h, 0)
    actual = coupling_matrices(np.array([h]), ((4 + .1j, 4 + .1j, 20),), 4 + .1j,
                               a=a, c=c, N=N, rtol=2e-5)[0]
    assert_allclose(actual, (1 - (4 + .1j)) / (1 + (4 + .1j)) * expected,
                    rtol=3e-4, atol=1e-7)


def test_direct_isotropic_signal_matches_gicm_and_tensor_changes_film_signal():
    common = dict(amplitude=60, M=8, N=6, quadQ=32, K=16, harmonics=(2, 3, 4))
    om = np.array([800., 1500.])
    # Resolve the image-table quadrature independently of the exact direct kernel.
    scalar_profile, scalar = film_profile(om, 4 + .2j, method="gicm", **common)
    direct_profile, direct = film_profile(om, 4 + .2j, method="fim", **common)
    assert_allclose(direct, scalar, rtol=2e-3)
    assert_allclose(np.angle(direct), np.angle(scalar), atol=2e-3)
    assert_allclose(direct_profile, scalar_profile, rtol=2e-3, atol=1e-4)
    anisotropic = film_profile(om, None, axes=(9 + .3j, 4 + .1j), method="fim", **common)[1]
    assert np.max(abs(anisotropic[1] - direct[1])) > 1e-15


def test_lossless_hyperbolic_film_retains_large_k_oscillations():
    k = np.array([1.0, 1.04, 1.08])
    values = reflection(k, ((-3 + 0j, 2 + 0j, 20),), 2.25)
    assert np.max(abs(np.diff(values))) > 0.01


def test_identical_film_and_substrate_is_homogeneous_halfspace():
    k = np.linspace(0, 1, 30)
    film = 4 + .3j
    assert_allclose(reflection(k, ((film, film, 5),), film),
                    (1 - film) / (1 + film), rtol=1e-13)


def hbn(frequency, damping=1):
    def axis(e_inf, nu_to, nu_lo, gamma):
        return e_inf * (1 + (nu_lo**2 - nu_to**2) /
                        (nu_to**2 - frequency**2 - 1j * damping * gamma * frequency))
    return axis(4.87, 1370, 1610, 5), axis(2.95, 780, 830, 4)


def surface_quadrature_basis(momentum, a, c, N, quad_q=192):
    """Independent surface projection with SciPy Bessel values."""
    k = np.asarray(momentum, dtype=float)
    x, w = gauss_legendre_rule(quad_q)
    px = np.polynomial.legendre.legvander(x, N)[:, 1:].T * w
    radius = np.sqrt(a*a - c*c) * np.sqrt(1 - x*x)
    projected = (j0(k[:, None] * radius)
                 * np.exp(-k[:, None] * (2*a - a*x - c))) @ px.T
    scaled = np.zeros((k.size, N))
    positive = k > 0
    t = k[positive, None] * c
    scaled[positive] = np.sqrt(np.pi / (2*t)) * ive(np.arange(1, N+1) + .5, t)
    return c * projected[:, :, None] * scaled[:, None, :]


@pytest.mark.parametrize("radius", [1., 25., 150.])
def test_momentum_basis_matches_surface_integral_without_surface_discretization(radius):
    from snom_calculator._uniaxial_esm import _momentum_basis
    a, N = 300., 16
    c = np.sqrt(a * (a - radius))
    k = np.r_[0., np.geomspace(1e-6, 3., 60)]
    expected = surface_quadrature_basis(k, a, c, N)
    # Surface order no longer limits the direct momentum basis.
    actual = _momentum_basis(k, a, c, N)
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-12 * abs(expected).max())


@pytest.mark.parametrize("frequency", [780.1, 800., 829.9, 1370.1, 1500., 1609.9])
def test_analytic_hbn_signal_matches_surface_quadrature_near_resonances(frequency, monkeypatch):
    from snom_calculator import _uniaxial_esm as esm
    perpendicular, parallel = hbn(frequency, damping=.05)
    common = dict(amplitude=60, M=12, N=12, quadQ=32, axes=(perpendicular, parallel),
                  harmonics=(2, 3, 4), momentum_rtol=1e-8)
    actual = film_profile([frequency], None, **common)[1]
    with monkeypatch.context() as context:
        context.setattr(esm, '_momentum_basis', lambda k, a, c, N: surface_quadrature_basis(k, a, c, N))
        expected = film_profile([frequency], None, **common)[1]
    # Same bulk reference for both, so signal agreement is contrast agreement.
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-12 * abs(expected).max())


@pytest.mark.parametrize("frequency", [700, 780.5, 800, 829.5, 1000, 1370.5, 1450, 1500, 1609.5, 1700])
@pytest.mark.parametrize("depth", [5, 10, 20, 50, 100])
@pytest.mark.parametrize("substrate", [2.25, 11.7])
def test_hbn_reflection_grid_is_finite_and_passive(frequency, depth, substrate):
    perpendicular, parallel = hbn(frequency)
    assert perpendicular.imag > 0 and parallel.imag > 0
    gamma = _gamma(perpendicular, parallel)
    assert gamma.real > 0
    k = np.linspace(0, 1, 251)
    values = reflection(k, ((perpendicular, parallel, depth),), substrate)
    assert np.isfinite(values).all()
    assert np.max(abs(values)) < 1e7


@pytest.mark.parametrize("frequency", [800, 1000, 1500])
def test_hbn_coupling_quadrature(frequency):
    perpendicular, parallel = hbn(frequency, damping=.2)
    layers = ((perpendicular, parallel, 20),)
    a = 300.
    c = np.sqrt(a * (a - 25))
    heights = np.array([2., 7., 26., 63., 122.])
    kwargs = dict(a=a, c=c, N=5)
    reference = coupling_matrices(heights, layers, 11.7, rtol=2e-6, **kwargs)
    coarse = coupling_matrices(heights, layers, 11.7, rtol=2e-5, **kwargs)
    assert_allclose(coarse, reference, rtol=2e-3, atol=1e-6)


def test_scaled_spherical_bessel_matches_scipy():
    x = np.r_[0.0, np.geomspace(1e-8, 2e3, 400)]
    N = 80
    got = _scaled_spherical_i(x, N)
    positive = x[1:, None]
    expected = np.sqrt(np.pi / (2 * positive)) * ive(np.arange(1, N + 1) + .5, positive)
    assert np.all(got[0] == 0)
    assert_allclose(got[1:], expected, rtol=1e-12, atol=1e-300)


def test_gauss_kronrod_rule_matches_quadpack_and_is_exact():
    x, kronrod, gauss, weights = _gauss_kronrod_rule()
    # QUADPACK qk31 outermost node and weights.
    assert_allclose([x[-1], kronrod[-1], weights[-1]],
                    [0.998002298693397, 0.005377479872923349, 0.030753241996117268], rtol=1e-12)
    for power in range(47):
        exact = 2 / (power + 1) if power % 2 == 0 else 0
        assert abs(kronrod @ x**power - exact) < 1e-14
        if power < 30:
            assert abs(weights @ x[gauss]**power - exact) < 1e-14
