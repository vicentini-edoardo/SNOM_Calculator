"""FIM (rotated-contour skeleton) couplings against the real-axis integrator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from snom_calculator import _fim_coupling as cc
from snom_calculator import _uniaxial_esm as esm
from _reference import film_profile
from snom_calculator._quadrature import demodulation_weights, tapping_heights
from snom_calculator._spheroid_engine import _solve_coupled, _tip

R, A_TIP, HMIN, AMP, M, N, QUAD = 25.0, 300.0, 2.0, 40.0, 8, 10, 16
C = float(np.sqrt(A_TIP * (A_TIP - R)))
HEIGHTS = tapping_heights(HMIN, AMP, M)


def hbn(f, damping=1.0):
    def axis(e_inf, to, lo, g):
        return e_inf * (1 + (lo**2 - to**2) / (to**2 - f**2 - 1j * damping * g * f))
    return axis(4.9, 1360, 1614, 7), axis(2.95, 760, 825, 3)


def film(f, d, damping=1.0):
    return (*hbn(f, damping), d)


def signal(W):
    profile = _solve_coupled(W[None], _tip(R, 2 * A_TIP, -1e10 + 0j, N))[0]
    return demodulation_weights(M, (2, 3, 4)) @ profile


def solver():
    return cc.FIMCoupling(HEIGHTS, A_TIP, C, N)


STACKS = {
    "hBN 20 type I": (790.0, (film(790.0, 20),), 11.7),
    "hBN 300 type II": (1480.0, (film(1480.0, 300),), 11.7),
    "hBN 20 between bands": (1000.0, (film(1000.0, 20),), 11.7),
    "hBN/SiO2/hBN": (1420.0, (film(1420.0, 10), (2.0 + .1j, 2.0 + .1j, 20.0), film(1420.0, 10)), 11.7),
    "PMMA/hBN on Au": (790.0, ((2.2 + .02j, 2.2 + .02j, 10.0), film(790.0, 20)), -3000 + 900j),
}


@pytest.mark.parametrize("name", STACKS)
def test_fim_matches_tight_real_axis_reference(name):
    _, layers, substrate = STACKS[name]
    fim = solver()
    W = fim.matrices(layers, substrate)
    assert "fallback" not in fim.stats
    reference = esm.coupling_matrices(HEIGHTS, layers, substrate, a=A_TIP, c=C, N=N,
                                      rtol=1e-10)
    assert np.max(abs(W - reference)) <= 1e-9 * np.max(abs(reference))
    assert_allclose(signal(W), signal(reference), rtol=1e-8)


def test_vacuum_layer_equals_raised_tip():
    """r of [air t] + S is exp(-2kt) r_S, so W at H equals W of S at H + t."""
    f = 1480.0
    stack = (film(f, 10), (2.0 + .1j, 2.0 + .1j, 20.0), film(f, 10))
    lifted = cc.FIMCoupling(HEIGHTS + 5.0, A_TIP, C, N).matrices(stack, 11.7)
    covered = solver().matrices(((1 + 0j, 1 + 0j, 5.0),) + stack, 11.7)
    assert np.max(abs(covered - lifted)) <= 1e-12 * np.max(abs(lifted))


def test_pole_count_matches_single_film_formula():
    f, d, substrate = 1480.0, 50.0, 11.7
    parts = cc._stack_parts((film(f, d),), substrate)
    gamma, (a, b), _ = parts
    n = np.arange(-400, 401)
    poles = (np.log(-a * b) - 2j * np.pi * n) / (2 * gamma[0] * d)
    for sign in (1, -1):
        for angle in (5.0, 25.0):
            phi, k_radius = np.deg2rad(angle), 2.0
            inside = np.sum((abs(poles) < k_radius) & (sign * np.angle(poles) > 0)
                            & (sign * np.angle(poles) < phi))
            assert cc._sector_pole_count(parts, sign, phi, k_radius) == inside
    assert cc._sector_pole_count(parts, 1, np.deg2rad(25.0), 2.0) > 0


@pytest.mark.parametrize("f", [790.0, 1000.0, 1480.0])
@pytest.mark.parametrize("d", [2.0, 20.0, 300.0])
def test_film_pole_count_matches_argument_principle(f, d):
    for substrate in (11.7, 1.5 + .01j, -3000 + 900j):
        parts = cc._stack_parts((film(f, d),), substrate)
        for sign in (1, -1):
            for angle in (5.0, 15.0, 25.0):
                phi = np.deg2rad(angle)
                assert (cc._film_pole_count(parts, sign, phi, 2.5)
                        == cc._sector_pole_count(parts, sign, phi, 2.5))


def test_film_pole_count_defers_on_axis_poles_and_replaces_phase_tracking(monkeypatch):
    lossless = cc._stack_parts((film(1480.0, 20, damping=0.0),), 11.7)
    assert cc._film_pole_count(lossless, -1, np.deg2rad(25.0), 2.0) is None
    calls = []
    original = cc._sector_pole_count
    monkeypatch.setattr(cc, "_sector_pole_count", lambda *args: calls.append(1) or original(*args))
    fim = solver()
    fim.matrices((film(1480.0, 20),), 11.7)
    assert calls == []
    fim.matrices((film(1480.0, 10), (2.0 + .1j, 2.0 + .1j, 20.0)), 11.7)
    assert calls


def _graphene_on_hbn():
    """Graphene plasmons (poles above) on hBN type-I phonons (poles below)."""
    f = 800.0
    w = 2 * np.pi * 2.99792458e10 * f
    sigma = (1.602176634e-19**2 * 0.4 * 1.602176634e-19 / (np.pi * 1.054571817e-34**2)
             * 1j / (w + 1e13))
    graphene = (1 + 1j * sigma / (8.8541878128e-12 * w * 0.34e-9), 1 + 0j, 0.34)
    return (graphene, film(f, 20), (2.0 + .1j, 2.0 + .1j, 285.0))


def test_mixed_forward_and_backward_modes_use_real_axis_skeleton():
    """No rotated ray is pole free; the fixed real-axis skeleton replaces the fallback."""
    layers = _graphene_on_hbn()
    fim = solver()
    W = fim.matrices(layers, 11.7, rtol=1e-7)
    assert fim.stats == {"real axis": 1}
    reference = esm.coupling_matrices(HEIGHTS, layers, 11.7, a=A_TIP, c=C, N=N,
                                      rtol=1e-10)
    assert np.max(abs(W - reference)) <= 1e-9 * np.max(abs(reference))


def test_adaptive_fallback_when_no_skeleton_passes(monkeypatch):
    calls = []
    original = esm.coupling_matrices

    def spy(*args, **kwargs):
        calls.append(kwargs["rtol"])
        return original(*args, **kwargs)

    monkeypatch.setattr(esm, "coupling_matrices", spy)
    fim = solver()
    rays_only = fim._candidates
    monkeypatch.setattr(fim, "_candidates", lambda parts: rays_only(parts)[:-1])
    W = fim.matrices(_graphene_on_hbn(), 11.7, rtol=1e-7)
    assert fim.stats == {"fallback": 1}
    assert calls == [1e-7]
    assert np.isfinite(W).all()


def test_lossless_film_uses_limiting_absorption():
    f = 1480.0
    lossless = solver().matrices((film(f, 20, damping=0.0),), 11.7)
    lossy = solver().matrices((film(f, 20, damping=1e-3),), 11.7)
    assert np.isfinite(lossless).all()
    difference = np.max(abs(signal(lossless) - signal(lossy)) / abs(signal(lossy)))
    assert difference < 1e-2


def test_film_fim_matches_real_axis_and_validates_method():
    om = np.array([790., 1000., 1480.])
    common = dict(axes=hbn(om), radius=R, length=2 * A_TIP, gap=HMIN, amplitude=AMP,
                  M=M, N=N, quadQ=QUAD, harmonics=(2, 3))
    fim = film_profile(om, None, method="fim", **common)
    adaptive = film_profile(om, None, method="real-axis", momentum_rtol=1e-10, **common)
    for actual, expected in zip(fim, adaptive):
        assert_allclose(actual, expected, rtol=1e-8)
    with pytest.raises(ValueError, match="methods"):
        film_profile(om, None, method="residue", **common)


def test_real_axis_method_skips_the_fim(monkeypatch):
    used = []
    monkeypatch.setattr(cc.FIMCoupling, "matrices", lambda *args, **kwargs: used.append(1))
    film_profile(np.array([1000.]), 4 + .2j, M=M, N=N, quadQ=QUAD, harmonics=(2,), method="real-axis")
    assert used == []
