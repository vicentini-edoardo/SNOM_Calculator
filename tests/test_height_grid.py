"""Log-height interpolated tapping grid and per-height multipole orders."""

from dataclasses import replace

import numpy as np
import pytest

from snom_calculator import _quadrature, _spheroid_engine
from snom_calculator._spectra import calculate_spectra, solve_profile
from snom_calculator.materials import constant_material
from snom_calculator.models import NumericalSettings, TipParameters

FREQUENCY = np.array([900., 910., 920.])
TIP = TipParameters(25., 600., 40., 1., 1., -1e10 + 0j)


def _contrast(layers, sample, numerical, tip=TIP, checked=False):
    reference = constant_material(-2000 + 500j, FREQUENCY, "metal")
    results = calculate_spectra(FREQUENCY, layers, sample, reference, tip, numerical, (2, 3), checked=checked)
    return np.array([results[h].contrast for h in (2, 3)]), results


def test_interpolation_reproduces_polynomials_in_log_height():
    gap, amplitude = 0.5, 40.
    heights = _quadrature.tapping_heights(gap, amplitude, 200)
    nodes = _quadrature.height_nodes(gap, amplitude, 33)
    x = lambda h: _quadrature._log_position(gap, amplitude, h)  # noqa: E731
    values = x(nodes) ** 20 - 3 * x(nodes) ** 7
    expected = x(heights) ** 20 - 3 * x(heights) ** 7
    np.testing.assert_allclose(_quadrature.interpolation_matrix(gap, amplitude, heights, 33) @ values, expected,
                               atol=1e-12)
    assert nodes[0] == gap and nodes[-1] == gap + 2 * amplitude and np.all(np.diff(nodes) > 0)


@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("M", [120, 240])
def test_interpolated_grid_matches_every_tapping_height(film, M, monkeypatch):
    sample = constant_material(-1.2 + 0.3j, FREQUENCY, "resonant")
    layers = ((sample, 10.),) if film else ()
    substrate = constant_material(11.7 + 0.01j, FREQUENCY, "Si") if film else sample
    numerical = NumericalSettings(M=M, N=24, quadQ=48, K=6)
    nodes, results = _contrast(layers, substrate, numerical)
    assert results[2].metadata["height_nodes"] in (33, 65, 129)
    direct, _ = _contrast(layers, substrate, replace(numerical, height_nodes=0))
    # The pointwise interpolation error is about twice the reported Chebyshev tail.
    np.testing.assert_allclose(nodes, direct, rtol=5 * max(results[2].metadata["height_interpolation_tail"], 1e-12))


def test_nodes_double_while_the_chebyshev_tail_is_large():
    # A resonant film on Si 0.5 nm below the tip needs 65 nodes (33 leave a 3e-5 harmonic error).
    sample, substrate = (constant_material(-1.2 + 0.3j, FREQUENCY, "resonant"),
                         constant_material(11.7 + 0.01j, FREQUENCY, "Si"))
    tip = TipParameters(25., 600., 40., 0.5, 0.5, -1e10 + 0j)
    profile, info = solve_profile(((sample, 10.),), substrate, tip, 0.5, NumericalSettings(M=240, N=24, quadQ=48, K=8),
                                  (2,), "auto", 1)
    assert profile.shape[0] == info["height_nodes"] >= 65
    assert info["height_interpolation_tail"] <= _quadrature.HEIGHT_TAIL_LIMIT


def test_profiles_not_smooth_in_log_height_fall_back_to_every_tapping_height():
    # A near-resonant sample 0.2 nm below a tip tapping 100 nm leaves a Chebyshev tail of ~0.1.
    sample = constant_material(-1.05 + 0.02j, FREQUENCY, "resonant")
    tip = TipParameters(25., 600., 100., 0.2, 0.2, -1e10 + 0j)
    numerical = NumericalSettings(M=120, N=24, quadQ=48)
    profile, info = solve_profile((), sample, tip, 0.2, numerical, (2,), "auto", 1)
    assert info["height_nodes"] == 0 and info["height_tapping_fallback"] == 1
    assert info["height_interpolation_tail"] > _quadrature.HEIGHT_TAIL_LIMIT
    direct, _ = solve_profile((), sample, tip, 0.2, replace(numerical, height_nodes=0), (2,), "auto", 1)
    np.testing.assert_array_equal(profile, direct)


def test_gaps_whose_shared_span_is_not_smooth_are_solved_one_by_one():
    sample = constant_material(-1.05 + 0.02j, FREQUENCY, "resonant")
    tip = TipParameters(25., 600., 100., 0.2, 0.2, -1e10 + 0j)
    numerical = NumericalSettings(M=120, N=24, quadQ=48)
    gaps = (0.2, 1., 2.)
    averaged = calculate_spectra(FREQUENCY, (), sample, None, tip, numerical, (2, 3), checked=True, sample_gaps=gaps)
    runs = [calculate_spectra(FREQUENCY, (), sample, None, replace(tip, sample_gap_nm=gap), numerical, (2, 3),
                              checked=True) for gap in gaps]
    assert averaged[2].metadata["height_tapping_fallback"] == 1
    for h in (2, 3):
        np.testing.assert_allclose(averaged[h].sample_harmonic,
                                   np.mean([run[h].sample_harmonic for run in runs], axis=0), rtol=1e-12)
        assert averaged[h].metadata["N_check_max_error"] == pytest.approx(
            max(run[h].metadata["N_check_max_error"] for run in runs), rel=1e-12)


def test_interpolated_grid_checks_m_without_solving_and_reuses_profiles():
    sample = constant_material(-1.2 + 0.3j, FREQUENCY, "resonant")
    numerical = NumericalSettings(M=120, N=24, quadQ=48)
    _, results = _contrast((), sample, numerical, checked=True)
    assert np.isfinite(results[2].metadata["M_check_max_error"])
    profile, _ = solve_profile((), sample, TIP, 1., numerical, (2,), "auto", 1)
    assert profile.shape[0] in (33, 65, 129)
    doubled = NumericalSettings(M=240, N=24, quadQ=48)
    reused, _ = solve_profile((), sample, TIP, 1., doubled, (2,), "auto", 1, previous=profile)
    assert reused is profile


def test_height_orders_keep_n_at_contact_and_floor_above():
    heights = _quadrature.height_nodes(0.5, 40., 33)
    orders = _spheroid_engine.height_orders(heights, 0.5, 120)
    assert orders[0] == 120 and orders.min() >= 32 and np.all(np.diff(orders) <= 0)
    assert np.all(_spheroid_engine.height_orders(heights, 0.5, 40) == 40)


@pytest.mark.parametrize("film", [False, True])
def test_adaptive_orders_stay_within_the_uniform_n_error(film, monkeypatch):
    sample = constant_material(-1.2 + 0.3j, FREQUENCY, "resonant")
    layers = ((sample, 10.),) if film else ()
    substrate = constant_material(11.7 + 0.01j, FREQUENCY, "Si") if film else sample
    tip = TipParameters(25., 600., 40., 0.5, 0.5, -1e10 + 0j)
    numerical = NumericalSettings(M=60, N=80, quadQ=112, K=6)
    adaptive, _ = _contrast(layers, substrate, numerical, tip)
    monkeypatch.setattr(_spheroid_engine, "ADAPTIVE_N", False)
    uniform, _ = _contrast(layers, substrate, numerical, tip)
    finer, _ = _contrast(layers, substrate, NumericalSettings(M=60, N=120, quadQ=160, K=6), tip)
    scale = np.max(abs(finer), axis=1, keepdims=True)
    assert np.max(abs(adaptive - uniform) / scale) <= np.max(abs(uniform - finer) / scale)
