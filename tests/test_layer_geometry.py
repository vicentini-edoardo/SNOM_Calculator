import numpy as np
from numpy.testing import assert_allclose

from _reference import film_profile
from snom_calculator import _spheroid_engine as engine
from snom_calculator import _uniaxial_esm as esm
from snom_calculator._quadrature import gauss_legendre_rule, legendre_P_all, tapping_heights


def test_factored_coupling_matches_original_contraction():
    a, c, N = 300., np.sqrt(300 * 275), 5
    heights = np.array([2., 11., 52.])
    k = np.array([.003, .019, .06])
    weights = np.array([.1, .3, .4])
    layers = ((4 + .1j, 7 + .2j, 20.),)
    basis = esm._momentum_basis(k, a, c, N)
    kernel = np.exp(-2 * heights[:, None, None, None] * k[None, :, None, None]) * basis[None]
    expected = np.einsum('k,hkij,k->hij', weights, kernel, esm.reflection(k, layers, 11.7))
    factors = np.exp(-2 * heights[:, None] * k) * (weights * esm.reflection(k, layers, 11.7))
    assert_allclose((factors @ basis.reshape(k.size, -1)).reshape(-1, N, N), expected, rtol=1e-13)


def test_direct_couplings_are_streamed(monkeypatch):
    from snom_calculator._fim_coupling import FIMCoupling
    real = FIMCoupling.matrices
    made = []
    solved_sizes = []
    original_solve = engine._solve_coupled

    def counted(*args, **kwargs):
        result = real(*args, **kwargs)
        made.append(result)
        return result

    def counted_solve(couplings, *args):
        solved_sizes.append(couplings.shape[0])
        return original_solve(couplings, *args)

    monkeypatch.setattr(FIMCoupling, 'matrices', counted)
    monkeypatch.setattr(engine, '_solve_coupled', counted_solve)
    monkeypatch.setattr(engine, '_THREADS', 1)  # threads split each chunk
    common = dict(M=8, N=5, quadQ=12, method='fim')
    om = np.array([800., 900., 1000.])
    monkeypatch.setattr(engine, '_FREQUENCY_CHUNK', 1)
    small = film_profile(om, 4 + .2j, **common)
    monkeypatch.setattr(engine, '_FREQUENCY_CHUNK', 3)
    large = film_profile(om, 4 + .2j, **common)
    assert solved_sizes == [1, 1, 1, 3]
    assert len(made) == 6
    for actual, expected in zip(small, large):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_height_tables_interpolate_and_extend_without_extrapolation():
    engine.clear_caches()
    a, radius, quad_q, N = 300., 25., 24, 6
    c = np.sqrt(a * (a - radius))
    tables = engine._height_tables(radius, a, quad_q)
    requested = np.array([2., 3.7, 21., 113.])
    matrices = tables.tables(requested, N)
    x, w = gauss_legendre_rule(quad_q)
    px = legendre_P_all(N, x)[1:]
    exact = engine._surface_tables(px, x, w, N, a, c, requested)
    assert_allclose(matrices, exact, rtol=2e-5, atol=2e-8)
    shifted = engine._integral_tables(px, x, w, N, a, c, np.array([2., 21.]), np.array([40., 184.]))
    assert_allclose(shifted, tables.tables(np.array([22., 113.]), N), rtol=2e-5, atol=2e-8)
    assert tables.bounds[0] <= 2 and tables.bounds[1] >= 113
    assert_allclose(tables.tables(np.array([1.1, 250.]), N),
                    engine._surface_tables(px, x, w, N, a, c, np.array([1.1, 250.])),
                    rtol=2e-5, atol=2e-8)
    assert tables.bounds[0] <= 1.1 and tables.bounds[1] >= 250


def test_nested_geometry_and_incremental_n_k_match_fresh():
    om = np.array([950., 1000.])
    common = dict(frequency=om, layer=4 + .2j, quadQ=20, method='gicm')
    engine.clear_caches()
    film_profile(M=30, N=6, K=3, **common)
    tables = engine._height_tables(25., 300., 20)
    before = tables.stats.copy()
    refined = film_profile(M=60, N=8, K=4, **common)
    assert tables.stats['shared_hits'] > before['shared_hits']
    assert tables.stats['extended_blocks'] > before['extended_blocks']
    engine.clear_caches()
    fresh = film_profile(M=60, N=8, K=4, **common)
    for actual, expected in zip(refined, fresh):
        assert_allclose(actual, expected, rtol=3e-5, atol=1e-16)


def test_height_table_invalidation_bounds_and_cached_arrays():
    engine.clear_caches()
    first = engine._height_tables(25., 300., 16)
    assert first is engine._height_tables(25., 300., 16)
    assert first is not engine._height_tables(26., 300., 16)
    assert first is not engine._height_tables(25., 300., 20)
    first.trajectory(2., 50., 30, (0., 40.), 6)
    assert not first.values.flags.writeable
    assert all(not value.flags.writeable for _, value in first.samples.values())
    for gap in range(3, 30):
        first.trajectory(float(gap), 50., 30, (0., 40.), 6)
    assert len(first.samples) == 1024
    assert engine._height_tables.cache_info().currsize <= 4
    with np.testing.assert_raises(ValueError):
        first.tables(np.array([0.]), 6)
    with np.testing.assert_raises(ValueError):
        first.tables(np.array([3.]), 17)
    engine.clear_caches()
    assert engine._height_tables.cache_info().currsize == 0


def test_resonant_frequency_does_not_change_smooth_frequency():
    common = dict(M=30, N=12, quadQ=24, K=10, method='gicm')
    engine.clear_caches()
    mixed = film_profile(np.array([950., 1000.]), np.array([-1 + .01j, 4 + .2j]), **common)
    alone = film_profile(np.array([1000.]), 4 + .2j, **common)
    for actual, expected in zip(mixed, alone):
        assert_allclose(actual[:, 1], expected[:, 0], rtol=1e-10, atol=1e-16)


def test_node_contraction_matches_interpolated_image_tables():
    engine.clear_caches()
    tables = engine._height_tables(25., 300., 24)
    heights = tapping_heights(2., 50., 10)
    shifts = (40., 80., 120.)
    weights = tables.image_weights(heights, shifts, 6)
    nodes = tables.values[:, :6, :6].reshape(len(tables.nodes), -1)
    contracted = np.einsum('khn,nq->khq', weights, nodes)
    requested = np.concatenate([heights + shift / 2 for shift in shifts])
    expected = tables.tables(requested, 6).reshape(len(shifts), heights.size, -1)
    assert_allclose(contracted, expected, rtol=1e-12, atol=1e-14 * abs(expected).max())


def test_height_block_and_exact_mask_do_not_change_result(monkeypatch):
    common = dict(frequency=np.array([950., 1000., 1050.]), layer=np.array([4 + .2j, -1 + .05j, 3 + .1j]),
                  M=12, N=8, quadQ=20, K=6, method='gicm')
    results = []
    for block in (1, 5):
        monkeypatch.setattr(engine, '_HEIGHT_BLOCK', block)
        engine.clear_caches()
        results.append(film_profile(**common)[1])
    # every frequency exact vs every frequency interpolated
    original = engine._image_charge_profile
    for forced in (True, False):
        def solve(*args, forced=forced):
            args = list(args)
            args[-1] = np.full_like(args[-1], forced)
            return original(*args)
        monkeypatch.setattr(engine, '_image_charge_profile', solve)
        engine.clear_caches()
        results.append(film_profile(**common)[1])
    for other in results[1:]:
        assert_allclose(other, results[0], rtol=1e-8, atol=1e-16)
