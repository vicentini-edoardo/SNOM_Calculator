"""Independent Legendre-Q accuracy and work-count checks."""

from math import factorial

import numpy as np
import pytest
from scipy.special import lqmn

import snom_calculator._quadrature as quadrature


@pytest.mark.parametrize("n", [2, 40, 240])
def test_legendre_q_matches_scipy_with_one_continued_fraction(monkeypatch, n):
    x = np.array([1.0001, 1.044, 1.5, 2.])
    # Extra degrees converge SciPy's downward recurrence near x = 1;
    # requesting only n = 2 loses about 1e-9 relative accuracy there.
    expected = lqmn(0, max(n, 64), x)[0][0, :n + 1]
    original = quadrature._continued_fraction_Hn
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(quadrature, "_continued_fraction_Hn", counted)
    actual = quadrature.legendre_Q_all(n, x)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=1e-300)
    assert len(calls) == 1


def test_legendre_q_retains_large_distance_accuracy():
    # At these distances the leading asymptotic term's relative correction
    # is below double precision. Direct log((x+1)/(x-1)) loses digits here.
    x = np.array([1e12, 1e16])
    degrees = np.arange(9)
    leading = np.array([2**n * factorial(n)**2 / factorial(2*n + 1) for n in degrees])
    expected = leading[:, None] / x[None, :] ** (degrees[:, None] + 1)
    np.testing.assert_allclose(quadrature.legendre_Q_all(8, x), expected, rtol=5e-14, atol=0)


def test_legendre_q_warns_when_the_continued_fraction_does_not_converge():
    with pytest.warns(UserWarning, match="did not converge"):
        quadrature.legendre_Q_all(2, np.array([1 + 1e-12]))
