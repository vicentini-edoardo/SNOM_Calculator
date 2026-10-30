"""Independent single-point references shared by the engine tests."""

import numpy as np

from snom_calculator._quadrature import legendre_P_all, legendre_Q_all


def integral_table_for_shift(px, x, w, N, a, c, height, image_shift):
    """T[i, j] = int P_i(x) P_j(y(x)) Q_j(eta(x)) dx, i, j = 1..N, for one image point.

    Fixed Gauss-Legendre rule; (eta, y) are the spheroidal coordinates of the
    image ``image_shift`` below the surface for an apex at ``height`` above it.
    """
    s = 2 * a / c + 2 * height / c + image_shift / c - a / c * x
    expr = 1 + s ** 2 + (a ** 2 / c ** 2 - 1) * (1 - x ** 2)
    eta = np.sqrt((expr + np.sqrt(expr ** 2 - 4 * s ** 2)) / 2)
    y = s / eta
    return px @ (legendre_P_all(N, y)[1:N + 1] * legendre_Q_all(N, eta)[1:N + 1] * w).T


def film_profile(frequency, layer, substrate=11.7, depth=20., axes=None, *, radius=25., length=600.,
                 gap=2., amplitude=50., M=8, N=5, quadQ=12, K=10, method="auto",
                 momentum_rtol=1e-5, harmonics=(2, 3), epsilon_tip=-1e10):
    """Spheroid film engine on scalar inputs: profile (heights, F) and demodulated signals.

    ``axes`` = (epsilon_xx, epsilon_zz) makes the film uniaxial; otherwise it is isotropic.
    """
    from snom_calculator import _spheroid_engine
    from snom_calculator._quadrature import demodulation_weights
    from snom_calculator.materials import MaterialSpectrum
    from snom_calculator.models import NumericalSettings, TipParameters

    frequency = np.atleast_1d(np.asarray(frequency, dtype=float))

    def spectrum(value):
        return np.broadcast_to(np.asarray(value, dtype=complex), frequency.shape).copy()

    in_plane, normal = (layer, layer) if axes is None else axes
    film = MaterialSpectrum(spectrum(normal), spectrum(in_plane), "film")
    base = MaterialSpectrum(spectrum(substrate), spectrum(substrate), "substrate")
    tip = TipParameters(radius, length, amplitude, gap, gap, epsilon_tip)
    numerical = NumericalSettings(M=M, N=N, quadQ=quadQ, K=K, momentum_rtol=momentum_rtol)
    values, _ = _spheroid_engine.profile(((film, depth),), base, tip, gap, numerical, method=method)
    return values, demodulation_weights(0 if amplitude == 0 else M, harmonics) @ values

