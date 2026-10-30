"""Far-field drive and collection factor (1 + coefficient r_pp)^2 sin(theta).

Every engine multiplies its demodulated near-field signal by this factor.
Surface-normal uniaxial stacks use the closed-form Fresnel recursion; other
tensors use passive dielectric Maxwell scattering for p-polarized light.

With exp(-i omega t), z into the sample and k/k0=(sin(theta),0,q),
Maxwell's equations are k x E = H and k x H = -epsilon E (H in
vacuum-impedance units). Eliminating Ez, Hz gives q W = B W for
W=(Ex,Ey,Hx,Hy). Tangential continuity and decaying propagation factors
give a reflection recursion without exponentially growing transfer matrices.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .materials import MaterialSpectrum
from .models import FarFieldSettings


def _modes(e: np.ndarray, sine: float):
    if abs(e[2, 2]) <= np.finfo(float).eps * np.max(np.abs(e)):
        raise ValueError("Tensor far field requires nonzero laboratory epsilon_zz.")
    zx, zy = e[2, :2] / e[2, 2]
    matrix = np.array([
        [-sine*zx, -sine*zy, 0, 1-sine**2/e[2, 2]],
        [0, 0, -1, 0],
        [-e[1, 0]+e[1, 2]*zx, sine**2-e[1, 1]+e[1, 2]*zy, 0, sine*e[1, 2]/e[2, 2]],
        [e[0, 0]-e[0, 2]*zx, e[0, 1]-e[0, 2]*zy, 0, -sine*e[0, 2]/e[2, 2]],
    ], dtype=complex)
    q, vectors = np.linalg.eig(matrix)
    flux = np.real(vectors[0]*vectors[3].conj()-vectors[1]*vectors[2].conj())
    tolerance = 1e-10 * max(1., float(np.max(np.abs(q))))
    forward = np.where(np.abs(q.imag) > tolerance, q.imag > 0, flux > 0)
    if np.count_nonzero(forward) != 2 or np.linalg.cond(vectors) > 1e12:
        raise ValueError("Cannot resolve two outgoing Maxwell modes; singular or degenerate tensor.")
    return q[forward], vectors[:, forward], q[~forward], vectors[:, ~forward]


def _diagonal_reflection(tensors, depths, k0, sine, cosine):
    """Impedance recursion also handles a film exactly at its critical angle."""
    def roots(e):
        x, y, z = np.diag(e)
        if z == 0:
            raise ValueError("Tensor far field requires nonzero laboratory epsilon_zz.")
        q = np.sqrt(np.array([x-x/z*sine**2, y-sine**2], dtype=complex))
        return np.where(q.imag < 0, -q, q), x

    q, x = roots(tensors[-1])
    # Homogeneous impedance/admittance pairs keep exact ENZ and poles finite.
    p_num, p_den = (q[0], x) if x != 0 else (1+0j, 0j)
    s_num, s_den = q[1], 1+0j
    for e, depth in zip(tensors[-2::-1], depths[::-1]):
        q, x = roots(e)
        tangent = np.tan(q*depth*k0)
        tangent_over_q = np.divide(tangent, q, out=np.full(2, depth*k0, dtype=complex), where=q != 0)
        p_num, p_den = (p_num-1j*(1-sine**2/e[2, 2])*tangent_over_q[0]*p_den,
                        p_den-1j*x*tangent_over_q[0]*p_num)
        s_num, s_den = (s_num-1j*q[1]*tangent[1]*s_den,
                        s_den-1j*tangent_over_q[1]*s_num)
        p_scale, s_scale = max(abs(p_num), abs(p_den)), max(abs(s_num), abs(s_den))
        p_num, p_den = p_num/p_scale, p_den/p_scale
        s_num, s_den = s_num/s_scale, s_den/s_scale
    return np.diag([(cosine*p_den-p_num)/(cosine*p_den+p_num),
                    (cosine*s_den-s_num)/(cosine*s_den+s_num)])


def _validate(frequency, layers, settings):
    """Checked frequencies and the layers of nonzero depth (zero depth has no effect)."""
    frequency = np.asarray(frequency, dtype=float)
    if frequency.ndim != 1 or not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("Far-field frequencies must be finite, positive and one-dimensional.")
    angle = settings.angle_deg
    if not np.isfinite([angle, settings.coefficient]).all() or not 0 <= angle <= 90:
        raise ValueError("Far-field angle and coefficient must be finite; incidence must lie in [0,90].")
    depths = np.asarray([depth for _, depth in layers], dtype=float)
    if not np.isfinite(depths).all() or np.any(depths < 0):
        raise ValueError("Film depths must be finite and nonnegative.")
    return frequency, [(material, depth) for material, depth in layers if depth > 0]


def reflection_matrix(
    frequency: np.ndarray,
    layers: Sequence[tuple[MaterialSpectrum, float]],
    substrate: MaterialSpectrum,
    settings: FarFieldSettings,
) -> np.ndarray:
    """Return (F,2,2) reflected (p,s) amplitudes for incident (p,s).

    Dielectric tensors are symmetric and passive. Exact epsilon_zz=0 and
    unresolved defective Maxwell eigenspaces raise rather than being regularized.
    The diagonal-film critical-angle limit is evaluated analytically.
    """
    frequency, retained = _validate(frequency, layers, settings)
    angle = settings.angle_deg
    depths = np.array([depth for _, depth in retained])
    tensors = []
    for material in [*(material for material, _ in retained), substrate]:
        tensor = np.asarray(material.tensor, dtype=complex)
        if tensor.shape != (frequency.size, 3, 3) or not np.isfinite(tensor).all():
            raise ValueError("Material tensors must be finite (frequency,3,3) arrays.")
        tensors.append(tensor)
    sine, cosine = np.sin(np.radians(angle)), max(np.cos(np.radians(angle)), 1e-12)
    air_plus = np.array([[cosine, 0], [0, 1], [0, -cosine], [1, 0]], dtype=complex)
    air_minus = np.array([[-cosine, 0], [0, 1], [0, cosine], [1, 0]], dtype=complex)
    result = np.empty((frequency.size, 2, 2), dtype=complex)
    for index, value in enumerate(frequency):
        media = [tensor[index] for tensor in tensors]
        k0 = 2*np.pi*value*1e-7  # cm^-1 to nm^-1
        if all(np.max(np.abs(e-np.diag(np.diag(e)))) < 1e-14*np.max(np.abs(e)) for e in media):
            result[index] = _diagonal_reflection(media, depths, k0, sine, cosine)
            continue
        modes = [_modes(e, sine) for e in media]
        reflection = np.zeros((2, 2), dtype=complex)
        for j in range(len(media)-1, -1, -1):
            _, plus, _, minus = modes[j]
            outgoing = plus + minus @ reflection
            if j:
                qp, above_plus, qm, above_minus = modes[j-1]
            else:
                above_plus, above_minus = air_plus, air_minus
            reflection = np.linalg.solve(np.column_stack((above_minus, -outgoing)), -above_plus)[:2]
            if j:
                reflection = (np.exp(-1j*qm*depths[j-1]*k0)[:, None]*reflection
                              *np.exp(1j*qp*depths[j-1]*k0)[None, :])
        result[index] = reflection
    return result


def _uniaxial_rpp(frequency, tensors, depths, sine, cosine):
    """Closed-form r_pp of air / surface-normal uniaxial films / substrate."""
    in_plane = [np.ones(frequency.size, dtype=complex)] + [t[:, 0, 0] for t in tensors]
    kz = [np.full(frequency.size, cosine, dtype=complex)]
    for tensor in tensors:
        q = np.emath.sqrt(tensor[:, 0, 0] - sine**2 * tensor[:, 0, 0] / tensor[:, 2, 2])
        kz.append(np.where(np.imag(q) < 0, -q, q))
    rp = [(in_plane[j + 1] * kz[j] - in_plane[j] * kz[j + 1])
          / (in_plane[j + 1] * kz[j] + in_plane[j] * kz[j + 1]) for j in range(len(tensors))]
    reflection = rp[-1]
    k0 = 2 * np.pi * frequency * 1e-7  # cm^-1 to nm^-1
    for index in range(len(depths) - 1, -1, -1):
        phase = np.exp(2j * kz[index + 1] * depths[index] * k0)
        reflection = (rp[index] + reflection * phase) / (1 + rp[index] * reflection * phase)
    return reflection


def far_field_factor(
    frequency: np.ndarray,
    layers: Sequence[tuple[MaterialSpectrum, float]],
    substrate: MaterialSpectrum,
    settings: FarFieldSettings,
) -> np.ndarray:
    """(1 + coefficient r_pp)^2 sin(theta) per frequency; ones when disabled."""
    if not settings.enabled:
        return np.ones(np.asarray(frequency).size, dtype=complex)
    frequency, layers = _validate(frequency, layers, settings)
    tensors = [np.asarray(material.tensor, dtype=complex)
               for material in (*(material for material, _ in layers), substrate)]
    if all(np.count_nonzero(t - t * np.eye(3)) == 0 and np.array_equal(t[:, 0, 0], t[:, 1, 1])
           for t in tensors):
        theta = np.radians(settings.angle_deg)
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = _uniaxial_rpp(frequency, tensors, [float(depth) for _, depth in layers],
                               np.sin(theta), max(np.cos(theta), 1e-12))
        if not np.isfinite(rp).all():  # exact epsilon_xx = 0 is 0/0 here; Maxwell stays finite
            rp = reflection_matrix(frequency, layers, substrate, settings)[:, 0, 0]
    else:
        rp = reflection_matrix(frequency, layers, substrate, settings)[:, 0, 0]
    return (1 + settings.coefficient * rp) ** 2 * np.sin(np.radians(settings.angle_deg))
