from dataclasses import replace

import numpy as np
import pytest

from snom_calculator.models import (
    NumericalSettings,
    SpectrumResult,
    TipParameters,
    validate_inputs,
)


def test_geometry_rejects_length_not_greater_than_diameter():
    with pytest.raises(ValueError, match="greater than 2R"):
        validate_inputs(
            np.array([900.0, 901.0]),
            harmonic=3,
            tip=TipParameters(radius_nm=30, length_nm=60),
            numerical=NumericalSettings(M=30, N=40, quadQ=48),
        )


@pytest.mark.parametrize("tolerance", [0, -1e-5, 1, np.inf, np.nan])
def test_invalid_momentum_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="Momentum tolerance"):
        validate_inputs(
            np.array([900., 901.]), harmonic=3, tip=TipParameters(),
            numerical=NumericalSettings(momentum_rtol=tolerance),
        )


def test_result_phase_is_unwrapped_in_radians():
    contrast = np.exp(1j * np.array([3.0, -3.0, -2.8]))
    result = SpectrumResult.from_contrast(
        frequency_cm1=np.array([1.0, 2.0, 3.0]),
        sample_harmonic=contrast,
        reference_harmonic=np.ones(3, dtype=complex),
        contrast=contrast,
        permittivities={"sample": np.ones(3, dtype=complex)},
        metadata={},
    )
    assert np.max(np.abs(np.diff(result.phase_rad))) < np.pi


@pytest.mark.parametrize("field", ["radius_nm", "length_nm", "amplitude_nm", "sample_gap_nm", "reference_gap_nm", "epsilon"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_solver_rejects_nonfinite_tip_parameters(field, value):
    with pytest.raises(ValueError):
        validate_inputs(np.array([900., 901.]), harmonic=3,
                        tip=replace(TipParameters(), **{field: value}), numerical=NumericalSettings())


@pytest.mark.parametrize("field", ["M", "N", "quadQ", "K"])
@pytest.mark.parametrize("value", [40.5, True])
def test_solver_rejects_noninteger_orders(field, value):
    with pytest.raises(ValueError, match="integer"):
        validate_inputs(np.array([900., 901.]), harmonic=3, tip=TipParameters(),
                        numerical=replace(NumericalSettings(), **{field: value}))


@pytest.mark.parametrize("thickness", [np.nan, np.inf])
def test_solver_rejects_nonfinite_thickness(thickness):
    with pytest.raises(ValueError, match="thickness"):
        validate_inputs(np.array([900., 901.]), harmonic=3, tip=TipParameters(),
                        numerical=NumericalSettings(), thickness_nm=thickness)
