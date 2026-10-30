"""Per-solver golden snapshots of the current software, not independent physical references."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from snom_calculator import _spheroid_engine
from snom_calculator._far_field import far_field_factor
from snom_calculator._quadrature import demodulation_weights, tapping_heights
from snom_calculator._spectra import E0
from snom_calculator._calculation_cache import _run_calculations
from snom_calculator.calculations import run_calculations
from snom_calculator.bulk import calculate_bulk_harmonics
from snom_calculator.layer import calculate_stack_harmonics
from snom_calculator.materials import MaterialSpectrum, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters


CASES = (
    "bulk", "bulk-no-field", "bulk-tensor",
    "layer", "layer-zero", "stack", "stack-collapsed", "layer-uniaxial",
    "layer-hyperbolic", "direct-isotropic", "adaptive-uniaxial",
    "checked-bulk", "checked-layer", "checked-stack",
)
SOLVERS = {
    "bulk": ("bulk", "bulk-no-field", "bulk-tensor", "layer-zero", "checked-bulk"),
    "gicm": ("layer", "stack", "stack-collapsed", "checked-layer", "checked-stack"),
    "fim": ("layer-uniaxial", "layer-hyperbolic", "direct-isotropic"),
    "real_axis": ("adaptive-uniaxial",),
}
FIXTURES = {case: Path(__file__).parent / "golden" / f"solver_{solver}.npz"
            for solver, cases in SOLVERS.items() for case in cases}
assert set(FIXTURES) == set(CASES)
RESULT_FIELDS = ("frequency_cm1", "sample_harmonic", "reference_harmonic",
                 "contrast", "amplitude", "phase_rad")


def _snapshot(case, solve_processed=_run_calculations):
    frequency = np.array([900., 950., 1000.])
    sample = MaterialSpectrum(np.array([-3 + .5j, 4 + .2j, 6 + .1j]),
                              np.array([-3 + .5j, 4 + .2j, 6 + .1j]), "sample")
    substrate = constant_material(9 + .3j, frequency, "substrate")
    reference = constant_material(12 + .1j, frequency, "reference")
    second = constant_material(2 + .1j, frequency, "second")
    tip = TipParameters(30., 600., 40., 2., 3., -20 + 2j)
    numerical = NumericalSettings(M=12, N=6, quadQ=16, K=4)
    harmonics = (1, 3)
    # Spheroid films ignored the reflection coefficient when this baseline was
    # recorded, so their stored spectra are the coefficient-1 spectra.
    film = not case.startswith(("bulk", "checked-bulk", "layer-zero"))
    far_field = FarFieldSettings(enabled=case != "bulk-no-field", angle_deg=45.,
                                 coefficient=1. if film else .7)
    if case in {"bulk-tensor", "layer-uniaxial", "adaptive-uniaxial"}:
        sample = replace(sample, perpendicular=sample.parallel * 1.4)
    if case == "layer-hyperbolic":
        sample = replace(sample, parallel=np.full(3, 2 + .1j),
                         perpendicular=np.full(3, -3 + .2j))
    common = dict(reference=reference, tip=tip, numerical=numerical,
                  harmonics=harmonics, far_field=far_field)
    if case in {"direct-isotropic", "adaptive-uniaxial"}:
        return _momentum_snapshot(frequency, sample, substrate, reference, tip, numerical, harmonics,
                                  "real-axis" if case == "adaptive-uniaxial" else "fim")
    if case.startswith("checked-"):
        model = case.removeprefix("checked-")
        if model == "stack":
            model = "multilayer"
        materials = dict(reference=reference, sample=sample, layer=sample,
                         substrate=substrate, thickness_nm=20.,
                         layers=((sample, 20.), (second, 10.)))
        controls = dict(tip=tip, numerical=numerical, far_field=far_field, workers=1,
                        check_n=True, preset="Custom", average_gaps=(1., 3., 3),
                        resolution_fwhm_cm1=100.)
        results = solve_processed(model, frequency, materials, controls, harmonics)
    elif case.startswith("bulk"):
        results = calculate_bulk_harmonics(frequency, sample=sample, **common)
    else:
        layers = ((sample, 0. if case == "layer-zero" else 20.),)
        if case == "stack":
            layers += ((second, 10.),)
        elif case == "stack-collapsed":
            layers = ((sample, 10.), (sample, 10.), (substrate, 5.))
        results = calculate_stack_harmonics(frequency, layers=layers, substrate=substrate, **common)
    values = {}
    for harmonic, result in results.items():
        for field in RESULT_FIELDS:
            values[f"{harmonic}/{field}"] = getattr(result, field)
        for label, epsilon in result.permittivities.items():
            values[f"{harmonic}/epsilon/{label}"] = epsilon
        values[f"{harmonic}/model"] = np.asarray(result.metadata["model"])
        for key, value in result.metadata.items():
            if "check" in key or key in {"sample_gaps_nm", "resolution_fwhm_cm1"}:
                values[f"{harmonic}/metadata/{key}"] = np.asarray(value)
    return values


def _momentum_snapshot(frequency, sample, substrate, reference, tip, numerical, harmonics, method):
    """The former raw engine output (keys of its result dictionary) from the engine pieces."""
    layers = ((sample, 20.),)
    far_field = FarFieldSettings(angle_deg=45.)
    sample_profile, _ = _spheroid_engine.profile(layers, substrate, tip, tip.sample_gap_nm,
                                                 numerical, method=method)
    reference_profile, _ = _spheroid_engine.profile((), reference, tip, tip.reference_gap_nm, numerical)
    weights = demodulation_weights(numerical.M, harmonics)
    drive = far_field_factor(frequency, layers, substrate, far_field)
    values = {"om": frequency, "harmonics": np.asarray(harmonics), "eps_layer": sample.parallel,
              "eps_sub": substrate.parallel, "eps_ref": reference.parallel,
              "H_nm": tapping_heights(tip.sample_gap_nm, tip.amplitude_nm, numerical.M),
              "alpha_z_H": sample_profile.T * drive[:, None]}
    sample_signal = weights @ sample_profile * E0 * drive
    reference_signal = weights @ reference_profile * E0 * far_field_factor(frequency, (), reference, far_field)
    for index, n in enumerate(harmonics):
        contrast = sample_signal[index] / reference_signal[index]
        phase = np.unwrap(np.angle(contrast))
        values.update({f"pz{n}": sample_signal[index], f"pz{n}_ref": reference_signal[index],
                       f"contrast{n}": contrast, f"Ampl{n}": np.abs(contrast),
                       f"Phase{n}": phase + 2 * np.pi if phase[0] < 0 else phase})
    return values


@pytest.mark.parametrize("case", CASES)
def test_calculation_outputs_match_solver_golden(case):
    _assert_snapshot(case, _snapshot(case))


@pytest.mark.parametrize("case", ("checked-bulk", "checked-layer", "checked-stack"))
def test_uncached_domain_matches_solver_golden(case):
    _assert_snapshot(case, _snapshot(case, solve_processed=run_calculations))


@pytest.mark.parametrize("case", ("checked-bulk", "checked-layer", "checked-stack"))
def test_processing_snapshots_exercise_broadening(case):
    def without_broadening(model, frequency, materials, controls, harmonics):
        return run_calculations(model, frequency, materials,
                                {**controls, "resolution_fwhm_cm1": None}, harmonics)

    broadened = _snapshot(case, solve_processed=run_calculations)
    unbroadened = _snapshot(case, solve_processed=without_broadening)
    for harmonic in (1, 3):
        key = f"{harmonic}/contrast"
        assert np.max(np.abs(broadened[key] - unbroadened[key])) > 1e-3 * np.max(np.abs(unbroadened[key]))


def _assert_snapshot(case, actual):
    with np.load(FIXTURES[case], allow_pickle=False) as baseline:
        expected_keys = {key.removeprefix(f"{case}/") for key in baseline.files
                         if key.startswith(f"{case}/")}
        assert actual.keys() == expected_keys
        for key, value in actual.items():
            expected = baseline[f"{case}/{key}"]
            assert value.shape == expected.shape
            assert value.dtype == expected.dtype
            if value.dtype.kind in "US":
                np.testing.assert_array_equal(value, expected)
            else:
                # Same 1e-10 relative tolerance as existing bulk/direct checks.
                # No absolute allowance: small raw dipoles remain protected.
                np.testing.assert_allclose(value, expected, rtol=1e-10, atol=0,
                                           err_msg=f"{case}/{key}")
