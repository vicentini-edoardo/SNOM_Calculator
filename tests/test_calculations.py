"""Calculation orchestration checks, independent of widgets and plotting."""

import json
import subprocess
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import snom_calculator.calculations as domain
from snom_calculator.calculations import run_calculations
from snom_calculator._spectra import max_signal_error, normalized_contrast, solve_profile
from snom_calculator.materials import MaterialSpectrum, catalog_material, constant_material
from snom_calculator.models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters


def test_domain_imports_without_frontend_dependencies():
    script = """
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def without_frontend(name, *args, **kwargs):
    if name.split('.')[0] in {'streamlit', 'plotly'}:
        raise AssertionError(f'Domain imported frontend dependency: {name}')
    return original(name, *args, **kwargs)
builtins.__import__ = without_frontend
from snom_calculator import calculations
assert callable(calculations.run_calculations)
"""
    source_root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run([sys.executable, "-c", script, str(source_root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("gaps", [(0., 3., 3), (3., 1., 3), (1., 3., 1),
                                  (1., 3., 26), (1., np.nan, 3)])
def test_gap_range_errors_precede_solving(gaps):
    def forbidden(**kwargs):
        pytest.fail("Invalid gap ranges must fail before solving.")

    with pytest.raises(ValueError, match="positive, increasing sample gap range"):
        run_calculations("bulk", np.array([900., 950.]), {},
                         {"average_gaps": gaps}, (3,), solve_bulk=forbidden)


def test_gap_count_accepts_numpy_integers():
    frequency = np.array([900., 950.])
    seen = []

    def solve(**kwargs):
        seen.append(kwargs["tip"].sample_gap_nm)
        raise RuntimeError("stop after validation")

    with pytest.raises(RuntimeError):
        run_calculations("bulk", frequency, {"sample": None, "reference": None},
                         {"average_gaps": (1., 3., np.int64(3)), "tip": TipParameters(),
                          "numerical": NumericalSettings(), "far_field": FarFieldSettings()},
                         (3,), solve_bulk=solve)
    assert seen == [1.0]


def test_empty_layer_stack_is_rejected_cleanly():
    frequency = np.linspace(900., 1000., 5)
    material = constant_material(2.0, frequency, "a")
    with pytest.raises(ValueError, match="at least one layer"):
        domain.validate_structure(
            "multilayer", {"substrate": material, "reference": material, "layers": []})
    # The engine API treats an empty stack as the bulk substrate.
    args = (TipParameters(), NumericalSettings(M=8, N=8, quadQ=16), (2,), FarFieldSettings(enabled=False))
    empty = domain.calculate_stack_checked(frequency, (), material, material, *args, 1)[2]
    bulk = domain.calculate_bulk_checked(frequency, material, material, *args)[2]
    np.testing.assert_array_equal(empty.contrast, bulk.contrast)


@pytest.mark.parametrize("frequency, fwhm, message", [
    ([900., 950.], 0., "Gaussian FWHM must be positive"),
    ([900., 950.], np.nan, "Gaussian FWHM must be positive"),
    ([900., 950., 1020.], 7., "evenly spaced frequency grid"),
    ([900.], 7., "evenly spaced frequency grid"),
])
def test_broadening_preserves_validation_errors(frequency, fwhm, message):
    frequency = np.array(frequency)
    signal = np.ones(frequency.size, complex)
    result = SpectrumResult.from_contrast(
        frequency_cm1=frequency, sample_harmonic=signal, reference_harmonic=signal,
        contrast=signal, permittivities={}, metadata={},
    )
    controls = dict(tip=None, numerical=None, far_field=None, resolution_fwhm_cm1=fwhm)
    with pytest.raises(ValueError, match=message):
        run_calculations("bulk", frequency, {"sample": None, "reference": None},
                         controls, (3,), solve_bulk=lambda **kwargs: {3: result})


@pytest.mark.parametrize("reference, rejected", [(0., True), (1e-19, True), (1e-18, False)])
def test_normalization_threshold_is_strict(reference, rejected):
    arguments = (np.ones((1, 2), complex), np.full((1, 2), reference, complex), (3,))
    if rejected:
        with pytest.raises(ValueError, match="zero normalization signal"):
            normalized_contrast(*arguments)
    else:
        np.testing.assert_array_equal(normalized_contrast(*arguments)[0], np.full(2, 1 / reference))


def test_bulk_calculation_reports_embedded_check_max_errors():
    frequency = np.array([900., 905., 910.])
    results = domain.calculate_bulk_checked(
        frequency, catalog_material("SiC3C", frequency), catalog_material("Si", frequency),
        TipParameters(length_nm=600., sample_gap_nm=1., reference_gap_nm=1.),
        NumericalSettings(M=16, N=16, quadQ=24), (2, 3), FarFieldSettings(),
    )

    assert results[2].metadata["M_check_max_error"] > 1e-3
    assert results[2].metadata["N_check_max_error"] > 1e-3
    assert results[2].metadata["quadQ_check_max_error"] > 0
    assert results[2].metadata["quadQ_check_higher"] == 32


def test_isotropic_layer_has_no_k_check():
    frequency = np.array([900., 940.])
    tip = TipParameters(length_nm=600., sample_gap_nm=1., reference_gap_nm=1.)
    film, substrate = catalog_material("SiC3C", frequency), catalog_material("Si", frequency)
    reference = constant_material(11.7 + .05j, frequency, "reference")
    off = FarFieldSettings(enabled=False)
    # K comes from the FIM-validated table (_preset_defaults.film_k): only M and N are checked.
    checked = domain.calculate_layer_checked(frequency, film, substrate, reference, 20., tip,
                                             NumericalSettings(M=16, N=16, quadQ=24, K=8), (2, 3), off, 1)[3].metadata
    assert "M_check_max_error" in checked and "N_check_max_error" in checked
    assert not [key for key in checked if key.startswith(("K_check", "quadQ_check"))]
    bulk = domain.calculate_bulk_checked(frequency, film, substrate, tip, NumericalSettings(M=16, N=16, quadQ=24),
                                         (2, 3), off)
    assert "K_check_max_error" not in bulk[3].metadata
    unchecked = domain.calculate_layer_checked(frequency, film, substrate, reference, 20., tip,
                                               NumericalSettings(M=16, N=16, quadQ=24, K=8), (2, 3), off, 1,
                                               check_n=False)[3].metadata
    assert "M_check_max_error" in unchecked
    assert not [key for key in unchecked if key.startswith(("N_check", "K_check"))]
    bulk = domain.calculate_bulk_checked(frequency, film, substrate, tip, NumericalSettings(M=16, N=16, quadQ=24),
                                         (2, 3), off, check_n=False)[3].metadata
    assert not [key for key in bulk if key.startswith(("N_check", "quadQ_check"))]


@pytest.mark.parametrize("film, substrate_name", ((None, "SiC4H"), ("aSiO2", "Si"), ("hBN", "Si")))
def test_threaded_solve_matches_serial(monkeypatch, film, substrate_name):
    from snom_calculator import _spheroid_engine
    frequency = np.linspace(1300., 1400., 30) if film == "hBN" else np.linspace(900., 960., 30)
    tip = TipParameters(length_nm=600., sample_gap_nm=2., reference_gap_nm=2.)
    layers = ((catalog_material(film, frequency), 20.),) if film else ()
    substrate = catalog_material(substrate_name, frequency)

    def run(threads):
        monkeypatch.setattr(_spheroid_engine, "_THREADS", threads)
        _spheroid_engine.clear_caches()
        values, _ = solve_profile(layers, substrate, tip, 2., NumericalSettings(M=16, N=16, quadQ=24, K=10), (), "auto", 1)
        return values

    np.testing.assert_allclose(run(4), run(1), rtol=1e-10)


@pytest.mark.parametrize("model", ("bulk", "layer", "multilayer"))
@pytest.mark.parametrize("far_enabled", (False, True))
def test_calculations_without_reference_return_sample_signal(model, far_enabled):
    frequency = np.array([900., 910.])
    sample = constant_material(6 + .2j, frequency, "sample")
    substrate = constant_material(12 + .1j, frequency, "substrate")
    reference = constant_material(8 + .1j, frequency, "reference")
    materials = {"sample": sample, "layer": sample, "substrate": substrate,
                 "layers": ((sample, 20.), (reference, 10.)),
                 "thickness_nm": 20., "reference": reference}
    controls = {"tip": TipParameters(), "numerical": NumericalSettings(M=8, N=8, quadQ=16, K=2),
                "far_field": FarFieldSettings(enabled=far_enabled), "workers": 1,
                "preset": "Custom", "check_n": True}
    normalized = run_calculations(model, frequency, materials, controls, (2,))[2]
    materials["reference"] = None
    materials["reference_disabled"] = True
    domain.validate_structure(model, materials)
    result = run_calculations(model, frequency, materials, controls, (2,))[2]

    np.testing.assert_allclose(result.contrast, normalized.sample_harmonic)
    np.testing.assert_allclose(result.amplitude, np.abs(normalized.sample_harmonic))
    np.testing.assert_array_equal(result.reference_harmonic, np.ones(2))
    assert result.metadata["reference_material"] is None
    assert result.metadata["normalization"] == "none"
    assert "M_check_max_error" in result.metadata
    assert "N_check_max_error" in result.metadata


def test_no_reference_skips_reference_solver():
    frequency = np.array([900., 910.])
    sample = constant_material(6 + .2j, frequency, "sample")
    substrate = constant_material(12 + .1j, frequency, "substrate")
    def forbidden(layers, *args):
        if not layers:
            pytest.fail("Reference solver must be skipped.")
        return solve_profile(layers, *args)

    result = domain.calculate_layer_checked(
        frequency, sample, substrate, None, 20., TipParameters(),
        NumericalSettings(M=8, N=8, quadQ=16, K=2), (2,),
        FarFieldSettings(), 1, solve=forbidden,
    )[2]
    np.testing.assert_array_equal(result.contrast, result.sample_harmonic)


def test_zero_thickness_without_reference_uses_bulk_sample():
    frequency = np.array([900., 910.])
    sample = constant_material(6 + .2j, frequency, "sample")
    substrate = constant_material(12 + .1j, frequency, "substrate")
    args = (TipParameters(), NumericalSettings(M=8, N=8, quadQ=16, K=2),
            (2,), FarFieldSettings())
    bulk = domain.calculate_bulk_checked(frequency, substrate, None, *args)[2]
    layer = domain.calculate_layer_checked(frequency, sample, substrate, None, 0., *args, 1)[2]
    stack = domain.calculate_stack_checked(
        frequency, ((sample, 0.), (substrate, 20.)), substrate, None, *args, 1)[2]
    np.testing.assert_allclose(layer.contrast, bulk.contrast)
    np.testing.assert_allclose(stack.contrast, bulk.contrast)


def test_layer_checks_cover_zero_thickness_and_simplified_stack():
    frequency = np.array([900., 910.])
    layer = constant_material(6 + .2j, frequency, "layer")
    substrate = constant_material(12 + .1j, frequency, "substrate")
    reference = constant_material(8 + .1j, frequency, "reference")
    tip = TipParameters()
    numerical = NumericalSettings(M=8, N=8, quadQ=16, K=2)
    far_field = FarFieldSettings(enabled=False)

    for results in (
        domain.calculate_layer_checked(
            frequency, layer, substrate, reference, 0., tip, numerical, (2,), far_field, 1),
        domain.calculate_stack_checked(
            frequency, ((layer, 0.), (layer, 20.)), substrate, reference,
            tip, numerical, (2,), far_field, 1,
        ),
    ):
        assert "M_check_max_error" in results[2].metadata
        assert "N_check_max_error" in results[2].metadata


def test_bulk_check_catches_one_bad_frequency():
    full = np.ones((1, 20), dtype=complex)
    sample_half = full.copy()
    sample_half[0, 7] = 1.002

    error = max_signal_error(full, full, sample_half, full)

    np.testing.assert_allclose(error, [0.002])
    with_null = np.array([[1, 0]], dtype=complex)
    tiny_change = np.array([[1, 1e-7]], dtype=complex)
    np.testing.assert_allclose(
        max_signal_error(with_null, with_null, tiny_change, with_null),
        [1e-5],
    )


def test_run_calculations_batches_selected_harmonics():
    seen = []
    expected = {2: SimpleNamespace(metadata={}), 3: SimpleNamespace(metadata={})}

    def fake_bulk(**kwargs):
        seen.append(kwargs["harmonics"])
        return expected

    results = run_calculations(
        "bulk",
        np.array([600.0, 601.0]),
        {"sample": catalog_material("SiC4H", np.array([600., 601.])),
         "reference": constant_material(12, np.array([600., 601.]), "reference")},
        {"tip": TipParameters(), "numerical": NumericalSettings(),
         "far_field": FarFieldSettings(), "preset": "Weak"},
        (2, 3),
        solve_bulk=fake_bulk,
    )

    assert results is expected
    assert seen == [(2, 3)]
    metadata = results[3].metadata
    assert metadata["tip"]["radius_nm"] == 25
    assert metadata["illumination"]["angle_deg"] == 60
    assert metadata["numerical"]["momentum_rtol"] == 1e-5
    assert metadata["sample_structure"]["sample"]["name"] == "SiC4H"
    assert "definition" not in metadata["sample_structure"]["sample"]
    assert metadata["reference_material"]["epsilon"] == 12
    assert any(ref.get("doi") == "10.1021/acs.nanolett.5c01352" for ref in metadata["references"])
    assert any("github.com" in ref.get("url", "") for ref in metadata["references"])
    assert "epsilon" in metadata["tip"]

    # The provenance records the scalar epsilon the engines solve with, not a passed tip_material.
    results = run_calculations(
        "bulk", np.array([600.0, 601.0]),
        {"sample": catalog_material("SiC4H", np.array([600., 601.])),
         "reference": constant_material(12, np.array([600., 601.]), "reference")},
        {"tip": TipParameters(), "numerical": NumericalSettings(), "far_field": FarFieldSettings(),
         "preset": "Weak", "tip_material": constant_material(-2000 + 600j, np.array([600., 601.]), "Au")},
        (2, 3), solve_bulk=fake_bulk,
    )
    assert results[3].metadata["tip"]["epsilon"] == TipParameters().epsilon
    assert results[3].metadata["tip_material"]["epsilon"] == TipParameters().epsilon


@pytest.mark.parametrize("sample_range, reference_range, sample_gaps, reference_gaps", [
    ((2., 6., 3), (1., 5., 2), (2., 4., 6.), (1., 5.)),
    ((4., 4., 25), (3., 3., 25), None, None),
    ((4., 4., 25), (1., 5., 2), None, (1., 5.)),
])
def test_gap_ranges_reach_the_solver_once(sample_range, reference_range, sample_gaps, reference_gaps):
    frequency = np.array([900., 901.])
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        signal = np.full(2, 2 + 1j)
        return {3: SpectrumResult.from_contrast(
            frequency_cm1=frequency, sample_harmonic=signal, reference_harmonic=signal,
            contrast=signal / signal, permittivities={}, metadata={})}

    material = constant_material(4, frequency, "sample")
    run_calculations(
        "bulk", frequency, {"sample": material, "reference": material},
        {"tip": TipParameters(), "numerical": NumericalSettings(),
         "far_field": FarFieldSettings(), "preset": "Weak",
         "average_gaps": sample_range, "average_reference_gaps": reference_range},
        (3,), solve_bulk=solve)
    assert len(calls) == 1
    assert (calls[0]["tip"].sample_gap_nm, calls[0]["tip"].reference_gap_nm) == (sample_range[0], reference_range[0])
    assert calls[0].get("sample_gaps") == sample_gaps
    assert calls[0].get("reference_gaps") == reference_gaps


@pytest.mark.parametrize("M", [8, 120])  # every gap on its own tapping heights; one shared span profile
@pytest.mark.parametrize("model", ["bulk", "layer"])
def test_gap_average_is_the_mean_signal_with_the_worst_gap_errors(M, model):
    frequency = np.array([900., 950., 1000.])
    sample = MaterialSpectrum(np.array([-3 + .5j, 4 + .2j, 6 + .1j]), np.array([-3 + .5j, 4 + .2j, 6 + .1j]), "s")
    substrate = constant_material(9 + .3j, frequency, "substrate")
    reference = constant_material(12 + .1j, frequency, "reference")
    layers = () if model == "bulk" else ((sample, 20.),)
    tip = TipParameters(30., 600., 40., 1., 2., -20 + 2j)
    numerical = NumericalSettings(M=M, N=8, quadQ=16, K=4)
    sample_gaps, reference_gaps = (1., 2., 3.), (2., 4.)
    args = (frequency, layers, substrate if layers else sample, reference)
    options = dict(far_field=FarFieldSettings(angle_deg=45.), checked=True)
    averaged = domain.calculate_spectra(*args, tip, numerical, (2, 3), sample_gaps=sample_gaps,
                                        reference_gaps=reference_gaps, **options)
    runs = [domain.calculate_spectra(*args, replace(tip, sample_gap_nm=gap, reference_gap_nm=ref), numerical, (2, 3),
                                     **options)
            for gap in sample_gaps for ref in reference_gaps]
    spanned = M == 120
    assert averaged[2].metadata.get("height_nodes", 0) > 0 if spanned else "height_nodes" not in averaged[2].metadata
    for harmonic in (2, 3):
        rtol = 1e-7 if spanned else 1e-12  # the span interpolates to every gap's tapping heights
        np.testing.assert_allclose(averaged[harmonic].sample_harmonic, np.mean(
            [run[harmonic].sample_harmonic for run in runs[::len(reference_gaps)]], axis=0), rtol=rtol)
        np.testing.assert_allclose(averaged[harmonic].reference_harmonic, np.mean(
            [run[harmonic].reference_harmonic for run in runs[:len(reference_gaps)]], axis=0), rtol=rtol)
        for key in ("M_check_max_error", "N_check_max_error"):
            assert averaged[harmonic].metadata[key] == pytest.approx(
                max(run[harmonic].metadata[key] for run in runs), rel=1e-4 if spanned else 1e-12)


def test_run_calculations_convolves_complex_spectrum_by_fwhm():
    frequency = np.arange(11, dtype=float)
    contrast = np.zeros(11, dtype=complex)
    contrast[5] = 1 + 1j

    def fake_bulk(**kwargs):
        return {3: SpectrumResult.from_contrast(
            frequency_cm1=frequency,
            sample_harmonic=contrast,
            reference_harmonic=np.ones(11, dtype=complex),
            contrast=contrast,
            permittivities={}, metadata={},
        )}

    controls = {
        "tip": TipParameters(), "numerical": NumericalSettings(),
        "far_field": FarFieldSettings(), "preset": "Weak",
        "resolution_fwhm_cm1": 2 * np.sqrt(2 * np.log(2)),
    }

    result = run_calculations(
        "bulk", frequency, {"sample": constant_material(4, frequency, "sample"),
                            "reference": constant_material(12, frequency, "reference")}, controls, (3,),
        solve_bulk=fake_bulk,
    )[3]

    assert 0 < result.contrast[5].real < 1
    assert result.contrast[4].real / result.contrast[5].real == pytest.approx(np.exp(-0.5), rel=1e-3)
    np.testing.assert_allclose(result.contrast.real, result.contrast.imag)
    np.testing.assert_allclose(result.sample_harmonic, result.contrast)


def test_run_calculations_solves_multilayer_stack():
    frequency = np.array([950.0, 951.0])
    materials = {
        "layers": tuple(
            (constant_material(2 + index + 0.1j, frequency, f"film {index}"), 10.0)
            for index in range(4)
        ),
        "substrate": constant_material(12, frequency, "Si"),
        "reference": constant_material(12, frequency, "reference"),
    }
    controls = {
        "tip": TipParameters(30, 600, 50, 2, 2, -1e10 + 0j),
        "numerical": NumericalSettings(M=8, N=8, quadQ=16, K=12),
        "far_field": FarFieldSettings(),
        "workers": 1,
        "preset": "Custom",
    }

    results = run_calculations("multilayer", frequency, materials, controls, (2, 3))

    assert set(results) == {2, 3}
    assert np.isfinite(results[3].contrast).all()
    assert results[3].metadata["thicknesses_nm"] == (10.0,) * 4
    assert [layer["thickness_nm"] for layer in results[3].metadata["sample_structure"]["layers"]] == [10.] * 4
    from snom_calculator.io import combined_results_csv, read_experiment_columns

    exported = combined_results_csv(results)
    tip_line = next(line for line in exported.splitlines() if line.startswith("# tip: "))
    assert json.loads(tip_line.removeprefix("# tip: "))["epsilon"] == {"real": -1e10, "imag": 0}
    headers, rows = read_experiment_columns(BytesIO(exported.encode()))
    assert headers == ["wn", "O2A", "O2P", "O3A", "O3P"]
    np.testing.assert_allclose(np.array(rows, dtype=float)[:, 3], results[3].amplitude)


def test_fim_flag_forces_momentum_integration_for_isotropic_film():
    frequency = np.array([950.0, 951.0])
    materials = {
        "layer": constant_material(4 + 0.5j, frequency, "film"), "thickness_nm": 20.0,
        "substrate": constant_material(12, frequency, "Si"), "reference": None,
    }
    controls = {
        "tip": TipParameters(30, 600, 50, 2, 2, -1e10 + 0j),
        "numerical": NumericalSettings(M=8, N=8, quadQ=16, K=20, momentum_rtol=1e-6),
        "far_field": FarFieldSettings(), "workers": 1, "preset": "Custom",
    }
    image = run_calculations("layer", frequency, materials, controls, (3,))[3]
    fim = run_calculations("layer", frequency, materials, {**controls, "method": "fim"}, (3,))[3]
    assert fim.metadata["film_method"] == "fim"
    np.testing.assert_allclose(fim.amplitude, image.amplitude, rtol=0.05)
