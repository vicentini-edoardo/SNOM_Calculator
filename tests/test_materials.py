from io import BytesIO
import io

import numpy as np
import pytest

from materials_library import eps_tensor, load

from snom_calculator.materials import (
    MaterialSpectrum,
    catalog_material,
    catalog_names,
    constant_material,
    effective_permittivity,
    parse_material_table,
    with_drude,
)


def test_effective_permittivity_uses_passive_branch():
    eps = effective_permittivity(
        MaterialSpectrum(
            parallel=np.array([-4 + 0.2j]),
            perpendicular=np.array([9 + 0.3j]),
            label="uniaxial",
        )
    )
    assert eps.imag[0] >= 0
    np.testing.assert_allclose(eps**2, np.array([-4 + 0.2j]) * np.array([9 + 0.3j]))


@pytest.mark.parametrize("normal,in_plane,expected", [
    (-9, -4, -6), (9, 4, 6), (-9, 4, 6j), (9, -4, 6j), (0, 4, 0),
])
def test_lossless_uniaxial_axes_match_the_passive_limit(normal, in_plane, expected):
    for loss in (0, 1e-12):
        material = MaterialSpectrum(np.array([normal + 1j * loss]),
                                    np.array([in_plane + 1j * loss]), "uniaxial")
        np.testing.assert_allclose(effective_permittivity(material), [expected], atol=3e-6)


def test_lossless_negative_uniaxial_bulk_spectrum_matches_small_loss():
    from snom_calculator.bulk import calculate_bulk_harmonics
    from snom_calculator.models import FarFieldSettings, NumericalSettings, TipParameters

    frequency = np.array([900., 910.])
    signals = []
    for loss in (0, 1e-10):
        material = MaterialSpectrum(np.full(2, -9 + 1j * loss),
                                    np.full(2, -4 + 1j * loss), "negative axes")
        result = calculate_bulk_harmonics(
            frequency, sample=material, reference=None, tip=TipParameters(),
            numerical=NumericalSettings(M=8, N=8, quadQ=16), harmonics=(2,),
            far_field=FarFieldSettings(enabled=False),
        )
        signals.append(result[2].sample_harmonic)
    np.testing.assert_allclose(*signals, rtol=1e-8)


def test_zero_normal_axis_has_zero_electrostatic_admittance():
    material = MaterialSpectrum(np.array([0]), np.array([4]), "zero normal")
    np.testing.assert_array_equal(effective_permittivity(material), [0])


def test_material_upload_interpolates_isotropic_table():
    csv = io.BytesIO(
        b"frequency_cm1,eps_real,eps_imag\n800,4,0.1\n1000,6,0.3\n"
    )
    spectrum = parse_material_table(csv, np.array([850.0, 950.0]), "upload")
    np.testing.assert_allclose(spectrum.parallel, [4.5 + 0.15j, 5.5 + 0.25j])
    np.testing.assert_allclose(spectrum.perpendicular, spectrum.parallel)
    assert spectrum.metadata["definition"]["frequency_cm1"] == [800, 1000]


def test_material_provenance_survives_drude_addition():
    frequency = np.array([900., 901.])
    material = with_drude(catalog_material("Si", frequency), frequency, 100., 10.)
    assert material.metadata["definition"]["name"] == "Si"
    assert material.metadata["references"][0]["doi"] == "10.1063/1.1923612"
    assert material.metadata["added_drude"] == {"wp_cm1": 100., "gamma_cm1": 10.}


def test_scalar_only_catalog_functions_are_evaluated_over_the_grid():
    spectrum = catalog_material("Au", np.array([900.0, 1000.0]))
    assert spectrum.parallel.shape == (2,)
    assert np.isfinite(spectrum.parallel).all()


def test_sic6h_catalog_is_passive_over_default_frequency_grid():
    spectrum = catalog_material("SiC6H", np.linspace(600.0, 1200.0, 10))
    assert np.all(spectrum.parallel.imag >= 0)
    assert np.all(spectrum.perpendicular.imag >= 0)


def test_material_upload_accepts_whitespace_txt():
    table = io.BytesIO(
        b"frequency_cm1 eps_real eps_imag\n800 4 0.1\n1000 6 0.3\n"
    )
    spectrum = parse_material_table(table, np.array([900.0]), "upload")
    np.testing.assert_allclose(spectrum.parallel, [5 + 0.2j])


def test_with_drude_adds_the_same_term_to_every_axis():
    frequency = np.array([500.0, 1000.0])
    base = constant_material(4 + 0.1j, frequency, "base")
    drude = with_drude(base, frequency, 2000.0, 50.0)
    expected = 4 + 0.1j - 2000.0**2 / (frequency**2 + 1j * 50.0 * frequency)
    np.testing.assert_allclose(drude.parallel, expected)
    np.testing.assert_allclose(drude.perpendicular, expected)
    assert np.all(drude.parallel.imag > 0)
    assert drude.label == "base + Drude"
    assert with_drude(base, frequency, 0.0, 50.0) is base


@pytest.mark.parametrize("data", [b"800 4 .1\n1000 6 .3\n", b"800,4,.1\n1000,6,.3\n",
    b"800,9,.2,4,.1\n1000,11,.4,6,.3\n"])
def test_headerless_material_upload_retains_first_row(data):
    material = parse_material_table(BytesIO(data), np.array([800., 900., 1000.]), "upload")
    np.testing.assert_allclose(material.perpendicular, [4 + .1j, 5 + .2j, 6 + .3j])


def test_snom_builtin_catalog_contains_only_supported_models():
    catalog = catalog_names()
    assert "hBN" in catalog and "aSiO2" in catalog and "SiC3C" in catalog
    assert not {"vacuum", "graphene", "Ga2O3_beta", "BaTiO3", "SiC"} & set(catalog)


def test_snom_builtin_axes_match_shared_library():
    frequency = np.array([800.0, 1000.0])
    spectrum = catalog_material("hBN", frequency)
    expected = eps_tensor(load("hBN"), frequency)
    np.testing.assert_allclose(spectrum.parallel, expected[:, 2, 2])
    np.testing.assert_allclose(spectrum.perpendicular, expected[:, 0, 0])
