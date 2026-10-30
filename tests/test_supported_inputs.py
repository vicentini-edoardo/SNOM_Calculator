"""The public build accepts scalar tips and unrotated isotropic/uniaxial media."""
import io

import numpy as np
import pytest

from snom_calculator.materials import MaterialSpectrum, catalog_names, parse_material_table
from snom_calculator.models import TipParameters
from snom_calculator.material_library import normalize_snom_material_definition


def test_tip_rejects_frequency_dependent_permittivity():
    with pytest.raises(ValueError, match="scalar"):
        TipParameters(epsilon=np.array([-100 + 1j, -120 + 1j]))
    assert TipParameters(epsilon=-100 + 1j).epsilon == -100 + 1j


def test_material_rejects_biaxial_axes():
    values = np.array([4 + .1j, 4 + .1j])
    with pytest.raises(ValueError, match="Biaxial"):
        MaterialSpectrum(values, values, "unsupported", secondary_parallel=np.array([5 + .1j, 5 + .1j]))


def test_biaxial_catalog_and_tables_are_unsupported():
    assert "MoO3" not in catalog_names() and "hBN" in catalog_names()
    table = io.BytesIO(b"900,1,0,2,0,3,0\n1000,1,0,2,0,3,0\n")
    with pytest.raises(ValueError):
        parse_material_table(table, np.array([900., 950.]), "biaxial.csv")


def test_custom_definition_rejects_biaxial_and_rotated_media():
    def axis(eps):
        return {"source_type": "weak_lorentz", "eps_inf": eps, "oscillators": []}

    uniaxial = {"name": "u", "tensor_mode": "anisotropic_diagonal",
                "axes": {"xx": axis(2.), "yy": axis(2.), "zz": axis(3.)}}
    assert normalize_snom_material_definition(uniaxial)["axes"]["zz"]["eps_inf"] == 3.
    for definition in ({**uniaxial, "axes": {**uniaxial["axes"], "yy": axis(5.)}},
                       {**uniaxial, "twist_deg": 10.}):
        with pytest.raises(ValueError, match="[Bb]iaxial|rotat"):
            normalize_snom_material_definition(definition)
