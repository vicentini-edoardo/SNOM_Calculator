"""Material import boundaries and preservation of complete definitions."""

import json

import numpy as np
import pytest

from snom_calculator.custom_materials import (
    load_material_definition_json, material_definition_to_json_bytes, parse_epsilon_table,
)
from snom_calculator.material_library import SessionMaterialLibrary


@pytest.mark.parametrize("axis", [None, {}, {"source_type": "weak_lorentz", "eps_inf": 2, "oscillators": [{}]},
    {"source_type": "table", "table": {"frequency_cm1": [900, 901], "eps_real": [2], "eps_imag": [0, 0]}},
    {"source_type": "table", "table": {"frequency_cm1": [901, 900], "eps_real": [2, 2], "eps_imag": [0, 0]}},
    {"source_type": "table", "table": {"frequency_cm1": [900, 901], "eps_real": [2, np.inf], "eps_imag": [0, 0]}}])
def test_invalid_material_archive_does_not_modify_library(axis):
    library = SessionMaterialLibrary({})
    valid = {"name": "valid", "tensor_mode": "isotropic", "axes": {
        key: {"source_type": "weak_lorentz", "eps_inf": 2, "oscillators": []}
        for key in ("xx", "yy", "zz")}}
    invalid = {"name": "invalid", "axes": {key: axis for key in ("xx", "yy", "zz")}}
    payload = json.dumps({"format": "snom-calculator-material-library", "version": 1,
                          "materials": [valid, invalid]})
    with pytest.raises(ValueError):
        library.import_archive(payload)
    assert library.names() == []


@pytest.mark.parametrize("definition", [None, {}, [], {"name": None, "axes": {}}])
def test_invalid_material_objects_raise_value_error(definition):
    with pytest.raises(ValueError):
        SessionMaterialLibrary({}).save(definition)


def test_custom_epsilon_table_rejects_infinity():
    with pytest.raises(ValueError):
        parse_epsilon_table(b"900 2 0\n901 inf 0\n")


def test_custom_epsilon_table_skips_metadata():
    table = parse_epsilon_table(b"% material table\nsource: measured\n900,2,0\n901,3,0\n")
    assert table["eps_real"] == [2, 3]


def test_individual_material_json_preserves_drude_response():
    from snom_calculator.materials import custom_material

    definition = {"name": "Drude material", "tensor_mode": "isotropic", "axes": {
        axis: {"source_type": "weak_lorentz", "eps_inf": 2, "oscillators": []}
        for axis in ("xx", "yy", "zz")}, "drude": {
        axis: {"wp_cm1": 1000, "gamma_cm1": 50} for axis in ("xx", "yy", "zz")}}
    definition["references"] = [{"citation": "Measured material parameters", "url": "https://example.org/data"}]
    restored = load_material_definition_json(material_definition_to_json_bytes(definition))
    spectrum = custom_material(restored, np.array([1000.]))
    np.testing.assert_allclose(spectrum.parallel, [1.002493765586035 + .04987531172069825j])
    assert spectrum.metadata["references"] == definition["references"]
