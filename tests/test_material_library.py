import json

import numpy as np
import pytest

import snom_calculator.material_library as libraries

from snom_calculator.material_library import MaterialLibrary
from snom_calculator.materials import custom_material


def _weak_axis() -> dict[str, object]:
    return {
        "source_type": "weak_lorentz",
        "eps_inf": 2.0,
        "oscillators": [{"strength": 3.0, "w0_cm1": 200.0, "gamma_cm1": 10.0}],
    }


def _strong_axis() -> dict[str, object]:
    return {
        "source_type": "strong_lorentz",
        "eps_inf": 2.0,
        "oscillators": [{"to_cm1": 180.0, "lo_cm1": 240.0, "gamma_cm1": 8.0}],
    }


def _definition(name: str = "Test material") -> dict[str, object]:
    weak = _weak_axis()
    return {
        "name": name,
        "tensor_mode": "anisotropic_diagonal",
        "axes": {"xx": weak, "yy": weak, "zz": _strong_axis()},
    }


def test_custom_material_maps_uniaxial_axes_and_evaluates_lorentz_models():
    spectrum = custom_material(_definition(), np.array([100.0]))

    np.testing.assert_allclose(spectrum.perpendicular, [5.995560488346282 + 0.13318534961154274j])
    np.testing.assert_allclose(spectrum.secondary_parallel, spectrum.perpendicular)
    np.testing.assert_allclose(spectrum.parallel, [4.247133757961784 + 0.08025477707006369j])


def test_custom_material_adds_per_axis_drude_terms():
    definition = _definition()
    definition["drude"] = {
        "xx": {"wp_cm1": 100.0, "gamma_cm1": 10.0},
        "yy": {"wp_cm1": 100.0, "gamma_cm1": 10.0},
        "zz": {"wp_cm1": 200.0, "gamma_cm1": 20.0},
    }

    spectrum = custom_material(definition, np.array([100.0]))

    np.testing.assert_allclose(
        spectrum.perpendicular,
        [5.005461478445292 + 0.23219525060164175j],
    )
    np.testing.assert_allclose(
        spectrum.parallel,
        [0.4009799118079385 + 0.8494855462614539j],
    )


def test_library_saves_reloads_and_renames_json_definitions(tmp_path):
    library = MaterialLibrary(tmp_path)
    definition = _definition()
    definition["drude"] = {axis: {"wp_cm1": 100.0, "gamma_cm1": 10.0} for axis in ("xx", "yy", "zz")}
    library.save(definition)

    assert library.names() == ["Test material"]
    assert library.load("Test material")["axes"]["zz"] == _strong_axis()
    assert library.load("Test material")["drude"]["xx"]["wp_cm1"] == 100.0

    library.rename("Test material", "Renamed material")

    assert library.names() == ["Renamed material"]
    assert not any(path.name == "Test%20material.json" for path in tmp_path.iterdir())
    assert json.loads(next(tmp_path.iterdir()).read_text())["name"] == "Renamed material"


def test_library_delete_removes_only_selected_definition(tmp_path):
    library = MaterialLibrary(tmp_path)
    library.save(_definition("keep"))
    library.save(_definition("remove"))

    library.delete("remove")

    assert library.names() == ["keep"]


def test_library_archive_round_trips_all_materials(tmp_path):
    source = MaterialLibrary(tmp_path / "source")
    source.save(_definition("first"))
    source.save(_definition("second"))

    restored = MaterialLibrary(tmp_path / "restored")
    names = restored.import_archive(source.archive_json_bytes())

    assert names == ["first", "second"]
    assert restored.load("first") == source.load("first")
    assert restored.load("second") == source.load("second")


def test_library_archive_rejects_invalid_definition_without_writing_files(tmp_path):
    library = MaterialLibrary(tmp_path)
    payload = json.dumps({"format": "snom-calculator-material-library", "version": 1, "materials": [_definition(), {"name": "bad", "axes": {}}]}).encode()

    with pytest.raises(ValueError, match="missing axis"):
        library.import_archive(payload)

    assert library.names() == []


def test_file_library_persists_save_import_rename_and_delete(tmp_path):
    path = tmp_path / "custom_material_library.json"
    library = libraries.JsonMaterialLibrary({}, path)
    library.save(_definition("first"))
    assert libraries.JsonMaterialLibrary({}, path).names() == ["first"]
    source = libraries.SessionMaterialLibrary({})
    source.save(_definition("second"))
    library.import_archive(source.archive_json_bytes())
    library.rename("first", "renamed")
    library.delete("second")
    assert libraries.JsonMaterialLibrary({}, path).names() == ["renamed"]
    assert json.loads(path.read_text())["materials"][0]["name"] == "renamed"


def test_file_library_rejects_corruption_and_preserves_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "library.json"
    path.write_text("broken JSON")
    with pytest.raises(ValueError):
        libraries.JsonMaterialLibrary({}, path)
    assert path.read_text() == "broken JSON"
    path.unlink()
    definitions = {}
    library = libraries.JsonMaterialLibrary(definitions, path)
    library.save(_definition("original"))
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("disk write failed")

    monkeypatch.setattr(type(path), "replace", fail_replace)
    with pytest.raises(OSError):
        library.save(_definition("new"))
    assert path.read_bytes() == previous
    assert list(definitions) == ["original"]
