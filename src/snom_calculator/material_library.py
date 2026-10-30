"""Persistent custom-material definitions for the s-SNOM workspace."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from uuid import uuid4

from .custom_materials import normalize_material_definition


def normalize_snom_material_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a diagonal principal-axis definition and optional rotations."""
    normalized = _normalize(definition)
    _check_biaxial(normalized)
    return normalized


def _normalize(definition: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_material_definition(definition)
    tensor_mode = normalized["tensor_mode"]
    axes = normalized["axes"]
    if tensor_mode not in {"isotropic", "anisotropic_diagonal"}:
        raise ValueError("SNOM supports only isotropic or diagonal material definitions.")
    if tensor_mode == "isotropic" and not axes["xx"] == axes["yy"] == axes["zz"]:
        raise ValueError("Isotropic material definitions require εxx = εyy = εzz.")
    for angle in ("twist_deg", "tilt_deg"):
        if angle in definition:
            try:
                value = float(definition[angle])
            except (TypeError, ValueError) as exc:
                raise ValueError("Material rotations must be finite numbers.") from exc
            if not math.isfinite(value):
                raise ValueError("Material rotations must be finite numbers.")
            normalized[angle] = value
    raw_drude = definition.get("drude")
    if raw_drude is None:
        return normalized
    if not isinstance(raw_drude, Mapping):
        raise ValueError("Drude terms must map tensor axes to parameters.")
    drude: dict[str, dict[str, float]] = {}
    for axis in ("xx", "yy", "zz"):
        term = raw_drude.get(axis, {})
        if not isinstance(term, Mapping):
            raise ValueError(f"Drude ε{axis} parameters must be an object.")
        try:
            wp = float(term.get("wp_cm1", 0.0))
            gamma = float(term.get("gamma_cm1", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Drude ε{axis} parameters must be numbers.") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in (wp, gamma)):
            raise ValueError(f"Drude ε{axis} parameters must be finite and non-negative.")
        drude[axis] = {"wp_cm1": wp, "gamma_cm1": gamma}
    if tensor_mode == "isotropic" and not drude["xx"] == drude["yy"] == drude["zz"]:
        raise ValueError("Isotropic material definitions require identical Drude terms on every axis.")
    normalized["drude"] = drude
    return normalized


def _check_biaxial(normalized: Mapping[str, Any]) -> None:
    """Reject εxx ≠ εyy and rotations in the public calculator."""
    drude = normalized.get("drude", {})
    if (normalized["axes"]["xx"] != normalized["axes"]["yy"] or drude.get("xx") != drude.get("yy")
            or normalized.get("twist_deg", 0.0) or normalized.get("tilt_deg", 0.0)):
        raise ValueError("Biaxial and rotated materials are unsupported.")


class MaterialLibrary:
    """One validated JSON definition per material in a user-owned directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".snom-calculator" / "materials"

    @staticmethod
    def _filename(name: str) -> str:
        return f"{quote(name, safe='')}.json"

    def _path(self, name: str) -> Path:
        return self.root / self._filename(name.strip())

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        names = []
        for path in self.root.glob("*.json"):
            try:
                definition = json.loads(path.read_text())
                names.append(str(normalize_snom_material_definition(definition)["name"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(set(names), key=str.casefold)

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise ValueError(f"Custom material '{name}' is not in the local library.")
        try:
            return normalize_snom_material_definition(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Custom material '{name}' is not valid JSON.") from exc

    def save(self, definition: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_snom_material_definition(definition)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(str(normalized["name"]))
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return normalized

    def rename(self, old_name: str, new_name: str) -> dict[str, Any]:
        definition = self.load(old_name)
        definition["name"] = new_name
        saved = self.save(definition)
        if old_name != str(saved["name"]):
            self.delete(old_name)
        return saved

    def delete(self, name: str) -> None:
        self.load(name)
        self._path(name).unlink()

    def archive_json_bytes(self) -> bytes:
        """Serialize every local definition into one portable library archive."""
        materials = [self.load(name) for name in self.names()]
        return json.dumps(
            {
                "format": "snom-calculator-material-library",
                "version": 1,
                "materials": materials,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

    def import_archive(self, payload: bytes | str) -> list[str]:
        """Validate and restore a portable library archive, replacing name matches."""
        raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
        try:
            archive = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("The material-library archive is not valid JSON.") from exc
        if not isinstance(archive, Mapping) or archive.get("format") != "snom-calculator-material-library" or archive.get("version") != 1:
            raise ValueError("The uploaded JSON is not a SNOM material-library archive.")
        materials = archive.get("materials")
        if not isinstance(materials, list):
            raise ValueError("The material-library archive must contain a materials list.")
        normalized = [normalize_snom_material_definition(material) for material in materials]
        names = [str(definition["name"]) for definition in normalized]
        if len(set(names)) != len(names):
            raise ValueError("The material-library archive contains duplicate material names.")
        for definition in normalized:
            self.save(definition)
        return sorted(names, key=str.casefold)


class SessionMaterialLibrary(MaterialLibrary):
    """Keep custom definitions in one Streamlit session's state."""

    def __init__(self, definitions: MutableMapping[str, dict[str, Any]]) -> None:
        self.definitions = definitions

    def names(self) -> list[str]:
        return sorted(self.definitions, key=str.casefold)

    def load(self, name: str) -> dict[str, Any]:
        try:
            return deepcopy(self.definitions[name.strip()])
        except KeyError:
            raise ValueError(f"Custom material '{name}' is not in this session's library.") from None

    def save(self, definition: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_snom_material_definition(definition)
        self.definitions[str(normalized["name"])] = deepcopy(normalized)
        return normalized

    def delete(self, name: str) -> None:
        self.load(name)
        del self.definitions[name.strip()]


class JsonMaterialLibrary(SessionMaterialLibrary):
    """A session library backed by one portable JSON archive on local installs."""

    def __init__(self, definitions: MutableMapping[str, dict[str, Any]], path: Path, *, load: bool = True) -> None:
        super().__init__(definitions)
        self.path = path
        if load and path.exists():
            restored = SessionMaterialLibrary({})
            restored.import_archive(path.read_bytes())
            self.definitions.update(restored.definitions)

    def _commit(self, candidate: SessionMaterialLibrary) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(candidate.archive_json_bytes())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.definitions.clear()
        self.definitions.update(candidate.definitions)

    def save(self, definition: Mapping[str, Any]) -> dict[str, Any]:
        candidate = SessionMaterialLibrary(deepcopy(dict(self.definitions)))
        normalized = candidate.save(definition)
        self._commit(candidate)
        return normalized

    def delete(self, name: str) -> None:
        candidate = SessionMaterialLibrary(deepcopy(dict(self.definitions)))
        candidate.delete(name)
        self._commit(candidate)

    def import_archive(self, payload: bytes | str) -> list[str]:
        candidate = SessionMaterialLibrary(deepcopy(dict(self.definitions)))
        names = candidate.import_archive(payload)
        self._commit(candidate)
        return names


def material_library_for_state(state: MutableMapping[str, Any]) -> SessionMaterialLibrary:
    """Disk persistence is explicitly enabled by the local launcher only."""
    definitions = state.setdefault("_custom_materials", {})
    filename = os.environ.get("SNOM_MATERIAL_LIBRARY_FILE")
    if filename and "_local_material_library_error" not in state:
        try:
            library = JsonMaterialLibrary(definitions, Path(filename).expanduser(),
                                          load=state.get("_local_material_library_loaded") != filename)
        except (OSError, ValueError) as exc:
            # Never overwrite an unreadable file: this session keeps its materials in memory only.
            state["_local_material_library_error"] = f"{filename}: {exc}"
        else:
            state["_local_material_library_loaded"] = filename
            return library
    return SessionMaterialLibrary(definitions)
