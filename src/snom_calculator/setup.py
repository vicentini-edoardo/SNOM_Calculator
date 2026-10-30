"""Validated setup interchange; transient browser state is never restored."""

from __future__ import annotations

import json
from io import BytesIO
import math
import re
from collections.abc import Mapping, MutableMapping

import numpy as np

from .material_library import normalize_snom_material_definition
from .materials import MATERIAL_TABLE_COLUMNS, parse_material_table
from .io import read_numeric_table

CALCULATED_HARMONICS = (0, 1, 2, 3, 4, 5)
# 2: tapping points renamed N -> M and multipole order m -> N; older files would swap them.
SETUP_VERSION = 2
_CUSTOM_BUILDER_PREFIX = "custom_material_builder_"


def _default_setup() -> dict[str, object]:
    """Return a fresh set of UI defaults for a bulk sample."""
    return _core_defaults()


def _core_defaults() -> dict[str, object]:
    return {
        "frequency_start": 1000.0,
        "frequency_end": 1400.0,
        "frequency_points": 100,
        "harmonic_0": False,
        "harmonic_1": False,
        "harmonic_2": False,
        "harmonic_3": True,
        "harmonic_4": False,
        "harmonic_5": False,
        "finite_layer_ids": [],
        "next_layer_id": 1,
        "tip_R": 25.0,
        "tip_L": 600.0,
        "tip_A": 40.0,
        "tip_Hs": 2.0,
        "tip_Hr": 2.0,
        "tip_Hs_max": 2.0,
        "tip_Hr_max": 2.0,
        "tip_reference_gap_count": 4,
        "tip_eps_real": -1e10,
        "tip_eps_imag": 0.0,
        "substrate_source": "Catalog",
        "substrate_catalog": "aSiO2",
        "substrate_variant": "",
        "substrate_real": 12.0,
        "substrate_imag": 0.0,
        "substrate_custom": "",
        "substrate_drude_enabled": False,
        "substrate_drude_wp_cm1": 0.0,
        "substrate_drude_gamma_cm1": 0.0,
        "reference_source": "Constant",
        "reference_disabled": False,
        "reference_catalog": "Si",
        "reference_variant": "",
        "reference_real": -5000.0,
        "reference_imag": 800.0,
        "reference_custom": "",
        "reference_drude_enabled": False,
        "reference_drude_wp_cm1": 0.0,
        "reference_drude_gamma_cm1": 0.0,
        "precision_mode": "Automatic",
        "preset": "Weak",
        "check_n": True,
        "force_fim": False,
        "bulk_custom_M": 80,
        "bulk_custom_N": 80,
        "layer_custom_M": 60,
        "layer_custom_N": 80,
        "layer_custom_K": 20,
        "workers": 1,
        "far_enabled": True,
        "far_angle": 30.0,
        "far_coefficient": 1.0,
        "average_gaps_enabled": False,
        "average_gap_min": 1.0,
        "average_gap_max": 4.0,
        "average_gap_count": 4,
        "resolution_enabled": False,
        "resolution_fwhm": 7.0,
    }


def _decode_setup(data: bytes) -> dict[str, object]:
    """Decode the setup object; field validation happens before applying it."""
    try:
        setup = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid setup JSON") from exc
    if not isinstance(setup, dict):
        raise ValueError("setup must be a JSON object")
    if setup.get("setup_version") != SETUP_VERSION:
        raise ValueError("Setup saved before the M (tapping points) / N (multipole order) rename; "
                         "it cannot be loaded. Recreate it and save it again.")
    return setup


def _apply_setup(state: MutableMapping[str, object], setup: Mapping[str, object]) -> None:
    """Validate a complete setup before changing the caller's session state."""
    values = _default_setup()
    for key in values:
        if key in setup:
            values[key] = setup[key]
    # Recognize the former defaults before dropping the unused quadrature setting.
    if tuple(setup.get(key) for key in ("bulk_custom_M", "bulk_custom_quad", "bulk_custom_N")) == (30, 16, 8):
        values.update(bulk_custom_M=80, bulk_custom_N=80)
    if "precision_mode" not in setup and "preset" in setup:  # saved before Automatic: keep the chosen preset
        values["precision_mode"] = "Custom" if values["preset"] == "Custom" else "Presets"
    if values["preset"] == "Custom":
        values["preset"] = "Weak"
    for key in ("tip_Hs", "tip_Hr"):
        if f"{key}_max" not in setup:
            values[f"{key}_max"] = values[key]
    if "tip_Hs_max" not in setup and setup.get("average_gaps_enabled", False):
        values["tip_Hs"] = values["average_gap_min"]
        values["tip_Hs_max"] = values["average_gap_max"]
    frequency = setup.get("frequency")
    if isinstance(frequency, Mapping):
        for source, target in (("start", "frequency_start"), ("end", "frequency_end"), ("points", "frequency_points")):
            if source in frequency:
                values[target] = frequency[source]
    layer_fields = "source|catalog|variant|real|imag|custom|thickness|drude_enabled|drude_wp_cm1|drude_gamma_cm1"

    for key, value in setup.items():
        if re.fullmatch(rf"layer_[1-9]\d*_(?:{layer_fields})", key):
            values[key] = value

    try:
        raw_tables = setup.get("material_tables", {})
        if not isinstance(raw_tables, Mapping):
            raise ValueError
        uploads = {}
        for prefix, table in raw_tables.items():
            if not re.fullmatch(r"(?:substrate|layer_[1-9]\d*)", prefix) or not isinstance(table, Mapping):
                raise ValueError
            text, filename = table["text"], table["filename"]
            if not isinstance(text, str) or not isinstance(filename, str):
                raise ValueError
            upload = BytesIO(text.encode("utf-8"))
            upload.name = filename
            numeric = read_numeric_table(upload, headerless_columns=MATERIAL_TABLE_COLUMNS)
            upload.seek(0)
            parse_material_table(upload, numeric["frequency_cm1"], filename)
            upload.seek(0)
            uploads[f"{prefix}_upload"] = upload
        raw_materials = setup.get("material_definitions", {})
        if not isinstance(raw_materials, Mapping):
            raise ValueError
        materials = {str(name): normalize_snom_material_definition(definition)
                     for name, definition in raw_materials.items()}
        if any(name != definition["name"] for name, definition in materials.items()):
            raise ValueError
        for key, default in _default_setup().items():
            if isinstance(default, bool):
                if not isinstance(values[key], bool):
                    raise ValueError
            elif isinstance(default, int):
                values[key] = _integer(values[key])
            elif isinstance(default, float):
                values[key] = _number(values[key])
            elif isinstance(default, str) and not isinstance(values[key], str):
                raise ValueError
        start = _number(values["frequency_start"])
        end = _number(values["frequency_end"])
        points = _integer(values["frequency_points"])
        if start < 1 or end < 2 or start >= end or points < 2 or points > 1000:
            raise ValueError
        layer_ids = values["finite_layer_ids"]
        if not isinstance(layer_ids, list) or any(
            isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id <= 0
            for layer_id in layer_ids
        ) or len(set(layer_ids)) != len(layer_ids):
            raise ValueError
        if _integer(values["next_layer_id"]) <= max(layer_ids, default=0):
            raise ValueError
        for layer_id in layer_ids:
            prefix = f"layer_{layer_id}"

            source = values.get(f"{prefix}_source")
            fields = {
                "Catalog": ("catalog", "thickness"),
                "Constant": ("real", "imag", "thickness"),
                "Upload CSV/TXT": ("thickness",),
                "Custom library": ("custom", "thickness"),
            }.get(source)
            if fields is None or any(
                f"{prefix}_{field}" not in values
                or (field == "custom" and not str(values[f"{prefix}_{field}"]).strip())
                for field in fields
            ):
                raise ValueError
        if any(
            not isinstance(values[f"harmonic_{harmonic}"], bool)
            for harmonic in CALCULATED_HARMONICS
        ):
            raise ValueError
        if _number(values["tip_A"]) < 0:
            raise ValueError
        if _number(values["tip_L"]) <= 0:
            raise ValueError
        for key in ("tip_R", "tip_Hs", "tip_Hr"):
            if _number(values[key]) <= 0:
                raise ValueError
        for key in ("tip_Hs", "tip_Hr"):
            if _number(values[f"{key}_max"]) < _number(values[key]):
                raise ValueError
        if not 2 <= _integer(values["tip_reference_gap_count"]) <= 25:
            raise ValueError

        for key, value in values.items():
            if re.fullmatch(r"layer_[1-9]\d*_(?:real|imag)", key):
                values[key] = _number(value)
            if re.fullmatch(r"(?:layer_[1-9]\d*_thickness|(?:layer_[1-9]\d*|substrate|reference)_drude_(?:wp|gamma)_cm1)", key) and _number(value) < 0:
                raise ValueError
            if key.endswith("_drude_enabled") and not isinstance(value, bool):
                raise ValueError

        if not 1 <= _integer(values["workers"]) <= 4:
            raise ValueError
        if (values["precision_mode"] not in ("Automatic", "Presets", "Custom")
                or values["preset"] not in ("Weak", "Strong", "Very strong") or not 0.1 <= values["far_angle"] <= 89.9):
            raise ValueError
        if any(values[key] < 1 for key in (
            "bulk_custom_M", "bulk_custom_N", "layer_custom_M", "layer_custom_N", "layer_custom_K",
        )):
            raise ValueError
        if not isinstance(values["average_gaps_enabled"], bool) or not isinstance(values["resolution_enabled"], bool):
            raise ValueError
        if (_number(values["average_gap_min"]) < 0.01
                or _number(values["average_gap_max"]) < _number(values["average_gap_min"])
                or not 2 <= _integer(values["average_gap_count"]) <= 25
                or _number(values["resolution_fwhm"]) < 0.01):
            raise ValueError

        for prefix, allow_upload in (("substrate", True), ("reference", False)):
            source = values[f"{prefix}_source"]
            fields = {
                "Catalog": ("catalog",),
                "Constant": ("real", "imag"),
                "Upload CSV/TXT": (),
                "Custom library": ("custom",),
            }.get(source)
            if fields is None or (source == "Upload CSV/TXT" and not allow_upload):
                raise ValueError
            if any(not str(values[f"{prefix}_{field}"]).strip() for field in fields):
                raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Invalid setup values") from None

    values["frequency_start"] = start
    values["frequency_end"] = end
    values["frequency_points"] = points
    state.update(values)
    state.pop("bulk_custom_quad", None)
    state.pop("layer_custom_quad", None)
    # Merge in place: replacing the dict would drop the session library, and the
    # disk-backed library would then save only the setup's materials.
    state.setdefault("_custom_materials", {}).update(materials)
    for key in tuple(state):
        if re.fullmatch(r"(?:substrate|layer_[1-9]\d*)_upload", str(key)):
            state.pop(key)
    state.update(uploads)


def _number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _integer(value: object) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value:
        raise ValueError
    return integer


def _encode_setup(state: Mapping[str, object]) -> str:
    def serializable(value: object) -> object:
        if isinstance(value, np.generic):  # numpy scalars are not int/float: keep them numeric in JSON
            value = value.item()
        if isinstance(value, Mapping):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        filename = getattr(value, "name", None)
        return filename if isinstance(filename, str) else str(value)

    # Keep results, uploads and dialog state out of saved setups.
    transient = {
        "bulk_custom_quad",
        "layer_custom_quad",
        "_custom_defaults_migrated",
        "results",
        "results_history",
        "m_refinement_request",
        "m_refinement_next",
        "selected_history_id",
        "history_seq",
        "experimental_overlays",
        "setup_upload",
        "_setup_upload_id",
        "material_library_upload",
        "_material_library_upload_id",
        "experimental_overlay_upload",
        "experimental_import_open",
        "editor",
        "editor_snapshot",
        "editor_restore",
        "editor_cleanup",
        "_custom_materials",
        "_local_material_library_loaded",
        "_local_material_library_error",
    }
    saved = {key: value for key, value in state.items()
                      if key not in transient
                      and not key.startswith(("experimental_import_", _CUSTOM_BUILDER_PREFIX, "draft_", "quick_"))
                      and not key.endswith("_delete_custom_confirm")}
    tables = {}
    for key, value in state.items():
        if re.fullmatch(r"(?:substrate|layer_[1-9]\d*)_upload", key) and hasattr(value, "getvalue"):
            content = value.getvalue()
            text = content.decode("utf-8-sig") if isinstance(content, bytes) else str(content)
            tables[key[:-7]] = {"filename": str(getattr(value, "name", "material.csv")), "text": text}
    if tables:
        saved["material_tables"] = tables
    if definitions := state.get("_custom_materials"):
        saved["material_definitions"] = definitions
    saved["setup_version"] = SETUP_VERSION
    return json.dumps(
        serializable(saved),
        indent=2,
        ensure_ascii=False,
    )
