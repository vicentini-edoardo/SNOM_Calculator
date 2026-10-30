"""Material-library adapter and simple table parser."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import BinaryIO, Mapping

import numpy as np

from materials_library import eps_tensor, load, names, variants

from .custom_materials import build_custom_axes

from .io import read_numeric_table
from .material_library import normalize_snom_material_definition


CM1_TO_HZ = 299_792_458.0 * 100.0
MATERIAL_TABLE_COLUMNS = {
    3: ("frequency_cm1", "eps_real", "eps_imag"),
    5: ("frequency_cm1", "eps_parallel_real", "eps_parallel_imag",
        "eps_perpendicular_real", "eps_perpendicular_imag"),
}


def cm1_to_hz(value_cm1: np.ndarray) -> np.ndarray:
    return np.asarray(value_cm1) * CM1_TO_HZ


@dataclass(frozen=True)
class MaterialSpectrum:
    """Unrotated isotropic or uniaxial response.

    Historical field names: ``parallel`` is principal epsilon_zz,
    ``perpendicular`` is principal epsilon_xx, and ``secondary_parallel`` is
    epsilon_yy. Permittivities are dimensionless complex arrays on the same
    cm^-1 grid; the supported convention is passive Im(epsilon) >= 0.
    """

    parallel: np.ndarray
    perpendicular: np.ndarray
    label: str
    secondary_parallel: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parallel = np.asarray(self.parallel, dtype=complex)
        perpendicular = np.asarray(self.perpendicular, dtype=complex)
        if parallel.shape != perpendicular.shape or parallel.ndim != 1:
            raise ValueError("Material axes must be one-dimensional arrays with matching shapes.")
        if not np.isfinite(parallel).all() or not np.isfinite(perpendicular).all():
            raise ValueError(f"{self.label}: permittivity contains non-finite values.")
        object.__setattr__(self, "parallel", parallel)
        object.__setattr__(self, "perpendicular", perpendicular)
        if self.secondary_parallel is not None:
            secondary = np.asarray(self.secondary_parallel, dtype=complex)
            if secondary.shape != parallel.shape:
                raise ValueError("Material axes must have matching shapes.")
            if not np.isfinite(secondary).all():
                raise ValueError(f"{self.label}: permittivity contains non-finite values.")
            object.__setattr__(self, "secondary_parallel", secondary)
        if self.biaxial:
            raise ValueError("Biaxial materials are unsupported.")

    def take(self, ids) -> "MaterialSpectrum":
        """The same material on a subset of the frequency grid."""
        return replace(self, parallel=self.parallel[ids], perpendicular=self.perpendicular[ids],
                       secondary_parallel=(None if self.secondary_parallel is None
                                           else self.secondary_parallel[ids]))

    @property
    def isotropic(self) -> bool:
        return not self.biaxial and np.allclose(self.parallel, self.perpendicular, rtol=1e-7, atol=1e-9)

    @property
    def biaxial(self) -> bool:
        return self.secondary_parallel is not None and not np.allclose(
            self.perpendicular, self.secondary_parallel, rtol=1e-7, atol=1e-9)

    @property
    def tensor(self) -> np.ndarray:
        axes = np.column_stack((self.perpendicular, self.perpendicular, self.parallel))
        tensor = np.zeros((len(self.parallel), 3, 3), dtype=complex)
        tensor[:, np.arange(3), np.arange(3)] = axes
        return tensor

    @property
    def axisymmetric(self) -> bool:
        tensor = self.tensor
        scale = np.maximum(1.0, np.max(np.abs(tensor), axis=(1, 2)))
        return (np.allclose(tensor[:, 0, 0], tensor[:, 1, 1], rtol=1e-7, atol=1e-9)
                and np.all(np.abs(tensor[:, [0, 0, 1], [1, 2, 2]]) <= 1e-9 * scale[:, None]))


def _passive(values: np.ndarray, label: str, *, normalize_mixed: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    imag = values.imag
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values))))
    if np.all(imag >= -tolerance):
        return values
    if np.all(imag <= tolerance):
        return np.conjugate(values)
    if normalize_mixed:
        return np.where(imag < -tolerance, np.conjugate(values), values)
    raise ValueError(f"{label}: active or mixed-sign Im(epsilon) spectra are unsupported.")


def effective_permittivity(material: MaterialSpectrum) -> np.ndarray:
    if not material.axisymmetric:
        raise ValueError(f"{material.label}: only unrotated isotropic or uniaxial materials are supported.")
    if material.isotropic:
        return material.parallel.copy()
    # Separate passive roots retain the negative sign when both lossless axes are negative.
    epsilon = np.sqrt(material.parallel) * np.sqrt(material.perpendicular)
    epsilon = np.where(epsilon.imag < 0, -epsilon, epsilon)
    return epsilon


def catalog_names() -> tuple[str, ...]:
    tensors = {"isotropic", "uniaxial"}
    return tuple(name for name, tensor in _catalog_tensors() if tensor in tensors)


@lru_cache(maxsize=1)
def _catalog_tensors() -> tuple[tuple[str, str], ...]:
    """(name, tensor kind) of every usable catalog entry."""
    return tuple(
        (name, definition["tensor"]) for name in names()
        if (definition := load(name))["status"] != "pending" and name != "vacuum"
    )


def catalog_variants(name: str) -> tuple[str, ...]:
    """Alternative parameter sets of a catalog material; the default set is not listed."""
    return variants(name) if name in catalog_names() else ()


def catalog_variant_label(name, variant):
    """Display name of a parameter set; the default set is named by its sources."""
    return variant or " + ".join(load(name).get("sources", ())) or "Default"


def catalog_material(name: str, frequency_cm1: np.ndarray, variant: str = "") -> MaterialSpectrum:
    frequency = np.asarray(frequency_cm1, dtype=float)
    if name not in catalog_names():
        raise ValueError(f"Unsupported catalog material '{name}'.")
    if variant and variant not in variants(name):
        raise ValueError(f"'{name}' has no parameter set '{variant}'.")
    definition = load(name, variant or None)
    sources = definition.get("sources")
    references = [ref for ref in definition.get("references", []) if sources is None or ref["key"] in sources]
    tensor = eps_tensor(definition, frequency)
    return MaterialSpectrum(
        parallel=tensor[:, 2, 2],
        perpendicular=tensor[:, 0, 0],
        secondary_parallel=tensor[:, 1, 1],
        label=f"{name} ({variant})" if variant else name,
        metadata={"source": "Catalog", "definition": definition, "variant": variant, "references": references},
    )


def _evaluate_axis(function, frequency_hz: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(function(frequency_hz), dtype=complex)
    except ValueError:
        values = np.asarray([function(value) for value in frequency_hz], dtype=complex)
    return np.broadcast_to(values, frequency_hz.shape).copy()


def constant_material(value: complex, frequency_cm1: np.ndarray, label: str) -> MaterialSpectrum:
    values = _passive(
        np.full(np.asarray(frequency_cm1).shape, complex(value), dtype=complex),
        label,
    )
    return MaterialSpectrum(values, values, label=label,
                            metadata={"source": "Constant", "definition": {"epsilon": complex(values[0])},
                                      "references": []})


def drude_permittivity(frequency_cm1: np.ndarray, wp_cm1: float, gamma_cm1: float) -> np.ndarray:
    frequency = np.asarray(frequency_cm1, dtype=float)
    return -wp_cm1**2 / (frequency**2 + 1j * gamma_cm1 * frequency)


def with_drude(
    spectrum: MaterialSpectrum, frequency_cm1: np.ndarray, wp_cm1: float, gamma_cm1: float
) -> MaterialSpectrum:
    """Add one isotropic Drude term to every tensor axis."""
    if not wp_cm1:
        return spectrum
    term = drude_permittivity(frequency_cm1, wp_cm1, gamma_cm1)
    secondary = spectrum.secondary_parallel
    return replace(
        spectrum,
        parallel=spectrum.parallel + term,
        perpendicular=spectrum.perpendicular + term,
        secondary_parallel=None if secondary is None else secondary + term,
        label=f"{spectrum.label} + Drude",
        metadata={**spectrum.metadata, "added_drude": {"wp_cm1": wp_cm1, "gamma_cm1": gamma_cm1}},
    )


def custom_material(
    definition: dict[str, object], frequency_cm1: np.ndarray
) -> MaterialSpectrum:
    """Evaluate a saved upstream-compatible definition on SNOM's frequency grid."""
    normalized = normalize_snom_material_definition(definition)
    frequency = np.asarray(frequency_cm1, dtype=float)
    fx, fy, fz, _ = build_custom_axes(normalized)
    hz = cm1_to_hz(frequency)
    drude = normalized.get("drude", {})

    def evaluate(function, axis: str) -> np.ndarray:
        values = _evaluate_axis(function, hz)
        term = drude.get(axis, {})
        wp, gamma = float(term.get("wp_cm1", 0.0)), float(term.get("gamma_cm1", 0.0))
        return values + drude_permittivity(frequency, wp, gamma) if wp else values

    return MaterialSpectrum(
        parallel=_passive(evaluate(fz, "zz"), str(normalized["name"])),
        perpendicular=_passive(evaluate(fx, "xx"), str(normalized["name"])),
        secondary_parallel=_passive(evaluate(fy, "yy"), str(normalized["name"])),
        label=str(normalized["name"]),
        metadata={"source": "Custom library", "definition": normalized,
                  "references": normalized.get("references", [])},
    )


def parse_material_axes(file: BinaryIO) -> tuple[str, dict[str, dict[str, list[float]]]]:
    """Split a 3- or 5-column material table into custom-library axis tables.

    Returns the tensor type ("Isotropic", "Uniaxial") and xx/yy/zz
    tables; uniaxial parallel is zz and perpendicular is xx = yy.
    """
    try:
        table = read_numeric_table(file, headerless_columns=MATERIAL_TABLE_COLUMNS)
    except Exception as exc:
        raise ValueError("Could not read the material table.") from exc
    names = set(table.dtype.names or ())

    def axis(column: str) -> dict[str, list[float]]:
        return {"frequency_cm1": np.atleast_1d(table["frequency_cm1"]).tolist(),
                "eps_real": np.atleast_1d(table[f"{column}_real"]).tolist(),
                "eps_imag": np.atleast_1d(table[f"{column}_imag"]).tolist()}

    if "frequency_cm1" not in names:
        raise ValueError("The material table needs a frequency_cm1 column.")
    if {"eps_real", "eps_imag"} <= names:
        return "Isotropic", {name: axis("eps") for name in ("xx", "yy", "zz")}
    if {"eps_parallel_real", "eps_parallel_imag", "eps_perpendicular_real", "eps_perpendicular_imag"} <= names:
        return "Uniaxial", {"xx": axis("eps_perpendicular"), "yy": axis("eps_perpendicular"), "zz": axis("eps_parallel")}
    raise ValueError("Expected isotropic or parallel/perpendicular real/imag columns.")


def parse_material_table(
    file: BinaryIO,
    frequency_cm1: np.ndarray,
    label: str,
) -> MaterialSpectrum:
    try:
        table = read_numeric_table(file, headerless_columns=MATERIAL_TABLE_COLUMNS)
    except Exception as exc:
        raise ValueError(f"{label}: could not read the material table.") from exc
    names = set(table.dtype.names or ())
    if any(not np.isfinite(table[name]).all() for name in names):
        raise ValueError(f"{label}: material table contains non-finite values.")
    frequency = np.atleast_1d(table["frequency_cm1"]) if "frequency_cm1" in names else None
    if frequency is None or frequency.size < 2 or not np.all(np.diff(frequency) > 0):
        raise ValueError(f"{label}: frequency_cm1 must contain at least two increasing values.")
    target = np.asarray(frequency_cm1, dtype=float)
    if target.ndim != 1 or not target.size or not np.isfinite(target).all():
        raise ValueError(f"{label}: calculation frequencies must be finite and one-dimensional.")
    if np.min(target) < frequency[0] or np.max(target) > frequency[-1]:
        raise ValueError(f"{label}: the calculation range must stay inside the uploaded range.")

    isotropic = {"eps_real", "eps_imag"}
    metadata = {
        "source": "Upload CSV/TXT", "filename": label, "references": [],
        "definition": {name: np.atleast_1d(table[name]).tolist() for name in table.dtype.names},
    }
    uniaxial = {
        "eps_parallel_real",
        "eps_parallel_imag",
        "eps_perpendicular_real",
        "eps_perpendicular_imag",
    }
    if isotropic <= names:
        values = _interpolate_complex(
            frequency, table["eps_real"] + 1j * table["eps_imag"], target
        )
        values = _passive(values, label)
        return MaterialSpectrum(values, values, label=label, metadata=metadata)
    if uniaxial <= names:
        parallel = _passive(
            _interpolate_complex(
                frequency,
                table["eps_parallel_real"] + 1j * table["eps_parallel_imag"],
                target,
            ),
            label,
        )
        perpendicular = _passive(
            _interpolate_complex(
                frequency,
                table["eps_perpendicular_real"] + 1j * table["eps_perpendicular_imag"],
                target,
            ),
            label,
        )
        return MaterialSpectrum(parallel, perpendicular, label=label, metadata=metadata)
    raise ValueError(
        f"{label}: expected isotropic or parallel/perpendicular real/imag columns."
    )


def _interpolate_complex(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.interp(target, x, np.real(y)) + 1j * np.interp(target, x, np.imag(y))
