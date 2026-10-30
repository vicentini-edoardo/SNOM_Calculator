"""Framework-independent checked spectra and optional spectral processing.

Frequency grids and broadening widths are in cm^-1; geometry and gaps are in
nm. The profile solver (``_spectra.solve_profile``) can be replaced by the
Streamlit adapter's cached version; the default requires no UI or session state.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ._progress import report, stage
from ._spectra import calculate_spectra, solve_profile
from .materials import MaterialSpectrum
from .models import FarFieldSettings, NumericalSettings, SpectrumResult, TipParameters


def validate_structure(model: str, materials: dict[str, object]) -> None:
    """Validate the material roles resolved by the workspace before solving."""
    required = ("sample", "reference") if model == "bulk" else (
        ("substrate", "reference") if model == "multilayer" else ("layer", "substrate", "reference")
    )
    if materials.get("reference_disabled", False):
        required = tuple(key for key in required if key != "reference")
    missing = [key for key in required if not isinstance(materials.get(key), MaterialSpectrum)]
    if missing:
        raise ValueError("Select or upload every material before calculating.")
    if model == "layer" and float(materials["thickness_nm"]) < 0:
        raise ValueError("Layer thickness cannot be negative.")
    if model == "multilayer" and not materials["layers"]:
        raise ValueError("Add at least one layer before calculating.")
    if model == "multilayer" and any(
        not isinstance(material, MaterialSpectrum) or thickness < 0
        for material, thickness in materials["layers"]
    ):
        raise ValueError("Select every layer material and use non-negative thicknesses.")


def calculate_bulk_checked(
    frequency: np.ndarray,
    sample: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: tuple[int, ...],
    far_field: FarFieldSettings,
    check_n: bool = True,
    *,
    solve=solve_profile,
    sample_gaps=None,
    reference_gaps=None,
) -> dict[int, SpectrumResult]:
    """Bulk spectra with M, optional N and engine-specific convergence checks."""
    return calculate_spectra(frequency, (), sample, reference, tip, numerical, harmonics, far_field,
                             checked=True, check_n=check_n, solve=solve,
                             sample_gaps=sample_gaps, reference_gaps=reference_gaps)


def calculate_layer_checked(
    frequency: np.ndarray,
    layer: MaterialSpectrum,
    substrate: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    thickness_nm: float,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: tuple[int, ...],
    far_field: FarFieldSettings,
    workers: int,
    check_n: bool = True,
    *,
    solve=solve_profile,
    method: str = "auto",
    sample_gaps=None,
    reference_gaps=None,
) -> dict[int, SpectrumResult]:
    """Checked single-film spectra; zero thickness is the bulk substrate."""
    return calculate_stack_checked(frequency, ((layer, thickness_nm),), substrate, reference, tip,
                                   numerical, harmonics, far_field, workers, check_n, solve=solve, method=method,
                                   sample_gaps=sample_gaps, reference_gaps=reference_gaps)


def calculate_stack_checked(
    frequency: np.ndarray,
    layers: tuple[tuple[MaterialSpectrum, float], ...],
    substrate: MaterialSpectrum,
    reference: MaterialSpectrum | None,
    tip: TipParameters,
    numerical: NumericalSettings,
    harmonics: tuple[int, ...],
    far_field: FarFieldSettings,
    workers: int,
    check_n: bool = True,
    *,
    solve=solve_profile,
    method: str = "auto",
    sample_gaps=None,
    reference_gaps=None,
) -> dict[int, SpectrumResult]:
    """Checked top-to-bottom films; redundant interfaces are removed first."""
    return calculate_spectra(frequency, layers, substrate, reference, tip, numerical, harmonics,
                             far_field, workers=workers, method=method, checked=True, check_n=check_n, solve=solve,
                             sample_gaps=sample_gaps, reference_gaps=reference_gaps)


def _build_provenance(
    model: str, materials: Mapping[str, object], controls: Mapping[str, object],
    gaps, fwhm, frequency,
) -> dict[str, object]:
    """Structure, tip, numerical and citation metadata attached to every result."""
    references = [
        {"url": "https://github.com/vicentini-edoardo/SNOM_Calculator"},
        {"url": "https://github.com/vicentini-edoardo/Materials_Library"},
        {"citation": "K. V. Voronin, I. Herrero León, R. Hillenbrand, A. Y. Nikitin, Quantitative Analytical Spheroid Model for Scattering-Type Scanning Near-Field Optical Spectroscopy, Adv. Opt. Mater. 13, e01539 (2025)",
         "doi": "10.1002/adom.202501539", "scope": "Bulk spheroid model"},
        {"citation": "K. V. Voronin, E. Vicentini, L. Mester, R. Hillenbrand, A. Y. Nikitin, Quantitative Analytical Model for Scattering-type Scanning Near-field Optical Spectroscopy of Layered Samples (DOI forthcoming)",
         "scope": "Finite-layer model (GICM and FIM)"},
        {"citation": "B.-Y. Jiang et al., Generalized spectral method for near-field optical microscopy, J. Appl. Phys. 119, 054305 (2016)",
         "doi": "10.1063/1.4941343", "url": "https://arxiv.org/abs/1503.00221",
         "scope": "General quasistatic spheroid spectral framework"},
    ]

    def describe(material: MaterialSpectrum) -> dict[str, object]:
        for reference in material.metadata.get("references", []):
            if reference not in references:
                references.append(reference)
        # Identify the material instead of listing epsilon per frequency.
        metadata = material.metadata
        source = metadata.get("source")
        description = {"label": material.label, "source": source}
        if source == "Catalog":
            description.update(name=metadata["definition"]["name"], variant=metadata.get("variant", ""),
                               references=metadata.get("references", []))
        elif source == "Constant":
            description["epsilon"] = metadata["definition"]["epsilon"]
        elif source == "Custom library":
            description["name"] = metadata["definition"]["name"]
        elif source == "Upload CSV/TXT":
            description["filename"] = metadata.get("filename")
        if "added_drude" in metadata:
            description["added_drude"] = metadata["added_drude"]
        return description

    structure = {"superstrate": {"label": "Air", "epsilon": 1.0, "semi_infinite": True},
                 "model": model, "layer_order": "top to bottom"}
    if model == "bulk":
        structure["sample"] = {**describe(materials["sample"]), "semi_infinite": True}
    else:
        layers = (materials["layers"] if model == "multilayer" else
                  ((materials["layer"], float(materials["thickness_nm"])),))
        structure["layers"] = [{"material": describe(material), "thickness_nm": depth}
                               for material, depth in layers]
        structure["substrate"] = {**describe(materials["substrate"]), "semi_infinite": True}
    reference_material = (None if materials["reference"] is None else
                          {**describe(materials["reference"]), "semi_infinite": True})
    tip = controls["tip"]
    tip_geometry = asdict(tip)
    # The engines use only the scalar tip.epsilon, so record that rather than controls["tip_material"].
    tip_values = np.broadcast_to(tip.epsilon, frequency.shape)
    tip_description = describe(MaterialSpectrum(tip_values, tip_values, "Tip", metadata={
        "source": "Constant", "definition": {"epsilon": tip.epsilon},
    }))
    metadata = {
        "model": model, "preset": controls["preset"],
        "tip": tip_geometry,
        "tip_material": tip_description,
        "illumination": {**asdict(controls["far_field"]), "polarization": "p"},
        "numerical": asdict(controls["numerical"]),
        "check_n": controls.get("check_n", False), "workers": controls.get("workers", 1),
        "film_method": controls.get("method", "auto"),
        "average_gaps": gaps, "average_reference_gaps": controls.get("average_reference_gaps"),
        "resolution_fwhm_cm1": fwhm,
        "sample_structure": structure, "reference_material": reference_material,
        "references": references,
        "normalization": "none" if reference_material is None else "reference",
    }
    return metadata


def run_calculations(
    model: str,
    frequency: np.ndarray,
    materials: Mapping[str, object],
    controls: Mapping[str, object],
    harmonics: Sequence[int],
    *,
    solve_bulk=calculate_bulk_checked,
    solve_layer=calculate_layer_checked,
    solve_stack=calculate_stack_checked,
) -> dict[int, SpectrumResult]:
    """Average complex sample/reference signals independently, then broaden contrast."""
    gaps = controls.get("average_gaps")
    reference_gaps = controls.get("average_reference_gaps")

    def gap_grid(gap_range, label):
        if gap_range is None:
            return [None]
        minimum, maximum, count = gap_range
        if (not np.isfinite([minimum, maximum]).all() or minimum <= 0
                or maximum < minimum or isinstance(count, bool)
                or not isinstance(count, numbers.Integral) or not 2 <= count <= 25):
            raise ValueError(f"Use a positive, increasing {label} gap range with 2–25 distances.")
        return [minimum] if minimum == maximum else np.linspace(minimum, maximum, count)

    sample_gaps = gap_grid(gaps, "sample")
    reference_grid = gap_grid(reference_gaps, "reference")
    if materials.get("reference") is None:
        reference_grid = [None]

    tip = controls["tip"]
    if sample_gaps[0] is not None:
        tip = replace(tip, sample_gap_nm=float(sample_gaps[0]))
    if reference_grid[0] is not None:
        tip = replace(tip, reference_gap_nm=float(reference_grid[0]))
    common = dict(
        frequency=frequency,
        reference=materials["reference"],
        tip=tip,
        numerical=controls["numerical"],
        harmonics=tuple(harmonics),
        far_field=controls["far_field"],
        check_n=controls.get("check_n", False),
    )
    # The solvers average the complex signals of several gaps (sharing their tip heights when they can).
    if len(sample_gaps) > 1:
        common["sample_gaps"] = tuple(float(gap) for gap in sample_gaps)
    if len(reference_grid) > 1:
        common["reference_gaps"] = tuple(float(gap) for gap in reference_grid)
    fwhm = controls.get("resolution_fwhm_cm1")
    if fwhm is not None:  # validate before the expensive solve
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError("Gaussian FWHM must be positive.")
        steps = np.diff(frequency)
        if steps.size == 0 or not np.allclose(steps, steps[0]):
            raise ValueError("Gaussian convolution requires an evenly spaced frequency grid.")
        sigma_points = fwhm / (2 * np.sqrt(2 * np.log(2)) * steps[0])
    report("Preparing calculation", 0.05)
    with stage("", 0.05, 0.9):
        if model == "bulk":
            results = solve_bulk(sample=materials["sample"], **common)
        else:
            film = {"method": controls["method"]} if controls.get("method", "auto") != "auto" else {}
            if model == "layer":
                results = solve_layer(
                    layer=materials["layer"], substrate=materials["substrate"],
                    thickness_nm=float(materials["thickness_nm"]),
                    workers=controls["workers"], **common, **film,
                )
            else:
                results = solve_stack(
                    layers=materials["layers"], substrate=materials["substrate"],
                    workers=controls["workers"], **common, **film,
                )
    report("Averaging gap signals" if "sample_gaps" in common or "reference_gaps" in common
           else "Processing results", 0.9)
    for harmonic, result in tuple(results.items()):
        if gaps is None and reference_gaps is None and fwhm is None:
            continue
        sample, reference, contrast = result.sample_harmonic, result.reference_harmonic, result.contrast
        metadata = dict(result.metadata)
        if gaps is not None:
            metadata["sample_gaps_nm"] = tuple(float(gap) for gap in sample_gaps)
        if reference_gaps is not None and materials["reference"] is not None:
            metadata["reference_gaps_nm"] = tuple(float(gap) for gap in reference_grid)
        if fwhm is not None:
            contrast = gaussian_filter1d(contrast, sigma_points, mode="reflect")
            metadata["resolution_fwhm_cm1"] = fwhm
        results[harmonic] = SpectrumResult.from_contrast(
            frequency_cm1=result.frequency_cm1,
            sample_harmonic=(sample if fwhm is None or controls["tip"].amplitude_nm == 0 and harmonic > 0
                             else contrast * reference),
            reference_harmonic=reference,
            contrast=contrast,
            permittivities=result.permittivities,
            metadata=metadata,
        )
    metadata = _build_provenance(model, materials, controls, gaps, fwhm, frequency)
    for result in results.values():
        result.metadata.update(metadata)
    return results
