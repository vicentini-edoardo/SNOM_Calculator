# Development

## Repository layout

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit entry point |
| `src/snom_calculator/app.py` | Page configuration and three-column workspace coordination |
| `_ui_state.py` | Session initialization, setup/library restoration and material/parameter resolution |
| `_ui_editors.py` | Tip, far-field and material dialogs, custom builder and dielectric previews |
| `_ui_structure.py` | Workbench styling, physical scene and visual stack controls |
| `_ui_controls.py` | Frequency/precision widgets, calculation action and progress/error handling |
| `_ui_results.py` | Spectrum plots, history, experimental imports and export controls |
| `_ui_quick.py` | Quick parameter edits for numeric tip, layer and custom-material values |
| `_ui_save.py` | Browser downloads, or a native Save As dialog when `SNOM_NATIVE_SAVE_DIALOG=1` |
| `calculations.py` | Framework-independent checked spectra, gap averaging, broadening and provenance |
| `_calculation_cache.py` | Streamlit profile caches and calculation adapters |
| `_spectra.py` | Shared pipeline for every engine: stack simplification, engine selection, frequency-block parallelism, demodulation, convergence checks, far field, normalization and results |
| `_quadrature.py` | Shared Legendre functions, Gauss rule, tapping heights and demodulation weights |
| `_preset_defaults.py` | Automatic and preset M, N and K, and the oscillator strength from max \|β\| |
| `_leading_blocks.py` | Full-N matrices kept during a checked run, sliced for the N/2 check |
| `src/snom_calculator/setup.py` | Setup defaults, validation and JSON interchange |
| `bulk.py`, `layer.py` in the package | Public calculation adapters (spheroid tip, scalar and tensor samples) |
| `_spheroid_engine.py` | Spheroid tip: surface tables, bulk eigen solver, GICM and momentum film solves |
| `_uniaxial_esm.py`, `_fim_coupling.py` | Spheroid uniaxial momentum integration: adaptive real axis and checked FIM |
| `_far_field.py` | Far-field factor: closed-form uniaxial Fresnel recursion and full tensor Maxwell scattering |
| `materials.py`, `custom_materials.py`, `material_library.py` | Catalog adapters, material definitions and storage |
| `io.py`, `models.py`, `_progress.py` | Table/CSV interchange, parameter/result types and progress callbacks |
| `docs/` | Model, installation and development guides |

Install pinned build tools with `python -m pip install -c constraints.txt setuptools wheel`,
then the editable app with `python -m pip install --no-build-isolation -c constraints.txt -e .`.
Run `python -m pip check`.

Every engine module exposes the same entry point,
`profile(layers, substrate, tip, gap, numerical, harmonics, *, method, previous=None)`, returning
the vertical tip dipole at each tapping height for unit drive (heights x
frequencies) and scalar diagnostics. `_spectra.calculate_spectra` turns the sample
and reference profiles into spectra; checked runs add M/2 and N/2 halving errors
and, for scalar bulk spheroids, a quadQ refinement. Film and stack
signals use the drive amplitude `_spectra.E0`; bulk signals use 1.

The app asks before each M or N doubling when the M/2 or N/2 error exceeds 20% in every preset, including Custom (preset M and N from `_preset_defaults.py`). When 2N exceeds `models.max_multipoles` for the tip shape, it only offers to keep the current N. The cache adapter retains the latest sample/reference
profiles for every gap, and passes
them to the engines as `previous`. The engines interleave those values with
new midpoint-height solves. Checked APIs report errors without changing M.
Above M = 64, `_quadrature.profile_grid` returns log-height Chebyshev nodes instead of
tapping heights, and callers demodulate with `profile_weights`. `solve_profile` doubles
the nodes through the same `previous` path while their Chebyshev tail exceeds
`HEIGHT_TAIL_LIMIT`, and re-solves with `height_nodes=0` (tapping heights) when the tail
stays above it. A profile whose length is not M + 1 holds nodes. An M refinement reuses a
node profile unchanged and extends a tapping one on the tapping grid.
`calculate_spectra(sample_gaps=..., reference_gaps=...)` averages several gaps: a side whose
gaps are `spanned` is solved once at `gap_span` with `M = SPAN_M` and `fallback=False`
(so the profile does not depend on M and M doublings hit the cache), else as a list of
per-gap profiles. Signal rows are gap-major blocks of harmonics
until the checks take the worst gap and the result takes the mean.

## Tests

`python -m pip install -e ".[test]"` then `python -m pytest -q`. The suite covers the engines, the calculation orchestration, file formats, setups and the Streamlit UI (`AppTest`). `tests/golden/` holds the per-solver golden snapshots; see its README. The PowerShell installer tests skip without PowerShell.

## Dependencies

Deploy on Python 3.13; 3.11-3.13 are supported. `pyproject.toml` defines the required packages and broad
version ranges (Streamlit floor 1.56). `constraints.txt` fixes the tested runtime and build versions, and
`requirements.txt` applies them. The Materials Library commit is fixed by its archive URL in `pyproject.toml`; the
Update launchers replace it with upstream HEAD. Install build
tools under the same constraints before using `--no-build-isolation`. Reproducible versions do not imply
bitwise-identical results across operating systems or BLAS implementations.

Calculation orchestration runs without Streamlit or Plotly. Its optional profile
solver callbacks let the Streamlit adapter reuse the same code with cached
profiles. Streamlit decorators, cache hashing and worker-key exclusions live in
`_calculation_cache.py`; engine-specific mathematics remains in the original
engine modules.

Within one server process, profile solves and convergence checks run one calculation
at a time to protect the shared geometry caches. Each solver still uses its own
threads and worker processes. Progress callbacks and stage windows are local to each
calculation context, so separate browser sessions do not overwrite one another's progress.

The UI modules use explicit imports and never import `app.py`. Structure uses
editors and state; editors reuse result plot styling and state helpers; calculation
controls call the existing cache adapter and save runs through result history.
`app.py` only wires these modules together; import helpers from the module that
defines them. Widget keys, labels, dialog callbacks and rendering order were preserved
during the split.

## Current implementation

The app caches raw height profiles in `_calculation_cache.py`, independently of harmonic selection and
worker count, and obtains its reference profile from the bulk eigen solver; the public layer engine uses the
same solver for its reference and for zero-thickness samples. `method="auto"` selects GICM for isotropic
layers and direct tensor integration for anisotropic layers.
