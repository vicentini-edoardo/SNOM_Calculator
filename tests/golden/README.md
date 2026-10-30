# Numerical reference fixtures

Keep these development fixtures. They are current-behaviour snapshots used by
`test_calculation_regression.py`, one file per solver:

| File | Solver | Cases |
| --- | --- | --- |
| `solver_bulk.npz` | bulk spheroid | `bulk`, `bulk-no-field`, `bulk-tensor`, `layer-zero`, `checked-bulk` |
| `solver_gicm.npz` | GICM films and stacks | `layer`, `stack`, `stack-collapsed`, `checked-layer`, `checked-stack` |
| `solver_fim.npz` | FIM (uniaxial and hyperbolic films, direct isotropic) | `layer-uniaxial`, `layer-hyperbolic`, `direct-isotropic` |
| `solver_real_axis.npz` | adaptive real-axis integral | `adaptive-uniaxial` |

Keys are `<case>/<array>`. The cases are defined in `test_calculation_regression.py`:
bulk eigen/direct tip response, isotropic film/stack GICM, uniaxial and hyperbolic
FIM response, adaptive integration, zero thickness, merged films, and checked
gap-averaged, Gaussian-broadened spectra. The processed cases also run without UI
caches. They were captured with Python 3.13 and NumPy 2.4.6.

Snapshots protect raw sample/reference harmonics, contrast, amplitude, phase,
permittivities, low-level height profiles, shapes, dtypes and convergence values.
They use `rtol=1e-10, atol=0`, protecting tiny dipoles without an absolute floor.
This allowance accommodates floating-point library variation; it is not a
physical accuracy estimate. A 0.1% perturbation of one raw sample harmonic fails
the check.

These are current-behaviour fixtures, **not independent physics references**.
Do not regenerate them to make a refactor pass. Regenerate only after a deliberate
change of a result or diagnostic, update only the affected keys, and say in the
commit which keys changed and why.
