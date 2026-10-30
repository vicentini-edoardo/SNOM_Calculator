# Numerical models

The tip is a prolate spheroid. Lengths are in nm, wavenumbers in cm⁻¹,
momentum in nm⁻¹, and phase in radians. `TipParameters.length_nm` is the
full tip length; the engine's `a` is its half-length. The tip length must
exceed twice its apex radius.

`MaterialSpectrum.perpendicular` holds εxx = εyy, and `parallel` holds εzz.
Materials are isotropic or surface-normal uniaxial, for bulk samples, films,
substrates and references.

## Constant tip permittivity

`TipParameters.epsilon` is a constant complex scalar shared by every frequency
and by the sample and reference calculations. The tip is homogeneous and
quasistatic.

## Bulk and finite layers

For a surface-normal uniaxial material, the bulk calculation uses an effective scalar permittivity
εeff = √εxx √εzz, using each axis's passive square root. This retains the negative
sign when both lossless axes are negative. This is a scalar approximation, distinct from the full tensor
propagation used for finite uniaxial films.

Isotropic films use the generalised image charge method (GICM): a least-squares
fit of their electrostatic reflection coefficient to `K` image charges at real
depths `2n d_top`. Increase `K`
to check the fit near sharp resonances. The rank cutoff suppresses nearly
dependent terms; increasing `K` alone does not guarantee convergence.

Automatic and Presets take `K` from a table (`_preset_defaults.film_k` and `stack_k`, by the
strongest film's preset and the thinnest film): the cheapest `K` within 3 % of the FIM
(Fourier integral method) in every training config, with FIM wherever no `K <= 16` reaches
3 %. Very strong films and Strong or Very strong stacks always use FIM. There is no `K`
check: `K` against `K // 2` overstated the error of `K` by up to 30x. Training covered 11
isotropic materials (catalog and Lorentz oscillators), gaps 0.3-10 nm, Au and Si substrates
and 100 nm under-layers; validation on 9 unseen materials, other thicknesses, gaps, tips and
under-layers found no config above 3 % (worst 2.7 %). A `K` typed in Custom is not checked.

Surface-normal uniaxial films use their full tensor. For in-plane momentum
`k`, the normal decay is `q = k sqrt(εxx/εzz)` and the electrostatic
admittance is `Y = εzz sqrt(εxx/εzz)`. The passive branch determines decay
and propagation. The direct solver integrates the stack reflection against
the spheroid kernel. Its default FIM (Fourier integral method) checks for poles and
quadrature error at each frequency, then falls back to adaptive real-axis
integration when no FIM ray passes. Use
`calculate_layer_harmonics(..., method="real-axis")` to force that path, or
`method="gicm"` / `"fim"` to pick the isotropic-film coupling.

Higher multipoles matter only near contact. For `N >= 64`, spheroid bulk and GICM
solves use order `N` at the closest approach and `8 ceil(1.5 N (gap/h)^0.4 / 8)` (at
least 32) at height `h` above it, solving leading blocks of the order-N tables. On 15
lab cases (0.2-5 nm gaps, tip radii 10-50 nm, amplitudes 10-100 nm) this added at most
0.2x the uniform-N truncation error, for 36-64 % of the solve cost at N >= 80. FIM and
real-axis films keep `N` at every height: their per-height coupling setup costs more
than the smaller solves save.

Layers are ordered from air to substrate. Zero-thickness films are removed,
adjacent identical films are combined, and an empty stack reduces to the
bulk substrate. The substrate and normalization reference are independent.

## Demodulation and normalization

The tapping response is sampled at `M+1` heights over half a period. Mirror
symmetry gives cosine-transform weights, with half weight at the endpoints.
Above `M = 64` the engines instead solve the tip at 33 Chebyshev-Lobatto nodes in
log height between the gap and the top of the cycle, and the profile is interpolated
to the `M+1` tapping heights before demodulation: the dipole is analytic in log height,
so a larger M costs no extra solves. The nodes double (nested, 65 then 129, fewer than
`M+1`) while the last Chebyshev coefficients of the profile exceed `1e-8` of the
largest; the pointwise interpolation error was about twice that ratio on 52 lab cases.
A ratio above `1e-3`, or one still above `1e-8` after the last doubling, means the
profile is not smooth in log height (seen for near-resonant coupling at 0.2-0.5 nm gaps
with large amplitudes or blunt tips): the engines then solve every tapping height, as
for small M. Results report `height_nodes` (0 for the tapping grid),
`height_tapping_fallback` (1 when sample or reference fell back) and
`height_interpolation_tail`. The M/2 check re-demodulates a node profile, so it costs no
solve. `NumericalSettings.height_nodes = 0` always solves the tapping heights.
Gap averaging (several sample or reference gaps) averages the complex signals of each
gap. When the gaps together have more tapping heights than `2 height_nodes - 1`, one
node profile spans the smallest gap to the top of the largest gap's cycle and is
interpolated to every gap's heights, so the cost is about that of one gap (the span's
log range is barely wider than the smallest gap's own). If that profile is not smooth,
each gap is solved on its own tapping heights. M and N follow the smallest gap, which
needs the most; the M and N checks report the worst gap.
The complex contrast is the sample harmonic divided by the reference
harmonic. Exported amplitude is its magnitude; phase is unwrapped in radians.

Far-field reflection optionally multiplies the driving field by
`(1 + ζ rp)**2 * sin(phi)`, where `rp` is the p-polarized reflection coefficient,
`ζ` the reflection coefficient setting and `phi` the incidence angle. Every
engine applies the same factor to sample and reference, after the convergence
checks: a closed-form Fresnel recursion over the principal axes for
surface-normal uniaxial stacks, with a Maxwell scattering fallback for singular
principal-axis cases. The app enables this correction by default.

## Numerical controls

| Control | Purpose |
| --- | --- |
| `M` | Tapping trajectory samples (log-height interpolation above 64) |
| `height_nodes` | Initial log-height nodes (33); 0 solves every tapping height |
| `N` | Spheroid multipole order; for N >= 64, bulk and GICM spheroids lower it above the closest approach |
| `quadQ` | Surface quadrature order for geometry tables; the app uses max(112, 3N/2) in every mode |
| `K` | GICM image-charge count for isotropic films |
| `momentum_rtol` | Direct solver's relative integration tolerance; fixed in the app, as it barely changes results or time |

Check sensitivity to these settings near resonances. The app's comparison
checks estimate sensitivity; they do not establish agreement with an
experiment or FEM simulation. Aligned hBN checks cover both hyperbolic bands,
smooth frequencies, and low-loss resonances. Independent simulation data
requires matching geometry, materials, normalization and field conventions.
