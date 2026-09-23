# Gravity inversion figures

Three independent entry points use `gravity_common.py`. Keep the four Python files together. Python 3.10+ is required; install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run these commands from the directory containing the two result directories and `RL05.shc` (adjust script paths as needed):

```bash
python gravity_figures/plot_field_timeseries.py --root solution_full_constellation --rl05 RL05.shc
python gravity_figures/plot_field_rms.py --root solution_all_constellations --rl05 RL05.shc
python gravity_figures/plot_parameter_correlations.py --root solution_full_constellation
```

Default outputs are `figures/field_timeseries.pdf`, `figures/field_rms.pdf`, and `figures/parameter_correlations.pdf`. Use `--output /path/name.png` or another Matplotlib-supported extension to change the destination/format. Existing output files are overwritten. Each command supports `--help`.

## Selected results and layouts

The full-constellation filename prefix is:

```text
rheology_ajisai_lageos1_lageos2_starlette_stella
```

The constrained branch is:

```text
sigma_E-13/constraints_G_trend_and_acceleration_and_annual
```

Files may be extensionless, as in the supplied trees, or have one filename extension. Multiple matching files are an error.

- **Time series:** 7 rows in the order C20, C21, S21, C40, C41, S41, C60; two columns. Left: constrained branch, `_tides`. Right: `_tides` directly under `solution_full_constellation`. All axes share x; each row shares y. Both columns contain RL05 and its ±1σ and ±3σ bands, and inversion values with ±3σ bands. Limits include all data and envelopes; no quantile clipping is applied. Matplotlib displays the large coefficient offset separately when appropriate.
- **RMS:** exactly 31 unique satellite combinations from `solution_all_constellations`, selecting exact `_tides` filenames and excluding `_solid_tides`. Rows run from one to five satellites, with ties ordered by Ajisai, LAGEOS-1, LAGEOS-2, Starlette, Stella. Columns follow the seven-harmonic order above. Cells contain colors only. One global linear blue-to-red scale spans the observed RMS minimum and maximum; the colorbar shows dimensionless coefficient differences. `--log-scale` optionally uses a global logarithmic scale.
- **Correlations:** the constrained full-constellation `_tides` matrix fills the left side; `_solid_tides` is top right and `_pole_tide` bottom right. All panels have square cells and share a fixed −1 (blue) to +1 (red) colorbar. Include fitted nondated parameters such as LQM, LTM and the field-model terms. Exclude dated GCN/GSN unknowns entirely and do not synthesize C/S RMS parameters. Fixed zero-sigma parameters are excluded. P-family labels become S; B, A, AA, C, S, P, Q superscripts preserve the source parameter codes. Nondated unknown names outside the recognized field family are retained literally rather than relabeled as LTM.

## Numerical conventions

For each constellation and harmonic, the RMS is:

```text
sqrt(mean((inverted_coefficient(t) - RL05_coefficient(t))**2))
```

The reference is evaluated at the inversion epochs, without nearest-neighbor matching or interpolation of a precomputed grid. By default, each harmonic uses the intersection of epochs across all selected constellations, This gives comparable sampling across rows. `--dates available` uses each constellation's own available epochs instead. Counts are printed. Empty overlap, missing harmonics, duplicate constellation files, or fewer than 31 combinations raise errors. Use `--allow-partial` only for intentionally incomplete constellation sets.

The RMS directory in the supplied tree is flat and unconstrained. If needed for another directory structure, select exactly one nested branch using:

```bash
python gravity_figures/plot_field_rms.py --root solution_all_constellations --rl05 RL05.shc --branch sigma_E-13/constraints_G_trend_and_acceleration_and_annual
```

RL05 implements the **SHC record dialect used by the supplied old script**: `G_BIAS`, `G_DRIFT`, `G_COS`, `G_SIN` (also accepted without the underscore), followed by degree/order, C/S values, C/S standard deviations, two YYYYMMDD validity dates, and optional trailing fields. Fractional date suffixes are discarded, as in the old script. The first date is also the basis origin. Time is elapsed days / 365 by default, matching the old code; `--year-days 365.25` changes that convention. COS/SIN terms are annual; trailing metadata is not interpreted as another period. Other coefficient-record dialects or term types need an explicit loader adaptation.

As explicitly requested, RL05 reference uncertainty preserves the original direct signed sum:

```text
sigma_legacy(t) = sum(basis_k(t) * sigma_k)
```

This is the original uncertainty convention, not quadrature propagation. Signed uncertainties are retained exactly when drawing the ±1σ and ±3σ bands. Both interval endpoints are included and contributions at shared boundaries accumulate, as in the original code. Values and uncertainties are initialized to zero outside covered intervals, also as before; ensure the input model covers the epochs being compared. No coefficient normalization, tide-system conversion, or background-model offset correction is applied: RL05 and the inversion values must already use the same conventions.

`gravity_common.gravity_timeseries()` preserves the old callable interface and its default 30-day grid from 1991-01-11 through 2025-12-31. The figures evaluate the same equations at inversion epochs for date-matched comparison. The RL05 numerical conventions (365-day basis, annual trigonometric terms, signed uncertainty addition, inclusive validity bounds) are unchanged.

The inversion parser takes the third numerical solution column as the final value and the fourth as formal sigma. It reads the nonzero-sigma unknowns in source order for matrix indexing. The inverse matrix is the packed lower triangle in 20-character fields, with arbitrary line wrapping. Only the nondated submatrix is retained in memory, while the entire triangle is counted. Correlations are Qij / sqrt(Qii Qjj); any common variance factor cancels. Incomplete matrices raise an error rather than filling missing entries. Time-series and RMS loaders stop before reading matrix values, so they can use the supplied abbreviated extract.

## Validation and remaining data check

Run the included numerical tests from this directory:

```bash
python -m unittest -v
```

Tests cover signed wrapped-matrix correlations against an independent dense calculation, fixed/datetime parameter exclusion, truncated matrix rejection, analytic RL05 values and uncertainty, adjacent validity boundaries, all 31 combinations with solid-tide distractors, and known RMS offsets. All three layouts were also rendered and inspected using synthetic fixtures.

The supplied real extract was parsed successfully: 2,839 parameters, 2,800 dated, 2,837 fitted. Its abbreviated inverse matrix contains 91 of the required 4,025,703 values, so it cannot produce a real correlation figure. The complete inversion tree and the actual RL05 file were not attached; production values and the RL05 dialect must still be checked against those files. No synthetic result plots are included as scientific output.
