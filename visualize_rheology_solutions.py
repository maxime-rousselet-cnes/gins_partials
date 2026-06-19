#!/usr/bin/env python3
"""
Parse rheology solution files and visualize LAM/LDM estimates, uncertainties,
and correlations with the other estimated parameters.

Usage
-----
python visualize_rheology_solutions.py /path/to/solution_directory --out /path/to/output_directory

The script expects files containing:
  - a SOLUTION section: parameter label + initial value + correction + solution
    value + solution-section formal uncertainty
  - an INVERSE MATRIX section: one header line containing the matrix dimension
    followed immediately by the variance unit factor, then the symmetric inverse
    matrix stored as lower-triangular fixed-width fields of 20 characters.

For direct GCN/GSN solutions, correlations to individual dated GCN/GSN terms are
summarized as root-mean-square correlations rather than plotted one-by-one.

Important convention
--------------------
The uncertainty reported in lam_ldm_summary.csv is computed from the inverse
matrix, not copied from the SOLUTION section:

    uncertainty = sqrt(variance_unit_factor * inverse_matrix_diagonal)

The original uncertainty value printed in the SOLUTION section is kept in the
column solution_section_uncertainty for checks.

Current grouping convention
---------------------------
Figures are split by mode only:

    figures/<mode>/...

The satellite label "multi_satellite" in filenames is reported in all tables and
plots as:

    starlette+ajisai+lageos2
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

FLOAT_WIDTH = 20
TARGETS = ("LAM", "LDM")

# Filename token -> public/display label used in CSV files and figures.
SATELLITE_PATTERNS = [
    ("multi_satellite", "starlette+ajisai+lageos2"),
    ("multi-satellite", "starlette+ajisai+lageos2"),
    ("multisatellite", "starlette+ajisai+lageos2"),
    ("starlette", "starlette"),
    ("ajisai", "ajisai"),
    ("stella", "stella"),
    ("lageos1", "lageos1"),
    ("lageos2", "lageos2"),
]
SATELLITE_ORDER = [
    "starlette",
    "ajisai",
    "stella",
    "lageos1",
    "lageos2",
    "starlette+ajisai+lageos2",
]
MODES = ["solid_tide", "pole_tide", "tides"]

# Longest first so more specific suffixes are found before their prefixes.
G_MODEL_PATTERNS = [
    ("G_trend_and_acceleration_and_annual", "trend+acceleration+annual"),
    ("G_trend_and_acceleration", "trend+acceleration"),
    ("G_trend_and_annual", "trend+annual"),
    ("G_trend_annual", "trend+annual"),
    ("FIX_G", "fixed_G"),
    ("G_trend", "trend"),
]
G_MODEL_ORDER = [
    "no_G_model/direct_GCN_GSN",
    "fixed_G",
    "trend",
    "trend+acceleration",
    "trend+annual",
    "trend+acceleration+annual",
]

# Separate color maps for the three plot families.
# Correlations are always diverging and symmetric around zero, with limits set
# independently for each heatmap: [-max(abs(values)), +max(abs(values))].
CORRELATION_CMAP = "RdBu_r"
SOLUTION_CMAP = "plasma"
UNCERTAINTY_CMAP = "viridis"

# Fixed clipping for solution heatmaps.
SOLUTION_COLOR_LIMITS = {
    "LAM": (0.15, 0.40),
    "LDM": (4.0, 15.0),
}

# Fixed upper clipping for formal-uncertainty heatmaps.
UNCERTAINTY_COLOR_LIMITS = {
    "LAM": (0.0, 0.125),
    "LDM": (0.0, 5.5),
}


@dataclass
class ParameterRow:
    label: str
    raw_label: str
    initial: float
    correction: float
    solution: float
    uncertainty: float
    solution_order: int


@dataclass
class FileMeta:
    file: str
    satellite: str
    mode: str
    g_model: str


def fortran_float(text: str) -> float:
    """Convert Fortran-like floats, accepting D exponents and missing leading zero."""
    s = text.strip().replace("D", "E").replace("d", "E")
    if s.startswith("-."):
        s = "-0" + s[1:]
    elif s.startswith("+."):
        s = "+0" + s[1:]
    elif s.startswith("."):
        s = "0" + s
    return float(s)


def normalize_label(tokens: list[str]) -> str:
    """Normalize labels so plotting/pivoting is stable across observed formats."""
    if not tokens:
        return ""

    head = tokens[0]
    if head in TARGETS:
        return head

    # Direct dated Stokes coefficients, e.g. "GCN   2  0 19910210".
    if head in {"GCN", "GSN"} and len(tokens) >= 4:
        deg = int(tokens[1]) if tokens[1].lstrip("+-").isdigit() else tokens[1]
        order = int(tokens[2]) if tokens[2].lstrip("+-").isdigit() else tokens[2]
        date = tokens[3]
        if isinstance(deg, int) and isinstance(order, int):
            return f"{head}_{deg:02d}_{order:02d}_{date}"
        return "_".join([head, str(deg), str(order), date])

    # Polynomial-like coefficients may appear as CA_41 or as "CAA 4 1".
    if len(tokens) == 3 and tokens[1].lstrip("+-").isdigit() and tokens[2].lstrip("+-").isdigit():
        deg = int(tokens[1])
        order = int(tokens[2])
        return f"{head}_{deg}{order}"

    return "_".join(tokens)


def direct_g_family(label: str) -> Optional[str]:
    """Return a family label for dated direct GCN/GSN coefficients, or None."""
    m = re.match(r"^(GCN|GSN)_(\d{2})_(\d{2})_(\d{8})$", label)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_RMS"


def parse_metadata(path: Path) -> FileMeta:
    name = path.name
    lowered = name.lower()

    satellite = "unknown_satellite"
    for token, label in sorted(SATELLITE_PATTERNS, key=lambda x: len(x[0]), reverse=True):
        if token.lower() in lowered:
            satellite = label
            break

    mode = "unknown_mode"
    for candidate in sorted(MODES, key=len, reverse=True):
        if candidate.lower() in lowered:
            mode = candidate
            break

    g_model = "no_G_model/direct_GCN_GSN"
    for pattern, label in G_MODEL_PATTERNS:
        if pattern.lower() in lowered:
            g_model = label
            break

    return FileMeta(file=name, satellite=satellite, mode=mode, g_model=g_model)


def find_section(lines: list[str], name: str) -> int:
    for i, line in enumerate(lines):
        if line.strip().startswith(name):
            return i
    raise ValueError(f"Could not find section {name!r}")


def parse_solution(lines: list[str]) -> tuple[list[ParameterRow], dict[str, int]]:
    i_sol = find_section(lines, "SOLUTION")
    i_inv = find_section(lines, "INVERSE MATRIX")
    header_tokens = lines[i_sol + 1].split()
    counts = {
        "n_total_declared": int(header_tokens[0]) if len(header_tokens) > 0 else -1,
        "n_free_declared": int(header_tokens[1]) if len(header_tokens) > 1 else -1,
        "n_fixed_declared": int(header_tokens[2]) if len(header_tokens) > 2 else -1,
    }

    rows: list[ParameterRow] = []
    for order, line in enumerate(lines[i_sol + 2 : i_inv]):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            initial, correction, solution, uncertainty = [fortran_float(x) for x in parts[-4:]]
        except ValueError:
            continue
        raw_tokens = parts[:-4]
        raw_label = " ".join(raw_tokens)
        label = normalize_label(raw_tokens)
        rows.append(
            ParameterRow(
                label=label,
                raw_label=raw_label,
                initial=initial,
                correction=correction,
                solution=solution,
                uncertainty=uncertainty,
                solution_order=order,
            )
        )
    return rows, counts


def parse_inverse_matrix(lines: list[str]) -> tuple[np.ndarray, int, float]:
    i_inv = find_section(lines, "INVERSE MATRIX")
    if i_inv + 1 >= len(lines):
        raise ValueError("INVERSE MATRIX header line is missing")

    # The first fixed-width chunk contains an integer id and the matrix dimension.
    # The second fixed-width chunk is the variance unit factor.
    header_line = lines[i_inv + 1]
    header_left = header_line[:FLOAT_WIDTH]
    ints = re.findall(r"[-+]?\d+", header_left)
    if len(ints) < 2:
        raise ValueError(f"Could not read matrix dimension from {header_line!r}")
    n = int(ints[-1])

    variance_chunk = header_line[FLOAT_WIDTH : 2 * FLOAT_WIDTH].strip()
    if not variance_chunk:
        raise ValueError(f"Could not read variance unit factor from {header_line!r}")
    variance_unit_factor = fortran_float(variance_chunk)

    expected = n * (n + 1) // 2
    values: list[float] = []
    for line in lines[i_inv + 2 :]:
        for start in range(0, len(line), FLOAT_WIDTH):
            chunk = line[start : start + FLOAT_WIDTH].strip()
            if not chunk:
                continue
            values.append(fortran_float(chunk))
            if len(values) == expected:
                break
        if len(values) == expected:
            break

    if len(values) != expected:
        raise ValueError(f"Matrix dimension {n} expects {expected} values, found {len(values)}")

    mat = np.zeros((n, n), dtype=float)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = values[k]
            mat[j, i] = values[k]
            k += 1
    return mat, n, variance_unit_factor


def select_free_rows(
    rows: list[ParameterRow], matrix_n: int, counts: dict[str, int], filename: str
) -> list[ParameterRow]:
    """Align solution rows to the inverse matrix dimension.

    In FIX_G files the SOLUTION section may still list fixed GCN/GSN rows, but
    their formal uncertainties are zero and the inverse matrix only covers the
    free parameters.
    """
    if len(rows) == matrix_n:
        return rows

    nonzero_unc = [r for r in rows if abs(r.uncertainty) > 0.0]
    if len(nonzero_unc) == matrix_n:
        return nonzero_unc

    n_free = counts.get("n_free_declared", -1)
    if n_free == matrix_n and len(nonzero_unc) >= matrix_n:
        return nonzero_unc[:matrix_n]

    # Last-resort fallback: LAM/LDM are normally at the end when many fixed rows precede them.
    print(
        f"WARNING: {filename}: could not confidently align matrix dimension {matrix_n} "
        f"with {len(rows)} solution rows / {len(nonzero_unc)} nonzero-uncertainty rows. "
        f"Using the last {matrix_n} rows."
    )
    return rows[-matrix_n:]


def covariance_to_correlation(mat: np.ndarray) -> np.ndarray:
    diag = np.diag(mat).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(np.outer(diag, diag))
        corr = mat / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def formal_uncertainty_from_matrix(variance_unit_factor: float, diag_value: float) -> float:
    value = variance_unit_factor * diag_value
    if not np.isfinite(value) or value < 0:
        return np.nan
    return float(np.sqrt(value))


def rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr * arr)))


def parse_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lines = path.read_text(errors="replace").splitlines()
    meta = parse_metadata(path)
    rows, counts = parse_solution(lines)
    mat, matrix_n, variance_unit_factor = parse_inverse_matrix(lines)
    free_rows = select_free_rows(rows, matrix_n, counts, path.name)

    if len(free_rows) != matrix_n:
        raise ValueError(f"{path.name}: matrix/parameter alignment failed")

    labels = [r.label for r in free_rows]
    free_row_by_label = {r.label: r for r in free_rows}
    corr = covariance_to_correlation(mat)
    meta_dict = asdict(meta)

    summary_records = []
    for target in TARGETS:
        if target not in labels:
            continue
        r = free_row_by_label[target]
        matrix_index = labels.index(target)
        inverse_matrix_diagonal = float(mat[matrix_index, matrix_index])
        rec = {
            **meta_dict,
            "target": target,
            "initial": r.initial,
            "correction": r.correction,
            "solution": r.solution,
            "uncertainty": formal_uncertainty_from_matrix(
                variance_unit_factor, inverse_matrix_diagonal
            ),
            "solution_section_uncertainty": r.uncertainty,
            "variance_unit_factor": variance_unit_factor,
            "inverse_matrix_diagonal": inverse_matrix_diagonal,
            "matrix_index": matrix_index,
            "matrix_dimension": matrix_n,
            **counts,
        }
        if "LAM" in labels and "LDM" in labels:
            rec["corr_LAM_LDM"] = corr[labels.index("LAM"), labels.index("LDM")]
        else:
            rec["corr_LAM_LDM"] = np.nan
        summary_records.append(rec)

    corr_records = []
    for target in TARGETS:
        if target not in labels:
            continue
        i = labels.index(target)

        direct_values_by_family: dict[str, list[float]] = {}
        direct_values_all: list[float] = []

        for j, other in enumerate(labels):
            if j == i:
                continue
            c = float(corr[i, j])
            family = direct_g_family(other)
            if family is not None:
                direct_values_all.append(c)
                direct_values_by_family.setdefault(family, []).append(c)
            else:
                corr_records.append(
                    {
                        **meta_dict,
                        "target": target,
                        "other_parameter": other,
                        "correlation": c,
                        "correlation_kind": "individual",
                        "abs_correlation": abs(c) if np.isfinite(c) else np.nan,
                    }
                )

        if direct_values_all:
            all_rms = rms(direct_values_all)
            corr_records.append(
                {
                    **meta_dict,
                    "target": target,
                    "other_parameter": "ALL_DIRECT_GCN_GSN_RMS",
                    "correlation": all_rms,
                    "correlation_kind": "direct_GCN_GSN_rms_all",
                    "abs_correlation": all_rms,
                }
            )
            for family, vals in sorted(direct_values_by_family.items()):
                val = rms(vals)
                corr_records.append(
                    {
                        **meta_dict,
                        "target": target,
                        "other_parameter": family,
                        "correlation": val,
                        "correlation_kind": "direct_GCN_GSN_rms_by_family",
                        "abs_correlation": val,
                    }
                )

    diagnostics = {
        **meta_dict,
        **counts,
        "variance_unit_factor": variance_unit_factor,
        "n_solution_rows_parsed": len(rows),
        "matrix_dimension": matrix_n,
        "n_free_rows_used": len(free_rows),
        "free_labels_first": ";".join(labels[:5]),
        "free_labels_last": ";".join(labels[-5:]),
    }

    return pd.DataFrame(summary_records), pd.DataFrame(corr_records), diagnostics


def discover_files(directory: Path) -> list[Path]:
    files = []
    skip_suffixes = {
        ".py",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".xlsx",
        ".parquet",
        ".zip",
    }
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() in skip_suffixes:
            continue
        try:
            text = p.read_text(errors="replace")
        except UnicodeDecodeError:
            continue
        if "SOLUTION" in text and "INVERSE MATRIX" in text:
            files.append(p)
    return files


def ordered_categorical(series: pd.Series, order: list[str]) -> pd.Categorical:
    observed_extra = [x for x in series.dropna().astype(str).unique() if x not in order]
    return pd.Categorical(
        series.astype(str), categories=order + sorted(observed_extra), ordered=True
    )


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_") or "unknown"


def correlation_column_order(data: pd.DataFrame) -> list[str]:
    labels = sorted(data["other_parameter_label"].dropna().astype(str).unique())
    priority = [
        "LDM",
        "LAM",
        "ALL_DIRECT_GCN_GSN_RMS",
    ]
    g_rms = [x for x in labels if re.match(r"^(GCN|GSN)_\d{2}_\d{2}_RMS$", x)]
    others = [x for x in labels if x not in set(priority) and x not in set(g_rms)]
    return [x for x in priority if x in labels] + sorted(g_rms) + sorted(others)


def make_heatmap(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    output_path: Path,
    annotate: bool = False,
    value_format: str = ".2g",
    max_height: float = 60.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    clip: bool = False,
    row_order: Optional[list[str]] = None,
    col_order: Optional[list[str]] = None,
    symmetric_around_zero: bool = False,
):
    if data.empty:
        return

    table = data.pivot_table(
        index=index, columns=columns, values=values, aggfunc="mean", observed=False
    )
    if row_order is not None:
        present = [x for x in row_order if x in table.index]
        extra = [x for x in table.index if x not in present]
        table = table.reindex(present + extra)
    if col_order is not None:
        present = [x for x in col_order if x in table.columns]
        extra = [x for x in table.columns if x not in present]
        table = table.reindex(columns=present + extra)

    table = table.dropna(how="all").dropna(axis=1, how="all")
    if table.empty:
        return

    arr = table.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return

    if symmetric_around_zero:
        max_abs = float(np.nanmax(np.abs(finite)))
        if max_abs == 0 or not np.isfinite(max_abs):
            max_abs = 1.0
        vmin, vmax = -max_abs, max_abs
    else:
        if vmin is None:
            vmin = float(np.nanmin(finite))
        if vmax is None:
            vmax = float(np.nanmax(finite))
        if math.isclose(float(vmin), float(vmax)):
            pad = abs(float(vmin)) * 0.05 if vmin else 1.0
            vmin = float(vmin) - pad
            vmax = float(vmax) + pad

    nrows, ncols = arr.shape
    fig_w = max(8.0, min(34.0, 0.55 * ncols + 5.0))
    fig_h = max(4.0, min(max_height, 0.34 * nrows + 2.6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(table.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(table.index.astype(str))
    fig.colorbar(im, ax=ax, shrink=0.8)

    if annotate and nrows * ncols <= 300:
        for i in range(nrows):
            for j in range(ncols):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, value_format), ha="center", va="center", fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def mode_dir(base: Path, mode: str) -> Path:
    return base / safe_name(mode)


def save_solution_heatmaps(summary_df: pd.DataFrame, out_dir: Path, annotate: bool):
    if summary_df.empty:
        return
    df = summary_df.copy()
    df["satellite_label"] = df["satellite"].astype(str)
    df["g_model_label"] = df["g_model"].astype(str)

    for mode in MODES:
        sub_mode = df[df["mode"].astype(str) == mode]
        if sub_mode.empty:
            continue
        this_out = mode_dir(out_dir, mode)

        for target in TARGETS:
            sub_target = sub_mode[sub_mode["target"].astype(str) == target]
            if sub_target.empty:
                continue

            sol_lim = SOLUTION_COLOR_LIMITS.get(target)
            make_heatmap(
                sub_target,
                index="satellite_label",
                columns="g_model_label",
                values="solution",
                title=f"{target} solution — {mode}",
                output_path=this_out / f"{target}_solution.png",
                annotate=annotate,
                row_order=SATELLITE_ORDER,
                col_order=G_MODEL_ORDER,
                vmin=sol_lim[0] if sol_lim else None,
                vmax=sol_lim[1] if sol_lim else None,
                clip=bool(sol_lim),
                cmap=SOLUTION_CMAP,
            )

            unc_lim = UNCERTAINTY_COLOR_LIMITS.get(target)
            make_heatmap(
                sub_target,
                index="satellite_label",
                columns="g_model_label",
                values="uncertainty",
                title=f"{target} formal uncertainty — {mode}",
                output_path=this_out / f"{target}_uncertainty.png",
                annotate=annotate,
                row_order=SATELLITE_ORDER,
                col_order=G_MODEL_ORDER,
                vmin=unc_lim[0] if unc_lim else None,
                vmax=unc_lim[1] if unc_lim else None,
                clip=bool(unc_lim),
                cmap=UNCERTAINTY_CMAP,
            )

        sub_corr = sub_mode[
            (sub_mode["target"].astype(str) == "LAM") & sub_mode["corr_LAM_LDM"].notna()
        ]
        if not sub_corr.empty:
            make_heatmap(
                sub_corr,
                index="satellite_label",
                columns="g_model_label",
                values="corr_LAM_LDM",
                title=f"Correlation LAM-LDM — {mode}",
                output_path=this_out / "LAM_LDM_correlation.png",
                annotate=annotate,
                row_order=SATELLITE_ORDER,
                col_order=G_MODEL_ORDER,
                cmap=CORRELATION_CMAP,
                symmetric_around_zero=True,
                clip=True,
            )


def save_correlation_heatmaps(corr_df: pd.DataFrame, out_dir: Path, annotate: bool):
    if corr_df.empty:
        return
    df = corr_df.copy()
    df["satellite_label"] = df["satellite"].astype(str)
    df["other_parameter_label"] = df["other_parameter"].astype(str)

    for (mode, target, g_model), sub in df.groupby(
        ["mode", "target", "g_model"], sort=False, observed=True
    ):
        if sub.empty:
            continue
        this_out = mode_dir(out_dir, str(mode))
        safe_model = safe_name(str(g_model))
        make_heatmap(
            sub,
            index="satellite_label",
            columns="other_parameter_label",
            values="correlation",
            title=f"{target} correlations — {str(g_model)}, {str(mode)}",
            output_path=this_out / f"corr_{target}_{safe_model}.png",
            annotate=annotate,
            row_order=SATELLITE_ORDER,
            col_order=correlation_column_order(sub),
            max_height=28.0,
            cmap=CORRELATION_CMAP,
            symmetric_around_zero=True,
            clip=True,
        )


def run(input_dir: Path, out_dir: Path, annotate: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(input_dir)
    if not files:
        raise SystemExit(f"No solution files found in {input_dir}")

    all_summary = []
    all_corr = []
    diagnostics = []
    failures = []

    for path in files:
        try:
            summary, corr, diag = parse_file(path)
            if not summary.empty:
                all_summary.append(summary)
            if not corr.empty:
                all_corr.append(corr)
            diagnostics.append(diag)
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})
            print(f"ERROR: {path.name}: {exc}")

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    diagnostics_df = pd.DataFrame(diagnostics)
    failures_df = pd.DataFrame(failures, columns=["file", "error"])

    # Stable ordering for tables/plots.
    for df in [summary_df, corr_df, diagnostics_df]:
        if not df.empty:
            if "g_model" in df:
                df["g_model"] = ordered_categorical(df["g_model"], G_MODEL_ORDER)
            if "mode" in df:
                df["mode"] = ordered_categorical(df["mode"], MODES)
            if "satellite" in df:
                df["satellite"] = ordered_categorical(df["satellite"], SATELLITE_ORDER)

    summary_df.to_csv(out_dir / "lam_ldm_summary.csv", index=False)
    corr_df.to_csv(out_dir / "correlations_long.csv", index=False)
    diagnostics_df.to_csv(out_dir / "parse_diagnostics.csv", index=False)
    failures_df.to_csv(out_dir / "parse_failures.csv", index=False)

    if not corr_df.empty:
        wide = corr_df.pivot_table(
            index=["file", "satellite", "mode", "g_model", "target"],
            columns="other_parameter",
            values="correlation",
            aggfunc="mean",
            observed=False,
        ).reset_index()
        wide.to_csv(out_dir / "correlations_wide.csv", index=False)

    save_solution_heatmaps(summary_df, out_dir / "figures", annotate=annotate)
    save_correlation_heatmaps(corr_df, out_dir / "figures", annotate=annotate)

    print(f"Parsed {len(files)} files with {len(failures)} failures.")
    print(f"Tables written to: {out_dir}")
    print(f"Figures written to: {out_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize LAM/LDM solution files and correlations."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing solution files")
    parser.add_argument(
        "--out", type=Path, default=Path("rheology_visualization_output"), help="Output directory"
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Print numeric values inside small heatmap cells"
    )
    args = parser.parse_args()
    run(args.input_dir, args.out, annotate=args.annotate)
