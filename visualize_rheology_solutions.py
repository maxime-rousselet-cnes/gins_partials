#!/usr/bin/env python3
"""
Parse rheology solution files and visualize LAM/LDM estimates, uncertainties,
correlations, and reconstructed gravity-field time series.

Usage
-----
python visualize_rheology_solutions.py /path/to/solution_directory --out /path/to/output_directory

The script expects files containing:
  - a SOLUTION section: parameter label + initial value + correction + solution
    value + solution-section formal uncertainty
  - an INVERSE MATRIX section: one header line containing the matrix dimension
    followed immediately by the variance unit factor, then the symmetric inverse
    matrix stored as lower-triangular fixed-width fields of 20 characters.

For direct dated GCN/GSN solutions, LAM/LDM correlations to individual dated
GCN/GSN terms are summarized as root-mean-square correlations rather than
plotted one-by-one.

Important convention
--------------------
The uncertainty reported in lam_ldm_summary.csv is computed from the inverse
matrix, not copied from the SOLUTION section:

    uncertainty = sqrt(variance_unit_factor * inverse_matrix_diagonal)

The original uncertainty value printed in the SOLUTION section is kept in the
column solution_section_uncertainty for checks.

Gravity-field convention
------------------------
The reconstructed gravity-field plots use the date 2000-01-01 as t = 0.
For polynomial G models the assumed form is:

    C_lm(t) = CB_lm + CA_lm*t + CAA_lm*t^2 + CC_lm*cos(2*pi*t) + CS_lm*sin(2*pi*t)
    S_lm(t) = SB_lm + SA_lm*t + SAA_lm*t^2 + SC_lm*cos(2*pi*t) + SS_lm*sin(2*pi*t)

where t is in Julian years from 2000-01-01. Missing terms are treated as zero.
Direct dated GCN/GSN rows are plotted at their listed dates.
"""

from __future__ import annotations

import argparse
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FLOAT_WIDTH = 20
TARGETS = ("LAM", "LDM")
T0 = pd.Timestamp("2000-01-01")
JULIAN_YEAR_DAYS = 365.25

# Filename token -> public/display label used in CSV files and figures.
SATELLITE_PATTERNS = [
    ("multi_satellite", "multi_satellite"),
    ("multi-satellite", "multi_satellite"),
    ("multisatellite", "multi_satellite"),
    ("starlette", "starlette"),
    ("stella", "stella"),
    ("ajisai", "ajisai"),
    ("lageos1", "lageos1"),
    ("lageos2", "lageos2"),
]
SATELLITE_ORDER = ["starlette", "stella", "ajisai", "lageos1", "lageos2", "multi_satellite"]

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

# Separate color maps for the three heatmap families.
CORRELATION_CMAP = "RdBu_r"  # blue / white / red, dynamic symmetric scale per heatmap
SOLUTION_CMAP = "plasma"  # solution values
UNCERTAINTY_CMAP = "viridis"  # formal uncertainties

SOLUTION_COLOR_LIMITS = {
    "LAM": (0.15, 0.40),
    "LDM": (4.0, 15.0),
}
UNCERTAINTY_COLOR_LIMITS = {
    "LAM": (0.0, 0.125),
    "LDM": (0.0, 5.5),
}

# Fixed styles for gravity-field line plots.
SATELLITE_COLORS = {
    "starlette": "tab:blue",
    "stella": "tab:orange",
    "ajisai": "tab:green",
    "lageos1": "tab:red",
    "lageos2": "tab:purple",
    "multi_satellite": "tab:brown",
}
G_MODEL_LINESTYLES = {
    "no_G_model/direct_GCN_GSN": (0, (1, 1)),
    "fixed_G": "-",
    "trend": "--",
    "trend+acceleration": "-.",
    "trend+annual": ":",
    "trend+acceleration+annual": (0, (3, 1, 1, 1)),
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

    header = lines[i_inv + 1]
    header_left = header[:FLOAT_WIDTH]
    ints = re.findall(r"[-+]?\d+", header_left)
    if len(ints) < 2:
        raise ValueError(f"Could not read matrix dimension from {header!r}")
    n = int(ints[-1])

    variance_text = header[FLOAT_WIDTH : 2 * FLOAT_WIDTH].strip()
    if not variance_text:
        parts = header.split()
        if len(parts) < 2:
            raise ValueError(f"Could not read variance unit factor from {header!r}")
        # In fixed-width output the dimension and variance may touch, e.g. '160.xxx'.
        variance_text = parts[-1]
    variance_unit_factor = fortran_float(variance_text)

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
    """Align SOLUTION rows to the inverse-matrix dimension.

    In FIX_G or partially constrained files, the SOLUTION section can list fixed
    rows that are not included in the inverse matrix. These usually have zero
    formal uncertainty in the SOLUTION section.
    """
    if len(rows) == matrix_n:
        return rows

    nonzero_unc = [r for r in rows if abs(r.uncertainty) > 0.0]
    if len(nonzero_unc) == matrix_n:
        return nonzero_unc

    n_free = counts.get("n_free_declared", -1)
    if n_free == matrix_n and len(nonzero_unc) >= matrix_n:
        return nonzero_unc[:matrix_n]

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


def matrix_formal_uncertainty(variance_unit_factor: float, matrix_diagonal: float) -> float:
    """Compute formal uncertainty from variance unit factor and matrix diagonal."""
    value = variance_unit_factor * matrix_diagonal
    if not np.isfinite(value) or value < 0:
        return np.nan
    return float(math.sqrt(value))


def rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr * arr)))


def parse_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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

    # Summary table for LAM/LDM values and matrix-derived uncertainties.
    summary_records = []
    for target in TARGETS:
        if target not in free_row_by_label:
            continue

        r = free_row_by_label[target]
        idx = labels.index(target)
        matrix_diagonal = float(mat[idx, idx])
        matrix_uncertainty = matrix_formal_uncertainty(variance_unit_factor, matrix_diagonal)
        rec = {
            **meta_dict,
            "target": target,
            "initial": r.initial,
            "correction": r.correction,
            "solution": r.solution,
            "uncertainty": matrix_uncertainty,
            "solution_section_uncertainty": r.uncertainty,
            "matrix_diagonal": matrix_diagonal,
            "matrix_variance_unit_factor": variance_unit_factor,
            "matrix_index": idx,
            "matrix_dimension": matrix_n,
            **counts,
        }
        if "LAM" in labels and "LDM" in labels:
            rec["corr_LAM_LDM"] = corr[labels.index("LAM"), labels.index("LDM")]
        else:
            rec["corr_LAM_LDM"] = np.nan
        summary_records.append(rec)

    # Correlations of LAM/LDM with every other free parameter.
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
            all_val = rms(direct_values_all)
            corr_records.append(
                {
                    **meta_dict,
                    "target": target,
                    "other_parameter": "ALL_DIRECT_GCN_GSN_RMS",
                    "correlation": all_val,
                    "correlation_kind": "direct_GCN_GSN_rms_all",
                    "abs_correlation": all_val,
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

    gravity_param_records = []
    for r in rows:  # Use all rows so FIX_G/direct dated gravity rows are kept even if not free.
        parsed = parse_gravity_parameter_label(r.label)
        if parsed is None:
            continue
        gravity_param_records.append(
            {
                **meta_dict,
                **parsed,
                "solution": r.solution,
                "initial": r.initial,
                "correction": r.correction,
            }
        )

    diagnostics = {
        **meta_dict,
        **counts,
        "n_solution_rows_parsed": len(rows),
        "matrix_dimension": matrix_n,
        "matrix_variance_unit_factor": variance_unit_factor,
        "n_free_rows_used": len(free_rows),
        "free_labels_first": ";".join(labels[:5]),
        "free_labels_last": ";".join(labels[-5:]),
    }

    return (
        pd.DataFrame(summary_records),
        pd.DataFrame(corr_records),
        pd.DataFrame(gravity_param_records),
        diagnostics,
    )


def discover_files(directory: Path) -> list[Path]:
    files = []
    skip_suffixes = {".py", ".csv", ".png", ".jpg", ".jpeg", ".pdf", ".xlsx", ".parquet", ".zip"}
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
    return pd.Categorical(series.astype(str), categories=order, ordered=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_") or "unknown"


def ordered_existing(values: Iterable[str], preferred_order: list[str]) -> list[str]:
    unique = [str(v) for v in pd.unique(pd.Series(list(values)).dropna())]
    preferred = [x for x in preferred_order if x in unique]
    rest = sorted([x for x in unique if x not in preferred])
    return preferred + rest


def correlation_column_order(data: pd.DataFrame) -> list[str]:
    if data.empty or "other_parameter" not in data:
        return []
    cols = [str(x) for x in data["other_parameter"].dropna().unique()]
    all_direct = [c for c in cols if c == "ALL_DIRECT_GCN_GSN_RMS"]
    family_rms = sorted([c for c in cols if c.endswith("_RMS") and c not in all_direct])
    individual = sorted([c for c in cols if c not in set(all_direct + family_rms)])
    return all_direct + family_rms + individual


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
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
):
    if data.empty:
        return
    table = data.pivot_table(
        index=index, columns=columns, values=values, aggfunc="mean", observed=False
    )

    if row_order is not None:
        table = table.reindex([r for r in row_order if r in table.index])
    if col_order is not None:
        table = table.reindex(columns=[c for c in col_order if c in table.columns])

    table = table.dropna(how="all").dropna(axis=1, how="all")
    if table.empty:
        return

    arr = table.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return

    if center_zero:
        max_abs = float(np.nanmax(np.abs(finite)))
        if max_abs == 0 or not np.isfinite(max_abs):
            max_abs = 1.0
        vmin = -max_abs
        vmax = max_abs
    else:
        if vmin is None:
            vmin = float(np.nanmin(finite))
        if vmax is None:
            vmax = float(np.nanmax(finite))
        if math.isclose(vmin, vmax):
            pad = abs(vmin) * 0.05 if vmin else 1.0
            vmin -= pad
            vmax += pad

    nrows, ncols = arr.shape
    fig_w = max(7.0, min(34.0, 0.65 * ncols + 4.0))
    fig_h = max(3.4, min(max_height, 0.42 * nrows + 2.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
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


def save_solution_heatmaps(summary_df: pd.DataFrame, out_dir: Path, annotate: bool):
    if summary_df.empty:
        return

    df = summary_df.copy()
    df["row"] = df["satellite"].astype(str)
    df["column"] = df["g_model"].astype(str)
    row_order = ordered_existing(df["row"], SATELLITE_ORDER)
    col_order = ordered_existing(df["column"], G_MODEL_ORDER)

    for mode, mode_df in df.groupby("mode", sort=False, observed=False):
        if mode_df.empty:
            continue
        mode_dir = out_dir / safe_name(str(mode))

        for target in TARGETS:
            sub = mode_df[mode_df["target"] == target]
            if sub.empty:
                continue
            make_heatmap(
                sub,
                index="row",
                columns="column",
                values="solution",
                title=f"{target} solution — {mode}",
                output_path=mode_dir / f"{target}_solution.png",
                annotate=annotate,
                cmap=SOLUTION_CMAP,
                vmin=SOLUTION_COLOR_LIMITS[target][0],
                vmax=SOLUTION_COLOR_LIMITS[target][1],
                row_order=row_order,
                col_order=col_order,
            )
            make_heatmap(
                sub,
                index="row",
                columns="column",
                values="uncertainty",
                title=f"{target} formal uncertainty — {mode}",
                output_path=mode_dir / f"{target}_uncertainty.png",
                annotate=annotate,
                cmap=UNCERTAINTY_CMAP,
                vmin=UNCERTAINTY_COLOR_LIMITS[target][0],
                vmax=UNCERTAINTY_COLOR_LIMITS[target][1],
                row_order=row_order,
                col_order=col_order,
            )

        sub = mode_df[(mode_df["target"] == "LAM") & mode_df["corr_LAM_LDM"].notna()].copy()
        if not sub.empty:
            make_heatmap(
                sub,
                index="row",
                columns="column",
                values="corr_LAM_LDM",
                title=f"Correlation LAM-LDM — {mode}",
                output_path=mode_dir / "LAM_LDM_correlation.png",
                cmap=CORRELATION_CMAP,
                center_zero=True,
                annotate=annotate,
                row_order=row_order,
                col_order=col_order,
            )


def save_correlation_heatmaps(corr_df: pd.DataFrame, out_dir: Path, annotate: bool):
    if corr_df.empty:
        return

    df = corr_df.copy()
    df["row"] = df["satellite"].astype(str)
    row_order = ordered_existing(df["row"], SATELLITE_ORDER)

    for (mode, target, g_model), sub in df.groupby(
        ["mode", "target", "g_model"], sort=False, observed=False
    ):
        if sub.empty:
            continue
        mode_dir = out_dir / safe_name(str(mode))
        safe_model = safe_name(str(g_model))
        make_heatmap(
            sub,
            index="row",
            columns="other_parameter",
            values="correlation",
            title=f"{target} correlations — {mode} — {g_model}",
            output_path=mode_dir / f"corr_{target}_{safe_model}.png",
            cmap=CORRELATION_CMAP,
            center_zero=True,
            annotate=annotate,
            max_height=80.0,
            row_order=row_order,
            col_order=correlation_column_order(sub),
        )


# -----------------------------------------------------------------------------
# Gravity-field reconstruction and plots
# -----------------------------------------------------------------------------


def parse_gravity_parameter_label(label: str) -> Optional[dict]:
    """Parse normalized gravity parameter labels.

    Returns fields describing either a direct dated coefficient or a polynomial
    model term. Coefficient names use the normalized GCN_02_00 / GSN_04_01 form.
    """
    m = re.match(r"^(GCN|GSN)_(\d{2})_(\d{2})_(\d{8})$", label)
    if m:
        kind, degree, order, date_text = m.groups()
        try:
            date = pd.to_datetime(date_text, format="%Y%m%d")
        except ValueError:
            return None
        return {
            "gravity_row_kind": "direct",
            "coefficient": f"{kind}_{degree}_{order}",
            "component": "C" if kind == "GCN" else "S",
            "degree": int(degree),
            "order": int(order),
            "term": "direct",
            "date": date,
        }

    m = re.match(r"^(CAA|CA|CB|CC|CS|SAA|SA|SB|SC|SS)_(\d)(\d)$", label)
    if m:
        term, degree, order = m.groups()
        component = "C" if term.startswith("C") else "S"
        kind = "GCN" if component == "C" else "GSN"
        return {
            "gravity_row_kind": "model_parameter",
            "coefficient": f"{kind}_{int(degree):02d}_{int(order):02d}",
            "component": component,
            "degree": int(degree),
            "order": int(order),
            "term": term,
            "date": pd.NaT,
        }
    return None


def decimal_years_from_t0(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dates = pd.to_datetime(dates)
    delta_days = (dates - T0) / pd.Timedelta(days=1)
    return np.asarray(delta_days, dtype=float) / JULIAN_YEAR_DAYS


def build_monthly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        start, end = end, start
    return pd.date_range(start=start, end=end, freq="MS")


def evaluate_gravity_models(gravity_params_df: pd.DataFrame) -> pd.DataFrame:
    """Build direct and model-evaluated gravity-field time series."""
    if gravity_params_df.empty:
        return pd.DataFrame()

    df = gravity_params_df.copy()

    # Determine a plotting/evaluation range from direct dated rows when possible.
    direct_dates = pd.to_datetime(
        df.loc[df["gravity_row_kind"] == "direct", "date"], errors="coerce"
    ).dropna()
    if not direct_dates.empty:
        start = direct_dates.min()
        end = direct_dates.max()
    else:
        start = pd.Timestamp("1991-01-01")
        end = pd.Timestamp(datetime.now().date())
    monthly_dates = build_monthly_grid(start, end)
    t_monthly = decimal_years_from_t0(monthly_dates)

    records: list[dict] = []

    # Direct dated coefficients.
    direct = df[df["gravity_row_kind"] == "direct"].copy()
    if not direct.empty:
        direct["date"] = pd.to_datetime(direct["date"])
        direct["t_years_since_2000_01_01"] = decimal_years_from_t0(direct["date"])
        direct["value"] = direct["solution"]
        for _, row in direct.iterrows():
            records.append(
                {
                    "file": row["file"],
                    "satellite": row["satellite"],
                    "mode": row["mode"],
                    "g_model": row["g_model"],
                    "coefficient": row["coefficient"],
                    "component": row["component"],
                    "degree": row["degree"],
                    "order": row["order"],
                    "date": row["date"],
                    "t_years_since_2000_01_01": row["t_years_since_2000_01_01"],
                    "value": row["value"],
                    "series_kind": "direct",
                }
            )

    # Polynomial/annual model parameters.
    model_params = df[df["gravity_row_kind"] == "model_parameter"].copy()
    group_cols = [
        "file",
        "satellite",
        "mode",
        "g_model",
        "coefficient",
        "component",
        "degree",
        "order",
    ]
    for keys, sub in model_params.groupby(group_cols, sort=False, observed=False):
        key_dict = dict(zip(group_cols, keys))
        terms = {str(row["term"]): float(row["solution"]) for _, row in sub.iterrows()}
        component = key_dict["component"]
        if component == "C":
            intercept = terms.get("CB", 0.0)
            trend = terms.get("CA", 0.0)
            accel = terms.get("CAA", 0.0)
            annual_cos = terms.get("CC", 0.0)
            annual_sin = terms.get("CS", 0.0)
        else:
            intercept = terms.get("SB", 0.0)
            trend = terms.get("SA", 0.0)
            accel = terms.get("SAA", 0.0)
            annual_cos = terms.get("SC", 0.0)
            annual_sin = terms.get("SS", 0.0)

        values = intercept + trend * t_monthly + accel * t_monthly**2
        values = (
            values
            + annual_cos * np.cos(2.0 * np.pi * t_monthly)
            + annual_sin * np.sin(2.0 * np.pi * t_monthly)
        )

        for date, t, value in zip(monthly_dates, t_monthly, values):
            records.append(
                {
                    **key_dict,
                    "date": date,
                    "t_years_since_2000_01_01": float(t),
                    "value": float(value),
                    "series_kind": "model_evaluated",
                }
            )

    return pd.DataFrame(records)


def save_gravity_field_figures(gravity_ts_df: pd.DataFrame, out_dir: Path, show: bool = False):
    """Save gravity-field figures split by mode, coefficient, and satellite.

    Each figure contains one gravitational coefficient for one satellite. The
    satellite is still encoded by color so that visual identity is consistent
    across figures, and the line style distinguishes the G model.
    """
    if gravity_ts_df.empty:
        return

    df = gravity_ts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig_root = out_dir / "gravity_fields"

    for mode, mode_df in df.groupby("mode", sort=False, observed=False):
        if mode_df.empty:
            continue
        mode_dir = fig_root / safe_name(str(mode))

        for coefficient in sorted(mode_df["coefficient"].dropna().unique()):
            coef_df = mode_df[mode_df["coefficient"] == coefficient]
            if coef_df.empty:
                continue
            coef_dir = mode_dir / safe_name(str(coefficient))

            for satellite in ordered_existing(coef_df["satellite"], SATELLITE_ORDER):
                sub = coef_df[coef_df["satellite"].astype(str) == satellite]
                if sub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
                plotted = False
                color = SATELLITE_COLORS.get(satellite, None)

                for g_model in ordered_existing(sub["g_model"], G_MODEL_ORDER):
                    line_df = sub[sub["g_model"].astype(str) == g_model].sort_values("date")
                    if line_df.empty:
                        continue
                    linestyle = G_MODEL_LINESTYLES.get(g_model, "-")
                    marker = "o" if line_df["series_kind"].iloc[0] == "direct" else None
                    markersize = 2.2 if marker else 0
                    linewidth = 1.2 if marker else 1.8
                    alpha = 0.75 if marker else 0.95
                    ax.plot(
                        line_df["date"],
                        line_df["value"],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=markersize,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=str(g_model),
                    )
                    plotted = True

                if not plotted:
                    plt.close(fig)
                    continue

                ax.axvline(T0, color="0.45", linewidth=0.9, alpha=0.7)
                ax.set_title(
                    f"Gravity field {coefficient} — {satellite} — {mode}\n"
                    "Polynomial models referenced to t=0 at 2000-01-01"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel(coefficient)
                ax.grid(True, alpha=0.25)
                ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

                model_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color or "0.2",
                        lw=2,
                        linestyle=G_MODEL_LINESTYLES.get(m, "-"),
                        label=m,
                    )
                    for m in G_MODEL_ORDER
                    if m in set(sub["g_model"].astype(str))
                ]
                ax.legend(handles=model_handles, title="G model", loc="best", fontsize=8)

                coef_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    coef_dir / f"gravity_{safe_name(coefficient)}_{safe_name(satellite)}.png",
                    dpi=200,
                )
                if show:
                    plt.show()
                plt.close(fig)


def run(input_dir: Path, out_dir: Path, annotate: bool = False, show_gravity: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(input_dir)
    if not files:
        raise SystemExit(f"No solution files found in {input_dir}")

    all_summary = []
    all_corr = []
    all_gravity_params = []
    diagnostics = []
    failures = []

    for path in files:
        try:
            summary, corr, gravity_params, diag = parse_file(path)
            if not summary.empty:
                all_summary.append(summary)
            if not corr.empty:
                all_corr.append(corr)
            if not gravity_params.empty:
                all_gravity_params.append(gravity_params)
            diagnostics.append(diag)
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})
            print(f"ERROR: {path.name}: {exc}")

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    gravity_params_df = (
        pd.concat(all_gravity_params, ignore_index=True) if all_gravity_params else pd.DataFrame()
    )
    gravity_ts_df = (
        evaluate_gravity_models(gravity_params_df)
        if not gravity_params_df.empty
        else pd.DataFrame()
    )
    diagnostics_df = pd.DataFrame(diagnostics)
    failures_df = pd.DataFrame(failures, columns=["file", "error"])

    # Stable ordering for tables/plots.
    for df in [summary_df, corr_df, gravity_params_df, gravity_ts_df, diagnostics_df]:
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
    gravity_params_df.to_csv(out_dir / "gravity_model_parameters.csv", index=False)
    gravity_ts_df.to_csv(out_dir / "gravity_field_timeseries.csv", index=False)

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
    save_gravity_field_figures(gravity_ts_df, out_dir / "figures", show=show_gravity)

    print(f"Parsed {len(files)} files with {len(failures)} failures.")
    print(f"Tables written to: {out_dir}")
    print(f"Figures written to: {out_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize LAM/LDM solution files, correlations, and gravity fields."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing solution files")
    parser.add_argument(
        "--out", type=Path, default=Path("rheology_visualization_output"), help="Output directory"
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Print numeric values inside small heatmap cells"
    )
    parser.add_argument(
        "--show-gravity",
        action="store_true",
        help="Call plt.show() after each gravity-field figure is saved, for quick interactive inspection",
    )
    args = parser.parse_args()
    run(args.input_dir, args.out, annotate=args.annotate, show_gravity=args.show_gravity)
