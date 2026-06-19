import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


SATELLITE_ORDER = [
    "starlette",
    "stella",
    "lageos1",
    "lageos2",
    "multi_satellite",
]

SATELLITE_CODES = {
    "starlette": "Sta",
    "stella": "Ste",
    "lageos1": "L1",
    "lageos2": "L2",
    "multi_satellite": "Multi",
}

SATELLITE_LABELS = {
    "starlette": "Starlette",
    "stella": "Stella",
    "lageos1": "LAGEOS-1",
    "lageos2": "LAGEOS-2",
    "multi_satellite": "Multi-satellite",
}

MODE_ORDER = [
    "pole_tide",
    "solid_tide",
    "tides",
]

MODE_CODES = {
    "pole_tide": "PT",
    "solid_tide": "ST",
    "tides": "Tides",
}

MODE_LABELS = {
    "pole_tide": "pole tide",
    "solid_tide": "solid tide",
    "tides": "tides",
}

MODEL_ORDER = [
    None,
    "trend",
    "trend_and_acceleration",
    "trend_and_acceleration_and_annual",
    "trend_and_annual",
]

MODEL_CODES = {
    None: "base",
    "trend": "Gtr",
    "trend_and_acceleration": "GtrAcc",
    "trend_and_acceleration_and_annual": "GtrAccAnn",
    "trend_and_annual": "GtrAnn",
}

MODEL_LABELS = {
    None: "no G model",
    "trend": "G trend",
    "trend_and_acceleration": "G trend + acceleration",
    "trend_and_acceleration_and_annual": "G trend + acceleration + annual",
    "trend_and_annual": "G trend + annual",
}

CONSTRAINT_ORDER = [
    None,
    "constraint_LAM",
    "constraint_LDM",
]

CONSTRAINT_CODES = {
    None: "free",
    "constraint_LAM": "cLAM",
    "constraint_LDM": "cLDM",
}

CONSTRAINT_LABELS = {
    None: "none",
    "constraint_LAM": "LAM constrained",
    "constraint_LDM": "LDM constrained",
}


def fortran_float(text):
    return float(text.strip().replace("D", "E").replace("d", "E"))


def parse_solution_filename(path):
    filename = Path(path).stem

    for satellite in sorted(SATELLITE_ORDER, key=len, reverse=True):
        prefix = f"{satellite}_rheology_from_"

        if not filename.startswith(prefix):
            continue

        after_satellite = filename[len(prefix) :]

        for mode in sorted(MODE_ORDER, key=len, reverse=True):
            if after_satellite == mode:
                return {
                    "file": filename,
                    "satellite": satellite,
                    "mode": mode,
                    "model": None,
                    "constraint": None,
                }

            if after_satellite.startswith(mode + "_"):
                after_mode = after_satellite[len(mode) :]

                for constraint in CONSTRAINT_ORDER[1:]:
                    if after_mode == f"_{constraint}":
                        return {
                            "file": filename,
                            "satellite": satellite,
                            "mode": mode,
                            "model": None,
                            "constraint": constraint,
                        }

                if after_mode.startswith("_G_"):
                    after_g = after_mode[len("_G_") :]

                    for model in sorted([m for m in MODEL_ORDER if m is not None], key=len, reverse=True):
                        if after_g == model:
                            return {
                                "file": filename,
                                "satellite": satellite,
                                "mode": mode,
                                "model": model,
                                "constraint": None,
                            }

                        for constraint in CONSTRAINT_ORDER[1:]:
                            if after_g == f"{model}_{constraint}":
                                return {
                                    "file": filename,
                                    "satellite": satellite,
                                    "mode": mode,
                                    "model": model,
                                    "constraint": constraint,
                                }

    return None


def sort_key_from_metadata(metadata):
    return (
        SATELLITE_ORDER.index(metadata["satellite"]),
        MODE_ORDER.index(metadata["mode"]),
        MODEL_ORDER.index(metadata["model"]),
        CONSTRAINT_ORDER.index(metadata["constraint"]),
        metadata["file"],
    )


def discover_solution_files(solution_dir, solution_pattern):
    paths_and_metadata = []

    for path in sorted(Path(solution_dir).glob(solution_pattern)):
        if not path.is_file():
            continue

        metadata = parse_solution_filename(path)

        if metadata is not None:
            paths_and_metadata.append((path, metadata))

    paths_and_metadata.sort(key=lambda item: sort_key_from_metadata(item[1]))
    return paths_and_metadata


def read_solution_file(solution_file):
    lines = Path(solution_file).read_text().splitlines()
    sol_idx = next(i for i, line in enumerate(lines) if line.strip() == "SOLUTION")
    inv_idx = next(i for i, line in enumerate(lines) if line.strip() == "INVERSE MATRIX")

    adjusted_values = {}
    matrix_parameters = []

    for line in lines[sol_idx + 2 : inv_idx]:
        if not line.strip():
            continue

        parameter = line[:24].strip()
        values = [fortran_float(value) for value in line[24:].split()]
        adjusted_values[parameter] = values[2]

        if values[-1] != 0.0:
            matrix_parameters.append(parameter)

    meta = lines[inv_idx + 1]
    n_matrix = int(meta[10:20])
    variance_coefficient = fortran_float(meta[20:40])

    lower_triangle_values = []

    for line in lines[inv_idx + 2 :]:
        s = line.strip()

        if not s:
            continue

        for k in range(0, len(s), 20):
            lower_triangle_values.append(fortran_float(s[k : k + 20]))

    inverse_matrix = np.zeros((n_matrix, n_matrix), dtype=float)
    k = 0

    for i in range(n_matrix):
        for j in range(i + 1):
            value = lower_triangle_values[k]
            inverse_matrix[i, j] = value
            inverse_matrix[j, i] = value
            k += 1

    return matrix_parameters, inverse_matrix * variance_coefficient, adjusted_values


def compute_parameter_correlations(parameters, covariance_matrix, target_parameter):
    i_target = parameters.index(target_parameter)

    return {
        parameter: covariance_matrix[i, i_target]
        / (covariance_matrix[i, i] * covariance_matrix[i_target, i_target]) ** 0.5
        for i, parameter in enumerate(parameters)
    }


def collapse_gcn_gsn_correlations(correlations):
    collapsed = {}
    gcn_values = []
    gsn_values = []

    for parameter, correlation in correlations.items():
        if "GCN" in parameter:
            gcn_values.append(correlation)
        elif "GSN" in parameter:
            gsn_values.append(correlation)
        else:
            collapsed[parameter] = correlation

    if gcn_values:
        collapsed["GCN_AM"] = np.mean(np.abs(gcn_values))

    if gsn_values:
        collapsed["GSN_AM"] = np.mean(np.abs(gsn_values))

    return collapsed


def build_lam_ldm_correlation_dataframes(solution_dir, solution_pattern="*"):
    paths_and_metadata = discover_solution_files(solution_dir, solution_pattern)

    lam_by_file = {}
    ldm_by_file = {}
    metadata_by_file = {}

    for solution_file, metadata in paths_and_metadata:
        parameters, covariance_matrix, adjusted_values = read_solution_file(solution_file)
        file_label = solution_file.stem

        lam_correlations = collapse_gcn_gsn_correlations(
            compute_parameter_correlations(parameters, covariance_matrix, "LAM")
        )
        ldm_correlations = collapse_gcn_gsn_correlations(
            compute_parameter_correlations(parameters, covariance_matrix, "LDM")
        )

        lam_correlations["LAM"] = adjusted_values["LAM"]
        ldm_correlations["LDM"] = adjusted_values["LDM"]

        lam_by_file[file_label] = lam_correlations
        ldm_by_file[file_label] = ldm_correlations
        metadata_by_file[file_label] = metadata

    lam = pd.DataFrame(lam_by_file)
    ldm = pd.DataFrame(ldm_by_file)

    all_parameters = list(dict.fromkeys(list(lam.index) + list(ldm.index)))
    all_files = list(metadata_by_file.keys())

    lam = lam.reindex(index=all_parameters, columns=all_files)
    ldm = ldm.reindex(index=all_parameters, columns=all_files)

    return lam, ldm, metadata_by_file


def make_file_label(metadata):
    parts = [
        SATELLITE_CODES[metadata["satellite"]],
        MODE_CODES[metadata["mode"]],
        MODEL_CODES[metadata["model"]],
    ]

    if metadata["constraint"] is not None:
        parts.append(CONSTRAINT_CODES[metadata["constraint"]])

    return "_".join(parts)


def make_file_labels(columns, metadata_by_file):
    labels = {}

    for column in columns:
        labels[column] = make_file_label(metadata_by_file[column])

    return labels


def draw_heatmap(ax, data, title, column_labels, norm, cmap="RdBu_r"):
    values = data.to_numpy(dtype=float)
    x_labels = [column_labels[column] for column in data.columns]
    y_labels = list(data.index)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("lightgrey")

    image = ax.imshow(values, aspect="auto", cmap=colormap, norm=norm)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.45, color="white")
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    n_columns = max(1, values.shape[1])
    annotation_size = max(4.2, min(6.4, 80 / n_columns))

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]

            if np.isfinite(value):
                text_color = "white" if abs(value) >= 0.55 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=text_color,
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def make_mapping_blocks(label_metadata, columns):
    blocks = []

    for column in columns:
        metadata = label_metadata[column]
        label = make_file_label(metadata)
        lines = [label]
        lines.append(f"  satellite : {SATELLITE_LABELS[metadata['satellite']]}")
        lines.append(f"  mode      : {MODE_LABELS[metadata['mode']]}")
        lines.append(f"  G model   : {MODEL_LABELS[metadata['model']]}")
        lines.append(f"  constraint: {CONSTRAINT_LABELS[metadata['constraint']]}")
        lines.append(
            "  file   : "
            + textwrap.fill(
                column,
                width=68,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        blocks.append("\n".join(lines))

    return blocks


def draw_mapping(ax, label_metadata, columns):
    ax.axis("off")

    blocks = make_mapping_blocks(label_metadata, columns)

    if len(blocks) <= 8:
        n_columns = 1
    elif len(blocks) <= 20:
        n_columns = 2
    else:
        n_columns = 3

    chunk_size = int(np.ceil(len(blocks) / n_columns))
    x_positions = np.linspace(0, 1, n_columns + 1)[:-1]

    for column_index in range(n_columns):
        start = column_index * chunk_size
        end = start + chunk_size
        text = "Filename label mapping\n\n" if column_index == 0 else "\n\n"
        text += "\n\n".join(blocks[start:end])

        ax.text(
            x_positions[column_index],
            1,
            text,
            va="top",
            ha="left",
            fontsize=6.8,
            family="monospace",
            linespacing=1.10,
            transform=ax.transAxes,
        )


def plot_lam_ldm_heatmaps(solution_dir, solution_pattern, out_prefix):
    lam, ldm, metadata_by_file = build_lam_ldm_correlation_dataframes(
        solution_dir=solution_dir,
        solution_pattern=solution_pattern,
    )

    column_labels = make_file_labels(lam.columns, metadata_by_file)
    corr_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    n_files = max(1, len(lam.columns))
    n_parameters = max(1, len(lam.index))
    figure_width = max(16, min(34, 0.70 * n_files + 7.0))
    heatmap_height = max(7.0, 0.34 * n_parameters + 4.5)
    legend_rows = int(np.ceil(n_files / (1 if n_files <= 8 else 2 if n_files <= 20 else 3)))
    legend_height = max(3.0, 0.95 * legend_rows + 1.0)
    figure_height = heatmap_height + legend_height

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1, 0.028],
        height_ratios=[1, 1, legend_height / max(heatmap_height, 1)],
        wspace=0.045,
        hspace=0.34,
    )

    lam_ax = fig.add_subplot(grid[0, 0])
    ldm_ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[:2, 1])
    mapping_ax = fig.add_subplot(grid[2, :])

    image_lam = draw_heatmap(
        lam_ax,
        lam,
        "LAM correlations by result file",
        column_labels,
        corr_norm,
    )
    draw_heatmap(
        ldm_ax,
        ldm,
        "LDM correlations by result file",
        column_labels,
        corr_norm,
    )

    lam_ax.set_ylabel("Parameter")
    ldm_ax.set_ylabel("Parameter")
    ldm_ax.set_xlabel("Result file label")

    colorbar = fig.colorbar(image_lam, cax=colorbar_ax)
    colorbar.set_label("Correlation / adjusted self-value", labelpad=12)

    draw_mapping(mapping_ax, metadata_by_file, list(lam.columns))

    fig.suptitle(
        "LAM and LDM parameter correlations\n"
        "Self-correlation cells show the adjusted parameter value",
        fontsize=16,
        y=0.992,
    )

    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", default=".")
    parser.add_argument("--solution-pattern", default="*")
    parser.add_argument("--out-prefix", default="lam_ldm_correlation_heatmaps")
    args = parser.parse_args()

    plot_lam_ldm_heatmaps(
        solution_dir=args.solution_dir,
        solution_pattern=args.solution_pattern,
        out_prefix=args.out_prefix,
    )
