"""
plot_graivty_test.py example
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from re import compile, fullmatch
from time import time
from typing import Optional

import matplotlib.dates as mdates
from base_models import DATA_PATH
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import close, figure, subplots
from numpy import arange, array, cos, ma, ndarray, ones, pi, quantile, sin, zeros, zeros_like
from pandas import date_range, to_datetime

SIGMA_VALUES = [-12, -13, -14]
N_SIGMAS = 3
CHECKPOINT_TUPLES = []
FLOAT_REGEX = compile(r"[+-]?\d+\.\d+E[+-]\d+|[+-]?\.\d+E[+-]\d+")
SAFETY_DIVERGENCE_FACTOR = 3


class ParameterType(Enum):
    """
    Assimilable to Signaletic element prefix.
    """

    S = auto()
    C = auto()
    LAM = auto()
    LQM = auto()
    LDM = auto()
    LTM = auto()


class ParameterMode(Enum):
    """
    Time-dependent base components for gravity field modeling.
    """

    ACC = auto()
    TREND = auto()
    COS = auto()
    SIN = auto()
    BIAS = auto()
    ADD_COS = auto()
    ADD_SIN = auto()


TYPE_MAP = {
    "C": ParameterType.C,
    "P": ParameterType.S,
}
MODE_MAP = {
    "AA": ParameterMode.ACC,
    "A": ParameterMode.TREND,
    "C": ParameterMode.COS,
    "S": ParameterMode.SIN,
    "B": ParameterMode.BIAS,
    "P": ParameterMode.ADD_COS,
    "Q": ParameterMode.ADD_SIN,
}
LAM_KEY: Parameter = (ParameterType.LAM, None, None, None, None)
LQM_KEY: Parameter = (ParameterType.LQM, None, None, None, None)
LDM_KEY: Parameter = (ParameterType.LDM, None, None, None, None)
LTM_KEY: Parameter = (ParameterType.LTM, None, None, None, None)
MODE_LETTERS = {
    ParameterMode.ACC: "AA",
    ParameterMode.TREND: "A",
    ParameterMode.COS: "C",
    ParameterMode.SIN: "S",
    ParameterMode.BIAS: "B",
}

Parameter = tuple[ParameterType, int | None, int | None, datetime | None, ParameterMode | None]


def gravity_timeseries(
    filename: str = "RL05.shc",
    start: str = "1991-01-11",
    end: str = "2025-12-31",
    step_days: int = 30,
) -> tuple[ndarray, dict[str, ndarray], dict[str, ndarray]]:
    """
    Extract C20, C21, C40, C41, C60 and their uncertainties from a
    time-dependent gravity-field model.
    """

    wanted = {(2, 0), (2, 1), (4, 0), (4, 1), (6, 0)}
    dates = date_range(start, end, freq=f"{step_days}D")
    values = {
        coeff: zeros_like(a=dates, dtype=float)
        for coeff in ["C_20", "C_21", "S_21", "C_40", "C_41", "S_41", "C_60"]
    }
    uncertainties = {coeff: zeros_like(a=dates, dtype=float) for coeff in values.keys()}

    with open(filename) as f:

        for line in f:

            if line[0] != "G":

                continue

            p = line.split()
            parameter_type = p[0].lower()
            n, m = int(p[1]), int(p[2])

            if (n, m) not in wanted:

                continue

            c_value, s_value = float(p[3]), float(p[4])
            c_sigma, s_sigma = float(p[5]), float(p[6])
            t0 = to_datetime(p[-3].split(".")[0], format="%Y%m%d")
            t1 = to_datetime(p[-2].split(".")[0], format="%Y%m%d")
            indices = (dates >= t0) * (dates <= t1)
            shift_dates = dates[indices] - t0
            normalized_dates = shift_dates.total_seconds() / 86400 / 365

            for coeff in "C" if m == 0 else "CS":

                value = c_value if coeff == "C" else s_value
                sigma = c_sigma if coeff == "C" else s_sigma

                if "bias" in parameter_type:

                    values[coeff + "_" + str(n) + str(m)][indices] += value
                    uncertainties[coeff + "_" + str(n) + str(m)][indices] += sigma

                elif "drift" in parameter_type:

                    values[coeff + "_" + str(n) + str(m)][indices] += value * normalized_dates
                    uncertainties[coeff + "_" + str(n) + str(m)][indices] += (
                        sigma * normalized_dates
                    )

                elif "cos" in parameter_type:

                    values[coeff + "_" + str(n) + str(m)][indices] += value * cos(
                        2 * pi * normalized_dates
                    )
                    uncertainties[coeff + "_" + str(n) + str(m)][indices] += sigma * cos(
                        2 * pi * normalized_dates
                    )

                elif "sin" in parameter_type:

                    values[coeff + "_" + str(n) + str(m)][indices] += value * sin(
                        2 * pi * normalized_dates
                    )
                    uncertainties[coeff + "_" + str(n) + str(m)][indices] += sigma * sin(
                        2 * pi * normalized_dates
                    )

            continue

    return dates, values, uncertainties


def parse_parameter_name(
    name: str,
) -> Parameter:
    """
    Parse a parameter name. Only manages the ParameterType enum parameters.
    """

    name = name.strip()

    m = fullmatch(r"G([SC])N\s+(\d+)\s+(\d+)\s+(\d{8})", name)

    if m:

        typ, degree, order, date = m.groups()

        return (
            ParameterType[typ],
            int(degree),
            int(order),
            datetime.strptime(date, "%Y%m%d"),
            None,
        )

    m = fullmatch(r"(C|P)(A|B|AA|C|S|P|Q)_(\d)(\d)", name)

    if m:

        family, suffix, degree, order = m.groups()

        return (
            TYPE_MAP[family],
            int(degree),
            int(order),
            None,
            MODE_MAP[suffix],
        )

    return (
        LAM_KEY
        if name == "LAM"
        else (LQM_KEY if name == "LQM" else (LDM_KEY if name == "LDM" else LTM_KEY))
    )


def ingest_dynamo_d_solution(
    file: Path,
) -> tuple[
    dict[Parameter, float], dict[Parameter, float], dict[tuple[Parameter, Parameter], float]
]:
    """
    Extracts relevant data from a Dynamo D solution file.
    Returns solutions, formal uncertainties and correlations.
    Concerning GCN/GSN parameters, reduces the correlations to their root mean square value.
    """

    solutions: dict[Parameter, float] = {}
    formal_uncertainties: dict[Parameter, float] = {}
    correlations: dict[tuple[Parameter, Parameter], float] = {}
    n_c, n_s = 0, 0

    with open(file, "r", errors="ignore") as f:

        lines = f.readlines()

    # Gets solutions.
    start = None

    for i, line in enumerate(lines):

        if line.strip() == "SOLUTION":

            start = i + 2

            break

    for i, line in enumerate(lines[start:]):

        if line.strip() == "INVERSE MATRIX":

            start = start + i + 2
            break

        values = FLOAT_REGEX.findall(line[24:])
        uncertainty = float(values[3])
        parameter = parse_parameter_name(name=line[:24])
        solutions[parameter] = float(values[2])

        if uncertainty != 0.0:

            formal_uncertainties[parameter] = uncertainty

        if parameter[3] is not None:

            if parameter[0] == ParameterType.C:

                n_c += 1

            elif parameter[0] == ParameterType.S:

                n_s += 1

    # Gets matrix.
    parameters = list(formal_uncertainties.keys())

    if len(parameters) == 0:

        print(file)
        return {}, {}, {}

    i_parameter = 0
    j_parameter = 0
    matrix = ones(shape=(len(parameters), len(parameters)))

    for line in lines[start:]:

        if line.strip() == "":

            break

        n_parameters_line = len(line.strip()) // 20

        for k_parameter in range(n_parameters_line):

            matrix[i_parameter, j_parameter] = float(
                line[20 * k_parameter : 20 * (k_parameter + 1)]
            )
            j_parameter += 1

        if j_parameter > i_parameter:

            j_parameter = 0
            i_parameter += 1

    # Formats matrix.
    for i_parameter, parameter_i in enumerate(parameters):

        if parameter_i[3] is None:

            sum_abs = {
                parameter_type: {
                    d_o: 0.0
                    for d_o in [(2, 1), (4, 1)]
                    + ([(2, 0), (4, 0), (6, 0)] if parameter_type == ParameterType.C else [])
                }
                for parameter_type in [ParameterType.C, ParameterType.S]
            }

            for j_parameter in range(i_parameter + 1):

                parameter_j = parameters[j_parameter]
                correlation = (
                    matrix[i_parameter, j_parameter]
                    / (matrix[i_parameter, i_parameter] * matrix[j_parameter, j_parameter]) ** 0.5
                )

                if parameter_j[3] is None:

                    correlations[(parameter_i, parameter_j)] = correlation
                    correlations[(parameter_j, parameter_i)] = correlations[
                        (parameter_i, parameter_j)
                    ]

                else:

                    sum_abs[parameter_j[0]][(parameter_j[1], parameter_j[2])] += abs(correlation)

            for parameter_type, degree_order_abs in sum_abs.items():

                for (degree, order), abs_value in degree_order_abs.items():

                    correlations[(parameter_i, (parameter_type, degree, order, None, None))] = (
                        abs_value / (n_c if parameter_type == ParameterType.C else n_s)
                    )
                    correlations[((parameter_type, degree, order, None, None), parameter_i)] = (
                        correlations[(parameter_i, (parameter_type, degree, order, None, None))]
                    )

    return solutions, formal_uncertainties, correlations


def create_parallel_path(root: Path, file: Path, output_root: Path) -> Path:
    """
    Creates a parallel path for the output file based on the input file's path.
    """

    output_path = output_root / file.relative_to(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def produce_uncrossed_figures(
    root: Path,
    file: Path,
    output_root: Path,
    reference_dates: ndarray,
    reference_field: tuple[dict[str, ndarray], dict[str, ndarray]],
    get_anyway: bool = False,
) -> tuple[
    dict[Parameter, float], dict[Parameter, float], dict[tuple[Parameter, Parameter], float]
]:
    """
    Produces uncrossed solution figures for a given solution file.
    """

    file_path_to_save = create_parallel_path(root=root, file=file, output_root=output_root)
    solutions, formal_uncertainties, correlations = ingest_dynamo_d_solution(file=file)
    file_to_save = file_path_to_save.parent.joinpath(file_path_to_save.name + ".pdf")

    if not get_anyway and (solutions == {} or file_to_save.exists()):

        return {}, {}, {}

    panels = 7 if get_anyway else (4 if "pole" in file.name else (3 if "solid" in file.name else 7))
    figure, axes = subplots(
        panels,
        1,
        figsize=(14, 4 * panels - 2),
        sharex=True,
    )
    i_ax = 0

    for degree in [2, 4, 6]:

        for order in [0, 1] if degree < 5 else [0]:

            for parameter_type in (
                [ParameterType.C, ParameterType.S] if order == 1 else [ParameterType.C]
            ):

                if not get_anyway and (
                    (order == 0 and "pole" in file.name) or (order == 1 and "solid" in file.name)
                ):

                    continue

                coeff = (
                    (r"$C_{" if parameter_type == ParameterType.C else r"$S_{")
                    + str(degree)
                    + str(order)
                    + "}$"
                )
                dates = [
                    parameter[3]
                    for parameter in solutions.keys()
                    if parameter[0] == parameter_type
                    and parameter[1] == degree
                    and parameter[2] == order
                    and parameter[3] is not None
                ]
                dates.sort()
                values = [solutions[(parameter_type, degree, order, date, None)] for date in dates]
                sigmas = array(
                    object=[
                        (
                            0
                            if (parameter_type, degree, order, date, None)
                            not in formal_uncertainties
                            else formal_uncertainties[(parameter_type, degree, order, date, None)]
                        )
                        for date in dates
                    ],
                    dtype=float,
                )
                ax: Axes = axes[i_ax]
                ax.fill_between(
                    dates,
                    values - N_SIGMAS * abs(sigmas),
                    values + N_SIGMAS * abs(sigmas),
                    color="b" if array(object=sigmas > 0, dtype=bool).all() else "r",
                    alpha=0.4,
                    label=rf"Solution ${N_SIGMAS}\sigma$",
                )
                ax.scatter(
                    dates,
                    values,
                    color="b" if array(object=sigmas > 0, dtype=bool).all() else "r",
                    label="Solution",
                )
                identifier = (
                    ("C" if parameter_type == ParameterType.C else "S")
                    + "_"
                    + str(degree)
                    + str(order)
                )
                reference_values = reference_field[0][identifier]
                reference_sigmas = reference_field[1][identifier]
                ax.fill_between(
                    reference_dates,
                    reference_values - N_SIGMAS * reference_sigmas,
                    reference_values + N_SIGMAS * reference_sigmas,
                    color="orange",
                    alpha=0.4,
                    label=rf"Reference ${N_SIGMAS}\sigma$",
                )
                ax.fill_between(
                    reference_dates,
                    reference_values - reference_sigmas,
                    reference_values + reference_sigmas,
                    color="red",
                    alpha=0.6,
                    label=r"Reference $1\sigma$",
                )
                ax.scatter(
                    reference_dates, reference_values, color="r", marker="x", label="Reference"
                )
                ax.set_title(coeff)
                ax.set_xlabel("Date")
                ax.set_ylabel("Solution")
                ax.grid(True)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                q2max, q3max = quantile(values + N_SIGMAS * abs(sigmas), (0.5, 0.75))
                q1min, q2min = quantile(values - N_SIGMAS * abs(sigmas), (0.25, 0.5))
                ax.set_ylim(
                    min(
                        min(reference_values) - 2 * max(reference_sigmas),
                        q2min - 2 * (q2min - q1min),
                    ),
                    max(
                        max(reference_values) + 2 * max(reference_sigmas),
                        q2max + 2 * (q3max - q2max),
                    ),
                )

                if i_ax == 0:

                    ax.legend()

                i_ax += 1

    figure.autofmt_xdate()
    figure.tight_layout()
    # indices = [date in reference_dates for date in dates]
    # reference_indices = [date in dates for date in reference_dates]
    figure.savefig(file_to_save)
    close(figure)

    return solutions, formal_uncertainties, correlations


def plot_solutions(
    lam: bool,
    lqm: bool,
    ldm: bool,
    ltm: bool,
    root: Path = Path("solution"),
    output_root: Path = DATA_PATH.joinpath("solution_figures"),
) -> None:
    """
    Iterates on all solutions of the root directory and produces uncrossed solution figures.
    """

    t_0 = time()

    lam_subdirectory = root / ("fix_lam_" + str(lam).lower())
    lqm_subdirectory = lam_subdirectory / ("fix_lqm_" + str(lqm).lower())
    ldm_subdirectory = lqm_subdirectory / ("fix_ldm_" + str(ldm).lower())
    ltm_subdirectory = ldm_subdirectory / ("fix_ltm_" + str(ltm).lower())
    checkpoint_tuple = (lam, lqm, ldm, ltm)
    reference_dates, reference_values, reference_uncertainties = gravity_timeseries()

    if checkpoint_tuple not in CHECKPOINT_TUPLES:

        gathered: dict[
            str,
            dict[
                str,
                tuple[
                    dict[Parameter, float],
                    dict[Parameter, float],
                    dict[tuple[Parameter, Parameter], float],
                ],
            ],
        ] = {}

        for g_subdirectory in ltm_subdirectory.iterdir():

            fix_g = g_subdirectory.name.split("_")[-1] == "true"

            for sub in g_subdirectory.iterdir():

                if sub.is_file():

                    gathered.setdefault(sub.stem, {})["fix_g" if fix_g else "no_g_model"] = (
                        produce_uncrossed_figures(
                            root=root,
                            file=sub,
                            output_root=output_root,
                            reference_dates=reference_dates,
                            reference_field=(reference_values, reference_uncertainties),
                        )
                    )

                if not fix_g:

                    if sub.is_dir():

                        if int(sub.name.split("E")[-1]) in SIGMA_VALUES:

                            for g_model_subdirectory in sub.iterdir():

                                for file in g_model_subdirectory.iterdir():

                                    gathered.setdefault(file.stem, {})[
                                        sub.name + "/" + g_model_subdirectory.name
                                    ] = produce_uncrossed_figures(
                                        root=root,
                                        file=file,
                                        output_root=output_root,
                                        reference_dates=reference_dates,
                                        reference_field=(reference_values, reference_uncertainties),
                                    )

        if gathered:

            file_path_to_save = create_parallel_path(
                root=root, file=ltm_subdirectory, output_root=output_root
            )
            plot_comparative(gathered=gathered, output_path=file_path_to_save)

    print(lam, lqm, ldm, ltm, time() - t_0)


def format_parameter(parameter: Parameter) -> Optional[str]:
    """
    Builds a short, human readable label for a parameter tuple.
    """

    p_type, degree, order, date, mode = parameter

    if p_type in (ParameterType.LAM, ParameterType.LQM, ParameterType.LDM, ParameterType.LTM):

        return (
            r"$\alpha$"
            if p_type == ParameterType.LAM
            else (
                r"$\log_{10} Q_\mu$"
                if p_type == ParameterType.LQM
                else (
                    r"$\log_{10} \Delta$" if p_type == ParameterType.LDM else r"$\log_{10} \tau_m$"
                )
            )
        )

    type_letter = "$C_{" if p_type == ParameterType.C else "$S_{"

    if date is not None:

        return None

    if mode is None:

        return type_letter + str(degree) + str(order) + "}$ (epochs AM)"

    return type_letter + str(degree) + str(order) + "}^{" + MODE_LETTERS[mode] + "}$"


def format_column_labels(column: str) -> str:
    """
    Builds a short, human readable label for a gravity field model depending on modes and sigma.
    """

    if column == "fix_g":

        return "Unadjusted Gravity field"

    if column == "no_g_model":

        return "Adjusted unmodeled Gravity field"

    modes = column.split("_G_")[1].replace("and_", "").split("_")

    return " & ".join([mode[:3] for mode in modes])


def order_columns(labels: list[str]) -> list[str]:
    """
    Orders comparative columns: fix_g case first, then no g model case, then g models by sigma
    value, then by mode.
    """

    priority = {"fix_g": 0, "no_g_model": 1}

    return sorted(
        labels,
        key=lambda label: (priority.get(label, 2), label.split("/")[0], label.split("/")[-1]),
    )


def order_parameters(parameters: list[Parameter]) -> list[Parameter]:
    """
    Orders parameters: Rheological parameters first, then per spherical harmonic, then per mode.
    """

    priority = {
        ParameterType.LAM: 0,
        ParameterType.LQM: 1,
        ParameterType.LDM: 2,
        ParameterType.LTM: 3,
    }

    return sorted(
        parameters,
        key=lambda parameter: (
            priority.get(parameter[0], 4),
            parameter[1],
            parameter[2],
            parameter[0].value,
            0 if parameter[4] is None else parameter[4].value,
        ),
    )


def normalize_solution(tab: ndarray) -> ndarray:
    """
    Normalizes every line between 0 and 1 using its maximal value and its minimal value.
    """

    output = zeros(shape=tab.shape)

    for i_line, line in enumerate(tab):

        non_null_line = [element for element in line if element != 0]
        min_line = min(non_null_line)
        max_line = max(non_null_line)

        for j_column, value in enumerate(line):

            output[i_line, j_column] = (
                -1
                if tab[i_line, j_column] == 0
                else (value - min_line) / (max_line - min_line + 1e-15)
            )

    return output


def normalize_uncertainty(solution_heatmap: ndarray, uncertainty_heatmap: ndarray) -> ndarray:
    """
    Normalizes uncertainties with respect to the solution they refer to.
    """

    output = zeros(shape=solution_heatmap.shape)

    for i_line, (solution_line, uncertainty_line) in enumerate(
        zip(solution_heatmap, uncertainty_heatmap)
    ):

        for j_column, (solution, uncertainty) in enumerate(zip(solution_line, uncertainty_line)):

            output[i_line, j_column] = abs(uncertainty) / (abs(solution) + 1e-15)

    return output


def plot_comparative(
    gathered: dict[
        str,
        dict[
            str,
            tuple[
                dict[Parameter, float],
                dict[Parameter, float],
                dict[tuple[Parameter, Parameter], float],
            ],
        ],
    ],
    output_path: Path,
) -> None:
    """
    Plots solutions, formal uncertainties, and available rheology correlations.

    The upper row has two panels. The lower row divides evenly among the
    available LAM, LQM, LDM, and LTM correlation panels, in that order.
    Missing correlations are blank; self-correlations remain unannotated.
    Existing output files and the original NEG_ naming rule are preserved.
    """

    rheology_keys = (LAM_KEY, LQM_KEY, LDM_KEY, LTM_KEY)
    baseline_columns = {"fix_g", "no_g_model"}
    output_path.mkdir(parents=True, exist_ok=True)

    def draw_heatmap(
        ax,
        values,
        row_labels,
        title,
        cmap,
        vmin,
        vmax,
        colorbar_label,
        colorbar_ticks,
        colorbar_ticklabels,
    ):
        """Apply the common image, ticks, cell grid, and colorbar styling."""
        ax.set_title(title, fontweight="bold")
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(columns)), labels=column_labels)
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.tick_params(axis="x", labelrotation=-90)
        ax.set_xticks(arange(-0.5, values.shape[1], 1), minor=True)
        ax.set_yticks(arange(-0.5, values.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = ax.figure.colorbar(image, ax=ax)
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_ticklabels(colorbar_ticklabels)
        colorbar.set_label(colorbar_label, rotation=-90, ha="center")

    def annotate(ax, column, row, value, color="black"):
        ax.text(column, row, f"{value:.2e}", ha="center", va="center", color=color)

    for filename, by_column in gathered.items():
        if any(
            (output_path / f"{prefix}{filename}_comparative.pdf").exists()
            for prefix in ("", "NEG_")
        ):
            continue

        ordered_columns = order_columns(labels=list(by_column))
        parameter_set = {
            parameter
            for column in ordered_columns
            for parameter, _ in by_column[column][2]
            if parameter[3] is None
        }
        parameters = order_parameters(
            parameters=[
                parameter
                for parameter in parameter_set
                if not (
                    ("pole_tide" in filename and parameter[2] == 0)
                    or ("solid_tide" in filename and parameter[2] == 1)
                )
            ]
        )
        parameters_subset = [
            parameter
            for parameter in parameters
            if not (parameter[2] is not None and parameter[4] is None)
        ]
        # Include modeled columns only when a gravity-model parameter is present.
        # A numeric '> 3' test would now admit four rheological parameters alone.
        columns = [
            column
            for column in ordered_columns
            if column in baseline_columns
            or any(
                parameter not in rheology_keys and parameter in by_column[column][0]
                for parameter in parameters_subset
            )
        ]
        if (
            not parameters_subset
            or not columns
            or (len(parameters_subset) == 2 and len(columns) == 2)
        ):
            continue

        active_keys = [
            key for key in rheology_keys if key in parameters and len(parameters_subset) > 1
        ]
        shape = (len(parameters_subset), len(columns))
        solution_heatmap = zeros(shape=shape)
        uncertainty_heatmap = zeros(shape=shape)
        solution_present = zeros(shape=shape, dtype=bool)
        uncertainty_present = zeros(shape=shape, dtype=bool)
        correlation_heatmaps = {
            key: ma.masked_all((len(parameters), len(columns))) for key in active_keys
        }

        for j, column in enumerate(columns):
            solutions, uncertainties, correlations = by_column[column]
            for i, parameter in enumerate(parameters_subset):
                if parameter in solutions:
                    solution_present[i, j] = True
                    solution_heatmap[i, j] = solutions[parameter]
                    if parameter in uncertainties:
                        uncertainty_present[i, j] = True
                        uncertainty_heatmap[i, j] = uncertainties[parameter]
            for i, parameter in enumerate(parameters):
                is_epoch_mean = parameter[2] is not None and parameter[4] is None
                if parameter not in solutions and not is_epoch_mean:
                    continue
                for key, heatmap in correlation_heatmaps.items():
                    pair = (key, parameter)
                    if parameter != key and pair in correlations:
                        heatmap[i, j] = correlations[pair]

        # Use dictionary membership to distinguish missing entries from real zeros.
        normalized_solutions = zeros(shape=shape) - 1
        for i, present in enumerate(solution_present):
            if present.any():
                values = solution_heatmap[i, present]
                normalized_solutions[i, present] = (values - values.min()) / (
                    values.max() - values.min() + 1e-15
                )
        normalized_uncertainties = ma.array(
            normalize_uncertainty(solution_heatmap, uncertainty_heatmap),
            mask=~uncertainty_present,
        )
        parameter_labels = [format_parameter(p) for p in parameters]
        subset_labels = [format_parameter(p) for p in parameters_subset]
        column_labels = [format_column_labels(column) for column in columns]

        fig = figure(figsize=(max(30, 10 * len(active_keys)), 22 if active_keys else 11))
        try:
            # Twelve columns divide equally into one, two, three, or four panels.
            grid = GridSpec(nrows=2 if active_keys else 1, ncols=12, figure=fig)
            ax_solutions = fig.add_subplot(grid[0, :6])
            ax_uncertainty = fig.add_subplot(grid[0, 6:])
            draw_heatmap(
                ax_solutions,
                normalized_solutions,
                subset_labels,
                "A. Solution",
                "copper",
                -1,
                1,
                "Adjusted value",
                [-1, 0, 1],
                ["Not", "Lowest", "Highest"],
            )
            draw_heatmap(
                ax_uncertainty,
                normalized_uncertainties,
                subset_labels,
                "B. Formal Uncertainty",
                "Reds",
                0,
                1,
                r"Uncertainty normalized by solution value $\frac{\sigma_p}{|p|}$",
                [0, 1],
                ["0", "1"],
            )
            correlation_axes = {}
            for index, (key, heatmap) in enumerate(correlation_heatmaps.items()):
                width = 12 // len(active_keys)
                ax = fig.add_subplot(grid[1, index * width : (index + 1) * width])
                correlation_axes[key] = ax
                draw_heatmap(
                    ax,
                    heatmap,
                    parameter_labels,
                    f"{chr(ord('C') + index)}. Correlations with {format_parameter(key)}",
                    "RdBu",
                    -1,
                    1,
                    "Correlation",
                    [-1, 0, 1],
                    ["-1", "0", "1"],
                )

            has_negative_uncertainties = zeros(len(columns), dtype=bool)
            for j, column in enumerate(columns):
                for i, parameter in enumerate(parameters_subset):
                    if not solution_present[i, j]:
                        continue
                    value = solution_heatmap[i, j]
                    annotate(ax_solutions, j, i, value)
                    if column in baseline_columns:
                        has_negative_uncertainties[j] = True
                    if not uncertainty_present[i, j]:
                        continue
                    sigma = uncertainty_heatmap[i, j]
                    color = (
                        "r" if sigma <= 0 else ("black" if sigma < 0.7 * abs(value) else "white")
                    )
                    annotate(ax_uncertainty, j, i, sigma, color)
                    if sigma < 0:
                        has_negative_uncertainties[j] = True

            for key, ax in correlation_axes.items():
                heatmap = correlation_heatmaps[key]
                for i, parameter in enumerate(parameters):
                    for j, column in enumerate(columns):
                        if ma.is_masked(heatmap[i, j]):
                            continue
                        if parameter not in by_column[column][0] and column == "fix_g":
                            continue
                        value = heatmap[i, j]
                        annotate(ax, j, i, value, "black" if abs(value) < 0.8 else "white")

            fig.suptitle(" ".join(word.capitalize() for word in filename[9:].split("_")))
            fig.tight_layout()
            prefix = "NEG_" if has_negative_uncertainties.all() else ""
            fig.savefig(output_path / f"{prefix}{filename}_comparative.pdf", bbox_inches="tight")
        finally:
            close(fig)


def parse_job_args() -> Namespace:
    """
    Defines a parsing function for command-line arguments.
    """

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--lam", action="store_true", default=False)
    parser.add_argument("--lqm", action="store_true", default=False)
    parser.add_argument("--ldm", action="store_true", default=False)
    parser.add_argument("--ltm", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_job_args()
    plot_solutions(lam=args.lam, lqm=args.lqm, ldm=args.ldm, ltm=args.ltm, root=Path(args.root))
