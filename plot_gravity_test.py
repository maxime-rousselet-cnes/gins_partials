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
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import close, figure, setp, subplots, tight_layout
from numpy import arange, array, cos, ndarray, ones, pi, quantile, sin, zeros, zeros_like
from pandas import date_range, to_datetime

SIGMA_VALUES = [-13, -14]
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
}
LAM_KEY: Parameter = (ParameterType.LAM, None, None, None, None)
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

    m = fullmatch(r"(C|P)(A|B|AA|C|S)_(\d)(\d)", name)

    if m:

        family, suffix, degree, order = m.groups()

        return (
            TYPE_MAP[family],
            int(degree),
            int(order),
            None,
            MODE_MAP[suffix],
        )

    return LAM_KEY if name == "LAM" else (LDM_KEY if name == "LDM" else LTM_KEY)


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
                parameter_type: {d_o: 0.0 for d_o in [(2, 0), (2, 1), (4, 0), (4, 1), (6, 0)]}
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
) -> tuple[
    dict[Parameter, float], dict[Parameter, float], dict[tuple[Parameter, Parameter], float]
]:
    """
    Produces uncrossed solution figures for a given solution file.
    """

    file_path_to_save = create_parallel_path(root=root, file=file, output_root=output_root)
    solutions, formal_uncertainties, correlations = ingest_dynamo_d_solution(file=file)
    file_to_save = file_path_to_save.parent.joinpath(file_path_to_save.name + ".pdf")

    if solutions == {} or file_to_save.exists():

        return {}, {}, {}

    figure, axes = subplots(7, 1, figsize=(14, 26), sharex=True)
    i_ax = 0

    for degree in [2, 4, 6]:

        for order in [0, 1] if degree < 5 else [0]:

            for parameter_type in (
                [ParameterType.C, ParameterType.S] if order == 1 else [ParameterType.C]
            ):

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
                    alpha=0.6,
                    label=rf"Reference ${N_SIGMAS}\sigma$",
                )
                ax.fill_between(
                    reference_dates,
                    reference_values - reference_sigmas,
                    reference_values + reference_sigmas,
                    color="red",
                    alpha=0.8,
                    label=r"Reference $1\sigma$",
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
    alpha: bool,
    delta: bool,
    tau_m: bool,
    root: Path = Path("solution"),
    output_root: Path = Path("solution_figures"),
) -> None:
    """
    Iterates on all solutions of the root directory and produces uncrossed solution figures.
    """

    t_0 = time()

    alpha_subdirectory = root / ("fix_alpha_" + str(alpha).lower())
    delta_subdirectory = alpha_subdirectory / ("fix_log10_delta_" + str(delta).lower())
    tau_m_subdirectory = delta_subdirectory / ("fix_log10_tau_m_" + str(tau_m).lower())
    checkpoint_tuple = (
        alpha,
        delta,
        tau_m,
    )
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

        for g_subdirectory in tau_m_subdirectory.iterdir():

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
                root=root, file=tau_m_subdirectory, output_root=output_root
            )
            plot_comparative(gathered=gathered, output_path=file_path_to_save)

    print(alpha, delta, tau_m, time() - t_0)


def format_parameter(parameter: Parameter) -> Optional[str]:
    """
    Builds a short, human readable label for a parameter tuple.
    """

    p_type, degree, order, date, mode = parameter

    if p_type in (ParameterType.LAM, ParameterType.LDM, ParameterType.LTM):

        return (
            r"$\alpha$"
            if p_type == ParameterType.LAM
            else (r"$\log_{10} \Delta$" if p_type == ParameterType.LDM else r"$\log_{10} \tau_m$")
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

    sigma_exponent = column.split("E")[1].split("/")[0]
    modes = column.split("_G_")[1].replace("and_", "").split("_")

    return (
        r"$\sigma_G = 10^{"
        + str(sigma_exponent)
        + "}$: "
        + " & ".join([mode[:3] for mode in modes])
    )


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

    priority = {ParameterType.LAM: 0, ParameterType.LDM: 1, ParameterType.LTM: 2}

    return sorted(
        parameters,
        key=lambda parameter: (
            priority.get(parameter[0], 3),
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
    Comparative plot of solutions (upper left panel), uncertainty (upper right panel), correlations
    with alpha (lower left panel), correlations with delta (lower middle panel) and correlations
    with tau_m (lower right panel). Produces one figure per filename i.e. one figure per satellite
    combination and per tide mode.
    """

    by_column: dict[str, tuple[dict, dict, dict]]

    for filename, by_column in gathered.items():

        if (output_path / (filename + "_comparative.pdf")).exists() or (
            output_path / ("NEG_" + filename + "_comparative.pdf")
        ).exists():

            continue

        ordered_columns = order_columns(labels=list(by_column.keys()))
        parameter_set = set(
            sum(
                [
                    [
                        (parameter_type, degree, order, date, mode)
                        for (parameter_type, degree, order, date, mode), _ in by_column[
                            column_g_model
                        ][2].keys()
                        if date is None
                    ]
                    for column_g_model in ordered_columns
                ],
                start=[],
            )
        )
        parameters = []
        columns = ordered_columns[:2]

        for parameter_type, degree, order, date, mode in parameter_set:

            if not (
                ("pole_tide" in filename and order == 0)
                or ("solid_tide" in filename and order == 1)
            ):

                parameters += [(parameter_type, degree, order, date, mode)]

        parameters = order_parameters(parameters=parameters)
        parameters_subset = [
            (parameter_type, degree, order, date, mode)
            for parameter_type, degree, order, date, mode in parameters
            if not (order is not None and mode is None)
        ]

        for column in ordered_columns[2:]:

            if (
                len(
                    [
                        parameter
                        for parameter in parameters_subset
                        if parameter in by_column[column][0]
                    ]
                )
                > 3
            ):

                columns += [column]

        # Uninformative case.
        if len(parameters_subset) == 0 or (len(parameters_subset) == 2 and len(columns) == 2):

            continue

        solution_heatmap = zeros(shape=(len(parameters_subset), len(columns)))
        uncertainty_heatmap = zeros(shape=(len(parameters_subset), len(columns)))
        alpha_correlation_heatmap = zeros(shape=(len(parameters), len(columns)))
        delta_correlation_heatmap = zeros(shape=(len(parameters), len(columns)))
        tau_m_correlation_heatmap = zeros(shape=(len(parameters), len(columns)))

        for i_column_g_model, column_g_model in enumerate(columns):

            for i_parameter, parameter in enumerate(parameters_subset):

                if parameter in by_column[column_g_model][0]:

                    solution_heatmap[i_parameter, i_column_g_model] = by_column[column_g_model][0][
                        parameter
                    ]
                    uncertainty_heatmap[i_parameter, i_column_g_model] = by_column[column_g_model][
                        1
                    ][parameter]

            for i_parameter, parameter in enumerate(parameters):

                if parameter in by_column[column_g_model][0] or (
                    parameter[2] is not None and parameter[4] is None
                ):

                    if parameter != LAM_KEY and LAM_KEY in parameters:

                        alpha_correlation_heatmap[i_parameter, i_column_g_model] = by_column[
                            column_g_model
                        ][2][(LAM_KEY, parameter)]

                    if parameter != LDM_KEY and LDM_KEY in parameters:

                        delta_correlation_heatmap[i_parameter, i_column_g_model] = by_column[
                            column_g_model
                        ][2][(LDM_KEY, parameter)]

                    if parameter != LTM_KEY and LTM_KEY in parameters:

                        tau_m_correlation_heatmap[i_parameter, i_column_g_model] = by_column[
                            column_g_model
                        ][2][(LTM_KEY, parameter)]

        parameter_labels = [format_parameter(parameter=parameter) for parameter in parameters]
        parameter_subset_labels = [
            format_parameter(parameter=parameter) for parameter in parameters_subset
        ]
        column_labels = [format_column_labels(column=column) for column in columns]
        fig = figure(figsize=(30, 22))
        grid = GridSpec(nrows=2, ncols=6, figure=fig)

        ax_solutions = fig.add_subplot(grid[0, 0:3])
        ax_solutions.set_title("A. Solution", fontweight="bold")
        im_solution = ax_solutions.imshow(
            normalize_solution(tab=solution_heatmap), aspect="auto", cmap="copper", vmin=-1, vmax=1
        )
        ax_solutions.set_xticks(ticks=range(len(columns)), labels=column_labels)
        ax_solutions.set_yticks(ticks=range(len(parameters_subset)), labels=parameter_subset_labels)
        ax_solutions.tick_params(axis="x", labelrotation=-90)
        ax_solutions.set_xticks(arange(-0.5, solution_heatmap.shape[1], 1), minor=True)
        ax_solutions.set_yticks(arange(-0.5, solution_heatmap.shape[0], 1), minor=True)
        ax_solutions.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax_solutions.tick_params(which="minor", bottom=False, left=False)
        cbar_solutions = ax_solutions.figure.colorbar(im_solution, ax=ax_solutions)
        cbar_solutions.set_ticks([-1, 0, 1.0])
        cbar_solutions.set_label(r"Adjusted value", rotation=-90, ha="center")
        cbar_solutions.set_ticklabels(["Not", "Lowest", "Highest"])
        setp(cbar_solutions.ax.get_xticklabels(), rotation=-90, ha="center")

        ax_uncertainty = fig.add_subplot(grid[0, 3:6])
        ax_uncertainty.set_title("B. Formal Uncertainty", fontweight="bold")
        im_uncertainty = ax_uncertainty.imshow(
            normalize_uncertainty(
                solution_heatmap=solution_heatmap, uncertainty_heatmap=uncertainty_heatmap
            ),
            aspect="auto",
            cmap="Reds",
            vmin=0,
            vmax=1,
        )
        ax_uncertainty.set_xticks(ticks=range(len(columns)), labels=column_labels)
        ax_uncertainty.set_yticks(
            ticks=range(len(parameters_subset)), labels=parameter_subset_labels
        )
        ax_uncertainty.tick_params(axis="x", labelrotation=-90)
        ax_uncertainty.set_xticks(arange(-0.5, uncertainty_heatmap.shape[1], 1), minor=True)
        ax_uncertainty.set_yticks(arange(-0.5, uncertainty_heatmap.shape[0], 1), minor=True)
        ax_uncertainty.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax_uncertainty.tick_params(which="minor", bottom=False, left=False)
        cbar_uncertainties = ax_uncertainty.figure.colorbar(im_uncertainty, ax=ax_uncertainty)
        cbar_uncertainties.set_ticks([0, 1.0])
        cbar_uncertainties.set_label(
            r"Uncertainty normalized by solution value $\frac{\sigma_p}{|p|}$",
            rotation=-90,
            ha="center",
        )
        cbar_uncertainties.set_ticklabels(["0", "1"])
        setp(cbar_uncertainties.ax.get_xticklabels(), rotation=-90, ha="center")

        if len(parameters_subset) > 1:

            if LAM_KEY in parameters:

                ax_lam = fig.add_subplot(
                    grid[1, 0:2]
                    if LDM_KEY in parameters and LTM_KEY in parameters
                    else (
                        grid[1, 0:3]
                        if LDM_KEY in parameters or LTM_KEY in parameters
                        else grid[1, 0:6]
                    )
                )
                ax_lam.set_title(r"C. Correlations with $\alpha$", fontweight="bold")
                im_lam = ax_lam.imshow(
                    alpha_correlation_heatmap, aspect="auto", cmap="RdBu", vmin=-1, vmax=1
                )
                ax_lam.set_xticks(ticks=range(len(columns)), labels=column_labels)
                ax_lam.set_yticks(ticks=range(len(parameters)), labels=parameter_labels)
                ax_lam.tick_params(axis="x", labelrotation=-90)
                ax_lam.set_xticks(arange(-0.5, alpha_correlation_heatmap.shape[1], 1), minor=True)
                ax_lam.set_yticks(arange(-0.5, alpha_correlation_heatmap.shape[0], 1), minor=True)
                ax_lam.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
                ax_lam.tick_params(which="minor", bottom=False, left=False)
                cbar_lam = ax_lam.figure.colorbar(im_lam, ax=ax_lam)
                cbar_lam.set_ticks([-1.0, 0, 1.0])
                cbar_lam.set_label(r"Correlation", rotation=-90, ha="center")
                cbar_lam.set_ticklabels(["-1", "0", "1"])
                setp(cbar_lam.ax.get_xticklabels(), rotation=-90, ha="center")

            if LDM_KEY in parameters:

                ax_ldm = fig.add_subplot(
                    grid[1, 2:4]
                    if LAM_KEY in parameters and LTM_KEY in parameters
                    else (
                        grid[1, 0:3]
                        if LTM_KEY in parameters
                        else (grid[1, 3:6] if LAM_KEY in parameters else grid[1, 0:6])
                    )
                )
                ax_ldm.set_title(r"D. Correlations with $\log_{10} \Delta$", fontweight="bold")
                im_ldm = ax_ldm.imshow(
                    delta_correlation_heatmap, aspect="auto", cmap="RdBu", vmin=-1, vmax=1
                )
                ax_ldm.set_xticks(ticks=range(len(columns)), labels=column_labels)
                ax_ldm.set_yticks(ticks=range(len(parameters)), labels=parameter_labels)
                ax_ldm.tick_params(axis="x", labelrotation=-90)
                ax_ldm.set_xticks(arange(-0.5, delta_correlation_heatmap.shape[1], 1), minor=True)
                ax_ldm.set_yticks(arange(-0.5, delta_correlation_heatmap.shape[0], 1), minor=True)
                ax_ldm.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
                ax_ldm.tick_params(which="minor", bottom=False, left=False)
                cbar_ldm = ax_ldm.figure.colorbar(im_ldm, ax=ax_ldm)
                cbar_ldm.set_ticks([-1.0, 0, 1.0])
                cbar_ldm.set_label(r"Correlation", rotation=-90, ha="center")
                cbar_ldm.set_ticklabels(["-1", "0", "1"])
                setp(cbar_ldm.ax.get_xticklabels(), rotation=-90, ha="center")

            if LTM_KEY in parameters:

                ax_ltm = fig.add_subplot(
                    grid[1, 4:6]
                    if LDM_KEY in parameters and LAM_KEY in parameters
                    else (
                        grid[1, 3:6]
                        if LDM_KEY in parameters or LAM_KEY in parameters
                        else grid[1, 0:6]
                    )
                )
                ax_ltm.set_title(r"E. Correlations with $\log_{10} \tau_m$", fontweight="bold")
                im_ltm = ax_ltm.imshow(
                    tau_m_correlation_heatmap, aspect="auto", cmap="RdBu", vmin=-1, vmax=1
                )
                ax_ltm.set_xticks(ticks=range(len(columns)), labels=column_labels)
                ax_ltm.set_yticks(ticks=range(len(parameters)), labels=parameter_labels)
                ax_ltm.tick_params(axis="x", labelrotation=-90)
                ax_ltm.set_xticks(arange(-0.5, delta_correlation_heatmap.shape[1], 1), minor=True)
                ax_ltm.set_yticks(arange(-0.5, delta_correlation_heatmap.shape[0], 1), minor=True)
                ax_ltm.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
                ax_ltm.tick_params(which="minor", bottom=False, left=False)
                cbar_ltm = ax_ltm.figure.colorbar(im_ltm, ax=ax_ltm)
                cbar_ltm.set_ticks([-1.0, 0, 1.0])
                cbar_ltm.set_label(r"Correlation", rotation=-90, ha="center")
                cbar_ltm.set_ticklabels(["-1", "0", "1"])
                setp(cbar_ltm.ax.get_xticklabels(), rotation=-90, ha="center")

        has_negative_uncertainties = zeros(shape=(len(columns)), dtype=bool)

        for i_parameter, parameter in enumerate(parameters_subset):

            for i_column_g_model, column_g_model in enumerate(columns):

                if parameter in by_column[column_g_model][0]:

                    ax_solutions.text(
                        i_column_g_model,
                        i_parameter,
                        f"{solution_heatmap[i_parameter, i_column_g_model]:.2g}",
                        ha="center",
                        va="center",
                        color="black",
                    )
                    ax_uncertainty.text(
                        i_column_g_model,
                        i_parameter,
                        f"{uncertainty_heatmap[i_parameter, i_column_g_model]:.2g}",
                        ha="center",
                        va="center",
                        color=(
                            (
                                "black"
                                if uncertainty_heatmap[i_parameter, i_column_g_model]
                                < 0.7 * abs(solution_heatmap[i_parameter, i_column_g_model])
                                else "white"
                            )
                            if uncertainty_heatmap[i_parameter, i_column_g_model] > 0
                            else "r"
                        ),
                    )

                    if uncertainty_heatmap[i_parameter, i_column_g_model] < 0 or column_g_model in [
                        "fix_g",
                        "no_g_model",
                    ]:

                        has_negative_uncertainties[i_column_g_model] = True

        for i_parameter, parameter in enumerate(parameters):

            for i_column_g_model, column_g_model in enumerate(columns):

                if parameter in by_column[column_g_model][0] or (
                    parameter[2] is not None and parameter[4] is None and column_g_model != "fix_g"
                ):

                    if len(parameters_subset) > 1:

                        if parameter != LAM_KEY and LAM_KEY in parameters:

                            ax_lam.text(
                                i_column_g_model,
                                i_parameter,
                                f"{alpha_correlation_heatmap[i_parameter, i_column_g_model]:.2g}",
                                ha="center",
                                va="center",
                                color=(
                                    "black"
                                    if abs(alpha_correlation_heatmap[i_parameter, i_column_g_model])
                                    < 0.8
                                    else "white"
                                ),
                            )

                        if parameter != LDM_KEY and LDM_KEY in parameters:

                            ax_ldm.text(
                                i_column_g_model,
                                i_parameter,
                                f"{delta_correlation_heatmap[i_parameter, i_column_g_model]:.2g}",
                                ha="center",
                                va="center",
                                color=(
                                    "black"
                                    if abs(delta_correlation_heatmap[i_parameter, i_column_g_model])
                                    < 0.8
                                    else "white"
                                ),
                            )

                        if parameter != LTM_KEY and LTM_KEY in parameters:

                            ax_ltm.text(
                                i_column_g_model,
                                i_parameter,
                                f"{tau_m_correlation_heatmap[i_parameter, i_column_g_model]:.2g}",
                                ha="center",
                                va="center",
                                color=(
                                    "black"
                                    if abs(tau_m_correlation_heatmap[i_parameter, i_column_g_model])
                                    < 0.8
                                    else "white"
                                ),
                            )

        fig.suptitle(" ".join([word.capitalize() for word in filename[9:].split("_")]))
        tight_layout()

        if has_negative_uncertainties.all():

            filename = "NEG_" + filename

        fig.savefig(output_path / (filename + "_comparative.pdf"), bbox_inches="tight")
        close(fig)


def parse_job_args() -> Namespace:
    """
    Defines a parsing function for command-line arguments.
    """

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="solution")
    parser.add_argument("--alpha", action="store_true", default=False)
    parser.add_argument("--delta", action="store_true", default=False)
    parser.add_argument("--tau_m", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_job_args()
    plot_solutions(alpha=args.alpha, delta=args.delta, tau_m=args.tau_m, root=Path(args.root))
