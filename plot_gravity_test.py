from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
from os import listdir
from pathlib import Path
from re import compile, fullmatch

import matplotlib.dates as mdates
from matplotlib.pyplot import close, subplots
from numpy import array, ones, quantile

FLOAT_REGEX = compile(r"[+-]?\d+\.\d+E[+-]\d+|[+-]?\.\d+E[+-]\d+")


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

    TREND = auto()
    BIAS = auto()
    ACC = auto()
    COS = auto()
    SIN = auto()


TYPE_MAP = {
    "C": ParameterType.C,
    "P": ParameterType.S,
}
MODE_MAP = {
    "A": ParameterMode.TREND,
    "B": ParameterMode.BIAS,
    "AA": ParameterMode.ACC,
    "C": ParameterMode.COS,
    "S": ParameterMode.SIN,
}

Parameter = tuple[ParameterType, int | None, int | None, datetime | None, ParameterMode | None]


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

    return (
        (ParameterType.LAM, None, None, None, None)
        if name == "LAM"
        else (
            (ParameterType.LDM, None, None, None, None)
            if name == "LDM"
            else (ParameterType.LTM, None, None, None, None)
        )
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

            sum_squares = {
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

                    sum_squares[parameter_j[0]][(parameter_j[1], parameter_j[2])] += correlation**2

            for parameter_type, degree_order_squares in sum_squares.items():

                for (degree, order), square in degree_order_squares.items():

                    correlations[(parameter_i, (parameter_type, degree, order, None, None))] = (
                        square**0.5
                    )

    return solutions, formal_uncertainties, correlations


def create_parallel_path(root: Path, file: Path, output_root: Path) -> Path:
    """
    Creates a parallel path for the output file based on the input file's path.
    """

    output_path = output_root / file.relative_to(root)
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def produce_uncrossed_figures(root: Path, file: Path, output_root: Path) -> None:
    """
    Produces uncrossed solution figures for a given solution file.
    """

    file_path_to_save = create_parallel_path(root=root, file=file, output_root=output_root)

    if listdir(file_path_to_save):

        return

    solutions, formal_uncertainties, correlations = ingest_dynamo_d_solution(file=file)

    for degree in [2, 4, 6]:

        for order in [0, 1] if degree < 5 else [0]:

            for parameter_type in (
                [ParameterType.C, ParameterType.S] if order == 1 else [ParameterType.C]
            ):

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
                sigmas = [
                    (
                        0
                        if (parameter_type, degree, order, date, None) not in formal_uncertainties
                        else formal_uncertainties[(parameter_type, degree, order, date, None)]
                    )
                    for date in dates
                ]
                figure, ax = subplots(figsize=(8, 4))
                ax.errorbar(
                    dates,
                    values,
                    yerr=abs(sigmas),
                    fmt="o-",
                    color="b" if array(object=sigmas > 0, dtype=bool).all() else "r",
                    capsize=3,
                    lw=1,
                )
                q1, q2, q3 = quantile(values, [0.25, 0.50, 0.75])
                ax.set_ylim(q1 - 2 * (q2 - q1), q3 + 2 * (q3 - q2))
                coeff = f"{'C' if parameter_type == ParameterType.C else 'S'}_{degree}{order}"
                ax.set_title(coeff)
                ax.set_xlabel("Date")
                ax.set_ylabel("Solution")
                ax.grid(True)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                figure.autofmt_xdate()
                figure.tight_layout()
                figure.savefig(str(file_path_to_save / coeff) + ".pdf")
                close(figure)


def plot_solutions(
    root: Path = Path("solution"), output_root: Path = Path("solution_figures")
) -> None:
    """
    Iterates on all solutions of the root directory and produces uncrossed solution figures.
    """

    for alpha_subdirectory in root.iterdir():

        for delta_subdirectory in alpha_subdirectory.iterdir():

            for tau_m_subdirectory in delta_subdirectory.iterdir():

                for g_subdirectory in tau_m_subdirectory.iterdir():

                    fix_g = g_subdirectory.name.split("_")[-1] == "true"

                    for sub in g_subdirectory.iterdir():

                        if sub.is_file():

                            produce_uncrossed_figures(root=root, file=sub, output_root=output_root)

                        if not fix_g:

                            if sub.is_dir():

                                for g_model_subdirectory in sub.iterdir():

                                    for file in g_model_subdirectory.iterdir():

                                        produce_uncrossed_figures(
                                            root=root, file=file, output_root=output_root
                                        )


if __name__ == "__main__":

    plot_solutions()
