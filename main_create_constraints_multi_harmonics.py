"""
Generate gravity-field constraint parameter files.

Example:

    python main_create_constraints_multi_harmonics.py \
        --output constraints_G_trend_and_acceleration_and_annual \
        --start 19910101 \
        --end 19910106 \
        --annual \
        --acceleration
"""

from argparse import ArgumentParser, ArgumentTypeError
from datetime import date, datetime, timedelta
from math import cos, floor, log10, pi, sin
from pathlib import Path

DEFAULT_COEFFS = [
    ("C", 2, 0),
    ("C", 2, 1),
    ("S", 2, 1),
    ("C", 4, 0),
    ("C", 4, 1),
    ("S", 4, 1),
    ("C", 6, 0),
]

PERIODIC_TERM_YEARS = 1.0
OMEGA = 2 * pi / PERIODIC_TERM_YEARS
DAYS_PER_YEAR = 365.25


def parse_date(value: str) -> date:
    """
    For Modified timespan to model.
    """

    value = value.strip()

    if "-" in value:

        return datetime.strptime(value, "%Y-%m-%d").date()

    return datetime.strptime(value, "%Y%m%d").date()


START_DATE = parse_date(value="19910101")
END_DATE = parse_date(value="20260101")


def iter_dates(start: date = START_DATE, end: date = END_DATE):
    """
    Iterator.
    """

    current = start
    one_day = timedelta(days=1)

    while current <= end:

        yield current
        current += one_day


def fortran_sci(value: float, decimals: int = 13) -> str:
    """
    Formats like:
        0.1000000000000E+01
        -.8100000000000E+02
        0.0000000000000E+00

    This uses a mantissa in [0.1, 1), matching the text format.
    """

    if abs(value) == 0.0:

        return f"0.{''.join(['0'] * decimals)}E+00"

    sign = "-" if value < 0 else ""
    x = abs(value)
    exponent = floor(log10(x)) + 1
    mantissa = x / (10.0**exponent)

    # Protect against rounding from 0.9999999999999 to 1.0000000000000.
    rounded = round(mantissa, decimals)

    if rounded >= 1.0:

        mantissa /= 10.0
        exponent += 1

    mantissa_text = f"{mantissa:.{decimals}f}"

    # The sample omits the zero after the minus sign: -.810..., not -0.810...
    if sign and mantissa_text.startswith("0."):

        mantissa_text = mantissa_text[1:]

    return f"{sign}{mantissa_text}E{exponent:+03d}"


def label(text: str) -> str:
    """
    DYNAMO-readable 24 character format.
    """

    return f"[{text:<24}]"


def harmonic_label(kind: str, degree: int, order: int, date_time: date) -> str:
    """
    Creates a signaletic element.
    """

    family = "GCN" if kind == "C" else "GSN"
    inside = f"{family:<3}{degree:4d}{order:3d} {date_time:%Y%m%d}     "

    return label(inside)


def parameter_names(
    kind: str, degree: int, order: int, annual: bool, acceleration: bool
) -> list[str]:
    """
    Generates parameter signaletic elements.
    """

    suffix = f"{degree}{order}"
    names: list[str] = []

    if kind == "C":

        if acceleration:

            names.append(f"CAA_{suffix}")

        if annual:

            names.extend([f"CC_{suffix}", f"CS_{suffix}"])

        names.extend([f"CA_{suffix}", f"CB_{suffix}"])

    elif kind == "S":

        if order == 0:

            raise ValueError("S_l0 terms are not valid for order 0.")

        if acceleration:

            names.append(f"SAA_{suffix}")

        if annual:

            names.extend([f"SC_{suffix}", f"SS_{suffix}"])

        names.extend([f"SA_{suffix}", f"SB_{suffix}"])

    else:

        raise ValueError(f"Unknown coefficient kind: {kind}")

    return names


def model_coefficients(
    t: float, annual: bool, acceleration: bool, omega: float = OMEGA
) -> list[float]:
    """
    Creates the constraint:
        X_lm(t) - t^2 XAA - cos(omega*t) XC - sin(omega*t) XS - t XA - XB = 0
    where omega = 2*pi by default.
    """

    coeffs: list[float] = []

    if acceleration:

        coeffs.append(-(t**2))

    if annual:

        coeffs.extend(
            [
                -cos(omega * t),
                -sin(omega * t),
            ]
        )

    coeffs.extend(
        [
            -t,
            -1.0,
        ]
    )

    return coeffs


def fractional_year_from_2000(date_time: date, days_per_year: float = DAYS_PER_YEAR) -> float:
    """
    Decimal-year convention.
    """

    day_of_year = date_time.timetuple().tm_yday

    return (date_time.year - 2000) + ((day_of_year - 1) / days_per_year)


def coefficient_summary(coeffs: list[tuple[str, int, int]]) -> str:
    """
    To create the needed parameters to describe the model.
    """

    pieces: list[str] = []
    used: set[tuple[str, int, int]] = set()

    for kind, degree, order in coeffs:

        key = (kind, degree, order)

        if key in used:

            continue

        if kind == "C" and ("S", degree, order) in coeffs:

            pieces.append(f"C_{degree}{order} and S_{degree}{order}")
            used.add(("C", degree, order))
            used.add(("S", degree, order))

        else:

            pieces.append(f"{kind}_{degree}{order}")
            used.add(key)

    return ", ".join(pieces)


def write_parameter_creation_block(
    lines: list[str], coeffs: list[tuple[str, int, int]], annual: bool, acceleration: bool
) -> None:
    """
    Creates the needed parameters to describe the model.
    """

    summary = coefficient_summary(coeffs=coeffs)
    lines.append(
        " nb cle     coef                   label                     coef"
        + "                 label                     valeur           sigma"
    )
    lines.append(f"##COM## Creation des parametres pour {summary}")

    for kind, degree, order in coeffs:

        for name in parameter_names(
            kind=kind, degree=degree, order=order, annual=annual, acceleration=acceleration
        ):

            lines.append(
                f"  0                       {label(name)}"
                + "                                                = {fortran_sci(0.0)}"
            )


def write_constraint_block(
    lines: list[str],
    kind_degree_order: tuple[str, int, int],
    dates: list[date],
    options: tuple[bool, bool],
    sigma: str,
) -> None:
    """
    Creates a single contraint line in the constrain file.
    """

    lines.append(f"##COM## {kind_degree_order[0]}_{kind_degree_order[1]}{kind_degree_order[2]}")
    param_names = parameter_names(
        kind=kind_degree_order[0],
        degree=kind_degree_order[1],
        order=kind_degree_order[2],
        annual=options[0],
        acceleration=options[1],
    )

    for date_time in dates:

        t = fractional_year_from_2000(date_time=date, days_per_year=DAYS_PER_YEAR)
        param_coeffs = model_coefficients(
            t=t, annual=options[0], acceleration=options[1], omega=OMEGA
        )
        linked_count = 1 + len(param_names)
        first_param_name = param_names[0]
        lines.append(
            f"{linked_count:3d} 1 {fortran_sci(value=1.0)} "
            + harmonic_label(
                kind=kind_degree_order[0],
                degree=kind_degree_order[1],
                order=kind_degree_order[2],
                date_time=date_time,
            )
            + f"{fortran_sci(value=param_coeffs[0])} {label(text=first_param_name)} "
            f"= {fortran_sci(value=0.0)} {sigma}"
        )
        remaining = list(zip(param_coeffs[1:], param_names[1:]))

        for i in range(0, len(remaining), 2):

            chunk = remaining[i : i + 2]
            chunk_text = " ".join(
                f"{fortran_sci(value=coeff)} {label(text=name)}" for coeff, name in chunk
            )
            lines.append(f"      {chunk_text}")


def generate_file(
    output: Path,
    coeffs: list[tuple[str, int, int]],
    options: tuple[bool, bool],
    sigma: str,
) -> None:
    """
    Generates a Gravity field model contraint file for DYNAMO.
    """

    dates = list(iter_dates(start=START_DATE, end=END_DATE))
    lines: list[str] = ["\n"] * 2
    write_parameter_creation_block(
        lines=lines, coeffs=coeffs, annual=options[0], acceleration=options[1]
    )
    summary = coefficient_summary(coeffs=coeffs)
    lines.append(f"##COM## Contraintes strictes sur {summary}")

    for kind_degree_order in coeffs:

        write_constraint_block(
            lines=lines,
            kind_degree_order=kind_degree_order,
            dates=dates,
            options=options,
            sigma=sigma,
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_coeff(value: str) -> tuple[str, int, int]:
    """
    Accepts forms like:
        C20
        S21
        C_20
        S_41
    """

    value = value.strip().upper().replace("_", "")

    if len(value) < 3:

        raise ArgumentTypeError(f"Invalid coefficient: {value}")

    kind = value[0]

    if kind not in {"C", "S"}:

        raise ArgumentTypeError(f"Coefficient must start with C or S: {value}")

    digits = value[1:]

    if not digits.isdigit():

        raise ArgumentTypeError(f"Invalid coefficient digits: {value}")

    if len(digits) == 2:

        degree = int(digits[0])
        order = int(digits[1])

    else:

        degree = int(digits[:-1])
        order = int(digits[-1])

    if kind == "S" and order == 0:

        raise ArgumentTypeError("S_l0 coefficients are invalid.")

    return kind, degree, order


def main() -> None:
    """
    Manages argument parsing and gravity field model constraint file generation.
    """

    parser = ArgumentParser(description="Generate a gravity-field parameter constraint text file.")
    parser.add_argument(
        "-o",
        "--output",
        default="constraints.txt",
        type=Path,
        help="Output text file path.",
    )
    parser.add_argument(
        "--coeff",
        action="append",
        type=parse_coeff,
        help="Coefficient to generate, e.g. C20, ... Defaults to C20 C21 S21 C40 C41 S41 C60.",
    )
    parser.add_argument(
        "--annual",
        action="store_true",
        help="Include annual cos/sin terms.",
    )
    parser.add_argument(
        "--acceleration",
        action="store_true",
        help="Include quadratic acceleration term.",
    )
    parser.add_argument(
        "--sigma",
        default="1.0000E-19",
        help="Sigma value written at the end of each constraint line.",
    )
    args = parser.parse_args()
    coeffs = args.coeff if args.coeff else DEFAULT_COEFFS
    generate_file(
        output=args.output,
        coeffs=coeffs,
        options=(args.annual, args.acceleration),
        sigma=args.sigma,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":

    main()
