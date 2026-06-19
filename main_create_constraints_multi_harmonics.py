"""
Create a DYNAMO D ready constraint file for one or several C_lm/S_lm harmonics.

Base model: Bias + trend.
Optional command-line flags add acceleration and/or annual terms:
    --acceleration / --no-acceleration
    --annual       / --no-annual

Single-harmonic mode is kept for backward compatibility:
    -d / --degree
    -o / --order

Multi-harmonic mode uses:
    -H / --harmonic / --harmonics

Examples:
    # Original C_21/S_21 behaviour
    python main_create_constraints_multi_harmonics.py -d 2 -o 1

    # Equivalent multi-harmonic syntax for C_21/S_21
    python main_create_constraints_multi_harmonics.py -H 21

    # Several harmonics in one file: C_21/S_21, C_41/S_41, C_20, C_40, C_60
    python main_create_constraints_multi_harmonics.py -H 21,41,20,40,60

    # Explicit degree:order syntax, useful for degrees or orders with several digits
    python main_create_constraints_multi_harmonics.py -H 2:1 4:1 2:0 4:0 6:0

    # C_32/S_32 with bias + trend + acceleration
    python main_create_constraints_multi_harmonics.py -d 3 -o 2 --acceleration

    # C_20 only, because S_l0 does not exist
    python main_create_constraints_multi_harmonics.py -d 2 -o 0

    # Custom output path
    python main_create_constraints_multi_harmonics.py -H 21,41,20,40,60 -f constraints_selected_harmonics
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta
from pathlib import Path

LABEL_WIDTH = 24
DEFAULT_DATE_DEBUT = "1991-01-01"
DEFAULT_DATE_FIN = "2026-12-31"
DEFAULT_CLE_STRICTE = 1
DEFAULT_SIGMA_STRICTE = 1e-13
DEFAULT_DEGREE = 2
DEFAULT_ORDER = 1
Harmonic = tuple[int, int]


def parse_date(d: str | date) -> date:
    """
    Accepts either a python date, either 'YYYY-MM-DD' or 'YYYYMMDD'.
    """

    if isinstance(d, date):

        return d

    d = d.strip()

    if len(d) == 8 and d.isdigit():

        return datetime.strptime(d, "%Y%m%d").date()

    return datetime.strptime(d, "%Y-%m-%d").date()


def fractional_year_since_2000(d: date) -> float:
    """
    Fractional year since 2000 from real year length.

    Exemple:
        2000-01-01 -> 0.0
        2001-01-01 -> 1.0
    """

    start_year = date(d.year, 1, 1)
    start_next_year = date(d.year + 1, 1, 1)
    year_length = (start_next_year - start_year).days
    day_in_year = (d - start_year).days

    return (d.year - 2000) + day_in_year / year_length


def iter_days(start: date, end: date):
    """
    Inclusive day iterator: start and end are included.
    """

    d = start

    while d <= end:

        yield d
        d += timedelta(days=1)


def validate_harmonic(degree: int, order: int) -> None:
    """
    Validates a spherical harmonic degree/order pair.
    """

    if degree < 0:

        raise ValueError("degree must be >= 0.")

    if order < 0:

        raise ValueError("order must be >= 0.")

    if order > degree:

        raise ValueError("order must be <= degree.")


def harmonic_code(degree: int, order: int) -> str:
    """
    Compact human-readable code used in filenames and variable suffixes.

    Examples:
        degree=2, order=1   -> 21
        degree=6, order=0   -> 60
        degree=10, order=1  -> 10_1
    """

    validate_harmonic(degree, order)

    if degree < 10 and order < 10:

        return f"{degree}{order}"

    return f"{degree}_{order}"


def harmonic_suffix(degree: int, order: int, *, enabled: bool) -> str:
    """
    Returns the suffix appended to model parameters in multi-harmonic mode.

    Single-harmonic mode keeps the historic parameter names CA, CB, SA, SB, ...
    Multi-harmonic mode uses independent variables such as CA_21, CB_21,
    SA_41, SB_41, ... so each harmonic has its own bias/trend/etc.
    """

    if not enabled:

        return ""

    return "_" + harmonic_code(degree, order)


def parse_harmonic_spec(text: str) -> Harmonic:
    """
    Parses one harmonic specification.

    Accepted forms:
        21      -> degree 2, order 1
        C21     -> degree 2, order 1; leading C/S/G is ignored
        2:1     -> degree 2, order 1
        2/1     -> degree 2, order 1

    The compact form is intentionally limited to two digits, because 101 is
    ambiguous. Use 10:1 instead.
    """

    original = text
    text = text.strip().upper()

    if not text:

        raise argparse.ArgumentTypeError("Empty harmonic specification.")

    for prefix in ("GCN", "GSN", "CN", "SN", "C", "S", "G"):

        if text.startswith(prefix):

            text = text[len(prefix) :]
            break

    text = text.strip().replace("_", ":")

    for separator in (":", "/"):

        if separator in text:

            parts = text.split(separator)

            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):

                raise argparse.ArgumentTypeError(
                    f"Invalid harmonic {original!r}. Use LM, L:M or L/M syntax."
                )

            degree = int(parts[0])
            order = int(parts[1])

            try:
                validate_harmonic(degree, order)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid harmonic {original!r}: {exc}"
                ) from exc

            return degree, order

    if text.isdigit() and len(text) == 2:

        degree = int(text[0])
        order = int(text[1])

        try:
            validate_harmonic(degree, order)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid harmonic {original!r}: {exc}"
            ) from exc

        return degree, order

    raise argparse.ArgumentTypeError(
        f"Invalid or ambiguous harmonic {original!r}. "
        "Use compact two-digit LM syntax, e.g. 21, or explicit L:M syntax, e.g. 10:1."
    )


def parse_harmonic_arguments(values: list[list[str]] | None) -> list[Harmonic] | None:
    """
    Parses -H/--harmonic arguments.

    Allows all of these equivalent forms:
        -H 21,41,20
        -H 21 41 20
        -H 21 -H 41 -H 20
        -H 2:1 4:1 2:0
    """

    if not values:

        return None

    parsed: list[Harmonic] = []
    seen: set[Harmonic] = set()

    for group in values:

        for item in group:

            for token in item.split(","):

                token = token.strip()

                if not token:

                    continue

                harmonic = parse_harmonic_spec(token)

                if harmonic not in seen:

                    parsed.append(harmonic)
                    seen.add(harmonic)

    if not parsed:

        raise argparse.ArgumentTypeError("At least one harmonic must be specified.")

    return parsed


def has_s_harmonic(order: int) -> bool:
    """
    Returns False for zonal terms because S_l0 does not exist.
    """

    return order != 0


def label(content: str) -> str:
    """
    Formats a label in the [....] DYNAMO ready format with 24 characters.
    """

    if len(content) > LABEL_WIDTH:

        raise ValueError(
            f"Label trop long: {content!r} contient {len(content)} caracteres, "
            f"maximum {LABEL_WIDTH}."
        )

    return f"[{content:<{LABEL_WIDTH}}]"


def label_harmonic(kind: str, degree: int, order: int, d: date) -> str:
    """
    Formats the label for a chosen harmonic and date.

    Examples:
        C_21 -> [GCN   2  1 20150316     ]
        S_21 -> [GSN   2  1 20150316     ]
        C_32 -> [GCN   3  2 20150316     ]
    """

    validate_harmonic(degree, order)

    kind = kind.upper()

    if kind == "C":

        prefix = "GCN"

    elif kind == "S":

        if not has_s_harmonic(order):

            raise ValueError("S_l0 does not exist for order 0.")

        prefix = "GSN"

    else:

        raise ValueError("kind must be 'C' or 'S'.")

    return label(f"{prefix} {degree:3d} {order:2d} {d:%Y%m%d}")


def label_c_harmonic(d: date, *, degree: int, order: int) -> str:
    """
    Formats the label for C_lm for a given date.
    """

    return label_harmonic("C", degree, order, d)


def label_s_harmonic(d: date, *, degree: int, order: int) -> str:
    """
    Formats the label for S_lm for a given date.
    """

    return label_harmonic("S", degree, order, d)


# Backward-compatible names for callers that imported these helpers.
def label_c21(d: date) -> str:
    """
    Formats the label for C_21 for a given date.
    """

    return label_c_harmonic(d, degree=2, order=1)


def label_s21(d: date) -> str:
    """
    Formats the label for S_21 for a given date.
    """

    return label_s_harmonic(d, degree=2, order=1)


def fmt_float_19(x: float) -> str:
    """
    Scientific fortran-ready format in 19 characters.

    Exemples:
        1.0  -> 0.1000000000000E+01
       -1.0  -> -.1000000000000E+01
        0.0  -> 0.0000000000000E+00
    """

    if x == 0.0:

        return "0.0000000000000E+00"

    sign = "-" if x < 0 else ""
    ax = abs(x)
    exponent = math.floor(math.log10(ax)) + 1
    mantissa = ax / (10.0**exponent)
    mantissa_str = f"{mantissa:.13f}"

    if mantissa_str.startswith("1."):

        mantissa /= 10.0
        exponent += 1
        mantissa_str = f"{mantissa:.13f}"

    if sign:

        mantissa_str = mantissa_str[1:]

    return f"{sign}{mantissa_str}E{exponent:+03d}"


def fmt_sigma(x: float) -> str:
    """
    Formats sigma, ex: 1.0000E-02.
    """

    return f"{x:.4E}"


def term(coef: float, lab: str) -> str:
    """
    Formats a term of the constraint, ex: -0.5000000000000E+00 [CC                     ].
    """

    return f"{fmt_float_19(coef)} {lab}"


def creation_line(var_name: str, initial_value: float = 0.0) -> str:
    """
    Creates a line for variable creation with initial value.
    """

    return f"{0:3d}" + " " * 23 + label(var_name) + " " * 48 + f"= {fmt_float_19(initial_value)}"


def constraint_lines(
    terms: list[tuple[float, str]],
    *,
    cle: int = 1,
    value: float,
    sigma: float,
) -> list[str]:
    """
    Creates a multi-parameter constraint.
    """

    n = len(terms)

    if n == 0:

        raise ValueError("One term at least.")

    first_terms = terms[:2]
    first_line = (
        f"{n:3d} {cle:d} "
        + " ".join(term(coef, lab) for coef, lab in first_terms)
        + f" = {fmt_float_19(value)} {fmt_sigma(sigma)}"
    )

    lines = [first_line]

    for i in range(2, n, 2):

        continuation_terms = terms[i : i + 2]
        lines.append("      " + " ".join(term(coef, lab) for coef, lab in continuation_terms))

    return lines


def model_variables(
    *,
    include_acceleration: bool = False,
    include_annual: bool = False,
    include_s_terms: bool = True,
    suffix: str = "",
) -> list[str]:
    """
    Returns the parameter creation order for the selected model.
    """

    c_variables: list[str] = []
    s_variables: list[str] = []

    if include_acceleration:
        c_variables.append("CAA")

        if include_s_terms:
            s_variables.append("SAA")

    if include_annual:
        c_variables.extend(["CC", "CS"])

        if include_s_terms:
            s_variables.extend(["SC", "SS"])

    c_variables.extend(["CA", "CB"])

    if include_s_terms:
        s_variables.extend(["SA", "SB"])

    return [var + suffix for var in c_variables + s_variables]


def build_terms(
    d: date,
    *,
    degree: int = DEFAULT_DEGREE,
    order: int = DEFAULT_ORDER,
    include_acceleration: bool = False,
    include_annual: bool = False,
    variable_suffix: str = "",
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """
    Builds the C_lm and S_lm constraint terms for one date.

    With all options enabled and order > 0:
        C_lm(t) - t^2 CAA - cos(t) CC - sin(t) CS - t CA - CB = 0
        S_lm(t) - t^2 SAA - cos(t) SC - sin(t) SS - t SA - SB = 0

    For order = 0, only C_l0 constraints are generated because S_l0 does not exist.

    The annual terms intentionally use cos(t) and sin(t), matching the original
    annual script that this merged script replaces.
    """

    validate_harmonic(degree, order)

    t = fractional_year_since_2000(d)
    include_s_terms = has_s_harmonic(order)

    c_terms = [(1.0, label_c_harmonic(d, degree=degree, order=order))]
    s_terms = (
        [(1.0, label_s_harmonic(d, degree=degree, order=order))]
        if include_s_terms
        else []
    )

    if include_acceleration:
        c_terms.append((-(t**2), label("CAA" + variable_suffix)))

        if include_s_terms:
            s_terms.append((-(t**2), label("SAA" + variable_suffix)))

    if include_annual:
        cos_t = math.cos(t)
        sin_t = math.sin(t)

        c_terms.extend(
            [(-cos_t, label("CC" + variable_suffix)), (-sin_t, label("CS" + variable_suffix))]
        )

        if include_s_terms:
            s_terms.extend(
                [(-cos_t, label("SC" + variable_suffix)), (-sin_t, label("SS" + variable_suffix))]
            )

    c_terms.extend([(-t, label("CA" + variable_suffix)), (-1.0, label("CB" + variable_suffix))])

    if include_s_terms:
        s_terms.extend([(-t, label("SA" + variable_suffix)), (-1.0, label("SB" + variable_suffix))])

    return c_terms, s_terms


def harmonic_description(degree: int, order: int) -> str:
    """
    Returns a human-readable harmonic description.
    """

    validate_harmonic(degree, order)

    code = harmonic_code(degree, order)

    if has_s_harmonic(order):

        return f"C_{code} and S_{code}"

    return f"C_{code}"


def model_description(
    *,
    degree: int = DEFAULT_DEGREE,
    order: int = DEFAULT_ORDER,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> str:
    """
    Returns a human-readable model description.
    """

    parts = [harmonic_description(degree, order), "bias", "trend"]

    if include_acceleration:
        parts.append("acceleration")

    if include_annual:
        parts.append("annual")

    return " + ".join(parts)


def harmonics_description(harmonics: list[Harmonic]) -> str:
    """
    Returns a human-readable description for several harmonics.
    """

    return ", ".join(harmonic_description(degree, order) for degree, order in harmonics)


def model_suffixes(
    *,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> list[str]:
    """
    Returns filename suffixes for the selected temporal model.
    """

    suffixes = ["trend"]

    if include_acceleration:
        suffixes.append("acceleration")

    if include_annual:
        suffixes.append("annual")

    return suffixes


def default_output_path(
    *,
    degree: int = DEFAULT_DEGREE,
    order: int = DEFAULT_ORDER,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> str:
    """
    Returns a default output filename for the selected harmonic and model.
    """

    validate_harmonic(degree, order)

    suffixes = model_suffixes(
        include_acceleration=include_acceleration,
        include_annual=include_annual,
    )

    if has_s_harmonic(order):

        base = f"constraints_C_{degree}_{order}_S_{degree}_{order}"

    else:

        base = f"constraints_C_{degree}_{order}"

    return base + "_" + "_and_".join(suffixes)


def default_output_path_for_harmonics(
    harmonics: list[Harmonic],
    *,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> str:
    """
    Returns a default output filename for one or several harmonics.
    """

    if len(harmonics) == 1:

        degree, order = harmonics[0]

        return default_output_path(
            degree=degree,
            order=order,
            include_acceleration=include_acceleration,
            include_annual=include_annual,
        )

    for degree, order in harmonics:
        validate_harmonic(degree, order)

    codes = "_".join(harmonic_code(degree, order) for degree, order in harmonics)
    suffixes = model_suffixes(
        include_acceleration=include_acceleration,
        include_annual=include_annual,
    )

    return f"constraints_harmonics_{codes}_" + "_and_".join(suffixes)


def initial_value_for(var_name: str, initial_values: dict[str, float]) -> float:
    """
    Returns an initial value for a variable.

    Exact suffixed names have priority, e.g. CA_21=1e-10. In multi-harmonic
    mode, a base name such as CA=1e-10 is used as a fallback for all variables
    whose name starts with CA_.
    """

    if var_name in initial_values:

        return initial_values[var_name]

    base_name = var_name.split("_", 1)[0]

    return initial_values.get(base_name, 0.0)


def parse_initial_value(text: str) -> tuple[str, float]:
    """
    Parses a command-line initial value of the form VAR=VALUE.
    """

    if "=" not in text:
        raise argparse.ArgumentTypeError("Initial values must use VAR=VALUE syntax.")

    var_name, value = text.split("=", 1)
    var_name = var_name.strip()

    if not var_name:
        raise argparse.ArgumentTypeError("Initial value variable name cannot be empty.")

    try:
        return var_name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value in {text!r}.") from exc


def generate_harmonics_constraints(
    date_debut: str | date,
    date_fin: str | date,
    harmonics: list[Harmonic],
    *,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
    suffix_variables: bool | None = None,
) -> str:
    """
    Generates the complete content of the text file for one or several harmonics.

    Base equations imposed for each day and each harmonic:
        C_lm(t) - t CA_lm - CB_lm = 0
        S_lm(t) - t SA_lm - SB_lm = 0, when order > 0

    Optional terms:
        Acceleration: - t^2 CAA_lm and - t^2 SAA_lm
        Annual:       - cos(t) CC_lm - sin(t) CS_lm and
                      - cos(t) SC_lm - sin(t) SS_lm

    By default, variables are suffixed only when more than one harmonic is
    requested. This preserves the historic CA/CB/SA/SB names for a single
    harmonic while keeping multi-harmonic constraints independent.
    """

    if not harmonics:

        raise ValueError("At least one harmonic must be specified.")

    for degree, order in harmonics:
        validate_harmonic(degree, order)

    start = parse_date(date_debut)
    end = parse_date(date_fin)

    if end < start:

        raise ValueError("date_fin doit etre posterieure ou egale a date_debut.")

    if initial_values is None:
        initial_values = {}

    if suffix_variables is None:
        suffix_variables = len(harmonics) > 1

    lines: list[str] = []

    lines.append(
        " nb cle     coef                   label                     "
        "coef                 label                     valeur           sigma"
    )

    lines.append(
        "##COM## Creation des parametres pour "
        f"{harmonics_description(harmonics)}"
    )

    all_variables: list[str] = []

    for degree, order in harmonics:

        suffix = harmonic_suffix(degree, order, enabled=suffix_variables)
        include_s_terms = has_s_harmonic(order)
        variables = model_variables(
            include_acceleration=include_acceleration,
            include_annual=include_annual,
            include_s_terms=include_s_terms,
            suffix=suffix,
        )
        all_variables.extend(variables)

    for var in all_variables:
        lines.append(creation_line(var, initial_value_for(var, initial_values)))

    lines.append(
        "##COM## Contraintes strictes sur "
        f"{harmonics_description(harmonics)}"
    )

    for degree, order in harmonics:

        suffix = harmonic_suffix(degree, order, enabled=suffix_variables)

        if len(harmonics) > 1:

            lines.append("##COM## " + harmonic_description(degree, order))

        for d in iter_days(start, end):
            c_terms, s_terms = build_terms(
                d,
                degree=degree,
                order=order,
                include_acceleration=include_acceleration,
                include_annual=include_annual,
                variable_suffix=suffix,
            )

            lines.extend(
                constraint_lines(
                    c_terms,
                    cle=cle_stricte,
                    value=0.0,
                    sigma=sigma_stricte,
                )
            )

            if s_terms:
                lines.extend(
                    constraint_lines(
                        s_terms,
                        cle=cle_stricte,
                        value=0.0,
                        sigma=sigma_stricte,
                    )
                )

    return "\n".join(lines) + "\n"


def generate_harmonic_constraints(
    date_debut: str | date,
    date_fin: str | date,
    *,
    degree: int = DEFAULT_DEGREE,
    order: int = DEFAULT_ORDER,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> str:
    """
    Generates the complete content of the text file for one harmonic.

    This wrapper preserves the old single-harmonic API and variable names.
    """

    return generate_harmonics_constraints(
        date_debut,
        date_fin,
        [(degree, order)],
        cle_stricte=cle_stricte,
        sigma_stricte=sigma_stricte,
        initial_values=initial_values,
        include_acceleration=include_acceleration,
        include_annual=include_annual,
        suffix_variables=False,
    )


# Backward-compatible name for callers that imported the old generator.
def generate_c21_s21_constraints(
    date_debut: str | date,
    date_fin: str | date,
    *,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> str:
    """
    Generates C_21/S_21 constraints, preserving the old public function name.
    """

    return generate_harmonic_constraints(
        date_debut,
        date_fin,
        degree=2,
        order=1,
        cle_stricte=cle_stricte,
        sigma_stricte=sigma_stricte,
        initial_values=initial_values,
        include_acceleration=include_acceleration,
        include_annual=include_annual,
    )


def write_harmonics_file(
    output_path: str | Path,
    date_debut: str | date,
    date_fin: str | date,
    harmonics: list[Harmonic],
    *,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
    suffix_variables: bool | None = None,
) -> None:
    """
    Ecrit directement le fichier texte pour un ou plusieurs harmoniques.
    """

    content = generate_harmonics_constraints(
        date_debut,
        date_fin,
        harmonics,
        cle_stricte=cle_stricte,
        sigma_stricte=sigma_stricte,
        initial_values=initial_values,
        include_acceleration=include_acceleration,
        include_annual=include_annual,
        suffix_variables=suffix_variables,
    )

    Path(output_path).write_text("\n\n\n" + content, encoding="ascii")


def write_harmonic_file(
    output_path: str | Path,
    date_debut: str | date,
    date_fin: str | date,
    *,
    degree: int = DEFAULT_DEGREE,
    order: int = DEFAULT_ORDER,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> None:
    """
    Ecrit directement le fichier texte pour un harmonique.
    """

    write_harmonics_file(
        output_path,
        date_debut,
        date_fin,
        [(degree, order)],
        cle_stricte=cle_stricte,
        sigma_stricte=sigma_stricte,
        initial_values=initial_values,
        include_acceleration=include_acceleration,
        include_annual=include_annual,
        suffix_variables=False,
    )


# Backward-compatible name for callers that imported the old writer.
def write_c21_s21_file(
    output_path: str | Path,
    date_debut: str | date,
    date_fin: str | date,
    *,
    cle_stricte: int = 1,
    sigma_stricte: float = 0.0,
    initial_values: dict[str, float] | None = None,
    include_acceleration: bool = False,
    include_annual: bool = False,
) -> None:
    """
    Writes C_21/S_21 constraints, preserving the old public function name.
    """

    write_harmonic_file(
        output_path,
        date_debut,
        date_fin,
        degree=2,
        order=1,
        cle_stricte=cle_stricte,
        sigma_stricte=sigma_stricte,
        initial_values=initial_values,
        include_acceleration=include_acceleration,
        include_annual=include_annual,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Builds the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Create a DYNAMO D ready constraint file for chosen C_lm/S_lm harmonics."
    )

    parser.add_argument(
        "-d",
        "--degree",
        type=int,
        default=DEFAULT_DEGREE,
        help=f"Harmonic degree l. Default: {DEFAULT_DEGREE}.",
    )
    parser.add_argument(
        "-o",
        "--order",
        type=int,
        default=DEFAULT_ORDER,
        help=f"Harmonic order m. Default: {DEFAULT_ORDER}.",
    )
    parser.add_argument(
        "-H",
        "--harmonic",
        "--harmonics",
        dest="harmonics",
        action="append",
        nargs="+",
        metavar="LM",
        help=(
            "One or several harmonics to constrain. Accepts compact LM syntax "
            "for single-digit degree/order, e.g. -H 21,41,20,40,60, and explicit "
            "L:M or L/M syntax, e.g. -H 2:1 4:1 2:0. May be repeated. "
            "When this option is used, it replaces -d/--degree and -o/--order."
        ),
    )
    parser.add_argument(
        "-f",
        "--output",
        default=None,
        help=(
            "Output file path. Defaults to a name based on the selected harmonic "
            "and model. Note: -o is now used for --order."
        ),
    )
    parser.add_argument(
        "--date-debut",
        "--start",
        default=DEFAULT_DATE_DEBUT,
        help=f"Start date, inclusive, as YYYY-MM-DD or YYYYMMDD. Default: {DEFAULT_DATE_DEBUT}.",
    )
    parser.add_argument(
        "--date-fin",
        "--end",
        default=DEFAULT_DATE_FIN,
        help=f"End date, inclusive, as YYYY-MM-DD or YYYYMMDD. Default: {DEFAULT_DATE_FIN}.",
    )
    parser.add_argument(
        "--cle-stricte",
        type=int,
        default=DEFAULT_CLE_STRICTE,
        help=f"Constraint key. Default: {DEFAULT_CLE_STRICTE}.",
    )
    parser.add_argument(
        "--sigma-stricte",
        type=float,
        default=DEFAULT_SIGMA_STRICTE,
        help=f"Constraint sigma. Default: {DEFAULT_SIGMA_STRICTE}.",
    )
    parser.add_argument(
        "--acceleration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include acceleration terms CAA and SAA.",
    )
    parser.add_argument(
        "--annual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include annual cosine/sine terms CC, CS, SC and SS.",
    )
    parser.add_argument(
        "--initial-value",
        action="append",
        type=parse_initial_value,
        default=[],
        metavar="VAR=VALUE",
        help="Initial value for a parameter. May be repeated, e.g. --initial-value CA=1e-10.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        harmonics = parse_harmonic_arguments(args.harmonics)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if harmonics is None:

        try:
            validate_harmonic(args.degree, args.order)
        except ValueError as exc:
            parser.error(str(exc))

        harmonics = [(args.degree, args.order)]

    elif args.degree != DEFAULT_DEGREE or args.order != DEFAULT_ORDER:

        parser.error(
            "Use either -d/--degree with -o/--order, or -H/--harmonics, not both. "
            "For example: -H 21,41,20,40,60."
        )

    initial_values = dict(args.initial_value)
    output = args.output or default_output_path_for_harmonics(
        harmonics,
        include_acceleration=args.acceleration,
        include_annual=args.annual,
    )

    write_harmonics_file(
        output,
        args.date_debut,
        args.date_fin,
        harmonics,
        cle_stricte=args.cle_stricte,
        sigma_stricte=args.sigma_stricte,
        initial_values=initial_values,
        include_acceleration=args.acceleration,
        include_annual=args.annual,
    )


if __name__ == "__main__":

    main()
