"""
Hard-codes the long-period solid Earth tide k20 model in f_marsol.f90.

The input Love-number periods are in years. The generated Fortran table uses
signed angular frequencies in rad/s and Hermitian Love numbers.
"""

from pathlib import Path

from alna import ALPHA_TAB, DELTA_TAB, PERIODS_TAB, load_love_numbers_for_gins
from base_models import DATA_PATH
from numpy import argsort, array, concatenate, linspace, ndindex, pi, zeros

F_MARSOL_FILE = DATA_PATH.absolute().parent.parent.parent.parent.joinpath(
    "gin/sub/obelix/src/f_marsol.f90"
)
SECONDS_PER_YEAR = 365.25 * 86400.0
OMEGA_LONG_PERIOD_MAX = 2.18679 * pi / 180.0 / 3600.0


START_DECL = "  ! SOLID_K2_TABLE_DECLARATIONS_BEGIN"
END_DECL = "  ! SOLID_K2_TABLE_DECLARATIONS_END"
START_VALUES = "  ! SOLID_K2_TABLE_VALUES_BEGIN"
END_VALUES = "  ! SOLID_K2_TABLE_VALUES_END"


def fmt(x: float) -> str:
    return f"{x:+.12E}"


def hard_code_fortran90(variable_name: str, x: array, k=100) -> tuple[str, str]:
    shape = ",".join(str(i) for i in x.shape)
    declaration = f"  real(kind=DP), dimension({shape}) :: {variable_name}\n"
    assignments = []

    if x.ndim == 1:
        vectors = [(f"{variable_name}(", x)]
    else:
        vectors = [
            (f"{variable_name}({','.join(str(i + 1) for i in idx)},", x[idx])
            for idx in ndindex(x.shape[:-1])
        ]

    for lhs, values in vectors:
        chunks = [values[i : i + k] for i in range(0, len(values), k)]
        i0 = 1
        for chunk in chunks:
            i1 = i0 + len(chunk) - 1
            body = ", ".join(fmt(v) for v in chunk)
            assignments.append(f"  {lhs}{i0}:{i1}) = (/ {body} /)\n")
            i0 = i1 + 1

    return declaration, "".join(assignments)


def insert_between_markers(path: Path, start_marker: str, end_marker: str, text: str) -> None:
    content = path.read_text(encoding="utf-8")
    start = content.find(start_marker) + len(start_marker)
    end = content.find(end_marker)
    path.write_text(
        content[:start] + "\n" + text.strip("\n") + "\n" + content[end:], encoding="utf-8"
    )


def positive_angular_frequencies() -> array:
    return 2.0 * pi / (array(PERIODS_TAB, dtype=float) * SECONDS_PER_YEAR)


def with_elastic_high_frequency_padding(omega, k2, k2_alpha, k2_delta, elastic_k2):
    keep = omega <= OMEGA_LONG_PERIOD_MAX
    omega = omega[keep]
    k2 = k2[:, :, keep]
    k2_alpha = k2_alpha[:, :, keep]
    k2_delta = k2_delta[:, :, keep]

    pad_omega = OMEGA_LONG_PERIOD_MAX * linspace(1.02, 1.10, 5)
    pad_shape = (*k2.shape[:2], len(pad_omega))
    k2_pad = zeros(pad_shape, dtype=complex) + elastic_k2
    partial_pad = zeros(pad_shape, dtype=complex)

    return (
        concatenate((omega, pad_omega)),
        concatenate((k2, k2_pad), axis=2),
        concatenate((k2_alpha, partial_pad), axis=2),
        concatenate((k2_delta, partial_pad), axis=2),
    )


def make_signed_hermitian(omega, k2, k2_alpha, k2_delta):
    return (
        concatenate((-omega[::-1], omega)),
        concatenate((k2[:, :, ::-1].conjugate(), k2), axis=2),
        concatenate((k2_alpha[:, :, ::-1].conjugate(), k2_alpha), axis=2),
        concatenate((k2_delta[:, :, ::-1].conjugate(), k2_delta), axis=2),
    )


def save_solid_k2_tables() -> None:
    elastic, love, love_alpha, love_delta = load_love_numbers_for_gins()

    omega = positive_angular_frequencies()
    order = argsort(omega)
    omega = omega[order]

    k2 = love[:, :, 0, order, 1]
    k2_alpha = love_alpha[:, :, 0, order, 1]
    k2_delta = love_delta[:, :, 0, order, 1]
    elastic_k2 = complex(elastic[0, 1], 0.0)

    omega, k2, k2_alpha, k2_delta = with_elastic_high_frequency_padding(
        omega, k2, k2_alpha, k2_delta, elastic_k2
    )
    omega, k2, k2_alpha, k2_delta = make_signed_hermitian(omega, k2, k2_alpha, k2_delta)

    definitions = [
        f"  integer :: n_alpha = {len(ALPHA_TAB)}\n",
        f"  integer :: n_delta = {len(DELTA_TAB)}\n",
        f"  integer :: n_frequencies = {len(omega)}\n",
    ]
    values = []

    arrays = {
        "solid_alpha_values": array(ALPHA_TAB, dtype=float),
        "solid_delta_values": array(DELTA_TAB, dtype=float),
        "solid_frequency_values": omega,
        "solid_k2_real": k2.real,
        "solid_k2_imag": k2.imag,
        "solid_k2_dalpha_real": k2_alpha.real,
        "solid_k2_dalpha_imag": k2_alpha.imag,
        "solid_k2_ddelta_real": k2_delta.real,
        "solid_k2_ddelta_imag": k2_delta.imag,
    }

    for name, table in arrays.items():
        declaration, assignment = hard_code_fortran90(name, table)
        definitions.append(declaration)
        values.append(assignment)

    insert_between_markers(F_MARSOL_FILE, START_DECL, END_DECL, "".join(definitions))
    insert_between_markers(F_MARSOL_FILE, START_VALUES, END_VALUES, "".join(values))


if __name__ == "__main__":
    save_solid_k2_tables()
