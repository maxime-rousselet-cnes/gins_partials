"""
Defines and hard-codes the solid pole tide correction driven by k2 only in GINS routine
f_marpolsol.f90.
The generated Fortran tables are gridded in the runtime interpolation variables
(alpha, log10(delta), log10(tau_m), date).
"""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from alna import (
    DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    MODELS,
    ROOT_PATH,
    SECONDS_PER_YEAR,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    generate_parameter_lines,
    load_love_numbers_for_gins,
    parameters_for_gins,
)
from base_models import (
    DATA_PATH,
    SteadyStateSignalParameters,
    build_steady_state_regime_signal,
    lagrange_order4,
    save_base_model,
)
from numpy import array, asarray, conjugate, log, log10, mean, ndarray, ndindex, unique, zeros
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm

from .utils import (
    DATA_DATES_LOWER_BOUND,
    DATA_DATES_MARGIN,
    DATA_DATES_UPPER_BOUND,
    JJUL_1970_REFERENCE_JJUL,
    JJUL_1970_REFERENCE_YEAR,
    K_2_IERS,
    PHI_CONSTANT,
    get_m1_m2_time_series,
)

LONG_TERM_HYPOTHESIS_PERIOD = 10000  # (yr).
MAX_STATEMENT_LENGTH = 20000
MAX_LINE_LENGTH = 6800
DEFAULT_POLE_TIDE_CORRECTION_FILE = ROOT_PATH.parent.parent.joinpath(
    "gin/sub/obelix/src/f_marpolsol.f90"
).resolve()
DEFAULT_SOLID_TIDE_CORRECTION_FILE = ROOT_PATH.parent.parent.joinpath(
    "gin/sub/obelix/src/f_marsol.f90"
).resolve()
POLE_MODELS_PATH = DATA_PATH.joinpath("pole")
TIDE_MODELS_PATH = DATA_PATH.joinpath("TIDE")
DEFAULT_SIGNAL_PARAMETERS = SteadyStateSignalParameters()
POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME = "pole_tide_correction_models"
SOLID_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME = "solid_tide_correction_models"
START_DECL = "  ! TIDE_TABLE_DECLARATIONS_BEGIN"
END_DECL = "  ! TIDE_TABLE_DECLARATIONS_END"
START_VALUES = "  ! TIDE_TABLE_VALUES_BEGIN"
END_VALUES = "  ! TIDE_TABLE_VALUES_END"
MODEL_NAMES = [
    "elastic",
    "anelastic",
    "alpha_partials",
    "log10_delta_partials",
    "log10_tau_m_partials",
    "IERS",
]

# IERS Conventions 2010, Chapter 6, Table 6.5b: long-period zonal tides for k20.
# Doodson IDs are written without the comma and multiplied by 1000, matching the
# nint(xnd(i) * 1000._DP) convention already used in f_marsol.f90 in GINS.
IERS_LONG_PERIOD_ZONAL_TIDES: tuple[tuple[int, float], ...] = (
    (55565, 0.00221),
    (55575, 0.00441),
    (56554, 0.04107),
    (57555, 0.08214),
    (57565, 0.08434),
    (58554, 0.12320),
    (63655, 0.47152),
    (65445, 0.54217),
    (65455, 0.54438),
    (65465, 0.54658),
    (65655, 0.55366),
    (73555, 1.01590),
    (75355, 1.08875),
    (75555, 1.09804),
    (75565, 1.10024),
    (75575, 1.10245),
    (83655, 1.56956),
    (85455, 1.64241),
    (85465, 1.64462),
    (93555, 2.11394),
    (95355, 2.18679),
)


def tide_angular_frequencies_to_cycle_per_yr(
    long_period_zonal_tides: tuple[tuple[int, float], ...] = IERS_LONG_PERIOD_ZONAL_TIDES,
) -> ndarray:
    """
    From degrees per hour to yr^-1.
    """

    return (
        array(object=[tide[1] for tide in long_period_zonal_tides], dtype=float)
        / 360.0
        / 3600.0
        * SECONDS_PER_YEAR
    )


def pole_motion_correction(
    i_signal: tuple[int, int],
    frequencies: ndarray,  # Already in steady-state.
    m_complex: ndarray,
    love_numbers_model: ndarray | complex | float = K_2_IERS,  # (n_periods).
    love_number_log_frequencies: Optional[ndarray] = None,
) -> tuple[ndarray, ndarray]:
    """
    Compute the coherent pole-tide C21 and S21 correction time series for one k2 model.
    """

    assert len(frequencies) == len(m_complex)

    if not isinstance(love_numbers_model, (ndarray, list)):

        love_number = complex(love_numbers_model)
        love_numbers = love_number * asarray(
            a=frequencies > 0, dtype=float
        ) + love_number.conjugate() * asarray(a=frequencies < 0, dtype=float)

    else:

        # Only interpolates on strictly positive frequencies, then build the Hermitian.
        love_numbers = zeros(shape=len(frequencies), dtype=complex)
        positive = frequencies > 0
        love_numbers[positive] = lagrange_order4(
            x=love_number_log_frequencies,
            y=love_numbers_model.real[0],
            new_x=log(frequencies[positive]),
        ) + 1j * lagrange_order4(
            x=love_number_log_frequencies,
            y=love_numbers_model.imag[0],
            new_x=log(frequencies[positive]),
        )

        freq_to_index = {round(float(f), 10): i for i, f in enumerate(frequencies)}

        for i_period, frequency in enumerate(frequencies):

            if frequency < 0:

                love_numbers[i_period] = conjugate(
                    love_numbers[freq_to_index[round(float(abs(frequency)), 10)]]
                )

            if abs(frequency) < 1 / LONG_TERM_HYPOTHESIS_PERIOD:

                m_complex[i_period] = 0

    # Solid Earth pole tide. The result is C21 + i*S21 in the frequency domain.
    phi_se_pt_complex: ndarray = -PHI_CONSTANT * love_numbers * m_complex
    coherent_pole_tide_correction: ndarray = ifft(phi_se_pt_complex)
    i_signal_start, i_signal_stop = i_signal
    coherent_pole_tide_correction = coherent_pole_tide_correction[
        i_signal_start : i_signal_start + i_signal_stop
    ]

    return coherent_pole_tide_correction.real, coherent_pole_tide_correction.imag


def fmt(x: float) -> str:
    """
    Formats like +7.90710E+03.
    """

    return f"{float(x):+.5E}"


def write_1d_slice(
    left_hand_side: str,
    values: list[str],
) -> str:
    """
    Prepares a slice of a 1D tab to hard-code in Fortran90.
    """

    statements: list[list[str]] = []
    current_stmt: list[str] = []
    current_len = 0

    for v in values:

        token = v + ", "

        if current_len + len(token) > MAX_STATEMENT_LENGTH:

            statements.append(current_stmt)
            current_stmt = [token]
            current_len = len(token)

        else:

            current_stmt.append(token)
            current_len += len(token)

    if current_stmt:

        statements.append(current_stmt)

    result = ""
    idx_start = 1

    for stmt in statements:

        lines: list[str] = []
        current = ""

        for token in stmt:

            if len(current) + len(token) > MAX_LINE_LENGTH:

                lines.append(current.rstrip())
                current = token

            else:

                current += token

        if current:

            lines.append(current.rstrip())

        values_str = "&\n  ".join(lines)
        idx_end = idx_start + len(stmt) - 1
        result += f"""  {left_hand_side}{idx_start}:{idx_end}) = (/ &
{values_str[:-1]} /)
"""
        idx_start = idx_end + 1

    return result


def hard_code_fortran90(
    variable_name: str,
    array_to_write: ndarray,
    float_option: bool = True,
) -> tuple[str, str]:
    """
    Returns a Fortran declaration and executable assignments for a real array.
    """

    shape_iterable_string = ",".join(str(s) for s in array_to_write.shape)
    declaration = (
        f"  real(kind=DP), dimension({shape_iterable_string}) :: {variable_name}\n"
        if float_option
        else f"  integer, dimension({shape_iterable_string}) :: {variable_name}\n"
    )
    result = ""

    if array_to_write.ndim == 1:

        result += write_1d_slice(
            left_hand_side=f"{variable_name}(",
            values=[fmt(x) if float_option else str(int(x)) for x in array_to_write],
        )

    else:

        for idx in ndindex(array_to_write.shape[:-1]):

            fixed_indices = ",".join(str(i + 1) for i in idx)
            left_hand_side = f"{variable_name}({fixed_indices},"
            result += write_1d_slice(
                left_hand_side=left_hand_side,
                values=[fmt(x) if float_option else str(int(x)) for x in array_to_write[idx]],
            )

    return declaration, result


def dates_to_jjul_dates(dates: ndarray) -> ndarray:
    """
    CNES Jour Julien conversion using the local reference.
    """

    return 365.25 * (dates - JJUL_1970_REFERENCE_YEAR) + JJUL_1970_REFERENCE_JJUL


def insert_between_markers(
    file_path: str | Path, start_marker: str, end_marker: str, multiline_text: str
) -> None:
    """
    Inserts multiline_text between two preserved markers.
    """

    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)

    if start_index == -1:

        raise ValueError(f"Start marker not found in {path}: {start_marker}")

    if end_index == -1:

        raise ValueError(f"End marker not found in {path}: {end_marker}")

    if start_index > end_index:

        raise ValueError(f"Start marker appears after end marker in {path}")

    insert_start = start_index + len(start_marker)
    updated_content = (
        content[:insert_start] + "\n" + multiline_text.strip("\n") + "\n" + content[end_index:]
    )
    path.write_text(updated_content, encoding="utf-8")


def interpolate_love_number_grid_to_solid_tides(
    model_grid: ndarray,
    love_number_log_frequencies: ndarray,
    solid_tide_frequencies: ndarray,
) -> ndarray:
    """
    Interpolates one k2 grid to the fixed IERS long-period zonal tide frequencies.
    """

    interpolated = zeros(
        shape=tuple(list(model_grid.shape[:3]) + [len(solid_tide_frequencies)]),
        dtype=complex,
    )
    target_log_frequencies = log(solid_tide_frequencies)

    for idx in ndindex(model_grid.shape[:3]):

        k2_series: ndarray = model_grid[idx][0]
        interpolated[idx] = lagrange_order4(
            x=love_number_log_frequencies,
            y=k2_series.real,
            new_x=target_log_frequencies,
        ) + 1j * lagrange_order4(
            x=love_number_log_frequencies,
            y=k2_series.imag,
            new_x=target_log_frequencies,
        )

    return interpolated


def generate_pole_tide_models(
    m_complex: ndarray,
    i_signal: tuple[int, int],
    initial_values: tuple[int, int],
    frequencies: ndarray,  # (yr^-1).
    file_path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH.joinpath(DEFAULT_FOR_GINS_OUTPUT_DIRECTORY),
) -> tuple[
    int, dict[str, ndarray], dict[str, dict[str, ndarray]]
]:  # n_parameter_values, Parameter tabs, then pole tide.
    """
    Build pole tide C/S correction series for k2 and its partials. Output model grids have shape
    (n_alpha, n_delta, n_tau_m, n_dates/n_periods). The second and third axes correspond to
    log10_delta_values and log10_tau_m_values.
    """

    n_parameter_values = len(
        unique(
            [
                file.name.split("alpha")[1].split("Delta")[0]
                for file in file_path.glob("*")
                if "period" not in file.name
            ]
        )
    )
    love_numbers_for_gins_tabs = generate_parameter_lines(
        parameters=parameters_for_gins(n_parameter_values=n_parameter_values), write=False
    )
    (
        love_number_periods,
        elastic,
        love_numbers,
        love_numbers_partials,
    ) = load_love_numbers_for_gins(
        dummy_variable=n_parameter_values,
        models=MODELS,
        path=file_path.parent,
        directory=file_path.name,
        love_numbers_for_gins_tabs=love_numbers_for_gins_tabs,
    )
    love_number_log_frequencies = log(1 / love_number_periods)  # (yr^-1).
    pole_tide_correction_models: dict[str, dict[str, ndarray]] = {
        component: {
            model_name: zeros(shape=tuple(list(love_numbers.shape[:3]) + [i_signal[1]]))
            for model_name in MODEL_NAMES
        }
        for component in "CS"
    }
    (
        pole_tide_correction_models["C"]["elastic"],
        pole_tide_correction_models["S"]["elastic"],
    ) = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex,
        love_numbers_model=elastic[0],  # n=2.
        love_number_log_frequencies=love_number_log_frequencies,
    )
    (
        pole_tide_correction_models["C"]["IERS"],
        pole_tide_correction_models["S"]["IERS"],
    ) = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex,
        love_numbers_model=K_2_IERS,
        love_number_log_frequencies=love_number_log_frequencies,
    )
    pole_tide_correction_models["C"]["elastic"] += (
        -PHI_CONSTANT * (K_2_IERS.real * initial_values[0] + K_2_IERS.imag * initial_values[1])
        - pole_tide_correction_models["C"]["elastic"][0]
    )
    pole_tide_correction_models["S"]["elastic"] += (
        -PHI_CONSTANT
        * (
            K_2_IERS.imag * initial_values[0]  # TODO: Change sign?
            - K_2_IERS.real * initial_values[1]
        )
        - pole_tide_correction_models["S"]["elastic"][0]
    )
    pole_tide_correction_models["C"]["IERS"] += (
        -PHI_CONSTANT * (K_2_IERS.real * initial_values[0] + K_2_IERS.imag * initial_values[1])
        - pole_tide_correction_models["C"]["IERS"][0]
    )
    pole_tide_correction_models["S"]["IERS"] += (
        -PHI_CONSTANT
        * (
            K_2_IERS.imag * initial_values[0]  # TODO: Change sign?
            - K_2_IERS.real * initial_values[1]
        )
        - pole_tide_correction_models["S"]["IERS"][0]
    )
    grid_indices = list(ndindex(love_numbers.shape[:3]))
    for model_name, model_grid in tqdm(
        zip(
            ["anelastic", "alpha_partials", "log10_delta_partials", "log10_tau_m_partials"],
            [love_numbers] + list(love_numbers_partials.values()),
        ),
        total=4,
    ):

        with Pool() as pool:

            results = pool.starmap(
                pole_motion_correction,
                [
                    (
                        i_signal,
                        frequencies,
                        m_complex,
                        model_grid[idx],
                        love_number_log_frequencies,
                    )
                    for idx in grid_indices
                ],
            )

        for idx, (c_model, s_model) in zip(grid_indices, results):

            pole_tide_correction_models["C"][model_name][idx] = c_model + (
                -PHI_CONSTANT
                * (K_2_IERS.real * initial_values[0] + K_2_IERS.imag * initial_values[1])
                - c_model[0]
            )
            pole_tide_correction_models["S"][model_name][idx] = s_model + (
                -PHI_CONSTANT
                * (
                    K_2_IERS.imag * initial_values[0]  # TODO: Change sign?
                    - K_2_IERS.real * initial_values[1]
                )
                - s_model[0]
            )

    return n_parameter_values, love_numbers_for_gins_tabs, pole_tide_correction_models


def generate_solid_tide_models(
    n_parameter_values: int,
    love_numbers_for_gins_tabs: dict[str, ndarray],
    file_path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH.joinpath(DEFAULT_FOR_GINS_OUTPUT_DIRECTORY),
) -> dict[str, dict[float, ndarray]]:
    """
    Build the interpolated k2 to zonal long-period tides.
    """

    n_parameter_values = len(
        unique(
            [
                file.name.split("alpha")[1].split("Delta")[0]
                for file in file_path.glob("*")
                if "period" not in file.name
            ]
        )
    )
    (
        love_number_periods,
        elastic,
        love_numbers,
        love_numbers_partials,
    ) = load_love_numbers_for_gins(
        dummy_variable=n_parameter_values,
        models=MODELS,
        path=file_path.parent,
        directory=file_path.name,
        love_numbers_for_gins_tabs=love_numbers_for_gins_tabs,
    )
    love_number_log_frequencies = log(1 / love_number_periods)  # (yr^-1).
    solid_tide_correction_models: dict[str, ndarray] = {
        model_name: zeros(
            shape=tuple(list(love_numbers.shape[:3]) + [len(IERS_LONG_PERIOD_ZONAL_TIDES)]),
            dtype=complex,
        )
        for model_name in MODEL_NAMES
    }
    solid_tide_frequencies = tide_angular_frequencies_to_cycle_per_yr()
    solid_tide_correction_models["elastic"][...] = elastic[0]
    solid_tide_correction_models["IERS"][...] = complex(K_2_IERS)
    for model_name, model_grid in tqdm(
        zip(
            ["anelastic", "alpha_partials", "log10_delta_partials", "log10_tau_m_partials"],
            [love_numbers] + list(love_numbers_partials.values()),
        ),
        total=4,
    ):

        solid_tide_correction_models[model_name][...] = interpolate_love_number_grid_to_solid_tides(
            model_grid=model_grid,
            love_number_log_frequencies=love_number_log_frequencies,
            solid_tide_frequencies=solid_tide_frequencies,
        )

    return solid_tide_correction_models


def save_pole_tide_corrections(
    dates: ndarray,
    parameter_tabs: dict[str, ndarray],
    pole_tide_correction_models: dict[str, dict[str, ndarray]],
    models_path: Path = TIDE_MODELS_PATH,
    pole_tide_corrections_file: Path = DEFAULT_POLE_TIDE_CORRECTION_FILE,
) -> None:
    """
    Hard-codes the pole tide corrections and their partials in f_marpolsol.f90.
    """

    alpha_values, delta_values, omega_m_values = tuple(parameter_tabs.values())
    log10_delta_values = log10(delta_values)
    log10_tau_m_values = log10(1 / omega_m_values)
    model_jjul_dates = dates_to_jjul_dates(dates=dates)
    model_mask = (model_jjul_dates >= DATA_DATES_LOWER_BOUND - DATA_DATES_MARGIN) & (
        model_jjul_dates <= DATA_DATES_UPPER_BOUND + DATA_DATES_MARGIN
    )
    definitions_to_hard_code = [
        "  ! Pole tide corrections generated from k2(alpha, log10(delta), log10(tau_m)).\n",
        "  ! Partial arrays are derivatives wrt alpha, log10(delta), and log10(tau_m).\n",
        f"  integer :: n_dates = {len(model_jjul_dates[model_mask])}\n",
        f"  integer :: n_alpha = {len(alpha_values)}\n",
        f"  integer :: n_delta = {len(log10_delta_values)}\n",
        f"  integer :: n_tau_m = {len(log10_tau_m_values)}\n",
    ]
    chunks_to_hard_code: list[str] = []

    for variable_name, array_to_write in {
        "jjul_dates": model_jjul_dates[model_mask],
        "alpha_values": alpha_values,
        "log10_delta_values": log10(delta_values),
        "log10_tau_m_values": log10(1 / omega_m_values),
    }.items():

        declaration, assignment = hard_code_fortran90(
            variable_name=variable_name, array_to_write=array_to_write
        )
        definitions_to_hard_code.append(declaration)
        chunks_to_hard_code.append(assignment)

    for component in "CS":

        for model_name, model in pole_tide_correction_models[component].items():

            # The IERS reference is saved to disk but not hard-coded in Fortran.
            if model_name == "IERS":

                continue

            declaration, assignment = hard_code_fortran90(
                variable_name="_".join((component, "potential", model_name)),
                array_to_write=asarray(model, dtype=float)[..., model_mask],
            )
            definitions_to_hard_code.append(declaration)
            chunks_to_hard_code.append(assignment)

    insert_between_markers(
        file_path=pole_tide_corrections_file,
        start_marker=START_DECL,
        end_marker=END_DECL,
        multiline_text="".join(definitions_to_hard_code),
    )
    insert_between_markers(
        file_path=pole_tide_corrections_file,
        start_marker=START_VALUES,
        end_marker=END_VALUES,
        multiline_text="".join(chunks_to_hard_code),
    )
    save_base_model(
        obj=pole_tide_correction_models,
        path=models_path,
        name=POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    )
    save_base_model(obj=model_jjul_dates, name="jjul_dates", path=models_path)
    save_base_model(obj=model_mask, name="model_mask", path=models_path)
    save_base_model(obj=alpha_values, name="alpha_values", path=models_path)
    save_base_model(obj=log10_delta_values, name="log10_delta_values", path=models_path)
    save_base_model(obj=log10_tau_m_values, name="log10_tau_m_values", path=models_path)


def save_solid_tide_corrections(
    parameter_tabs: dict[str, ndarray],
    solid_tide_correction_models: dict[str, ndarray],
    models_path: Path = TIDE_MODELS_PATH,
    solid_tide_corrections_file: Path = DEFAULT_SOLID_TIDE_CORRECTION_FILE,
) -> None:
    """
    Hard-codes interpolated k20 values and their partials in f_marsol.f90.

    The hard-coded arrays are gridded as:
        alpha, log10(delta), log10(tau_m), solid_tide_index

    The solid_tide_index axis follows IERS_LONG_PERIOD_ZONAL_TIDES.
    """

    alpha_values, delta_values, omega_m_values = tuple(parameter_tabs.values())
    log10_delta_values = log10(delta_values)
    log10_tau_m_values = log10(1 / omega_m_values)
    solid_tide_doodson_ids = array(
        object=[doodson_id for doodson_id, _ in IERS_LONG_PERIOD_ZONAL_TIDES],
        dtype=int,
    )
    solid_tide_frequency_values = tide_angular_frequencies_to_cycle_per_yr()
    definitions_to_hard_code = [
        "  ! Solid tide k20 generated from k2(alpha, log10(delta), log10(tau_m), tide).\n",
        "  ! Frequency interpolation is done at the IERS Table 6.5b long-period zonal tides.\n",
        "  ! Partial arrays are derivatives wrt alpha, log10(delta), and log10(tau_m).\n",
        f"  integer :: n_alpha = {len(alpha_values)}\n",
        f"  integer :: n_delta = {len(log10_delta_values)}\n",
        f"  integer :: n_tau_m = {len(log10_tau_m_values)}\n",
        f"  integer :: n_solid_tides = {len(solid_tide_doodson_ids)}\n",
    ]
    chunks_to_hard_code: list[str] = []
    real_arrays = {
        "solid_alpha_values": alpha_values,
        "solid_log10_delta_values": log10_delta_values,
        "solid_log10_tau_m_values": log10_tau_m_values,
        "solid_tide_frequency_values": solid_tide_frequency_values,
    }

    for variable_name, array_to_write in real_arrays.items():

        declaration, assignment = hard_code_fortran90(
            variable_name=variable_name,
            array_to_write=asarray(array_to_write, dtype=float),
        )
        definitions_to_hard_code.append(declaration)
        chunks_to_hard_code.append(assignment)

    declaration, assignment = hard_code_fortran90(
        variable_name="solid_tide_doodson_ids",
        array_to_write=solid_tide_doodson_ids,
        float_option=False,
    )
    definitions_to_hard_code.append(declaration)
    chunks_to_hard_code.append(assignment)
    model_to_fortran_variable = {
        "anelastic": "solid_k2",
        "alpha_partials": "solid_k2_dalpha",
        "log10_delta_partials": "solid_k2_dlog10_delta",
        "log10_tau_m_partials": "solid_k2_dlog10_tau_m",
    }

    for model_name, fortran_prefix in model_to_fortran_variable.items():

        model = asarray(solid_tide_correction_models[model_name])
        declaration, assignment = hard_code_fortran90(
            variable_name=f"{fortran_prefix}_real",
            array_to_write=asarray(model.real, dtype=float),
        )
        definitions_to_hard_code.append(declaration)
        chunks_to_hard_code.append(assignment)
        declaration, assignment = hard_code_fortran90(
            variable_name=f"{fortran_prefix}_imag",
            array_to_write=asarray(model.imag, dtype=float),
        )
        definitions_to_hard_code.append(declaration)
        chunks_to_hard_code.append(assignment)

    insert_between_markers(
        file_path=solid_tide_corrections_file,
        start_marker=START_DECL,
        end_marker=END_DECL,
        multiline_text="".join(definitions_to_hard_code),
    )
    insert_between_markers(
        file_path=solid_tide_corrections_file,
        start_marker=START_VALUES,
        end_marker=END_VALUES,
        multiline_text="".join(chunks_to_hard_code),
    )
    save_base_model(
        obj={element: tab.real for element, tab in solid_tide_correction_models.items()},
        path=models_path,
        name=SOLID_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME + "_real",
    )
    save_base_model(
        obj={element: tab.imag for element, tab in solid_tide_correction_models.items()},
        path=models_path,
        name=SOLID_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME + "_imag",
    )
    save_base_model(obj=solid_tide_doodson_ids, name="solid_tide_doodson_ids", path=models_path)
    save_base_model(
        obj=solid_tide_frequency_values, name="solid_tide_frequency_values", path=models_path
    )


def preprocess_and_save_tide_correction_partials(
    steady_state_signal_parameters: SteadyStateSignalParameters = DEFAULT_SIGNAL_PARAMETERS,
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
) -> None:
    """
    Builds and hard-codes pole tide corrections from k2 and its partials.
    """

    dates, m_1, m_2 = get_m1_m2_time_series(
        models_path=models_path, pole_motion_file=pole_motion_file
    )
    mean_m_1 = mean(a=m_1[: len(m_1)])
    mean_m_2 = mean(a=m_2[: len(m_2)])
    i_signal_start, steady_state_dates, steady_state_m_1 = build_steady_state_regime_signal(
        t=dates,
        signal=m_1 - mean_m_1,
        plateau_length=steady_state_signal_parameters.plateau_length,
        cubic_spline_length=steady_state_signal_parameters.cubic_spline_length,
    )
    _, _, steady_state_m_2 = build_steady_state_regime_signal(
        t=dates,
        signal=m_2 - mean_m_2,
        plateau_length=steady_state_signal_parameters.plateau_length,
        cubic_spline_length=steady_state_signal_parameters.cubic_spline_length,
    )
    frequencies = fftfreq(
        n=len(steady_state_dates), d=steady_state_dates[1] - steady_state_dates[0]
    )
    m_complex = fft(x=steady_state_m_1) - 1j * fft(x=steady_state_m_2)
    n_parameter_values, tabs, pole_models = generate_pole_tide_models(
        m_complex=m_complex,
        i_signal=(i_signal_start, len(dates)),
        initial_values=(m_1[0] + mean_m_1, m_2[0] + mean_m_2),
        frequencies=frequencies,
    )
    solid_models = generate_solid_tide_models(
        n_parameter_values=n_parameter_values, love_numbers_for_gins_tabs=tabs
    )
    save_pole_tide_corrections(
        dates=dates,
        parameter_tabs=tabs,
        pole_tide_correction_models=pole_models,
        models_path=TIDE_MODELS_PATH,
        pole_tide_corrections_file=DEFAULT_POLE_TIDE_CORRECTION_FILE,
    )
    save_solid_tide_corrections(
        parameter_tabs=tabs,
        solid_tide_correction_models=solid_models,
        models_path=TIDE_MODELS_PATH,
        solid_tide_corrections_file=DEFAULT_SOLID_TIDE_CORRECTION_FILE,
    )
