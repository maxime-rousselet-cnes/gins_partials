"""
Defines the pole tide correction.
"""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from alna import ALPHA_TAB, DELTA_TAB, INTEGRATION_PATH, load_love_numbers_for_gins
from base_models import (
    DATA_PATH,
    SteadyStateSignalParameters,
    build_steady_state_regime_signal,
    lagrange_order4,
    load_base_model,
    save_base_model,
)
from numpy import argmax, array, conjugate, log, ndarray, ndindex, zeros
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm

from .utils import (
    DATA_DATES_LOWER_BOUND,
    DATA_DATES_MARGIN,
    DATA_DATES_UPPER_BOUND,
    EARTH_RADIUS,
    H_2_IERS,
    JJUL_1970_REFERENCE_JJUL,
    JJUL_1970_REFERENCE_YEAR,
    K_2_IERS,
    PHI_CONSTANT,
    get_m1_m2_time_series,
)

DEFAULT_POLE_TIDE_CORRECTION_FILE = DATA_PATH.absolute().parent.parent.parent.parent.joinpath(
    "gin/sub/obelix/src/f_marpolsol.f90"
)
POLE_MODELS_PATH = DATA_PATH.joinpath("pole")
DEFAULT_SIGNAL_PARAMETERS = SteadyStateSignalParameters()
POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME = "pole_tide_correction_models"


def pole_motion_correction(
    i_signal: tuple[int, int],
    frequencies: ndarray,  # Already in steady-state.
    m_complex: ndarray,
    love_numbers_model: ndarray | complex | float = K_2_IERS,  # (n_periods).
    love_number_log_frequencies: Optional[ndarray] = None,
) -> tuple[ndarray, ndarray]:
    """
    Computes pole tide correction time series coherent with the given rheological model.
    """

    assert len(frequencies) == len(m_complex)

    if not isinstance(love_numbers_model, ndarray):

        love_numbers = love_numbers_model * array(
            object=frequencies > 0, dtype=float
        ) + love_numbers_model.conjugate() * array(object=frequencies < 0, dtype=float)

    else:

        # Only interpolates on striclty positive periods.
        love_numbers = zeros(shape=len(frequencies), dtype=complex)
        love_numbers[frequencies > 0] = lagrange_order4(
            x=love_number_log_frequencies,
            y=love_numbers_model.real,
            new_x=log(frequencies[frequencies > 0]),
        ) + 1j * lagrange_order4(
            x=love_number_log_frequencies,
            y=love_numbers_model.imag,
            new_x=log(frequencies[frequencies > 0]),
        )

        # Builds Hermitian.
        freq_to_index = {f: i for i, f in enumerate(frequencies)}

        for i_period, period in enumerate(frequencies):

            if period < 0:

                love_numbers[i_period] = conjugate(
                    love_numbers[round(number=freq_to_index[abs(period)], ndigits=10)]
                )

            elif period == 0:

                love_numbers[i_period] = love_numbers_model[argmax(love_number_log_frequencies)]

    # Solid Earth (SE) pole Tide (PT).
    phi_se_pt_complex: ndarray = -PHI_CONSTANT * love_numbers * m_complex

    # C_PT_SE_2_1 - 1j * S_PT_SE_2_1.
    coherent_pole_tide_correction = array(object=ifft(phi_se_pt_complex), dtype=complex)
    i_signal_start, i_signal_stop = i_signal
    coherent_pole_tide_correction = coherent_pole_tide_correction[
        i_signal_start : i_signal_start + i_signal_stop
    ]

    return coherent_pole_tide_correction.real, coherent_pole_tide_correction.imag


def fmt(x):
    """
    Formats like +7.9071E+03.
    """

    return f"{x:+.4E}"


def hard_code_fortran90(
    variable_name: str,
    array_to_write: ndarray,
    max_line_length: int = 6800,
    max_statement_length: int = 20000,
) -> tuple[str, str]:
    """
    Writes in a "fortran 90" ready format.
    """

    def _write_1d_slice(
        lhs: str,
        values: list[str],
    ) -> str:

        statements = []
        current_stmt = []
        current_len = 0

        for v in values:

            token = v + ", "

            if current_len + len(token) > max_statement_length:

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

            lines = []
            current = ""

            for token in stmt:

                if len(current) + len(token) > max_line_length:

                    lines.append(current.rstrip())
                    current = token

                else:

                    current += token

            if current:

                lines.append(current.rstrip())

            values_str = "&\n  ".join(lines)

            n_vals = len(stmt)
            idx_end = idx_start + n_vals - 1

            result += f"""  {lhs}{idx_start}:{idx_end}) = (/ &
  {values_str[:-1]} /)
"""

            idx_start = idx_end + 1

        return result

    shape_iterable_string = ",".join(str(s) for s in array_to_write.shape)

    declaration = f"  real(kind=DP), dimension({shape_iterable_string}) " f":: {variable_name}\n"

    result = ""

    if array_to_write.ndim == 1:

        values = [fmt(x) for x in array_to_write]

        result += _write_1d_slice(
            lhs=f"{variable_name}(",
            values=values,
        )

    else:

        for idx in ndindex(array_to_write.shape[:-1]):

            vec = array_to_write[idx]

            values = [fmt(x) for x in vec]

            fixed_indices = ",".join(str(i + 1) for i in idx)

            lhs = f"{variable_name}" f"({fixed_indices},"

            result += _write_1d_slice(
                lhs=lhs,
                values=values,
            )

    return declaration, result


def compute_delta_model(
    model,
    i_signal,
    frequencies,
    m_complex,
    love_number_log_frequencies,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    To parallelize the pole tide model processings.
    """

    # Potential (vertical component index 1)
    c_pot, s_pot = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex,
        love_numbers_model=model[0, :, 1],
        love_number_log_frequencies=love_number_log_frequencies,
    )

    # Deformation (vertical scaling)
    c_def, s_def = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex * EARTH_RADIUS,
        love_numbers_model=model[0, :, 0],
        love_number_log_frequencies=love_number_log_frequencies,
    )

    return c_pot, s_pot, c_def, s_def


def generate_pole_tide_models(
    m_complex: ndarray,
    i_signal: tuple[int, int],
    frequencies: ndarray,
    path: Path = INTEGRATION_PATH,
) -> dict[str, dict[str, dict[str, ndarray]]]:
    """
    Loads Already computed Love numbers and builds pole tide correction series and pole tide
    deformation correction series. Includes elastic and IERS models. The saved corrections have a
    similar and constant bias that can be retrieved from the elastic unbiased correction.
    """

    love_number_log_frequencies = log(load_base_model(name="periods_tab", path=path), dtype=float)
    models = dict(
        zip(
            [
                "elastic_love_numbers",
                "love_numbers",
                "love_numbers_alpha_partials",
                "love_numbers_delta_partials",
            ],
            # (n_alpha, n_delta, n_degrees, n_periods, n_directions) = (16, 13, 2, 40, 2).
            load_love_numbers_for_gins(),
        )
    )
    pole_tide_correction_models = {
        component: {
            correction_type: {
                model_name: zeros(
                    shape=tuple(list(models["love_numbers"].shape[:2]) + [i_signal[1]])
                )
                for model_name in models
                if "elastic" not in model_name
            }
            for correction_type in ["potential", "deformation"]
        }
        for component in "CS"
    }
    (
        pole_tide_correction_models["C"]["potential"]["IERS"],
        pole_tide_correction_models["S"]["potential"]["IERS"],
    ) = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex,
        love_number_log_frequencies=love_number_log_frequencies,
    )
    (
        pole_tide_correction_models["C"]["deformation"]["IERS"],
        pole_tide_correction_models["S"]["deformation"]["IERS"],
    ) = pole_motion_correction(
        i_signal=i_signal,
        frequencies=frequencies,
        m_complex=m_complex * EARTH_RADIUS,
        love_numbers_model=H_2_IERS,
        love_number_log_frequencies=love_number_log_frequencies,
    )

    for model_name, model_tab in models.items():

        if "elastic" in model_name:

            (
                pole_tide_correction_models["C"]["potential"][model_name],
                pole_tide_correction_models["S"]["potential"][model_name],
            ) = pole_motion_correction(
                i_signal=i_signal,
                frequencies=frequencies,
                m_complex=m_complex,
                love_numbers_model=model_tab[0, 1],  # For Degree 2, potential.
                love_number_log_frequencies=love_number_log_frequencies,
            )
            (
                pole_tide_correction_models["C"]["deformation"][model_name],
                pole_tide_correction_models["S"]["deformation"][model_name],
            ) = pole_motion_correction(
                i_signal=i_signal,
                frequencies=frequencies,
                m_complex=m_complex * EARTH_RADIUS,
                love_numbers_model=model_tab[0, 0],  # For Degree 2, vertical.
                love_number_log_frequencies=love_number_log_frequencies,
            )

            assert len(pole_tide_correction_models["C"]["potential"][model_name]) == i_signal[1]

        else:

            for i_alpha, model_array in tqdm(enumerate(model_tab)):

                with Pool() as p:

                    results = p.starmap(
                        compute_delta_model,
                        [
                            (
                                model,
                                i_signal,
                                frequencies,
                                m_complex,
                                love_number_log_frequencies,
                            )
                            for model in model_array
                        ],
                    )

                for i_delta, model in enumerate(results):

                    (
                        pole_tide_correction_models["C"]["potential"][model_name][i_alpha, i_delta],
                        pole_tide_correction_models["S"]["potential"][model_name][i_alpha, i_delta],
                        pole_tide_correction_models["C"]["deformation"][model_name][
                            i_alpha, i_delta
                        ],
                        pole_tide_correction_models["S"]["deformation"][model_name][
                            i_alpha, i_delta
                        ],
                    ) = model

    return pole_tide_correction_models


def dates_to_jjul_dates(dates: ndarray) -> ndarray:
    """
    "Jour Julien" conversion using reference.
    """

    return 365.25 * (dates - JJUL_1970_REFERENCE_YEAR) + JJUL_1970_REFERENCE_JJUL


def insert_between_markers(file_path: str, start_marker: str, end_marker: str, multiline_text: str):
    """
    Inserts multiline_text between the first occurrence of start_marker
    and end_marker in the file.
    The markers themselves are preserved.
    """

    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)

    if start_index == -1:

        raise ValueError(f"Start marker not found: {start_marker}")

    if end_index == -1:

        raise ValueError(f"End marker not found: {end_marker}")

    if start_index > end_index:

        raise ValueError("Start marker appears after end marker")

    insert_start = start_index + len(start_marker)
    updated_content = (
        content[:insert_start] + "\n" + multiline_text.strip("\n") + "\n" + content[end_index:]
    )
    path.write_text(updated_content, encoding="utf-8")


def save_pole_tide_corrections(
    dates: ndarray,
    pole_tide_correction_models: dict[str, dict[str, dict[str, ndarray]]],
    models_path: Path = POLE_MODELS_PATH,
    pole_tide_corrections_file: Path = DEFAULT_POLE_TIDE_CORRECTION_FILE,
    alpha_delta_tabs: tuple[ndarray, ndarray] = (ALPHA_TAB, DELTA_TAB),
) -> None:
    """
    Writes all pole tide corrections and their partials in a (.TXT) file.
    """

    model_jjul_dates = dates_to_jjul_dates(dates=dates)
    model_mask = (model_jjul_dates >= DATA_DATES_LOWER_BOUND - DATA_DATES_MARGIN) & (
        model_jjul_dates <= DATA_DATES_UPPER_BOUND + DATA_DATES_MARGIN
    )
    definitions_to_hard_code = [
        f"""  integer :: n_dates = {len(model_jjul_dates[model_mask])}\n""",
        f"""  integer :: n_alpha = {len(alpha_delta_tabs[0])}\n""",
        f"""  integer :: n_delta = {len(alpha_delta_tabs[1])}\n""",
    ]
    chuncks_to_hard_code = []
    a, b = hard_code_fortran90(
        variable_name="jjul_dates", array_to_write=model_jjul_dates[model_mask]
    )
    definitions_to_hard_code += [a]
    chuncks_to_hard_code += [b]
    a, b = hard_code_fortran90(variable_name="alpha_values", array_to_write=alpha_delta_tabs[0])
    definitions_to_hard_code += [a]
    chuncks_to_hard_code += [b]
    a, b = hard_code_fortran90(variable_name="delta_values", array_to_write=alpha_delta_tabs[1])
    definitions_to_hard_code += [a]
    chuncks_to_hard_code += [b]

    for (
        component,
        pole_tide_correction_models_per_correction_type,
    ) in pole_tide_correction_models.items():

        for (
            correction_type,
            pole_tide_correction_models_per_model,
        ) in pole_tide_correction_models_per_correction_type.items():

            for model_name, model in pole_tide_correction_models_per_model.items():

                a, b = hard_code_fortran90(
                    variable_name="_".join((component, correction_type, model_name)),
                    array_to_write=array(object=model, dtype=float)[..., model_mask],
                )
                definitions_to_hard_code += [a]
                chuncks_to_hard_code += [b]

    insert_between_markers(
        file_path=pole_tide_corrections_file,
        start_marker="  !Variables locales",
        end_marker="  integer :: std_crt",
        multiline_text="".join(definitions_to_hard_code),
    )

    insert_between_markers(
        file_path=pole_tide_corrections_file,
        start_marker="  !>Partie executive",
        end_marker="!Initialisation des variables",
        multiline_text="".join(chuncks_to_hard_code),
    )
    save_base_model(
        obj=pole_tide_correction_models,
        path=models_path,
        name=POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    )
    save_base_model(
        obj=model_jjul_dates,
        name="jjul_dates",
        path=models_path,
    )
    save_base_model(
        obj=model_mask,
        name="model_mask",
        path=models_path,
    )
    save_base_model(
        obj=alpha_delta_tabs[0],
        name="alpha_values",
        path=models_path,
    )
    save_base_model(
        obj=alpha_delta_tabs[1],
        name="delta_values",
        path=models_path,
    )


def preprocess_and_save_tide_correction_partials(
    steady_state_signal_parameters: SteadyStateSignalParameters = DEFAULT_SIGNAL_PARAMETERS,
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
    pole_tide_corrections_file: Path = DEFAULT_POLE_TIDE_CORRECTION_FILE,
) -> None:
    """
    Gets already computed Love numbers for a range of admissible physical quantities.
    Deduces the Love number arrays to write in GINS routines and their partial derivatives.
    Deduces the pole tide correction arrays to write in GINS routines and their partial derivatives.
    Deduces the pole tide deformation correction arrays to write in GINS routines and their partial
    derivatives.
    """

    dates, m_1, m_2 = get_m1_m2_time_series(
        models_path=models_path, pole_motion_file=pole_motion_file
    )
    i_signal_start, steady_state_dates, steady_state_m_1 = build_steady_state_regime_signal(
        t=dates,
        signal=m_1 - m_1[0],
        plateau_length=steady_state_signal_parameters.plateau_length,
        cubic_spline_length=steady_state_signal_parameters.cubic_spline_length,
    )
    _, _, steady_state_m_2 = build_steady_state_regime_signal(
        t=dates,
        signal=m_2 - m_2[0],
        plateau_length=steady_state_signal_parameters.plateau_length,
        cubic_spline_length=steady_state_signal_parameters.cubic_spline_length,
    )
    frequencies = fftfreq(
        n=len(steady_state_dates), d=steady_state_dates[1] - steady_state_dates[0]
    )
    m_complex = fft(x=steady_state_m_1) - 1j * fft(x=steady_state_m_2)
    save_pole_tide_corrections(
        dates=dates,
        pole_tide_correction_models=generate_pole_tide_models(
            m_complex=m_complex,
            i_signal=(i_signal_start, len(dates)),
            frequencies=frequencies,
        ),
        models_path=models_path,
        pole_tide_corrections_file=pole_tide_corrections_file,
    )


def regenerate_fortran_code(
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "pole_tide_correction_models",
    pole_tide_corrections_file: Path = DEFAULT_POLE_TIDE_CORRECTION_FILE,
) -> None:
    """
    Regenerates the fortran90-ready code from the (.JSON) informations.
    """

    dates, _, _ = get_m1_m2_time_series(
        models_path=models_path, pole_motion_file="C01_pole_motion_time_series.txt"
    )
    pole_tide_correction_models = load_base_model(name=pole_motion_file, path=models_path)
    alpha_values = array(object=load_base_model(name="alpha_values", path=models_path), dtype=float)
    delta_values = array(object=load_base_model(name="delta_values", path=models_path), dtype=float)
    save_pole_tide_corrections(
        dates=dates,
        pole_tide_correction_models=pole_tide_correction_models,
        models_path=models_path,
        pole_tide_corrections_file=pole_tide_corrections_file,
        alpha_delta_tabs=(alpha_values, delta_values),
    )
