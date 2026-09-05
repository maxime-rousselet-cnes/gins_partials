"""
Defines and hard-codes the solid pole tide correction driven by k2 only in GINS routine
f_marpolsol.f90.
The generated Fortran tables are gridded in the runtime interpolation variables
(alpha, log10(Delta), log10(tau_m), date).
"""

from itertools import product
from pathlib import Path
from typing import NamedTuple, Optional

from alna import (
    ELASTIC_INTEGRATION_PATH,
    SECONDS_PER_YEAR,
    TO_GET_INVERSE_DERIVATIVES,
    TO_GET_LOG_DERIVATIVES,
    get_tabs_from_all_love_number_files,
    load_single_model_love_numbers_for_gins,
    load_solid_earth_numerical_model,
)
from base_models import (
    DATA_PATH,
    BoundaryCondition,
    Direction,
    SteadyStateSignalParameters,
    build_steady_state_regime_signal,
    lagrange_order4,
    load_base_model,
    save_base_model,
)
from numpy import (
    array,
    asarray,
    conjugate,
    dtype,
    flip,
    fromfile,
    log,
    mean,
    ndarray,
    ndindex,
    zeros,
)
from numpy.testing import assert_array_equal
from scipy.fft import fft, fftfreq, ifft

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

# TIDE_MODELS_PATH = Path("/work/GRGS/users/rousselm/public/tide_models").resolve()
TIDE_MODELS_PATH = DATA_PATH.joinpath("tide_models").resolve()
TIDE_DATA_PATH = DATA_PATH.joinpath("tide_binary_files").resolve()
LONG_TERM_HYPOTHESIS_PERIOD = 10000  # (yr).
POLE_MODELS_PATH = DATA_PATH.joinpath("pole").resolve()
DEFAULT_SIGNAL_PARAMETERS = SteadyStateSignalParameters()
POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME = "pole_tide_correction_models"
SOLID_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME = "solid_tide_correction_models"

# IERS Conventions 2010, Chapter 6, Table 6.5b: long-period zonal tides for k20.
# Doodson IDs are written without the comma and multiplied by 1000, matching the
# nint(xnd(i) * 1000._DP) convention already used in f_marsol.f90 in GINS.
REFERENCE_K2 = 0.30190
IERS_LONG_PERIOD_ZONAL_TIDES: tuple[tuple[int, float], ...] = (
    (55565, 0.00221, 0.01347, -0.00541),
    (55575, 0.00441, 0.01124, -0.00488),
    (56554, 0.04107, 0.00547, -0.00349),
    (57555, 0.08214, 0.00403, -0.00315),
    (57565, 0.08434, 0.00398, -0.00313),
    (58554, 0.12320, 0.00326, -0.00296),
    (63655, 0.47152, 0.00101, -0.00242),
    (65445, 0.54217, 0.00080, -0.00237),
    (65455, 0.54438, 0.00080, -0.00237),
    (65465, 0.54658, 0.00079, -0.00237),
    (65655, 0.55366, 0.00077, -0.00236),
    (73555, 1.01590, -0.00009, -0.00216),
    (75355, 1.08875, -0.00018, -0.00213),
    (75555, 1.09804, -0.00019, -0.00213),
    (75565, 1.10024, -0.00019, -0.00213),
    (75575, 1.10245, -0.00019, -0.00213),
    (83655, 1.56956, -0.00065, -0.00202),
    (85455, 1.64241, -0.00071, -0.00201),
    (85465, 1.64462, -0.00071, -0.00201),
    (93555, 2.11394, -0.00102, -0.00193),
    (95355, 2.18679, -0.00106, -0.00192),
)

NAMES_MAP = {
    r"\alpha^{MANTLE_0}": "lam",
    r"\log_{10}Q_\mu^{MANTLE_0}": "lqm",
    r"\log_{10}\Delta^{MANTLE_0}": "ldm",
    r"\log_{10}\tau_{m-inf}^{MANTLE_0}": "ltm",
}


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
            y=love_numbers_model.real,
            new_x=log(frequencies[positive]),
        ) + 1j * lagrange_order4(
            x=love_number_log_frequencies,
            y=love_numbers_model.imag,
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


def dates_to_jjul_dates(dates: ndarray) -> ndarray:
    """
    CNES Jour Julien conversion using the local reference.
    """

    return 365.25 * (dates - JJUL_1970_REFERENCE_YEAR) + JJUL_1970_REFERENCE_JJUL


class PoleMotionSignal(NamedTuple):
    """
    Original pole motion samples and their prepared frequency-domain signal.
    """

    dates: ndarray
    m_1: ndarray
    m_2: ndarray
    i_signal: tuple[int, int]
    frequencies: ndarray
    m_complex: ndarray


def prepare_pole_motion_signal(
    steady_state_signal_parameters: SteadyStateSignalParameters = DEFAULT_SIGNAL_PARAMETERS,
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
) -> PoleMotionSignal:
    """
    Load pole motion, remove its means, extend to steady state, and compute FFTs.
    Keep the original samples for reference offsets and output dates. The signal
    indices select those dates from the inverse-transformed steady-state signal.
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

    return PoleMotionSignal(
        dates=dates,
        m_1=m_1,
        m_2=m_2,
        i_signal=(i_signal_start, len(dates)),
        frequencies=frequencies,
        m_complex=m_complex,
    )


def tide_correction_model_generation(
    file_path: Path,
    steady_state_signal_parameters: SteadyStateSignalParameters = DEFAULT_SIGNAL_PARAMETERS,
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
    tide_models_path: Path = TIDE_MODELS_PATH,
) -> None:
    """
    Gets Love numbers for a given rheological model, generates the corresponding pole tide and
    solid Earth tide models and saves them together in a single (.JSON) file.
    """

    love_number_log_frequencies, love_numbers, love_number_partials = (
        load_single_model_love_numbers_for_gins(file_path=file_path)
    )

    pole_motion = prepare_pole_motion_signal(
        steady_state_signal_parameters=steady_state_signal_parameters,
        models_path=models_path,
        pole_motion_file=pole_motion_file,
    )
    corrections_to_save = {}
    solid_tide_frequencies = log(tide_angular_frequencies_to_cycle_per_yr())

    for model_name, model in zip(
        [""] + [NAMES_MAP[name] for name in love_number_partials.keys()],
        [love_numbers] + list(love_number_partials.values()),
    ):

        (
            corrections_to_save["_".join(("C", model_name)) if model_name else "C"],
            corrections_to_save["_".join(("S", model_name)) if model_name else "S"],
        ) = pole_motion_correction(
            i_signal=pole_motion.i_signal,
            frequencies=pole_motion.frequencies,
            m_complex=pole_motion.m_complex,
            love_numbers_model=model[0],  # Degree 2 only.
            love_number_log_frequencies=love_number_log_frequencies,
        )
        corrections_to_save["_".join(("k2", model_name, "real")) if model_name else "k2_real"] = (
            lagrange_order4(
                x=love_number_log_frequencies,
                y=model.real[0],  # Degree 2 only.
                new_x=solid_tide_frequencies,
            )
        )
        corrections_to_save["_".join(("k2", model_name, "imag")) if model_name else "k2_imag"] = (
            lagrange_order4(
                x=love_number_log_frequencies,
                y=model.imag[0],  # Degree 2 only.
                new_x=solid_tide_frequencies,
            )
        )

    save_base_model(
        obj=corrections_to_save,
        name=file_path.name,
        path=tide_models_path,
    )


def interpolate_love_number_grid_to_solid_tides(
    model_grid: ndarray,
    love_number_log_frequencies: ndarray,
    solid_tide_frequencies: ndarray,
) -> ndarray:
    """
    Interpolates one k2 grid to the fixed IERS long-period zonal tide frequencies.
    """

    interpolated = zeros(
        shape=tuple(
            list(model_grid.shape[: len(model_grid.shape) - 2]) + [len(solid_tide_frequencies)]
        ),
        dtype=complex,
    )
    target_log_frequencies = log(solid_tide_frequencies)

    for idx in ndindex(model_grid.shape[: len(model_grid.shape) - 2]):

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


def load_tide_correction_models(
    path: Path = TIDE_MODELS_PATH,
) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
    """
    Gets all individual tide correction models and their partials in a single dictionary.
    """

    tabs = get_tabs_from_all_love_number_files(path=path)
    all_correction_models = {}

    for iterators in product(*(range(len(tab)) for tab in tabs.values())):

        file_finder = list(
            path.glob(
                "*"
                + "*".join(
                    (f"{tab[iterator]:.2e}" for iterator, tab in zip(iterators, tabs.values()))
                )
                + "*"
            )
        )

        if not file_finder:

            raise NameError

        single_rheology_correction_models: dict[str, ndarray] = load_base_model(
            name=file_finder[0].name, path=file_finder[0].parent
        )

        for (
            correction_type,
            correction_model,
        ) in single_rheology_correction_models.items():

            correction_model = array(object=correction_model, dtype=float)

            if correction_type not in all_correction_models:

                all_correction_models[correction_type] = zeros(
                    shape=tuple([len(tab) for tab in tabs.values()] + list(correction_model.shape)),
                    dtype=complex,
                )

            all_correction_models[correction_type][iterators] = correction_model

    inverted_tabs = {}
    # Change of variables for inverse.
    for i_axis, parameter in enumerate(tabs):

        if parameter in TO_GET_INVERSE_DERIVATIVES:

            inverted_tabs[TO_GET_INVERSE_DERIVATIVES[parameter]] = 1 / flip(m=tabs[parameter])

            for correction_type in all_correction_models:

                all_correction_models[correction_type] = flip(
                    m=all_correction_models[correction_type],
                    axis=i_axis,
                )

        else:

            inverted_tabs[parameter] = tabs[parameter]

    log_inverted_tabs = {}
    # Change of variables for log.
    for i_axis, parameter in enumerate(inverted_tabs.keys()):

        if parameter in TO_GET_LOG_DERIVATIVES:

            log_inverted_tabs[r"\log_{10}" + parameter] = log(inverted_tabs[parameter]) / log(10)

        else:

            log_inverted_tabs[parameter] = inverted_tabs[parameter]

    return log_inverted_tabs, all_correction_models


def encode_tide_correction_models(
    path: Path = TIDE_MODELS_PATH,
    steady_state_signal_parameters: SteadyStateSignalParameters = DEFAULT_SIGNAL_PARAMETERS,
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
    to_save: bool = False,
) -> None:
    """
    Encodes in binary files fortran-compatible the pole tide and solid Earth tide corrections and
    their partials for all rheological models. Eventually saves them to (.JSON) files.
    """

    elastic = load_solid_earth_numerical_model(
        name="PREM",
        path=ELASTIC_INTEGRATION_PATH,
    ).love_numbers["real"][2][0][BoundaryCondition.POTENTIAL.value][Direction.POTENTIAL.value]
    tabs, all_correction_models = load_tide_correction_models(path=path)
    pole_motion = prepare_pole_motion_signal(
        steady_state_signal_parameters=steady_state_signal_parameters,
        models_path=models_path,
        pole_motion_file=pole_motion_file,
    )
    (
        all_correction_models["C_elastic"],
        all_correction_models["S_elastic"],
    ) = pole_motion_correction(
        i_signal=pole_motion.i_signal,
        frequencies=pole_motion.frequencies,
        m_complex=pole_motion.m_complex,
        love_numbers_model=elastic,
    )
    (
        all_correction_models["C_IERS"],
        all_correction_models["S_IERS"],
    ) = pole_motion_correction(
        i_signal=pole_motion.i_signal,
        frequencies=pole_motion.frequencies,
        m_complex=pole_motion.m_complex,
        love_numbers_model=K_2_IERS,
    )
    x_0 = pole_motion.m_1[0]
    y_0 = pole_motion.m_2[0]  # This is y_p - y_s, not IERS m_2.
    c_21_reference = -PHI_CONSTANT * (K_2_IERS.real * x_0 - K_2_IERS.imag * y_0)
    s_21_reference = PHI_CONSTANT * (K_2_IERS.real * y_0 + K_2_IERS.imag * x_0)

    for correction_type in all_correction_models:

        if (
            "lam" in correction_type
            or "lqm" in correction_type
            or "ldm" in correction_type
            or "ltm" in correction_type
        ) and "k" not in correction_type:

            all_correction_models[correction_type] -= all_correction_models[correction_type][
                ..., :1
            ]

        elif correction_type.startswith("C"):

            all_correction_models[correction_type] += (
                c_21_reference - all_correction_models[correction_type][..., :1]
            )

        elif correction_type.startswith("S"):

            all_correction_models[correction_type] += (
                s_21_reference - all_correction_models[correction_type][..., :1]
            )

    TIDE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    save_tabs(
        dates=pole_motion.dates,
        tabs=tabs,
        path=path,
        models_path=models_path,
        to_save=to_save,
    )
    save_pole_tide_corrections(
        dates=pole_motion.dates,
        pole_tide_correction_models={
            correction_type: correction_model.real
            for correction_type, correction_model in all_correction_models.items()
            if "C" in correction_type or "S" in correction_type
        },
        path=path,
        models_path=models_path,
        to_save=to_save,
    )
    save_solid_tide_corrections(
        solid_tide_correction_models={
            correction_type: correction_model.real
            for correction_type, correction_model in all_correction_models.items()
            if not ("C" in correction_type or "S" in correction_type)
        },
        path=path,
        models_path=models_path,
        to_save=to_save,
    )


def write_binary_fortran(
    path: Path,
    array_to_write: ndarray,
    *,
    dtype_to_write: dtype,
) -> None:
    """
    Writes a NumPy array as a raw binary stream compatible with Fortran:
        open(..., form='unformatted', access='stream')
        read(unit) array
    The array is serialized in Fortran/column-major order.
    """

    asarray(array_to_write, dtype=dtype_to_write).ravel(order="F").tofile(path)


def save_tabs(
    dates: ndarray,
    tabs: dict[str, ndarray],
    path: Path = TIDE_MODELS_PATH,
    models_path: Path = POLE_MODELS_PATH,
    to_save: bool = False,
) -> None:
    """
    Saves the grid of rheological parameters for the pole tide and solid Earth tide corrections.
    Verifies lecture consistency.
    """

    lam_values = tabs[r"\alpha^{MANTLE_0}"]
    lqm_values = tabs[r"\log_{10}Q_\mu^{MANTLE_0}"]
    ldm_values = tabs[r"\log_{10}\Delta^{MANTLE_0}"]
    ltm_values = tabs[r"\log_{10}\tau_{m-inf}^{MANTLE_0}"]
    model_jjul_dates = dates_to_jjul_dates(dates=dates)
    model_mask = (model_jjul_dates >= DATA_DATES_LOWER_BOUND - DATA_DATES_MARGIN) & (
        model_jjul_dates <= DATA_DATES_UPPER_BOUND + DATA_DATES_MARGIN
    )
    grid_arrays = {
        "jjul_dates": model_jjul_dates[model_mask],
        "lam_values": lam_values,
        "lqm_values": lqm_values,
        "ldm_values": ldm_values,
        "ltm_values": ltm_values,
    }

    for name, array_to_write in grid_arrays.items():

        write_binary_fortran(
            path=path / f"{name}.bin",
            array_to_write=array_to_write,
            dtype_to_write=dtype("<f8"),
        )
        from_binary = fromfile(
            path / f"{name}.bin",
            dtype="<f8",
        ).reshape(array_to_write.shape, order="F")
        assert_array_equal(
            from_binary,
            array_to_write.astype("<f8"),
        )

    if to_save:

        save_base_model(obj=model_jjul_dates, name="jjul_dates", path=models_path)
        save_base_model(obj=model_mask, name="model_mask", path=models_path)
        save_base_model(obj=lam_values, name="lam_values", path=models_path)
        save_base_model(obj=lqm_values, name="lqm_values", path=models_path)
        save_base_model(obj=ldm_values, name="ldm_values", path=models_path)
        save_base_model(obj=ltm_values, name="ltm_values", path=models_path)


def save_pole_tide_corrections(
    dates: ndarray,
    pole_tide_correction_models: dict[str, ndarray],
    path: Path = TIDE_MODELS_PATH,
    models_path: Path = POLE_MODELS_PATH,
    to_save: bool = False,
) -> None:
    """
    Saves the pole tide corrections and their partials in binary files fortran-compatible.
    Verifies lecture consistency.
    """

    model_jjul_dates = dates_to_jjul_dates(dates=dates)
    model_mask = (model_jjul_dates >= DATA_DATES_LOWER_BOUND - DATA_DATES_MARGIN) & (
        model_jjul_dates <= DATA_DATES_UPPER_BOUND + DATA_DATES_MARGIN
    )

    for model_name, model in pole_tide_correction_models.items():

        if "IERS" in model_name:

            continue

        array_to_write = model[..., model_mask]
        write_binary_fortran(
            path / f"{model_name}.bin",
            array_to_write,
            dtype_to_write=dtype("<f4"),
        )
        from_binary = fromfile(
            path / f"{model_name}.bin",
            dtype="<f4",
        ).reshape(array_to_write.shape, order="F")
        assert_array_equal(
            from_binary,
            array_to_write.astype("<f4"),
        )

    if to_save:

        save_base_model(
            obj=pole_tide_correction_models,
            path=models_path,
            name=POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
        )


def save_solid_tide_corrections(
    solid_tide_correction_models: dict[str, ndarray],
    path: Path = TIDE_MODELS_PATH,
    models_path: Path = POLE_MODELS_PATH,
    to_save: bool = False,
) -> None:
    """
    Saves the solid Earth tide corrections and their partials in binary files fortran-compatible.
    Verifies lecture consistency.
    """

    for variable_name, array_to_write in solid_tide_correction_models.items():

        write_binary_fortran(
            path / f"{variable_name}.bin",
            array_to_write,
            dtype_to_write=dtype("<f4"),
        )
        from_binary = fromfile(
            path / f"{variable_name}.bin",
            dtype="<f4",
        ).reshape(array_to_write.shape, order="F")
        assert_array_equal(
            from_binary,
            array_to_write.astype("<f4"),
        )

    if to_save:

        save_base_model(
            obj=solid_tide_correction_models,
            path=models_path,
            name=SOLID_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
        )
