"""
Main interraction functions with data files and python.
"""

from datetime import datetime
from pathlib import Path

from base_models import DATA_PATH, EARTH_RADIUS
from numpy import ndarray, pi
from pandas import read_csv

ARC_SECOND_TO_RADIANS = 2 * pi / (60 * 60 * 360)
MILLI_ARC_SECOND_TO_RADIANS = ARC_SECOND_TO_RADIANS / 1000
MEAN_POLE_COEFFICIENTS = {"m_1": [55.0, 1.677], "m_2": [320.5, 3.460]}  # (mas/yr^index).
MEAN_POLE_T_0 = (2000.0,)  # (yr).

OMEGA = 2.0 * pi / 86164.0  # (Rad.s^-1)
MEAN_G_AT_EARTH_SURFACE = 9.8  # (m.s^-2).
PHI_CONSTANT = OMEGA**2 * EARTH_RADIUS / MEAN_G_AT_EARTH_SURFACE / 15**0.5  # (Unitless).

K_2_IERS = 0.3077 + 0.0036j
H_2_IERS = 0.36207

DATA_DATES_LOWER_BOUND = 14975  # TODO: modify later to include arcs earlier than 1991.
DATA_DATES_UPPER_BOUND = 27500
DATA_DATES_MARGIN = 100

JJUL_1970_REFERENCE_YEAR = 2018
JJUL_1970_REFERENCE_JJUL = 24837


def fractional_year(
    year: int, month: int, day: int, hour_minute_second: tuple[int, int, float] = (0, 0, 0.0)
):
    """
    From floats or ints.
    """

    hour, minute, second = hour_minute_second
    current = datetime(year, month, day, hour, minute, int(second))
    start = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    year_length = (next_year - start).total_seconds()
    elapsed = (current - start).total_seconds()

    return year + elapsed / year_length


def get_c01_pole_motion_time_series(
    models_path: Path = DATA_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
) -> tuple[ndarray[float], ndarray[float], ndarray[float]]:
    """
    Gets dates and long term pole motion x and y as arrays.
    """

    df = read_csv(models_path.joinpath(pole_motion_file), sep=r"\s+", engine="python", header=1)

    return df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 3].values


def correct_from_mean_pole(
    dates: ndarray[float], x: ndarray[float], y: ndarray[float]
) -> tuple[ndarray[float], ndarray[float]]:
    """
    Computes m1 and m2 from x and y pole motion time series and get sure to substract mean pole
    Convention and give m1 and m2 in radians.
    """

    return MILLI_ARC_SECOND_TO_RADIANS * (
        x
        - (
            MEAN_POLE_COEFFICIENTS["m_1"][0]
            + MEAN_POLE_COEFFICIENTS["m_1"][1] * (dates - MEAN_POLE_T_0)
        )
    ), MILLI_ARC_SECOND_TO_RADIANS * (
        y
        - (
            MEAN_POLE_COEFFICIENTS["m_2"][0]
            + MEAN_POLE_COEFFICIENTS["m_2"][1] * (dates - MEAN_POLE_T_0)
        )
    )


def get_m1_m2_time_series(
    models_path: Path = DATA_PATH, pole_motion_file: str = "C01_pole_motion_time_series.txt"
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Gets m_1 and m_2 in radians from the C01 pole motion time series file.
    """

    dates, x, y = get_c01_pole_motion_time_series(
        models_path=models_path, pole_motion_file=pole_motion_file
    )
    m_1, m_2 = correct_from_mean_pole(dates=dates, x=x, y=y)

    return dates, m_1, m_2
