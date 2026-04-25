"""
Figures for validation purposes.
"""

from pathlib import Path
from typing import Optional

from alna import save_figure
from base_models import lagrange_order4, load_base_model
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from numpy import array, ndarray, zeros

from .tide_correction_model import (
    POLE_MODELS_PATH,
    POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    dates_to_jjul_dates,
)
from .utils import get_m1_m2_time_series

GINS_ARC_MONITORING_SHORTCUT_PLOTTER = 100
GINS_ARC_MONITORING_START_JJUL = 25080
GINS_ARC_MONITORING_END_JJUL = 25110
GINS_ARC_MONITORING_JJUL_MARGIN = 30
DEFAULT_MODEL_VALUES_TO_PLOT = [(0.2, 5), (0.2, 10), (0.25, 5), (0.25, 10)]


def get_gins_pole_motion_time_series(
    path: Path = Path("."),
    file: str = "gins_listing",
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
]:
    """
    Gets the GINS pole motion and mean pole on a monitored arc.
    """

    dates, u, v, u0, v0 = [], [], [], [], []

    with open(path.joinpath(file), "r", encoding="utf-8") as f:

        lines = f.readlines()
        i = 0

        while i < len(lines):

            parts = lines[i].strip().split()

            if parts and parts[0] == "pole_motion_monitor":

                dates.append(float(parts[1]))
                u.append(float(parts[2]))
                u0.append(float(parts[3]))
                next_parts = lines[i + 1].strip().split()
                v.append(float(next_parts[0]))
                v0.append(float(next_parts[1]))
                i += 2  # Move to the line after the next

            else:

                i += 1

    return (
        array(object=dates),
        array(object=u) - array(object=u0),
        array(object=v) - array(object=v0),
    )


def get_gins_pole_tide(
    path: Path = Path("."),
    file: str = "gins_listing",
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Gets the GINS pole tide on a monitored arc.
    """

    dates, c, s = [], [], []

    with open(path.joinpath(file), "r", encoding="utf-8") as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            parts = line.strip().split()

            if parts and parts[0] == "pole_tide_monitor" and i >= 2:

                date_parts = lines[i - 2].strip().split()
                dates.append(float(date_parts[1]))
                c.append(float(parts[1]))
                s.append(float(parts[2]))

    return array(object=dates), array(object=c), array(object=s)


def plot_pole_motion(
    path: Path = Path("."),
    file: str = "gins_listing",
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = "C01_pole_motion_time_series.txt",
) -> None:
    """
    Compares the pole motiuon model with the GINS pole motion on a monitored arc.
    """

    gins_dates, u, v = get_gins_pole_motion_time_series(path=path, file=file)
    dates, m_1, m_2 = get_m1_m2_time_series(
        models_path=models_path, pole_motion_file=pole_motion_file
    )
    jjul_dates = dates_to_jjul_dates(dates=dates)
    mask = (GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= jjul_dates) * (
        jjul_dates >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN
    )
    jjul_dates = jjul_dates[mask]
    m_1 = m_1[mask]
    m_2 = m_2[mask]
    mask = (GINS_ARC_MONITORING_END_JJUL >= gins_dates) * (
        gins_dates >= GINS_ARC_MONITORING_START_JJUL
    )
    gins_dates = gins_dates[mask]
    u = u[mask]
    v = v[mask]
    axes: list[Axes]
    figure, axes = subplots(2, 1, figsize=(8, 8))
    target_dates = gins_dates[::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]
    axes[0].plot(
        target_dates,
        lagrange_order4(x=jjul_dates, y=m_1, new_x=target_dates),
        label=r"C01",
    )
    axes[0].plot(gins_dates, u, label="Including sub-diurnal")
    axes[1].plot(
        target_dates,
        lagrange_order4(x=jjul_dates, y=m_2, new_x=target_dates),
    )
    axes[1].plot(gins_dates, v)
    axes[0].set_ylabel(ylabel=r"$m_1$")
    axes[1].set_ylabel(ylabel=r"$m_2$")
    axes[1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0].legend()
    save_figure(figure=figure, figure_title="pole_motion")


def plot_pole_tide_models(
    path: Path = Path("."),
    file: str = "gins_listing",
    models_path: Path = POLE_MODELS_PATH,
    pole_motion_file: str = POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    model_values_to_plot: Optional[list[tuple[float, float]]] = None,
) -> None:
    """
    Compares the pole motiuon model with the GINS pole motion on a monitored arc.
    """

    if model_values_to_plot is None:

        model_values_to_plot = DEFAULT_MODEL_VALUES_TO_PLOT

    gins_dates, gins_c, gins_s = get_gins_pole_tide(path=path, file=file)
    mask = (GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= gins_dates) * (
        gins_dates >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN
    )
    gins_dates = gins_dates[mask][::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]
    gins_c = gins_c[mask][::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]
    gins_s = gins_s[mask][::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]
    pole_tide_correction_models = load_base_model(name=pole_motion_file, path=models_path)
    jjul_dates = array(object=load_base_model(name="jjul_dates", path=models_path), dtype=float)
    alpha_values = array(object=load_base_model(name="alpha_values", path=models_path), dtype=float)
    delta_values = array(object=load_base_model(name="delta_values", path=models_path), dtype=float)
    model_mask = array(object=load_base_model(name="model_mask", path=models_path), dtype=bool)
    mask = (GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= jjul_dates) * (
        jjul_dates >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN
    )
    jjul_dates = jjul_dates[mask]
    axes: list[Axes]
    figure, axes = subplots(2, 1, figsize=(8, 8))
    axes[0].plot(gins_dates, gins_c, label="GINS")
    axes[1].plot(gins_dates, gins_s)

    for component, gins_component, ax in zip("CS", [gins_c, gins_s], axes):

        sub_diurnal_correction = gins_component - lagrange_order4(
            x=jjul_dates,
            y=array(
                object=pole_tide_correction_models[component]["potential"]["elastic_love_numbers"],
                dtype=float,
            )[model_mask][mask],
            new_x=gins_dates,
        )

        for alpha, delta in model_values_to_plot:

            pole_tide_correction_array = zeros(shape=(len(delta_values), len(jjul_dates)))

            for i_delta, _ in enumerate(delta_values):

                for i_date, _ in enumerate(jjul_dates):

                    pole_tide_correction_array[i_delta, i_date] = lagrange_order4(
                        x=alpha_values,
                        y=array(
                            object=pole_tide_correction_models[component]["potential"][
                                "love_numbers"
                            ],
                            dtype=float,
                        )[:, i_delta, model_mask][:, mask][:, i_date],
                        new_x=[alpha],
                    )[0]

            pole_tide_correction_tab = zeros(shape=len(jjul_dates))

            for i_date, _ in enumerate(jjul_dates):

                pole_tide_correction_tab[i_date] = lagrange_order4(
                    x=delta_values,
                    y=pole_tide_correction_array[:, i_date],
                    new_x=[delta],
                )[0]

            ax.plot(
                gins_dates,
                lagrange_order4(x=jjul_dates, y=pole_tide_correction_tab, new_x=gins_dates)
                + sub_diurnal_correction,
                label=rf"$\alpha={alpha}$  $\Delta={delta}$",
            )

    axes[0].set_ylabel(ylabel=r"$C_{21}$")
    axes[1].set_ylabel(ylabel=r"$S_{21}$")
    axes[1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0].legend()
    save_figure(figure=figure, figure_title="pole_tide_models")
