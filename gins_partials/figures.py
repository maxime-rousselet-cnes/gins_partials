"""
Figures for validation purposes.
"""

from pathlib import Path
from typing import Iterable, Optional

from alna import save_figure
from base_models import lagrange_order4, load_base_model
from matplotlib.axes import Axes
from matplotlib.pyplot import show, subplots, tight_layout
from numpy import array, ndarray, zeros

from .listing_getters import read_for_partials
from .tide_correction_model import (
    POLE_MODELS_PATH,
    POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    TIDE_MODELS_PATH,
    dates_to_jjul_dates,
)
from .utils import get_m1_m2_time_series

GINS_ARC_MONITORING_SHORTCUT_PLOTTER = 100
GINS_ARC_MONITORING_START_JJUL = 25080
GINS_ARC_MONITORING_END_JJUL = 25110
GINS_ARC_MONITORING_JJUL_MARGIN = 30
DEFAULT_MODEL_VALUES_TO_PLOT = [(0.2, 0, 3.51), (0.2, -1, 3.51), (0.25, 0, 3.51), (0.25, -1, 3.51)]


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
    axes[0].scatter(
        target_dates, lagrange_order4(x=jjul_dates, y=m_1, new_x=target_dates), label=r"C01", s=2
    )
    axes[0].scatter(gins_dates, u, label="Including high frequencies", s=2)
    axes[1].scatter(target_dates, lagrange_order4(x=jjul_dates, y=m_2, new_x=target_dates), s=2)
    axes[1].scatter(gins_dates, v, s=2)
    axes[0].set_ylabel(ylabel=r"$m_1$")
    axes[1].set_ylabel(ylabel=r"$m_2$")
    axes[1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0].legend()
    save_figure(figure=figure, figure_title="pole_motion")


def interpolate_by_axis(
    alpha_delta_tau_m_values: tuple[ndarray, ndarray],
    jjul_dates: ndarray,
    alpha_delta_tau_m: float,
    pole_tide_correction_model: ndarray,
) -> ndarray:
    """
    Interpolates axis by axis for plot purposes.
    """

    alpha_values, log10_delta_values, log10_tau_m_values = alpha_delta_tau_m_values
    alpha, log10_delta, log10_tau_m = alpha_delta_tau_m
    pole_tide_correction_array = zeros(
        shape=(len(log10_delta_values), len(log10_tau_m_values), len(jjul_dates))
    )

    for i_delta, _ in enumerate(log10_delta_values):

        for i_tau_m, _ in enumerate(log10_tau_m_values):

            for i_date, _ in enumerate(jjul_dates):

                pole_tide_correction_array[i_delta, i_tau_m, i_date] = lagrange_order4(
                    x=alpha_values,
                    y=pole_tide_correction_model[:, i_delta, i_tau_m, i_date],
                    new_x=[alpha],
                )[0]

    pole_tide_correction_tab = zeros(shape=(len(log10_tau_m_values), len(jjul_dates)))

    for i_tau_m, _ in enumerate(log10_tau_m_values):

        for i_date, _ in enumerate(jjul_dates):

            pole_tide_correction_tab[i_tau_m, i_date] = lagrange_order4(
                x=log10_delta_values,
                y=pole_tide_correction_array[:, i_tau_m, i_date],
                new_x=[log10_delta],
            )[0]

    pole_tide_correction = zeros(shape=len(jjul_dates))

    for i_date, _ in enumerate(jjul_dates):

        pole_tide_correction[i_date] = lagrange_order4(
            x=log10_tau_m_values,
            y=pole_tide_correction_tab[:, i_date],
            new_x=[log10_tau_m],
        )[0]

    return pole_tide_correction


def plot_pole_tide_models(
    path: Path = Path("."),
    file: str = "gins_listing",
    tide_models_path: Path = TIDE_MODELS_PATH,
    pole_tide_file: str = POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    model_values_to_plot: Optional[list[tuple[float, float]]] = None,
) -> None:
    """
    Compares the pole motiuon model with the GINS pole motion on a monitored arc.
    """

    if model_values_to_plot is None:

        model_values_to_plot = DEFAULT_MODEL_VALUES_TO_PLOT

    gins_model: dict[str, ndarray] = {}
    gins_model["dates"], gins_model["C"], gins_model["S"] = get_gins_pole_tide(path=path, file=file)
    mask = (
        GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= gins_model["dates"]
    ) * (
        gins_model["dates"]
        >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN - 500000  # TODO.
    )

    for component in ["C", "S", "dates"]:

        gins_model[component] = gins_model[component][mask][::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]

    pole_tide_correction_models = load_base_model(name=pole_tide_file, path=tide_models_path)
    jjul_dates = array(
        object=load_base_model(name="jjul_dates", path=tide_models_path), dtype=float
    )
    alpha_values = array(
        object=load_base_model(name="alpha_values", path=tide_models_path), dtype=float
    )
    log10_delta_values = array(
        object=load_base_model(name="log10_delta_values", path=tide_models_path), dtype=float
    )
    log10_tau_m_values = array(
        object=load_base_model(name="log10_tau_m_values", path=tide_models_path), dtype=float
    )
    mask = (GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= jjul_dates) * (
        jjul_dates
        >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN - 500000  # TODO.
    )
    jjul_dates = jjul_dates[mask]
    axes: list[Axes]
    figure, axes = subplots(2, 1, figsize=(8, 8))
    axes[0].scatter(gins_model["dates"], gins_model["C"], label="GINS", s=2)
    axes[1].scatter(gins_model["dates"], gins_model["S"], s=2)

    for component, ax in zip("CS", axes):

        sub_diurnal_correction = gins_model[component] - lagrange_order4(
            x=jjul_dates,
            y=array(
                object=pole_tide_correction_models[component]["elastic"],
                dtype=float,
            )[mask],
            new_x=gins_model["dates"],
        )

        for alpha, log10_delta, log10_tau_m in model_values_to_plot:

            """
            TODO:
            ax.scatter(
                gins_model["dates"],
                lagrange_order4(
                    x=jjul_dates,
                    y=interpolate_by_axis(
                        alpha_delta_tau_m_values=(
                            alpha_values,
                            log10_delta_values,
                            log10_tau_m_values,
                        ),
                        jjul_dates=jjul_dates,
                        alpha_delta_tau_m=(alpha, log10_delta, log10_tau_m),
                        pole_tide_correction_model=array(
                            object=pole_tide_correction_models[component]["anelastic"],
                            dtype=float,
                        )[:, :, :, mask],
                    ),
                    new_x= gins_model["dates"],
                )+sub_diurnal_correction,
                label=rf"$\alpha={alpha}$  $\Delta={10**log10_delta}$",
                s=2,
            )
            REMOVE:
            """
            ax.plot(
                jjul_dates[::10],
                interpolate_by_axis(
                    alpha_delta_tau_m_values=(
                        alpha_values,
                        log10_delta_values,
                        log10_tau_m_values,
                    ),
                    jjul_dates=jjul_dates[::10],
                    alpha_delta_tau_m=(alpha, log10_delta, log10_tau_m),
                    pole_tide_correction_model=array(
                        object=pole_tide_correction_models[component]["anelastic"],
                        dtype=float,
                    )[:, :, :, mask][:, :, :, ::10],
                ),
                label=rf"$\alpha={alpha}$  $\Delta={10**log10_delta}$",
            )

    axes[0].set_ylabel(ylabel=r"$C_{21}$")
    axes[1].set_ylabel(ylabel=r"$S_{21}$")
    axes[1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0].legend()
    show()
    save_figure(figure=figure, figure_title="pole_tide_models")


def compare_acceleration_partials_to_finite_differences(
    d_parameter: float = 0.01,
    satellite: str = "ajisai",
) -> None:

    epochs, acceleration, alpha_formal, log10_delta_formal, log10_tau_m_formal = read_for_partials(
        filename=f"rheology_{satellite}_checkup.yml"
    )
    _, acceleration_alpha_plus_d_alpha, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_alpha_plus_" + str(d_parameter) + ".yml"
    )
    _, acceleration_log10_delta_plus_d_log10_delta, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_log10_delta_plus_" + str(d_parameter) + ".yml"
    )
    _, acceleration_tau_m_plus_d_log10_tau_m, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_log10_tau_m_plus_" + str(d_parameter) + ".yml"
    )
    alpha_finite_difference = (acceleration_alpha_plus_d_alpha - acceleration) / d_parameter
    log10_delta_finite_difference = (
        acceleration_log10_delta_plus_d_log10_delta - acceleration
    ) / d_parameter
    log10_tau_m_finite_difference = (
        acceleration_tau_m_plus_d_log10_tau_m - acceleration
    ) / d_parameter

    axes: Iterable[Iterable[Axes]]
    fig, axes = subplots(3, 3, figsize=(12, 10), sharex=True)

    for (i, ax_line), component in zip(enumerate(axes), ["X", "Y", "Z"]):

        for ax, parameter in zip(ax_line, ["alpha", "\log_10(\delta)", "\log_10(\tau_m)"]):

            if "alpha" in parameter:

                ax.scatter(
                    epochs,
                    alpha_formal[:, i],
                    color="b",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    alpha_finite_difference[:, i],
                    color="b",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            elif "elta" in parameter:

                ax.scatter(
                    epochs,
                    log10_delta_formal[:, i],
                    color="r",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    log10_delta_finite_difference[:, i],
                    color="r",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            else:

                ax.scatter(
                    epochs,
                    log10_tau_m_formal[:, i],
                    color="r",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    log10_tau_m_finite_difference[:, i],
                    color="r",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            if i == 0:

                ax.legend(ncol=2)
                ax.set_title(r"$\frac{\partial a}{\partial " + parameter + r"}$")

        ax.set_ylabel(f"{component}")
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("JJul")

    fig.suptitle("Finite difference comparison to formal partials")

    tight_layout()
    show()
