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
DEFAULT_MODEL_VALUES_TO_PLOT = [
    (0.1, 400, -0.5, 4),
    (0.25, 400, -0.5, 3.51),
    (0.2, 400, -0.5, 3.51),
    (0.25, 300, -0.5, 3.51),
    (0.25, 400, -0.1, 3.51),
    (0.25, 400, -0.5, 4),
]


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
    lam_values: ndarray,
    lqm_values: ndarray,
    ldm_values: ndarray,
    ltm_values: ndarray,
    lam: float,
    lqm: float,
    ldm: float,
    ltm: float,
    jjul_dates: ndarray,
    pole_tide_correction_model: ndarray,
) -> ndarray:
    """
    Interpolates axis by axis for plot purposes.
    """

    pole_tide_correction_volume = zeros(
        shape=(len(lqm_values), len(ldm_values), len(ltm_values), len(jjul_dates))
    )

    for i_lqm, _ in enumerate(lqm_values):

        for i_ldm, _ in enumerate(ldm_values):

            for i_ltm, _ in enumerate(ltm_values):

                for i_date, _ in enumerate(jjul_dates):

                    pole_tide_correction_volume[i_lqm, i_ldm, i_ltm, i_date] = lagrange_order4(
                        x=lam_values,
                        y=pole_tide_correction_model[:, i_lqm, i_ldm, i_ltm, i_date],
                        new_x=[lam],
                    )[0]

    pole_tide_correction_array = zeros(shape=(len(ldm_values), len(ltm_values), len(jjul_dates)))

    for i_ldm, _ in enumerate(ldm_values):

        for i_ltm, _ in enumerate(ltm_values):

            for i_date, _ in enumerate(jjul_dates):

                pole_tide_correction_array[i_ldm, i_ltm, i_date] = lagrange_order4(
                    x=lqm_values,
                    y=pole_tide_correction_volume[:, i_ldm, i_ltm, i_date],
                    new_x=[lqm],
                )[0]

    pole_tide_correction_tab = zeros(shape=(len(ltm_values), len(jjul_dates)))

    for i_ltm, _ in enumerate(ltm_values):

        for i_date, _ in enumerate(jjul_dates):

            pole_tide_correction_tab[i_ltm, i_date] = lagrange_order4(
                x=ldm_values,
                y=pole_tide_correction_array[:, i_ltm, i_date],
                new_x=[ldm],
            )[0]

    pole_tide_correction = zeros(shape=len(jjul_dates))

    for i_date, _ in enumerate(jjul_dates):

        pole_tide_correction[i_date] = lagrange_order4(
            x=ltm_values,
            y=pole_tide_correction_tab[:, i_date],
            new_x=[ltm],
        )[0]

    return pole_tide_correction


def plot_pole_tide_models(
    path: Path = Path("."),
    file: str = "gins_listing",
    tide_models_path: Path = TIDE_MODELS_PATH,
    pole_tide_file: str = POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    model_values_to_plot: Optional[list[tuple[float, float, float, float]]] = None,
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
    ) * (gins_model["dates"] >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN)

    for component in ["C", "S", "dates"]:

        gins_model[component] = gins_model[component][mask][::GINS_ARC_MONITORING_SHORTCUT_PLOTTER]

    pole_tide_correction_models = load_base_model(name=pole_tide_file, path=tide_models_path)
    jjul_dates = array(
        object=load_base_model(name="jjul_dates", path=tide_models_path), dtype=float
    )
    lam_values = array(
        object=load_base_model(name="lam_values", path=tide_models_path), dtype=float
    )
    lqm_values = array(
        object=load_base_model(name="lqm_values", path=tide_models_path), dtype=float
    )
    ldm_values = array(
        object=load_base_model(name="ldm_values", path=tide_models_path), dtype=float
    )
    ltm_values = array(
        object=load_base_model(name="ltm_values", path=tide_models_path), dtype=float
    )
    mask = (GINS_ARC_MONITORING_END_JJUL + GINS_ARC_MONITORING_JJUL_MARGIN >= jjul_dates) * (
        jjul_dates >= GINS_ARC_MONITORING_START_JJUL - GINS_ARC_MONITORING_JJUL_MARGIN
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
        ax.scatter(
            gins_model["dates"],
            lagrange_order4(
                x=jjul_dates,
                y=array(object=pole_tide_correction_models[component]["IERS"], dtype=float)[mask],
                new_x=gins_model["dates"],
            )
            + sub_diurnal_correction,
            label="IERS",
        )

        for lam, lqm, ldm, ltm in model_values_to_plot:

            ax.scatter(
                gins_model["dates"],
                lagrange_order4(
                    x=jjul_dates,
                    y=interpolate_by_axis(
                        lam_values=lam_values,
                        lqm_values=lqm_values,
                        ldm_values=ldm_values,
                        ltm_values=ltm_values,
                        lam=lam,
                        lqm=lqm,
                        ldm=ldm,
                        ltm=ltm,
                        jjul_dates=jjul_dates,
                        pole_tide_correction_model=array(
                            object=pole_tide_correction_models[component]["anelastic"],
                            dtype=float,
                        )[:, :, :, :, mask],
                    ),
                    new_x=gins_model["dates"],
                )
                + sub_diurnal_correction,
                label=rf"$\alpha={round(lam, 2)}$  $Q={round(lqm)}$  $\Delta={round(10**ldm, 2)}$  $\tau_m={round(10**(ltm))}$s",
                s=2,
            )

    axes[0].set_ylabel(ylabel=r"$C_{21}$")
    axes[1].set_ylabel(ylabel=r"$S_{21}$")
    axes[1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0].legend()
    save_figure(figure=figure, figure_title="pole_tide_models")


REFERENCE_PARAMETER_VALUES = {"lam": 0.1, "lqm": 400, "ldm": -0.5, "ltm": 3.5}


def compare_acceleration_partials_to_finite_differences(
    d_parameter: float = 0.01,
    satellite: str = "ajisai",
) -> None:
    """
    Partial derivatives validation figure for a single arc at a single parameter value.
    """

    epochs, acceleration, lam_formal, lqm_formal, ldm_formal, ltm_formal = read_for_partials(
        filename=f"rheology_{satellite}_checkup.yml"
    )
    _, acceleration_lam_plus_d_lam, _, _, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_lam_plus_" + str(d_parameter),
        parameter_index=1,
        parameter_value=REFERENCE_PARAMETER_VALUES["lam"] + d_parameter,
    )
    _, acceleration_lqm_plus_d_lqm, _, _, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_lqm_plus_" + str(100 * d_parameter),
        parameter_index=2,
        parameter_value=REFERENCE_PARAMETER_VALUES["lqm"] + 100 * d_parameter,
    )
    _, acceleration_ldm_plus_d_ldm, _, _, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_ldm_plus_" + str(d_parameter),
        parameter_index=3,
        parameter_value=REFERENCE_PARAMETER_VALUES["lam"] + d_parameter,
    )
    _, acceleration_ltm_plus_d_ltm, _, _, _, _ = read_for_partials(
        filename=f"rheology_{satellite}_checkup_ltm_plus_" + str(d_parameter),
        parameter_index=4,
        parameter_value=REFERENCE_PARAMETER_VALUES["ltm"] + d_parameter,
    )
    lam_finite_difference = (acceleration_lam_plus_d_lam - acceleration) / d_parameter
    lqm_finite_difference = (acceleration_lqm_plus_d_lqm - acceleration) / (100 * d_parameter)
    ldm_finite_difference = (acceleration_ldm_plus_d_ldm - acceleration) / d_parameter
    ltm_finite_difference = (acceleration_ltm_plus_d_ltm - acceleration) / d_parameter

    axes: Iterable[Iterable[Axes]]
    figure, axes = subplots(3, 3, figsize=(12, 10), sharex=True)

    for (i, ax_line), component in zip(enumerate(axes), ["X", "Y", "Z"]):

        ax: Axes

        for ax, parameter in zip(
            ax_line, [r"\alpha", r"Q_\mu", r"\log_{10}(\Delta)", r"\log_{10}(\tau_m)"]
        ):

            if "alpha" in parameter:

                ax.scatter(
                    epochs,
                    lam_formal[:, i],
                    c="b",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    lam_finite_difference[:, i],
                    c="b",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            elif "Q" in parameter:

                ax.scatter(
                    epochs,
                    lqm_formal[:, i],
                    c="orange",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    lqm_finite_difference[:, i],
                    c="orange",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )
            elif "elta" in parameter:

                ax.scatter(
                    epochs,
                    ldm_formal[:, i],
                    c="orange",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    ldm_finite_difference[:, i],
                    c="orange",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            else:

                ax.scatter(
                    epochs,
                    ltm_formal[:, i],
                    c="g",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs,
                    ltm_finite_difference[:, i],
                    c="g",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            if i == 0:

                ax.legend(ncol=2)
                ax.set_title(r"$\frac{\partial a}{\partial " + parameter + r"}$")

            ax.set_ylabel(f"{component}")
            ax.grid(True, alpha=0.3)

            if component == "Z":

                ax.set_xlabel("JJul")

    figure.suptitle("Finite difference comparison to formal partials " + str(satellite))
    ax: Axes = axes[0][0]
    ax.set_xlim(24970, 24970.5)
    tight_layout()
    save_figure(figure=figure, figure_title="Acceleration_partials")
    show()
