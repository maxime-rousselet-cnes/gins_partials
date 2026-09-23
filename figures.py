"""
Figures for validation purposes.
"""

from pathlib import Path
from typing import Iterable, Optional

from alna import save_figure
from alna.tide_correction_model import (
    POLE_MODELS_PATH,
    POLE_TIDE_CORRECTION_MODELS_DEFAULT_FILE_NAME,
    dates_to_jjul_dates,
    get_m1_m2_time_series,
    read_for_partials,
)
from base_models import lagrange_order4, load_base_model
from matplotlib.axes import Axes
from matplotlib.pyplot import show, subplots, tight_layout
from numpy import array, log10, ndarray, zeros
from numpy.testing import assert_allclose

GINS_ARC_MONITORING_SHORTCUT_PLOTTER = 100
GINS_ARC_MONITORING_START_JJUL = 25080
GINS_ARC_MONITORING_END_JJUL = 25110
GINS_ARC_MONITORING_JJUL_MARGIN = 30
DEFAULT_MODEL_VALUES_TO_PLOT = [
    (log10(0.17), -1.17, 1.17, 0.25),
    (log10(0.16), -1.17, 1.17, 0.25),
    (log10(0.17), -1.27, 1.17, 0.25),
    (log10(0.17), -1.17, log10(3), 0.25),
    (log10(0.17), -1.17, 1.17, 0.5),
]


REFERENCE_PARAMETER_VALUES = {
    parameter: value
    for parameter, value in zip(DEFAULT_MODEL_VALUES_TO_PLOT[0], ["lam", "lqm", "ldm", "ltm"])
}

JUMPER = 2
JJUL_MAX_FIGURE = 24970.5


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
    tide_models_path: Path = POLE_MODELS_PATH,
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
    axes: list[list[Axes]]
    figure, axes = subplots(2, 2, figsize=(16, 8), sharex=True)
    axes[0][0].scatter(gins_model["dates"], gins_model["C"], label="GINS", s=2)
    axes[1][0].scatter(gins_model["dates"], gins_model["S"], s=2)

    for component, ax_line in zip("CS", axes):

        sub_diurnal_correction = gins_model[component] - lagrange_order4(
            x=jjul_dates,
            y=array(
                object=pole_tide_correction_models[component + "_elastic"],
                dtype=float,
            )[mask],
            new_x=gins_model["dates"],
        )
        """
        ax.scatter(
            gins_model["dates"],
            lagrange_order4(
                x=jjul_dates,
                y=array(object=pole_tide_correction_models[component + "_IERS"], dtype=float)[mask],
                new_x=gins_model["dates"],
            )
            + sub_diurnal_correction,
            label="IERS",
        )
        """

        for (lam, lqm, ldm, ltm), color in zip(
            model_values_to_plot, ["orange", "green", "red", "purple", "pink"]
        ):

            values = lagrange_order4(
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
                        object=pole_tide_correction_models[
                            (
                                component.replace("__", "_")
                                if component in pole_tide_correction_models
                                else component + "_"
                            )
                        ],
                        dtype=float,
                    )[:, :, :, :, mask],
                ),
                new_x=gins_model["dates"],
            )
            ax_line[0].scatter(
                gins_model["dates"],
                values + sub_diurnal_correction,
                label=rf"$\alpha_{{Asth.}}={round(10**lam, 2)}$  $\alpha_{{non-Asth.}}={round(10**lqm, 2)}$  $\Delta_{{Asth.}}={round(10**ldm, 2)}$  $\Delta_{{non-Asth.}}={round(10**(ltm), 2)}$s",
                s=2,
                color=color,
            )
            error = values + sub_diurnal_correction - gins_model[component]
            ax_line[1].scatter(
                gins_model["dates"],
                error - error[0],
                s=2,
                color=color,
            )

    axes[0][0].set_title("Absolute correction values")
    axes[0][1].set_title("Differences to standard")
    axes[0][0].set_ylabel(ylabel=r"$C_{21}$")
    axes[1][0].set_ylabel(ylabel=r"$S_{21}$")
    axes[1][0].set_xlabel(xlabel=r"$J_{julian}$")
    axes[1][1].set_xlabel(xlabel=r"$J_{julian}$")
    axes[0][0].legend()
    save_figure(figure=figure, figure_title="pole_tide_models")


def compare_acceleration_partials_to_finite_differences(
    d_parameter: float = 0.1,
    satellite: str = "ajisai",
) -> None:
    """
    Validate formal partials against symmetric differences with step d_parameter.
    """

    epochs, _, lam_formal, lqm_formal, ldm_formal, ltm_formal = read_for_partials(
        filename=f"rheology_{satellite}_checkup.yml"
    )
    mask = epochs <= JJUL_MAX_FIGURE
    finite_differences = {}

    for parameter in ("lam", "lqm", "ldm", "ltm"):

        epochs_plus, acceleration_plus, _, _, _, _ = read_for_partials(
            filename=f"rheology_{satellite}_checkup_{parameter}_plus_" + str(d_parameter),
        )
        epochs_minus, acceleration_minus, _, _, _, _ = read_for_partials(
            filename=f"rheology_{satellite}_checkup_{parameter}_minus_" + str(d_parameter),
        )
        assert_allclose(epochs_plus, epochs, rtol=0.0, atol=1e-12)
        assert_allclose(epochs_minus, epochs, rtol=0.0, atol=1e-12)
        finite_differences[parameter] = (acceleration_plus[mask] - acceleration_minus[mask]) / (
            2 * d_parameter
        )

    lam_formal = lam_formal[mask]
    lqm_formal = lqm_formal[mask]
    ldm_formal = ldm_formal[mask]
    ltm_formal = ltm_formal[mask]
    epochs = epochs[mask]
    lam_finite_difference = finite_differences["lam"]
    lqm_finite_difference = finite_differences["lqm"]
    ldm_finite_difference = finite_differences["ldm"]
    ltm_finite_difference = finite_differences["ltm"]

    axes: Iterable[Iterable[Axes]]
    figure, axes = subplots(3, 4, figsize=(18, 12), sharex=True)

    for (i, ax_line), component in zip(enumerate(axes), ["X", "Y", "Z"]):

        ax: Axes

        for ax, parameter in zip(
            ax_line,
            [
                r"\log_{10}(\alpha_{Asth.})",
                r"\log_{10}(\alpha_{non-Asth.})",
                r"\log_{10}(\Delta_{Asth.})",
                r"\log_{10}(\Delta_{non-Asth.})",
            ],
        ):

            if "alpha_{A" in parameter:

                ax.scatter(
                    epochs[::JUMPER],
                    lam_formal[::JUMPER, i],
                    c="b",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs[::JUMPER],
                    lam_finite_difference[::JUMPER, i],
                    c="b",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            elif "alpha_{n" in parameter:

                ax.scatter(
                    epochs[::JUMPER],
                    lqm_formal[::JUMPER, i],
                    c="orange",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs[::JUMPER],
                    lqm_finite_difference[::JUMPER, i],
                    c="orange",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            elif "Delta_{A" in parameter:

                ax.scatter(
                    epochs[::JUMPER],
                    ldm_formal[::JUMPER, i],
                    c="green",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs[::JUMPER],
                    ldm_finite_difference[::JUMPER, i],
                    c="green",
                    marker="o",
                    label="finite differences" if i == 0 else None,
                )

            else:

                ax.scatter(
                    epochs[::JUMPER],
                    ltm_formal[::JUMPER, i],
                    c="red",
                    marker="x",
                    label="formal" if i == 0 else None,
                )
                ax.scatter(
                    epochs[::JUMPER],
                    ltm_finite_difference[::JUMPER, i],
                    c="red",
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

    ax: Axes = axes[0][0]
    ax.set_xlim(24970, JJUL_MAX_FIGURE)
    tight_layout()
    save_figure(figure=figure, figure_title="acceleration_partials_" + satellite)
    show()
