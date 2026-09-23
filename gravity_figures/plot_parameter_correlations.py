"""Full-constellation nondated parameter correlations: tides, solid tides, pole tide."""

from argparse import ArgumentParser
from re import fullmatch

from matplotlib import use
from matplotlib.colors import BoundaryNorm
from matplotlib.pyplot import close, figure, get_cmap
from matplotlib.transforms import ScaledTranslation
from numpy import arange, concatenate, ix_, linspace

from gravity_common import full_solution, load_solution, save_figure

use("Agg")

N_COLORBAR = 7
k_saturated_correlation = 0.95


def make_figure(root):

    cases = ("tides", "solid_tides", "pole_tide")
    harmonics = (
        "C_20",
        "C_40",
        "C_60",
        "C_21",
        "S_21",
        "C_41",
        "S_41",
    )

    allowed = {
        "tides": set(harmonics),
        "solid_tides": {"C_20", "C_40", "C_60"},
        "pole_tide": {"C_21", "S_21", "C_41", "S_41"},
    }

    mode_order = {
        "B": 0,
        "A": 1,
        "AA": 2,
        "C": 3,
        "S": 4,
        "P": 5,
        "Q": 6,
    }

    # Appearance settings.
    parameter_fontsize = 18
    group_fontsize = 22
    subtitle_fontsize = 24
    title_fontsize = 28
    colorbar_fontsize = 24
    colorbar_tick_fontsize = 22

    if N_COLORBAR < 3:
        raise ValueError("N_COLORBAR must be at least 3.")
    if not 0 < k_saturated_correlation < 1:
        raise ValueError("k_saturated_correlation must be between 0 and 1.")

    k = k_saturated_correlation

    boundaries = concatenate(
        (
            [-1.0],
            linspace(-k, k, N_COLORBAR - 1),
            [1.0],
        )
    )

    cmap = get_cmap("RdBu_r").resampled(N_COLORBAR)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    def field_parts(parameter):

        match = fullmatch(
            r"([CP])(AA|A|B|C|S|P|Q)_(\d)(\d)",
            parameter.name,
        )

        if match is None:

            return None

        family, mode, degree, order = match.groups()
        family = "S" if family == "P" else "C"

        return f"{family}_{degree}{order}", mode

    def parameter_order(parameter):

        if parameter.name in ("LQM", "LTM"):

            return (0, ("LQM", "LTM").index(parameter.name), 0)

        parts = field_parts(parameter)

        if parts is None:

            return (1, parameter.name, 0)

        harmonic, mode = parts

        return (2, harmonics.index(harmonic), mode_order[mode])

    def group_name(parameter):

        if parameter.name in ("LQM", "LTM"):

            return r"$(\alpha,\Delta)$"

        parts = field_parts(parameter)

        if parts is None:

            return parameter.name

        family, degree_order = parts[0].split("_")

        return rf"${family}_{{{degree_order}}}$"

    solutions = [
        load_solution(
            full_solution(root, case),
            with_correlation=True,
        )
        for case in cases
    ]

    for case, solution in zip(cases, solutions):

        parameters = solution.correlation_parameters
        keep = [
            index
            for index, parameter in enumerate(parameters)
            if (field_parts(parameter) is None or field_parts(parameter)[0] in allowed[case])
        ]
        keep.sort(key=lambda index: parameter_order(parameters[index]))
        solution.correlation = solution.correlation[ix_(keep, keep)]
        solution.correlation_parameters = [parameters[index] for index in keep]

    sizes = [len(s.correlation_parameters) for s in solutions]
    fig = figure(figsize=(26, 19), layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.65, 1),
        height_ratios=(sizes[1] + 5, sizes[2] + 5),
    )
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    ]

    for ax, solution, subtitle in zip(axes, solutions, ("Tides", "Solid tides", "Pole tide")):

        parameters = solution.correlation_parameters
        plot_names = {
            "LQM": r"$\alpha$",
            "LTM": r"$\Delta$",
        }
        labels = [plot_names.get(parameter.name, parameter.label) for parameter in parameters]
        im = ax.imshow(
            solution.correlation,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )

        # Keep ticks at cell centers, but draw labels manually in
        # alternating tiers. Offsets are in points, independent of
        # panel dimensions.
        positions = arange(len(labels))
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.tick_params(
            axis="both",
            labelbottom=False,
            labelleft=False,
            length=3,
        )

        tier_offsets = (10, 48)

        for index, label in enumerate(labels):

            offset = tier_offsets[index % 2]

            # Bottom labels: two staggered horizontal tiers.
            ax.annotate(
                label,
                xy=(index, 0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, -offset),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=parameter_fontsize,
                annotation_clip=False,
            )

            # Left labels: two staggered vertical tiers.
            ax.annotate(
                label,
                xy=(0, index),
                xycoords=ax.get_yaxis_transform(),
                xytext=(-offset, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=parameter_fontsize,
                annotation_clip=False,
            )

        ax.set_title(
            subtitle,
            fontsize=subtitle_fontsize,
            pad=16,
        )

        # Find contiguous thematic groups.
        groups = []

        for index, parameter in enumerate(parameters):

            name = group_name(parameter)

            if groups and groups[-1][0] == name:

                groups[-1][2] = index

            else:

                groups.append([name, index, index])

        bottom_bar = ax.get_xaxis_transform() + ScaledTranslation(0, -82 / 72, fig.dpi_scale_trans)
        bottom_text = ax.get_xaxis_transform() + ScaledTranslation(0, -91 / 72, fig.dpi_scale_trans)
        left_bar = ax.get_yaxis_transform() + ScaledTranslation(-100 / 72, 0, fig.dpi_scale_trans)
        left_text = ax.get_yaxis_transform() + ScaledTranslation(-112 / 72, 0, fig.dpi_scale_trans)

        for name, first, last in groups:

            midpoint = (first + last) / 2
            lower, upper = first - 0.35, last + 0.35

            ax.plot(
                [lower, upper],
                [0, 0],
                transform=bottom_bar,
                color="0.25",
                lw=1.2,
                clip_on=False,
            )
            ax.text(
                midpoint,
                0,
                name,
                transform=bottom_text,
                ha="center",
                va="top",
                fontsize=group_fontsize,
                clip_on=False,
            )

            ax.plot(
                [0, 0],
                [lower, upper],
                transform=left_bar,
                color="0.25",
                lw=1.2,
                clip_on=False,
            )
            ax.text(
                0,
                midpoint,
                name,
                transform=left_text,
                ha="center",
                va="center",
                rotation=90,
                fontsize=group_fontsize,
                clip_on=False,
            )

        for _, first, _ in groups[1:]:

            ax.axhline(first - 0.5, color="0.35", lw=0.6)
            ax.axvline(first - 0.5, color="0.35", lw=0.6)

    colorbar = fig.colorbar(
        im,
        ax=axes,
        boundaries=boundaries,
        spacing="proportional",
        drawedges=True,
        fraction=0.025,
        pad=0.035,
        shrink=0.85,
        ticks=boundaries,
        format="%.2f",
    )
    colorbar.set_label(
        "Correlation",
        fontsize=colorbar_fontsize,
        labelpad=15,
    )
    colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    fig.suptitle(
        "Reference inversion correlations",
        fontsize=title_fontsize,
    )

    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="solution_full_constellation")
    parser.add_argument("--output", default="figures/parameter_correlations.pdf")
    args = parser.parse_args()
    fig = make_figure(args.root)
    save_figure(fig, args.output)
    close(fig)


if __name__ == "__main__":
    main()
