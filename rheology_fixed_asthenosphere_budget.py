"""
Two-panel uncertainty budget with fixed asthenosphere parameters.
"""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from re import IGNORECASE, compile
from warnings import warn

from base_models import DATA_PATH
from matplotlib.lines import Line2D
from matplotlib.pyplot import close, rc_context, show, subplots
from numpy import array, errstate, power, ptp, std

PARAMETERS = {"LQM": "alpha", "LTM": "Delta"}
SYMBOLS = {
    "alpha": r"\alpha_{\mathrm{non\!\! -\! Asth.}}",
    "Delta": r"\Delta_{\mathrm{non\!\! -\! Asth.}}",
}
# Record key: ((lam level, ldm level), constellation, tide, gravity model, parameter).
SOURCES = {"constellation": 1, "lam_ldm": 0, "formal": None, "tide_mode": 2, "gravity_model": 3}
SOURCE_LABELS = (
    "Constellation",
    "Fixed lam/ldm\nvalues",
    "Formal\nuncertainty",
    "Tide mode",
    "Residual gravity\nfield model",
)
NAME_PATTERN = compile(
    r"^rheology_(?P<constellation>.+?)_(?P<tide>solid_tides|tides)_"
    r"(?P<lam>.+?)_lam_(?P<ldm>.+?)_ldm(?:\.[A-Za-z][A-Za-z0-9]*)?$",
    IGNORECASE,
)
ROW_PATTERN = compile(
    r"^\s*(?:\[\s*(?P<bracketed>LQM|LTM)\s*\]|(?P<bare>LQM|LTM))\s+(?P<columns>.*)$",
    IGNORECASE,
)
INVERSE_MATRIX_PATTERN = compile(r"^\s*INVERSE\s+MATRIX\b", IGNORECASE)


def read_parameters(file_path: Path, n_sigmas=3.0, *, value_column=2, sigma_column=3):
    """
    Return {parameter: (linear solution, transformed formal half-width)}.
    """

    result = {}

    with Path(file_path).open() as stream:

        for _, line in enumerate(stream, 1):

            if INVERSE_MATRIX_PATTERN.match(line):

                break

            match = ROW_PATTERN.search(line)

            if match is None:

                continue

            name = (match["bracketed"] or match["bare"]).upper()
            parameter = PARAMETERS[name]
            tokens = match["columns"].split()
            log_value, sigma = (
                float(tokens[i].replace("D", "E").replace("d", "e"))
                for i in (value_column, sigma_column)
            )

            with errstate(over="ignore", under="ignore", invalid="ignore"):

                lower, value, upper = power(
                    10.0, [log_value - n_sigmas * sigma, log_value, log_value + n_sigmas * sigma]
                )

            result[parameter] = (float(value), float(max(value - lower, upper - value)))

    return result


def load_records(path, n_sigmas=3.0):
    """
    Read DYNAMO solution files like:
        rheology_ajisai_lageos1_lageos2_starlette_solid_tides_high_lam_high_ldm
    """

    path = Path(path)

    if not path.is_dir():

        raise NotADirectoryError(path)

    records, origins = {}, {}
    model_paths = sorted(p for p in path.iterdir() if p.is_dir())

    for listing in (f for model_path in model_paths for f in sorted(model_path.iterdir())):

        if not listing.is_file() or listing.suffix.lower() == ".err":

            continue

        match = NAME_PATTERN.fullmatch(listing.name)
        constellation = tuple(sorted(match["constellation"].lower().split("_")))
        config = (
            (match["lam"].lower(), match["ldm"].lower()),
            constellation,
            match["tide"].lower(),
            listing.parent.name,
        )
        origins[config] = listing

        for parameter, pair in read_parameters(listing, n_sigmas).items():

            records[config + (parameter,)] = pair
    return records


def compute_uncertainty_samples(records, *, metric="range"):
    """
    Vary one source at a time; preserve the controlled configurations.
    """

    reducers = {
        "range": ptp,
        "half_range": lambda x: ptp(x) / 2,
        "std": lambda x: std(x, ddof=0),
    }
    samples, diagnostics = {}, []

    for parameter in PARAMETERS.values():

        rows = {key: pair for key, pair in records.items() if key[-1] == parameter}

        for source, dimension in SOURCES.items():

            bucket = samples[parameter, source] = {}

            if dimension is None:

                for key, (value, formal) in rows.items():

                    bucket[key] = dict(uncertainty=formal, levels=(), solutions=(value,))

                continue

            expected = (
                {"solid_tides", "tides"}
                if source == "tide_mode"
                else {key[dimension] for key in rows}
            )
            groups = defaultdict(dict)

            for key, (value, _) in rows.items():

                controlled = key[:dimension] + key[dimension + 1 :]
                groups[controlled][key[dimension]] = value

            for controlled, level_values in groups.items():

                levels = tuple(sorted(level_values))
                missing = tuple(sorted(expected - set(levels)))

                if len(levels) < 2 or missing:

                    diagnostics.append(
                        dict(
                            parameter=parameter,
                            source=source,
                            configuration=controlled,
                            missing_levels=missing,
                            status="skipped" if len(levels) < 2 else "partial",
                        )
                    )

                if len(levels) < 2:

                    continue

                values = tuple(level_values[level] for level in levels)
                bucket[controlled] = dict(
                    uncertainty=float(reducers[metric](values)), levels=levels, solutions=values
                )
    return samples, diagnostics


def summarize_uncertainties(samples):
    """
    Equal weight per comparison; population standard deviation (ddof=0).
    """

    summary = {}

    for key, bucket in samples.items():

        if not bucket:

            continue

        values = array([sample["uncertainty"] for sample in bucket.values()])
        summary[key] = dict(
            n=len(values),
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std(ddof=0)),
        )

    return summary


def summarize_overall_solutions(records):
    """
    Mean and population std of linear solutions across the entire ensemble.
    Exponentiation happens in read_parameters BEFORE these statistics. Formal
    uncertainties are reported separately and are not folded into ensemble std.
    """

    overall = {}

    for parameter in PARAMETERS.values():

        values = array(
            [value for key, (value, _) in records.items() if key[-1] == parameter], dtype=float
        )
        overall[parameter] = dict(
            n=len(values), mean=float(values.mean()), std=float(values.std(ddof=0))
        )

    return overall


def plot_uncertainty_budget(summary, records, *, metric="range", n_sigmas=3.0):
    """
    Two linear-space panels with ensemble mean/std on their y-axis labels.
    """

    source_order = [
        "gravity_model",
        "constellation",
        "lam_ldm",
        "tide_mode",
        "formal",
    ]
    labels_by_source = dict(zip(SOURCES, SOURCE_LABELS))

    overall = summarize_overall_solutions(records)

    with rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):

        fig, axes = subplots(2, 1, sharex=True, figsize=(11, 7.5))

        for ax, parameter, color in zip(axes, PARAMETERS.values(), ("#176b87", "#94602d")):

            limits = [0.0]

            for x, source in enumerate(source_order):

                stats = summary.get((parameter, source))

                if stats is None:
                    ax.text(
                        x,
                        0.90,
                        "No comparison",
                        ha="center",
                        fontsize=9,
                        transform=ax.get_xaxis_transform(),
                        color="#68717d",
                    )

                    continue

                low, high = stats["mean"] - stats["std"], stats["mean"] + stats["std"]
                ax.vlines(x, stats["min"], stats["max"], color=color, lw=1.5)
                ax.hlines([stats["min"], stats["max"]], x - 0.09, x + 0.09, color=color)
                ax.vlines(x, low, high, color=color, lw=8, alpha=0.4)
                ax.plot(x, stats["mean"], "o", color=color, mec="white", ms=8)
                ax.text(
                    x,
                    0.94,
                    f"n = {stats['n']}",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9,
                    color="#68717d",
                )
                limits.extend([stats["min"], stats["max"], low, high])

            lo, hi = min(limits), max(limits)
            span = hi - lo or 1.0
            ax.set_ylim(lo - 0.06 * span, hi + 0.24 * span)
            ax.set_xlim(-0.5, len(SOURCES) - 0.5)
            ensemble = overall[parameter]
            ax.set_ylabel(
                rf"${SYMBOLS[parameter]}={ensemble['mean']:.3g}\pm{3*ensemble['std']:.3g}$"
                "\n"
                r"Variability sources $\rightarrow$",
                rotation=0,
                ha="right",
                va="center",
            )
            ax.axhline(0, color="#aab2bb", lw=0.8)
            ax.grid(axis="y", color="#e5e9ed")
            ax.set_axisbelow(True)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

        axes[-1].set_xticks(
            range(len(source_order)),
            [labels_by_source[source] for source in source_order],
        )
        handles = [
            Line2D([], [], color="#465361", lw=1.5, marker="_", label="Min–max"),
            Line2D([], [], color="#465361", lw=8, alpha=0.4, label="Mean ± std"),
            Line2D([], [], color="#465361", lw=0, marker="o", label="Mean"),
        ]
        fig.suptitle("Non-asthenospheric rheology variability synthesis", y=0.98)
        fig.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.8, 0.94), ncol=3, frameon=False
        )
        fig.tight_layout()

    return fig, axes


def main():

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("rheology_fixed_asthenosphere_budget.pdf")
    )
    parser.add_argument("--metric", choices=("range", "half_range", "std"), default="range")
    parser.add_argument("--n-sigmas", type=float, default=3.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.data_path is None:

        args.data_path = DATA_PATH.joinpath("dynamo/rheology/solution_fix_asthenosphere")

    records = load_records(args.data_path, args.n_sigmas)
    samples, diagnostics = compute_uncertainty_samples(records, metric=args.metric)
    summary = summarize_uncertainties(samples)

    if diagnostics:

        skipped = sum(row["status"] == "skipped" for row in diagnostics)
        warn(
            f"{skipped} singleton comparisons skipped; "
            f"{len(diagnostics) - skipped} partial comparisons retained. "
            "Call compute_uncertainty_samples() to inspect diagnostics."
        )

    fig, _ = plot_uncertainty_budget(summary, records, metric=args.metric, n_sigmas=args.n_sigmas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")

    if not args.no_show:

        show()

    close(fig)


if __name__ == "__main__":

    main()
