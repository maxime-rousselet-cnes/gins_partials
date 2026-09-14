"""
Functions for main rheological solution uncertainty budget figure.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from re import search

from alna import SECONDS_PER_YEAR
from alna.rheological_formulas import find_tau_m_sup
from matplotlib.lines import Line2D
from matplotlib.pyplot import rc_context, subplots, tight_layout
from numpy import asarray, isfinite, mean, ptp, sqrt, std

SolutionKey = tuple[str, str, str, str, str]
# Context, fixed parameter, gravity model, constellation, tide mode, parameter.
RecordKey = tuple[str, str, str, str, str, str]
Records = dict[RecordKey, tuple[float, float]]
Samples = dict[tuple[str, str], dict[tuple[str, ...], dict]]
Summary = dict[tuple[str, str], dict[str, float | int]]

PARAMETERS = ("alpha", "tau", "Delta")
DIMENSIONS = (
    "context",
    "fixed_parameter",
    "gravity_model",
    "constellation",
    "tide_mode",
    "parameter",
)
SOURCE_DIMENSIONS = {
    "third_parameter": 0,
    "fixed_variable": 1,
    "constellation": 3,
    "gravity_model": 2,
    "tide_mode": 4,
    "formal": None,
}
SOURCE_LABELS = (
    "3rd parameter\ninfluence",
    "Fixed variable",
    "Constellation",
    "Residual gravity\nfield model",
    "Tide mode",
    "Formal\nuncertainty",
)
NAME_MAPPING = {"LAM": r"$\alpha$", "LQM": r"$\tau$", "LDM": r"$\Delta$"}


def get_parameter_values_and_sigma(
    parameters: list[str],
    file_path: Path,
    n_sigmas: int = 3,
    alpha: float = 0.25,
) -> dict[str, tuple[float, float]]:
    """
    Reads values and sigmas from listing.
    """

    names = {p.casefold(): p for p in parameters}
    result = {}

    with file_path.open() as f:

        for line in f:

            match = search(r"\[([^\]]+)\]\s+([+-]?\d.*)", line)

            if match and (key := match[1].strip().casefold()) in names:

                columns = match[2].split()
                value, sigma = (float(s.replace("D", "E").replace("d", "e")) for s in columns[2:4])

                if names[key] == "LDM":

                    sigma = max(
                        10**value - 10 ** (value - n_sigmas * sigma),
                        10 ** (value + n_sigmas * sigma) - 10**value,
                    )
                    value = 10**value

                result[NAME_MAPPING[names[key]]] = (value, sigma)

        if NAME_MAPPING["LQM"] in result:

            value, sigma = result[NAME_MAPPING["LQM"]]
            delta, _ = result[NAME_MAPPING["LDM"]]
            new_value = (
                find_tau_m_sup(
                    omega_m_inf=3.09e-4, period_unit=1, alpha=alpha, delta=delta, q_mu=10**value
                )
                / SECONDS_PER_YEAR
            )
            lower_bound = (
                find_tau_m_sup(
                    omega_m_inf=3.09e-4,
                    period_unit=1,
                    alpha=alpha,
                    delta=delta,
                    q_mu=10 ** (value - n_sigmas * sigma),
                )
                / SECONDS_PER_YEAR
            )
            upper_bound = (
                find_tau_m_sup(
                    omega_m_inf=3.09e-4,
                    period_unit=1,
                    alpha=alpha,
                    delta=delta,
                    q_mu=10 ** (value + n_sigmas * sigma),
                )
                / SECONDS_PER_YEAR
            )
            result[NAME_MAPPING["LQM"]] = (
                new_value,
                max(
                    new_value - lower_bound,
                    upper_bound - new_value,
                ),
            )

    return result


def get_parameter_values_and_sigma_for_all_pod_parametrizations(
    parameters: list[str],
    path: Path,
    fix_option: str,
    n_sigmas: int = 3,
    alpha: float = 0.25,
) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    """
    For a given rheological parametrization, gets needed rheological solutions.
    """

    solutions = {}

    for gravity_model_path in path.glob("*"):

        if "acceleration" not in gravity_model_path.name:

            continue

        gravity_model = ("with" if "annual" in gravity_model_path.name else "without") + " annual"

        for listing in gravity_model_path.glob("*"):

            if listing.name[-4:] == ".err":

                continue

            tide_parametrization = (
                "without" if ("solid" in listing.name) else "with"
            ) + " pole tide"
            constellation = (
                listing.name[17:]
                .split("_solid" if ("solid" in listing.name) else "tides")[0]
                .replace("_", " ")
            )
            listing_solutions = get_parameter_values_and_sigma(
                parameters=parameters, file_path=listing, n_sigmas=n_sigmas, alpha=alpha
            )

            for parameter, solution in listing_solutions.items():

                solutions[
                    (fix_option, gravity_model, constellation, tide_parametrization, parameter)
                ] = solution

    return solutions


def normalize_solutions(
    contexts: Mapping[str, Mapping[SolutionKey, tuple[float, float]]],
) -> Records:
    """Normalize whitespace and math labels; reject ambiguous or invalid rows.

    Satellite names are sorted so a constellation is independent of name order.
    Normalization collisions raise even if the associated values are identical.
    """
    records = {}
    for context, solutions in contexts.items():
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context names must be nonempty strings.")
        for key, pair in solutions.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 5
                or not all(isinstance(x, str) for x in key)
            ):
                raise ValueError(f"Invalid solution key: {key!r}")
            fixed, gravity, constellation, tide, parameter = (" ".join(x.split()) for x in key)
            parameter = parameter.replace("$", "").replace("\\", "").strip()
            if fixed not in ("LAM", "LQM") or parameter not in PARAMETERS:
                raise ValueError(f"Unknown fixed/evaluated parameter: {key!r}")
            allowed = {"LAM": ("tau", "Delta"), "LQM": ("alpha", "Delta")}
            if parameter not in allowed[fixed]:
                raise ValueError(f"Incompatible fixed/evaluated parameter: {key!r}")
            if not all((gravity, constellation, tide)):
                raise ValueError(f"Empty configuration label: {key!r}")
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Expected (solution, formal_uncertainty): {pair!r}")
            value, formal = map(float, pair)
            if not all(map(isfinite, (value, formal))) or formal < 0:
                raise ValueError(f"Nonfinite solution or invalid formal uncertainty: {pair!r}")
            normalized = (
                context.strip(),
                fixed,
                gravity,
                " ".join(sorted(constellation.split())),
                tide,
                parameter,
            )
            if normalized in records:
                raise ValueError(f"Duplicate normalized configuration: {normalized!r}")
            records[normalized] = (value, formal)
    if not records:
        raise ValueError("No solutions supplied.")
    return records


def uncertainty_metric(values: Sequence[float], *, metric: str = "range") -> float:
    """Reduce >=2 solutions to one sensitivity; no array keys or callable map."""
    values = asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not isfinite(values).all():
        raise ValueError("At least two finite scalar solutions are required.")
    if metric == "range":
        return float(ptp(values))
    if metric == "half_range":
        return float(ptp(values) / 2)
    if metric == "std":
        return float(std(values, ddof=0))
    raise ValueError(f"Unknown metric: {metric!r}")


def compute_uncertainty_samples(
    records: Records, *, metric: str = "range"
) -> tuple[Samples, list[dict]]:
    """Hold every dimension except the selected source constant.

    Return samples[(parameter, source)][controlled_configuration]. Each sample
    retains its uncertainty, compared source levels and solutions. Configuration
    keys contain DIMENSIONS with only the varying dimension removed (all six
    dimensions for formal uncertainty). Missing source settings are never imputed.
    Groups with one setting are skipped; partial groups with >=2 are used and
    reported. Diagnostics compare levels to the observed union for that panel.
    """
    if metric not in ("range", "half_range", "std"):
        raise ValueError(f"Unknown metric: {metric!r}")
    samples, diagnostics = {}, []
    for parameter in PARAMETERS:
        rows = {key: pair for key, pair in records.items() if key[-1] == parameter}
        for source, dimension in SOURCE_DIMENSIONS.items():
            if source == "third_parameter" and parameter == "Delta":
                continue
            if source == "fixed_variable" and parameter != "Delta":
                continue
            bucket = samples[(parameter, source)] = {}
            if source == "formal":
                for key, (value, formal) in rows.items():
                    bucket[key] = {
                        "uncertainty": formal,
                        "n_levels": 1,
                        "levels": (),
                        "solutions": (value,),
                    }
                continue
            groups = defaultdict(dict)
            expected_levels = {key[dimension] for key in rows}
            for key, (value, _) in rows.items():
                controlled = key[:dimension] + key[dimension + 1 :]
                groups[controlled][key[dimension]] = value
            for controlled, level_values in groups.items():
                levels = tuple(sorted(level_values))
                missing = tuple(sorted(expected_levels - set(levels)))
                if len(levels) < 2 or missing:
                    diagnostics.append(
                        {
                            "parameter": parameter,
                            "source": source,
                            "configuration": controlled,
                            "missing_levels": missing,
                            "status": "skipped" if len(levels) < 2 else "partial",
                        }
                    )
                if len(levels) < 2:
                    continue
                values = tuple(level_values[level] for level in levels)
                bucket[controlled] = {
                    "uncertainty": uncertainty_metric(values, metric=metric),
                    "n_levels": len(levels),
                    "levels": levels,
                    "solutions": values,
                }
    return samples, diagnostics


def summarize_uncertainties(samples: Samples, *, ddof: int = 0) -> Summary:
    """Equal weight per controlled configuration; population std by default.

    Empty groups remain absent. Insufficient samples for ddof raise explicitly.
    Mean +/- std is not clipped to zero or to the observed min/max.
    """
    if not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof must be a nonnegative integer.")
    summary = {}
    for key, bucket in samples.items():
        if not bucket:
            continue
        values = asarray([sample["uncertainty"] for sample in bucket.values()])
        if len(values) <= ddof:
            raise ValueError(f"Insufficient samples for ddof={ddof}: {key}")
        mean_value, std_value = float(mean(values)), float(std(values, ddof=ddof))
        summary[key] = {
            "n": len(values),
            "min": float(min(values)),
            "mean_minus_std": mean_value - std_value,
            "mean": mean_value,
            "mean_plus_std": mean_value + std_value,
            "max": float(max(values)),
            "std": std_value,
        }
    return summary


def summarize_overall_solutions(records: Records) -> dict[str, dict[str, float]]:
    """Pool solutions across contexts/configurations; population std (ddof=0).

    The combined label uncertainty is sqrt(solution_std**2 + max_formal**2).
    """
    overall = {}
    for parameter in PARAMETERS:
        pairs = [pair for key, pair in records.items() if key[-1] == parameter]
        if not pairs:
            raise ValueError(f"No solutions for {parameter}.")
        values = asarray([value for value, _ in pairs], dtype=float)
        solution_std = float(std(values, ddof=0))
        max_formal = max(formal for _, formal in pairs)
        overall[parameter] = {
            "mean": float(mean(values)),
            "solution_std": solution_std,
            "max_formal": max_formal,
            "combined_std": sqrt(solution_std**2 + max_formal**2),
        }
    return overall


def plot_uncertainty_budget(
    summary: Summary,
    records: Records,
    *,
    parameter_labels: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (11, 9),
):
    """Three shared-x panels, independent linear y scales, all statistics visible.

    Labels show overall solution mean +/- combined std; tau is in years.
    Counts show the number of comparisons or supplied formal uncertainties.
    Optional parameter_labels override the complete generated y-axis labels.
    """
    overall = summarize_overall_solutions(records)
    symbols = {"alpha": r"\alpha", "tau": r"\tau", "Delta": r"\Delta"}
    labels = {}
    for parameter, stats in overall.items():
        unit = " (yr)" if parameter == "tau" else ""
        labels[parameter] = (
            rf"${symbols[parameter]} = {stats['mean']:.3f} \pm {stats['combined_std']:.3f}$" + unit
        )
    if parameter_labels:
        labels.update(parameter_labels)
    colors = ("#176b87", "#94602d", "#6b5795")
    with rc_context(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    ):
        fig, axes = subplots(3, 1, sharex=True, figsize=figsize)
        for ax, parameter, color in zip(axes, PARAMETERS, colors):
            limits = [0.0]
            for x, source in enumerate(SOURCE_DIMENSIONS):
                stats = summary.get((parameter, source))
                if stats is None:
                    continue
                ax.vlines(x, stats["min"], stats["max"], color=color, linewidth=1.5, zorder=3)
                ax.hlines(
                    [stats["min"], stats["max"]],
                    x - 0.09,
                    x + 0.09,
                    color=color,
                    linewidth=1.5,
                    zorder=3,
                )
                ax.vlines(
                    x,
                    stats["mean_minus_std"],
                    stats["mean_plus_std"],
                    color=color,
                    linewidth=8,
                    alpha=0.40,
                    zorder=4,
                )
                ax.plot(
                    x,
                    stats["mean"],
                    "o",
                    color=color,
                    markeredgecolor="white",
                    markersize=8,
                    zorder=5,
                )
                ax.text(
                    x,
                    0.94,
                    f"n = {stats['n']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color="#68717d",
                    fontsize=9,
                )
                limits.extend(
                    [stats["min"], stats["max"], stats["mean_minus_std"], stats["mean_plus_std"]]
                )
            lo, hi = min(limits), max(limits)
            span = hi - lo or 1.0
            ax.set_ylim(lo - 0.06 * span, hi + 0.22 * span)
            ax.axhline(0, color="#aab2bb", linewidth=0.8)
            ax.grid(axis="y", color="#e5e9ed", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_ylabel(labels[parameter], rotation=90, fontsize=15)
            ax.tick_params(axis="x", length=0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            ax.set_xlim(-0.5, 5.5)
        axes[-1].set_xticks(range(6), SOURCE_LABELS)
        handles = [
            Line2D([0], [0], color="#465361", lw=1.5, marker="_", label="Min–max"),
            Line2D([0], [0], color="#465361", lw=8, alpha=0.4, label="Mean ± std"),
            Line2D([0], [0], color="#465361", lw=0, marker="o", label="Mean"),
        ]
        fig.suptitle("Uncertainty budget", x=0.09, ha="left", y=0.98, fontsize=19, weight="bold")
        fig.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.985), ncol=3, frameon=False
        )
        tight_layout()
    return fig, axes
