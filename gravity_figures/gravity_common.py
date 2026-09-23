"""Shared Dynamo-D parsing, RL05 evaluation, file selection and plotting helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARMONICS = ("C_20", "C_21", "S_21", "C_40", "C_41", "S_41", "C_60")
SATELLITES = ("ajisai", "lageos1", "lageos2", "starlette", "stella")
CONSTRAINT = "constraints_G_trend_and_acceleration_and_annual"
BRANCH = Path("sigma_E-13") / CONSTRAINT
FULL_NAME = "rheology_" + "_".join(SATELLITES)
# An explicit exponent width avoids swallowing the next adjacent matrix value.
NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]\d{2,3})?")
DATED = re.compile(r"G([CS])N\s+(\d+)\s+(\d+)\s+(\d{8})")
MODEL = re.compile(r"([CP])(AA|A|B|C|S|P|Q)_(\d)(\d)")


def floats(text):
    tokens = NUMBER.findall(text)
    remainder = NUMBER.sub("", text)
    if remainder.strip():
        raise ValueError(f"Unrecognized numeric data: {text[:100]!r}")
    return [float(x.replace("D", "E").replace("d", "e")) for x in tokens]


def date8(value):
    value = value.split(".")[0]
    return np.datetime64(f"{value[:4]}-{value[4:6]}-{value[6:8]}", "D")


def harmonic_label(name):
    family, nm = name.split("_")
    return rf"${family}_{{{nm}}}$"


@dataclass
class Parameter:
    name: str
    value: float
    sigma: float
    harmonic: str | None = None
    date: np.datetime64 | None = None

    @property
    def label(self):
        match = MODEL.fullmatch(self.name)
        if match:
            family, mode, n, m = match.groups()
            family = "S" if family == "P" else "C"
            return rf"${family}_{{{n}{m}}}^{{{mode}}}$"
        return self.name


@dataclass
class Solution:
    path: Path
    parameters: list[Parameter]
    correlation: np.ndarray | None = None
    correlation_parameters: list[Parameter] | None = None

    def series(self, harmonic):
        records = sorted(
            (p for p in self.parameters if p.harmonic == harmonic), key=lambda p: p.date
        )
        if not records:
            raise ValueError(f"{self.path}: no dated {harmonic} parameters")
        dates = np.array([p.date for p in records], dtype="datetime64[D]")
        if len(np.unique(dates)) != len(dates):
            raise ValueError(f"{self.path}: duplicate {harmonic} epochs")
        return dates, np.array([p.value for p in records]), np.array([p.sigma for p in records])


def load_solution(path, with_correlation=False):
    """Read solutions; optionally stream the complete packed lower triangle.

    Matrix order follows nonzero-sigma SOLUTION records, as in the old script.
    Keep only the nondated submatrix, without allocating the full inverse matrix.
    Zero-sigma (fixed) parameters are excluded from correlations.
    """
    path = Path(path)
    parameters = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "SOLUTION":
                break
        else:
            raise ValueError(f"{path}: missing SOLUTION section")
        header = next(stream, "").split()
        if len(header) != 3:
            raise ValueError(f"{path}: invalid SOLUTION count header")
        total, fitted, fixed = map(int, header)
        matrix_found = False
        names = set()
        for line in stream:
            if line.strip() == "INVERSE MATRIX":
                matrix_found = True
                break
            if not line.strip():
                continue
            name = " ".join(line[:24].split())
            data = floats(line[24:])
            if len(data) != 4 or not np.isfinite(data).all():
                raise ValueError(f"{path}: invalid solution record {line.rstrip()!r}")
            if name in names:
                raise ValueError(f"{path}: duplicate parameter {name}")
            names.add(name)
            match = DATED.fullmatch(name)
            harmonic = date = None
            if match:
                family, n, m, epoch = match.groups()
                harmonic, date = f"{family}_{int(n)}{int(m)}", date8(epoch)
            elif name.startswith(("GCN", "GSN")):
                raise ValueError(f"{path}: malformed dated parameter {name}")
            parameters.append(Parameter(name, data[2], data[3], harmonic, date))
        if len(parameters) != total:
            raise ValueError(f"{path}: expected {total} parameters; found {len(parameters)}")
        active = [p for p in parameters if p.sigma != 0]
        if len(active) != fitted or total != fitted + fixed:
            raise ValueError(f"{path}: count header disagrees with nonzero formal sigmas")
        result = Solution(path, parameters)
        if not with_correlation:
            return result
        if not matrix_found:
            raise ValueError(f"{path}: missing INVERSE MATRIX")
        matrix_header = next(stream, "")
        # Two width-10 integers followed by the variance factor in the extract.
        try:
            matrix_size = int(matrix_header[10:20])
        except ValueError as exc:
            raise ValueError(f"{path}: invalid inverse matrix header") from exc
        if matrix_size != fitted:
            raise ValueError(f"{path}: matrix dimension {matrix_size} != fitted count {fitted}")
        selected = [i for i, p in enumerate(active) if p.date is None]
        if not selected:
            raise ValueError(f"{path}: no fitted nondated parameters")
        lookup = {old: new for new, old in enumerate(selected)}
        covariance = np.full((len(selected), len(selected)), np.nan)
        i = j = count = 0
        expected = fitted * (fitted + 1) // 2
        for line in stream:
            if not line.strip():
                continue
            if count == expected:
                break  # Subsequent file sections are not matrix data.
            packed = line.rstrip("\r\n")
            if len(packed) % 20:
                raise ValueError(f"{path}: matrix line is not a multiple of 20 columns")
            for offset in range(0, len(packed), 20):
                value = float(packed[offset : offset + 20].replace("D", "E").replace("d", "e"))
                if count >= expected:
                    raise ValueError(f"{path}: too many matrix values")
                if not np.isfinite(value):
                    raise ValueError(f"{path}: nonfinite matrix value")
                if i in lookup and j in lookup:
                    a, b = lookup[i], lookup[j]
                    covariance[a, b] = covariance[b, a] = value
                count += 1
                j += 1
                if j > i:
                    i += 1
                    j = 0
        if count != expected:
            raise ValueError(f"{path}: incomplete inverse matrix: {count:,} / {expected:,} values")
        diagonal = np.diag(covariance)
        if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
            raise ValueError(f"{path}: invalid selected covariance diagonal/data")
        correlation = covariance / np.sqrt(diagonal[:, None] * diagonal[None, :])
        if np.max(np.abs(correlation)) > 1 + 1e-6:
            raise ValueError(f"{path}: covariance gives correlation outside [-1, 1]")
        # Rheology first; then harmonic order and temporal mode.
        mode_order = {"B": 0, "A": 1, "AA": 2, "C": 3, "S": 4, "P": 5, "Q": 6}

        def order(k):
            name = active[selected[k]].name
            match = MODEL.fullmatch(name)
            if match:
                family, mode, n, m = match.groups()
                h = ("S" if family == "P" else "C") + "_" + n + m
                return (1, HARMONICS.index(h) if h in HARMONICS else 99, mode_order[mode], name)
            return (0, 0, 0, name)

        permutation = sorted(range(len(selected)), key=order)
        result.correlation = np.clip(correlation[np.ix_(permutation, permutation)], -1, 1)
        result.correlation_parameters = [active[selected[k]] for k in permutation]
        return result


class RL05:
    """Evaluate the G_BIAS/G_DRIFT/G_COS/G_SIN SHC dialect in the old script.

    The first two YYYYMMDD[.fraction] tokens after the four coefficient fields
    define validity and origin. Annual terms use year_days (365 by default,
    matching the supplied code). Other trailing fields are ignored, as before.
    Values and uncertainties both use the direct signed sum of basis terms,
    exactly as in the supplied gravity_timeseries implementation.
    """

    def __init__(self, path, year_days=365.0):
        if year_days <= 0:
            raise ValueError("year_days must be positive")
        self.year_days = year_days
        self.records = {h: [] for h in HARMONICS}
        with Path(path).open() as stream:
            for lineno, line in enumerate(stream, 1):
                p = line.split()
                if not p or not p[0].upper().startswith("G"):
                    continue
                if len(p) < 3:
                    continue
                try:
                    n, m = int(p[1]), int(p[2])
                except ValueError:
                    continue  # Header rather than a coefficient record.
                if f"C_{n}{m}" not in HARMONICS:
                    continue

                parameter_type = p[0].lower()

                if "bias" in parameter_type:
                    mode = "BIAS"
                elif "drift" in parameter_type:
                    mode = "DRIFT"
                elif "cos" in parameter_type:
                    mode = "COS"
                elif "sin" in parameter_type:
                    mode = "SIN"
                else:
                    continue
                values = [float(x.replace("D", "E").replace("d", "e")) for x in p[3:7]]
                epochs = [x for x in p[7:] if re.fullmatch(r"\d{8}(?:\.\d+)?", x)]
                if len(values) != 4 or len(epochs) != 2 or not np.isfinite(values).all():
                    raise ValueError(
                        f"{path}:{lineno}: expected values, sigmas and two validity dates"
                    )
                start, end = map(date8, epochs)
                if start > end or min(values[2:]) < 0:
                    raise ValueError(f"{path}:{lineno}: invalid interval or sigma")
                for family, value, sigma in [
                    ("C", values[0], values[2]),
                    ("S", values[1], values[3]),
                ]:
                    h = f"{family}_{n}{m}"
                    if h in self.records:
                        self.records[h].append((mode, start, end, value, sigma))
        if not any(self.records.values()):
            raise ValueError(f"{path}: no supported RL05 records")

    def evaluate(self, harmonic, dates):
        """Original RL05 equations at requested dates; inclusive validity bounds.

        Keep the legacy signed uncertainty sum, including zero initialization
        outside covered intervals. Do not square terms or take absolute values.
        """
        dates = np.asarray(dates, dtype="datetime64[D]")
        values = np.zeros(len(dates))
        uncertainties = np.zeros(len(dates))
        for mode, start, end, value, sigma in self.records[harmonic]:
            mask = (dates >= start) & (dates <= end)
            years = (dates[mask] - start).astype(float) / self.year_days
            basis = {
                "BIAS": lambda: np.ones(len(years)),
                "DRIFT": lambda: years,
                "COS": lambda: np.cos(2 * np.pi * years),
                "SIN": lambda: np.sin(2 * np.pi * years),
            }[mode]()
            values[mask] += value * basis
            uncertainties[mask] += sigma * basis
        return values, uncertainties


def gravity_timeseries(filename="RL05.shc", start="1991-01-11", end="2025-12-31", step_days=30):
    """Legacy public interface: the same grid, equations and uncertainty sums.

    RL05.evaluate uses these same equations directly at inversion epochs for
    RMS comparisons, so no interpolation of this plotting grid is needed.
    """
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    dates = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(end, "D") + np.timedelta64(1, "D"),
        np.timedelta64(step_days, "D"),
    )
    model = RL05(filename)
    values, uncertainties = {}, {}
    for harmonic in HARMONICS:
        values[harmonic], uncertainties[harmonic] = model.evaluate(harmonic, dates)
    return dates, values, uncertainties


def full_solution(root, case="tides", constrained=True):
    parent = Path(root) / BRANCH if constrained else Path(root)
    name = FULL_NAME + "_" + case
    matches = [
        p for p in parent.glob(name + "*") if p.is_file() and (p.name == name or p.stem == name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one result named {name}[.ext] in {parent}; found {len(matches)}"
        )
    return matches[0]


def constellation_from_file(path):
    name = path.stem if path.suffix else path.name
    if not name.startswith("rheology_") or not name.endswith("_tides"):
        return None
    middle = name[len("rheology_") : -len("_tides")]
    satellites = middle.split("_")
    if len(set(satellites)) != len(satellites) or not set(satellites) <= set(SATELLITES):
        return None  # In particular, excludes solid_tides.
    return tuple(s for s in SATELLITES if s in satellites)


def constellation_files(root, branch="unconstrained"):
    """Find exact tides filenames; select a single branch, reject duplicates."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    suffix = None if branch == "unconstrained" else Path(branch).parts
    if suffix is not None and (Path(branch).is_absolute() or ".." in suffix):
        raise ValueError("branch must be a relative sigma/constraint path")
    found = {}
    for path in sorted(root.rglob("rheology_*")):
        if not path.is_file():
            continue
        constellation = constellation_from_file(path)
        if constellation is None:
            continue
        parents = path.relative_to(root).parts[:-1]
        if suffix is None:
            if any(p.startswith(("sigma_", "constraints_")) for p in parents):
                continue
        elif tuple(parents[-len(suffix) :]) != suffix:
            continue
        if constellation in found:
            raise ValueError(
                f"Duplicate constellation in selected branch: {found[constellation]} and {path}"
            )
        found[constellation] = path
    return sorted(
        found.items(), key=lambda item: (len(item[0]), tuple(SATELLITES.index(s) for s in item[0]))
    )


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved {output}")
