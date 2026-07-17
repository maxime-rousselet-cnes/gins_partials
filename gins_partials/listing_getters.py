"""
Functions to read GINS listings for validation purposes.
"""

from pathlib import Path

from base_models import DATA_PATH
from numpy import asarray, ndarray

GINS_LISTING_PATH = DATA_PATH.joinpath("listing")


def read_for_partials(
    filename: str,
    path: Path = GINS_LISTING_PATH,
    parameter_index: int = 1,
    parameter_value: float = 0.25,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Gets tabs of epochs, accelerations, and parameter partials.
    Handles vector values split across multiple lines.
    """

    epochs = []

    outputs = {
        "acc": [],
        "alpha": [],
        "log10_delta": [],
        "log10_tau_m": [],
    }

    current_key = None
    current_values = []

    with open(list(path.glob(filename + "*"))[0], "r", encoding="utf-8") as f:

        for line in f:

            if "ER:0" in line:

                break

            fields = line.split()

            if not fields:

                continue

            key = fields[0]

            if key == "time":

                epochs.append(float(fields[1]))
                current_key = None
                current_values = []

                if parameter_index < 3:

                    assert float(fields[1 + parameter_index]) == parameter_value

            elif key in outputs:

                current_key = key
                current_values = [float(value) for value in fields[1:]]

            elif current_key is not None:

                current_values.extend(float(value) for value in fields)

            if current_key is not None and len(current_values) == 3:

                outputs[current_key].append(current_values)
                current_key = None
                current_values = []

    return (
        asarray(epochs),
        asarray(outputs["acc"]),
        asarray(outputs["alpha"]),
        asarray(outputs["log10_delta"]),
        asarray(outputs["log10_tau_m"]),
    )
