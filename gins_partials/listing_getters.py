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
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Gets tabs of epochs, accelerations, and parameter partials.
    Handles vector values split across multiple lines.
    """

    epochs = []

    outputs = {
        "acc": [],
        "local_lam": [],
        "local_lqm": [],
        "local_ldm": [],
        "local_ltm": [],
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

            elif key in outputs:

                if key == "local_ldm" and len(outputs["local_ldm"]) == len(outputs["local_lqm"]):

                    key = "local_lqm"

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
        asarray(outputs["local_lam"]),
        asarray(outputs["local_lqm"]),
        asarray(outputs["local_ldm"]),
        asarray(outputs["local_ltm"]),
    )
