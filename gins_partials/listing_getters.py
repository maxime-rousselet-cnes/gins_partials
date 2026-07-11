"""
Functions to read GINS listings for validation purposes.
"""

from pathlib import Path

from base_models import DATA_PATH
from numpy import asarray, ndarray

GINS_LISTING_PATH = DATA_PATH.joinpath("listing")


def read_for_partials(
    filename: str, path: Path = GINS_LISTING_PATH
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Gets tabs of epochs, accelerations, and alpha and delta partials of the accelerations.
    """

    epochs = []
    acc = []
    alpha = []
    delta = []

    with open(path.joinpath(filename), "r", encoding="utf-8") as f:
        for line in f:
            fields = line.split()

            if not fields:
                continue

            key = fields[0]

            if key == "acceleration":
                epochs.append(float(fields[1]))
                acc.append([float(fields[2]), float(fields[3]), float(fields[4])])

            elif key == "alpha_partials":
                alpha.append([float(fields[2]), float(fields[3]), float(fields[4])])

            elif key == "delta_partials":
                delta.append([float(fields[2]), float(fields[3]), float(fields[4])])

    return (
        asarray(epochs),
        asarray(acc),
        asarray(alpha),
        asarray(delta),
    )
