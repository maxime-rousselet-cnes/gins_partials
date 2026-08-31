"""
Generates the time-dependent pole tide correction and the frequency-dependent solid Earth tide
correction for a given rheological model.
"""

from argparse import ArgumentParser
from pathlib import Path

from gins_partials import tide_correction_model_generation

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()
    tide_correction_model_generation(file_path=Path(args.file_path))
