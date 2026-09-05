"""
Prepares transient anelastic tide corrections (pole tide and solid Earth tide) and their partial
derivatives to be used in GINS software.
Reads the Love numbers and the C01 time series and computes the corresponding pole tide correction
and pole tide deformation correction. Saves all relevant information as fortran90-ready hard coded
arrays.
"""

import argparse

from gins_partials import encode_tide_correction_models

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Enable encoding",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enable encoding",
    )
    args = parser.parse_args()
    encode_tide_correction_models(
        to_encode=args.encode,
        to_save=args.save,
    )
