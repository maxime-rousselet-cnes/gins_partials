"""
Prepares transient anelastic tide corrections (pole tide and solid Earth tide) and their partial
derivatives to be used in GINS software.
Reads the Love numbers and the C01 time series and computes the corresponding pole tide correction
and pole tide deformation correction. Saves all relevant information as fortran90-ready hard coded
arrays.
"""

from gins_partials import hard_code_tide_correction_models

if __name__ == "__main__":

    hard_code_tide_correction_models(to_encode=False, to_save=True)
