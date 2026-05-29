"""
Prepares transient anelastic tide corrections (pole tide and solid Earth tide) and their partial
derivatives to be used in GINS software.
Reads the Love numbers and the C01 time series and computes the corresponding pole tide correction
and pole tide deformation correction. Saves all relevant information as fortran90-ready hard coded
arrays.
"""

from gins_partials import regenerate_fortran_code

if __name__ == "__main__":

    regenerate_fortran_code()
