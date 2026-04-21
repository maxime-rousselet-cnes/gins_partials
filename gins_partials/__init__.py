"""
Package to bridge ALNA with GINS the hard way.
"""

from .tide_correction_model import hard_code_fortran90, preprocess_and_save_tide_correction_partials

to_import = [hard_code_fortran90, preprocess_and_save_tide_correction_partials]
