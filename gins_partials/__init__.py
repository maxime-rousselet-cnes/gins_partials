"""
Package to bridge ALNA with GINS the hard way.
"""

from .figures import (
    compare_acceleration_partials_to_finite_differences,
    plot_pole_motion,
    plot_pole_tide_models,
)
from .tide_correction_model import (
    hard_code_fortran90,
    preprocess_and_save_tide_correction_partials,
    regenerate_fortran_code,
)

to_import = [
    compare_acceleration_partials_to_finite_differences,
    plot_pole_motion,
    plot_pole_tide_models,
    hard_code_fortran90,
    preprocess_and_save_tide_correction_partials,
    regenerate_fortran_code,
]
