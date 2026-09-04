"""
Validation figures.
"""

from gins_partials import (
    compare_acceleration_partials_to_finite_differences,
    plot_pole_motion,
    plot_pole_tide_models,
)

UNPLOT = [plot_pole_motion, plot_pole_tide_models]

if __name__ == "__main__":

    compare_acceleration_partials_to_finite_differences()
