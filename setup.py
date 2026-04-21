"""
To install via pip install -e . in the root of the repository. This will make the GINS_PARTIALS
package available in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="gins_partials",
    packages=find_packages(),
    version="0.0.1",
    description="Prepares rheological partial derivatives to include in GINS",
    author="Maxime Rousselet",
)
