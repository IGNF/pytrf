""" Setup script for pytrf """

import setuptools

setuptools.setup(
    name='pytrf',
    version='0.1',
    description='Python utilities for the analysis and combination of Terrestrial Reference Frames',
    author='Paul Rebischung',
    author_email='paul.rebischung@ign.fr',
    packages=setuptools.find_packages(),
    entry_points = {'gui_scripts': ['fits=pytrf.fits:main']},
)

