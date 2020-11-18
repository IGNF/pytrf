"""
pytrf
=====

pytrf is a collection of routines that can be used for the analysis and
combination of Terrestrial Reference Frames.

This module imports the following classes and subpackages into the main
namespace:

    date   : Class for handling dates in various formats
    sinex  : Class for reading, writing and manipulating SINEX files
    erp    : Class for reading, writing and manipulating ERP files
    
    const  : pytrf constants
    utils  : pytrf miscellaneous utilities
    math   : pytrf math utilities
    io     : pytrf I/O utilities
    igs    : pytrf IGS utilities
    ts     : pytrf time series utilities
    snxcmb : SINEX combination utilities

"""

__version__ = '0.1'
__author__ = 'Paul Rebischung'

# Import pytrf classes
from .date import date
from .sinex import sinex
from .erp import erp

# Import subpackages
from . import const
from . import utils
from . import math
from . import io
from . import igs
from . import ts
from . import snxcmb
