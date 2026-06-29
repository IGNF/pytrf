#-------------------------------------------------------------------------------
# Copyright (c) Institut national de l'information géographique et forestière
#
# Main author:
#  - Paul Rebischung
#
# This file is part of pytrf: https://github.com/IGNF/pytrf
#
# pytrf is licensed under the MIT license found in the LICENSE.md file
# in the root directory of this source tree.
#-------------------------------------------------------------------------------



"""
pytrf
=====

Python toolbox for the analysis of time series of terrestrial reference frames

This module imports the following classes and modules into the main
namespace:

    date   : Class for handling dates in various formats
    sinex  : Class for reading, writing and manipulating SINEX files
    erp    : Class for reading, writing and manipulating ERP files
    
    const  : pytrf constants
    utils  : pytrf miscellaneous utilities
    math   : pytrf math utilities
    io     : pytrf input/output utilities
    igs    : pytrf IGS utilities
    ts     : pytrf time series utilities
    snxcmb : SINEX combination utilities

"""



# Import pytrf classes
from .date import date
from .sinex import sinex
from .erp import erp

# Import pytrf modules
from . import const
from . import utils
from . import math
from . import io
from . import igs
from . import ts
from . import snxcmb
