"""
pytrf constants

This subpackage defines several useful constants including:
* conversion factors between angular units,
* offsets between different time scales,
* GRS80 ellipsoid parameters

"""

# External imports
#-----------------
import math
import numpy as np

# Data
#-----

# Agency
agency = 'IGN'

# Default DOMES number
default_domes = '     M   '

# Conversion factors from as, mas and uas to rad
as2rad  = math.pi / 180 / 3600
mas2rad = as2rad / 1000
uas2rad = mas2rad / 1000

# Conversion factor from s, ms and us to rad
s2rad  = math.pi / 12 / 3600
ms2rad = s2rad / 1000
us2rad = ms2rad / 1000

# Offset between GPS time and TT
tt_gps = (19 + 32.184) / 86400

# GPS-UTC leap seconds
gps_utc  = np.arange(-9, 19)
mjd_leap = np.array([41317, 41499, 41683, 42048, 42413, 42778, 43144,
                     43509, 43874, 44239, 44786, 45151, 45516, 46247,
                     47161, 47892, 48257, 48804, 49169, 49534, 50083,
                     50630, 51179, 53736, 54832, 56109, 57204, 57754])

# GRS80 parameters
ae = 6378137.
fe = 0.00335281068118
ee = math.sqrt(2*fe - fe**2)
be = ae * math.sqrt(1 - ee**2)

# Rate of advance of ERA
dera_dt = 1.00273781191135448


# periods codes & values
PERIODS = {
    "A001":365.25,
    "D001":351.5
    }
