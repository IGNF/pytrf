#--------------------------------------------------------------------------------------------------------------------------------
# This script demonstrates how to read GNSS station position time series in different formats with pytrf:
#  - IGS xyz format (ftp://igs-rf.ign.fr/pub/README_crd)
#  - NGL tenv3 format (https://geodesy.unr.edu/gps_timeseries/readmes/README_tenv3.txt)
#  - JPL series format (https://sideshow.jpl.nasa.gov/post/series.html)
#  - PBO pos format (https://www.unavco.org/data/gps-gnss/derived-products/docs/knowledgetree-docs-old/gps_timeseries_format.pdf)
#  - SPOTGINS enu format (https://www.poleterresolide.fr/geodesy-plotter)
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
import numpy as np
from pytrf import date
from pytrf.ts import ts



# 1 - IGS crd format
#-------------------

# Download example file
os.system('wget -c ftp://igs-rf.ign.fr/pub/crd/CKSV_igs.xyz')

# Read time series
# Note: No need to specify units since time and values are given in pytrf's default units (MJD and meters).
# Note: No need to specify dimension names. They're automatically set to 'East', 'North', 'Up' with rotate=True.
# Note: Correlations across dimensions are not used by pytrf when modeling time series.
#       Here, however, it is necessary to read them, in order to compute the ENH sigmas rigorously.
r = ts.read('CKSV_igs.xyz',
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),                              # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),     # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas; pairwise correlations
            dtrd=1,                                                                 # Remove (and store) mean linear trend
            rotate=True)                                                            # Rotate time series from geocentric (XYZ) to topocentric (ENH) frame

# Remove points with unusually large sigmas
r.clean_sigmas()

# Plot time series (with X-axis in decimal years)
r.plot(tunit='y')



# 2 - NGL tenv3 format
#---------------------

# Download example file
os.system('wget -c https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/CKSV.tenv3')

# Read time series
# Note: No need to specify units since time and values are given in pytrf's default units (MJD and meters).
r = ts.read('CKSV.tenv3',
            skiprows=1,                                         # Skip header line
            usecols=(3, 8, 10, 12, 14, 15, 16),                 # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'),      # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas
            dims=('East', 'North', 'Up'),                       # Dimension names
            dtrd=1)                                             # Remove (and store) mean linear trend

# Plot time series (with X-axis in decimal years)
r.plot(tunit='y')



# 3 - JPL series format
#----------------------

# Download example file
os.system('wget -c https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/CKSV.series')

# Read time series
# Note: No need to specify units since values are given in pytrf's default unit (meters),
#       and time will be converted to pytrf's default unit (MJD) below.
r = ts.read('CKSV.series',
            usecols=(10, 1, 2, 3, 4, 5, 6),                     # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'),      # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas
            dims=('East', 'North', 'Up'),                       # Dimension names
            dtrd=1)                                             # Remove (and store) mean linear trend

# Convert original dates (seconds since J2000) into MJDs
r.t = date.from_ymdhms(2000, 1, 1, 12).mjd + r.t/86400

# Round MJDs to nearest noon
# (For better efficiency in noise modeling, it is preferable, when possible, to get back to regularly spaced dates.)
r.t = np.ceil(r.t) - 0.5

# Consequently set time series integration interval to 1 day
r.T = 1

# Delete occasional points at the same dates as the next ones (a particularity of the JPL GNSS time series)
ind = np.where(np.diff(r.t) == 0)[0]
r.del_points(ind)

# Plot time series (with X-axis in decimal years)
r.plot(tunit='y')



# 4 - PBO pos format
#-------------------

# Download example file
os.system('wget -c https://geodesy-plotter.ipgp.fr/data/CKSV00TWN/IGS_CKSV00TWN.pos')

# Read time series
# Note: No need to specify units since time and values are given in pytrf's default units (MJD and meters).
r = ts.read('IGS_CKSV00TWN.pos',
            skiprows=37,                                        # Skip header lines
            usecols=(2, 15, 16, 17, 18, 19, 20),                # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'),      # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas
            dims=('East', 'North', 'Up'),                       # Dimension names
            dtrd=1)                                             # Remove (and store) mean linear trend

# Remove points with unusually large sigmas
r.clean_sigmas()

# Plot time series (with X-axis in decimal years)
r.plot(tunit='y')



# 5 - SPOTGINS enu format
#------------------------

# Download example file
os.system('wget -c https://geodesy-plotter.ipgp.fr/data/CKSV00TWN/SPOTGINS_CKSV00TWN.enu')

# Read time series
# Note: No need to specify units since time and values are given in pytrf's default units (MJD and meters).
r = ts.read('SPOTGINS_CKSV00TWN.enu',
            skiprows=20,                                        # Skip header lines
            usecols=range(7),                                   # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'),      # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas
            dims=('East', 'North', 'Up'),                       # Dimension names
            dtrd=1)                                             # Remove (and store) mean linear trend

# Round MJDs to nearest noon
# (For better efficiency in noise modeling, it is preferable, when possible, to get back to regularly spaced dates.)
r.t = np.ceil(r.t) - 0.5

# Consequently set time series integration interval to 1 day
r.T = 1

# Plot time series (with X-axis in decimal years)
r.plot(tunit='y')
