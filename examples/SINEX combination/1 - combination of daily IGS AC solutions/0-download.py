#--------------------------------------------------------------------------------------------------------------------------------
# This script downloads the files necessary for the combination example.
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#
# Requirement: The script downloads files from CDDIS. For that purpose, you need a ~/.netrc file as explained here:
#              https://cddis.nasa.gov/Data_and_Derived_Products/CreateNetrcFile.html
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
from pytrf import date
from pytrf.io import read_yaml



# Create "gen" and "inputs" directories if they don't exist already
if not(os.path.isdir('gen')):
    os.mkdir('gen')
if not(os.path.isdir('inputs')):
    os.mkdir('inputs')
    
# Download general files (DOMES number catalogue and files associated with the IGSR3 reference frame)
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/DOMES/codomes_gps_coord.snx') # DOMES number catalogue
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/IGSR3_2077.ssc')        # IGSR3 SINEX file (w/o covariance matrix)
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/soln_IGSR3.snx')        # IGSR3 discontinuity list
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/psd_IGSR3.snx')         # IGSR3 post-seismic deformation models

# Read list of combination inputs
w = 2124                                                                     # GPS week
d = 0                                                                        # Day of week
t = date.from_wd(w, d+0.5)                                                   # pytrf date object
inputs = read_yaml('inputs.yml', sed=True, t=t)                              # List of combination inputs

# Download combination inputs (daily repro3 SINEX solutions from different IGS ACs)
for ac in inputs:
    f = os.path.basename(ac.file)
    os.system('wget --auth-no-challenge -P inputs https://cddis.nasa.gov/archive/gnss/products/{0:04d}/repro3/{1}'.format(w, f))
