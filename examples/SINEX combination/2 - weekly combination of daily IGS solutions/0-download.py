#--------------------------------------------------------------------------------------------------------------------------------
# This script downloads the files necessary for the weekly combination example.
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
from pytrf import date
from pytrf.io import read_yaml



# Download general files (DOMES number catalogue and files associated with the IGSR3 reference frame)
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/DOMES/codomes_gps_coord.snx') # DOMES number catalogue
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/IGSR3_2077.ssc')        # IGSR3 SINEX file (w/o covariance matrix)
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/soln_IGSR3.snx')        # IGSR3 discontinuity list
os.system('wget -P gen ftp://igs-rf.ign.fr/pub/IGSR3/psd_IGSR3.snx')         # IGSR3 post-seismic deformation models

# Download combination inputs (daily IGS repro3 solutions of GPS week 2124)
w = 2124                                                                     # GPS week
for d in range(7):
    t = date.from_wd(w, d+0.5)
    f = 'IGS1R03SNX_{0.yyyy}{0.doy}0000_01D_01D_SOL.SNX.gz'.format(t)
    os.system('wget -P inputs ftp://igs-rf.ign.fr/pub/repro3/{0}/{1}'.format(w, f))
