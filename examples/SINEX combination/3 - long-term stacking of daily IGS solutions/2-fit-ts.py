#-------------------------------------------------------------------------------------------------------
# This script adjusts a trajectory model to each of the station position time series in directory "crd".
# The main purpose of these adjustments is to identify outliers in the station position time series, and
# report them in the daily outlier lists contained in the "del" directory. These outliers will then be
# removed from the daily input solutions to the long-term stacking by the script "3-preprocess.py", so
# that the long-term stacking itself can be run only once (i.e., without having to iteratively identify
# and remove outliers from the daily input solutions).
#
# The trajectory model adjusted to each station position time series is composed of:
#  - a linear trend,
#  - offsets and velocity changes as indicated in the "master" discontinuity list (soln_IGSR3.snx),
#  - post-seismic deformation models, fixed to those given in the "master" PSD file (psd_IGSR3.snx),
#  - annual and semi-annual sine waves,
#  - variable white noise only (which is sufficient for the purpose of outlier identification).
#
# Since the stacking example is limited to 52 stations that are all part of the IGSR3 reference frame,
# and to a time period that is covered by the IGSR3 discontinuity list and PSD models, it is fine to
# use as "master" discontinuity list and PSD file those which come with IGSR3. In a real-world stacking
# though, the "master" discontinuity list and PSD file should cover all stations included in the
# stacking, not only those in the reference frame. However, the information for reference frame stations
# in the "master" discontinuity list and PSD file should remain consistent with the information in the
# reference frame discontinuity list and PSD file, in order to ensure a proper alignment of the stacked
# solution to the reference frame.
#
# In addition to outlier lists, the script also produces figures showing the adjusted trajectory models
# and their residuals in the "fig" directory. They have little use in this example, in which we have a
# complete discontinuity list and PSD file available a priori. But in a real-world stacking, those
# figures could be used to identify missing discontinuities and PSD models in the "master" files. This
# script would then have to be re-run after the "master" discontinuity list and PSD file have been
# updated.
#
# For efficiency, this script is parallelized over the different stations.
#
# Warning: For the parallelization over stations to be efficient, each process should use a single CPU.
#          However, the numpy linear algebra operations performed by model.fit_iter() are parallelized
#          by default in most installations. Before calling this script numpy should therefore be told
#          to use a single CPU per process, by setting the environment variable "OMP_NUM_THREADS" to "1"
#           - On Linux and MacOS:
#              $ export OMP_NUM_THREADS=1
#              $ python 2-fit-ts.py
#           - On Windows:
#              $ set OMP_NUM_THREADS=1
#              $ python 2-fit-ts.py
#
# Requirement: The script uses the GNU command "rm".
#-------------------------------------------------------------------------------------------------------



# Imports
import os, glob
import multiprocessing as mp
import numpy as np
from pytrf import date, sinex
from pytrf.ts import ts, model
from pytrf.io import read_solns



# Function to be called for every station position time series
#-------------------------------------------------------------
def fit(f):
    print('Fit trajectory model to '+f)
    
    # Station name
    sta = os.path.basename(f)[:4]
    
    # Read time series
    r = ts.read(f, format=('t', 'x', 'y', 'z', 'qx', 'qxy', 'qxz', 'qy', 'qyz', 'qz'), dtrd=1, rotate=True)
    
    # Flag points with abnormally large formal errors as outliers
    r.clean_sigmas()
    
    # Intialize model from master discontinuity list and PSD SINEX file
    # Note the "fix_amp=True" and "fix_tau=True" arguments which mean that the
    # PSD models will be fixed to those in the PSD SINEX file, not re-adjusted.
    m = model.from_solns(r, solns, code=sta, per=[365.25,182.625], noise=['vw'], psd=psd, fix_amp=True, fix_tau=True)
    
    # Fit model without rejecting any outliers at first and draw "raw" figures
    # (In a real-world stacking, these "raw" figures would be the ones to look at in order to identify
    # possible missing  discontinuities and PSD models in the "master" files, rather than the "clean"
    # figures drawn after outlier rejection. The points after a missing discontinuity may indeed be
    # rejected as outliers, hence absent from the "clean" figures.)
    m.fit(quiet=True)
    m.plot_fit(tunit='y', output='fig/'+sta+'-raw-fit.png')
    m.plot_res(tunit='y', output='fig/'+sta+'-raw-res.png')

    # Iteratively fit model and reject outliers until there are no more normalized residuals larger than 5
    m.fit_iter(thr_norm=5, quiet=True)
    
    # Add another rejection criteria based on raw residuals
    # (At each iteration, a running median and running MAD of the residuals are computed over 1-year long windows.
    # Any residual falling out of the median +/- 5 * MAD envelope is flagged as an outlier.)
    m.fit_iter(thr_norm=5, thr_mad=5, win_mad=365.25, quiet=True)

    # Draw "clean" figures
    m.plot_fit(tunit='y', output='fig/'+sta+'-clean-fit.png')
    m.plot_res(tunit='y', output='fig/'+sta+'-clean-res.png')

    # Update daily outlier lists
    for i in range(len(r.tdel)):
        t = date.from_mjd(r.tdel[i])
        with open('del/{0.week}{0.dow}.del'.format(t), 'a') as fdel:
            print(sta, file=fdel)



# Start of main code
#-------------------

# Create "del" and "fig" directories if they don't exist already
if not(os.path.isdir('del')):
    os.mkdir('del')
if not(os.path.isdir('fig')):
    os.mkdir('fig')

# Clean "del" and "fig" directories
os.system('rm del/*')
os.system('rm fig/*')

# Read master discontinuity list & PSD models
solns = read_solns('gen/soln_IGSR3.snx')
psd = sinex.read('gen/psd_IGSR3.snx')

# Number of processes to run in parallel
# (By default: as many processes as there are CPUs on your computer. But you may specify some
# number instead of mp.cpu_count() if you don't want the script to use all available CPUs.)
nproc = mp.cpu_count()

# Call function "fit" in parallel over the different station position time series
files = np.sort(glob.glob('crd/*'))
with mp.Pool(nproc) as pool:
    pool.map(fit, files)
