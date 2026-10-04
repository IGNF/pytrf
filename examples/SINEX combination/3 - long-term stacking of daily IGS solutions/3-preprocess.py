#----------------------------------------------------------------------------------------------------------------
# This script prepares the inputs for the long-term stacking perfomed by "4-stack.py".
#
# Each of the sinex objects in the daily pickle files prepared by "1-extract-crd.py" is processed as follows:
#  - Solns are made consistent with the "master" discontinuity list, so that they don't have to be checked again
#    during the stacking.
#  - Station position outliers previously identified by "2-fit-ts.py" are reduced.
#  - Post-seismic deformation models are subtracted from station positions, so that purely piecewise linear
#    trajectory models can be adjusted during the stacking.
#  - Periodic signals are subtracted from station positions.
#  - The inverse of the covariance matrix is computed and stored in order to save time during the stacking.
#
# The preprocessed sinex objects, which will serve as inputs to snxcmb.combine() in "4-stack.py", are then dumped
# in pickle files in the "pkl-clean" directory.
#
# For efficiency, the script is parallelized over the daily pickle files.
#
# Warning: For the parallelization over daily pickle files to be efficient, each process should use a single CPU.
#          However, the numpy linear algebra operations (here: inversion of covariance matrix) are parallelized
#          by default in most installations. Before calling this script numpy should therefore be told to use a
#          single CPU per process, by setting the environment variable "OMP_NUM_THREADS" to "1".
#           - On Linux and MacOS:
#              $ export OMP_NUM_THREADS=1
#              $ python 3-preprocess.py
#           - On Windows:
#              $ set OMP_NUM_THREADS=1
#              $ python 3-preprocess.py
#
# Requirement: The script uses the GNU command "rm".
#----------------------------------------------------------------------------------------------------------------



# Imports
import os, glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pytrf import date, sinex
from pytrf.io import read_solns
from pytrf.math import invspd



# Function to be called for every daily pickle file
#--------------------------------------------------
def preprocess(f):
    print('Preprocess '+f)

    # Load sinex object from pickle file
    snx = sinex.load(f)
    
    # Solution epoch
    t = date.from_tsnx(snx.param[0].tref)

    # Check solution numbers
    snx.check_solns(solns, quiet=True)

    # Delete previously identified station position outliers
    if (os.path.isfile('del/{0.week}{0.dow}.del'.format(t))):
        stadel = np.loadtxt('del/{0.week}{0.dow}.del'.format(t), dtype='O', ndmin=1)
        snx.del_sta(stadel)

    # Remove PSD models
    snx.add_psd(psd, remove=True, update_cov=False)

    # Remove periodic signals using coefficients saved by model.fit() in the per_coeffs/ directory
    snx.remove_periodic('per_coeffs')

    # Invert covariance matrix to save time during stacking
    snx.N = invspd(snx.Q)

    # Overwrite pickle file with preprocessed sinex object
    snx.dump('pkl-clean/'+os.path.basename(f))



# Start of main code
#-------------------

# Create "pkl-clean" directory if it doesn't exist already
if not(os.path.isdir('pkl-clean')):
    os.mkdir('pkl-clean')

# Clean the "pkl-clean" directory
os.system('rm pkl-clean/*')

# Read master discontinuity list & PSD models
solns = read_solns('gen/soln_IGSR3.snx')
psd = sinex.read('gen/psd_IGSR3.snx')

# Number of processes to run in parallel
# (By default: as many processes as there are CPUs on your computer. But you may specify some
# number instead of mp.cpu_count() if you don't want the script to use all available CPUs.)
nproc = mp.cpu_count()

# Call function "preprocess" in parallel over daily pickle files prepared by "1-extract-crd.py"
files = np.sort(glob.glob('pkl/*.pkl'))
with mp.Pool(nproc) as pool:
    pool.map(preprocess, files)



# Final spectral analysis after corrections
#------------------------------------------

# (This part can be omitted and run only once as a test to check that periodic signal removal from the SINEX objects works).
# At this point, the SINEX objects have been corrected for outliers, PSD and periodic signals. Their time series can
# therefore be considered as the final ones, representing the observations to be stacked. Therefore, if the model is
# fitted without periodic terms for each station to obtain the amplitude spectrum of the time-series residuals, there
# should be no spectral peaks close to 365.25 days and 182.625 days, i.e. no annual and semi-annual periodic content.
# These final spectrum plots should be similar to those obtained in the previous step 2-fit-ts.py ('+sta+'-clean-spectrum.png),
# where the model was fitted with annual and semi-annual periodic terms.

from pytrf.ts import ts, model

if not(os.path.isdir('crd-clean')):
    os.mkdir('crd-clean')
os.system('rm crd-clean/*')

# Extract station time series from preprocessed SINEX pickle files
files = np.sort(glob.glob('pkl-clean/*.pkl'))

for f in files:
    snx = sinex.load(f)
    t = date.from_tsnx(snx.param[0].tref)
    
    # Loop over station positions
    for i in snx.ix:
        sta = snx.param[i].code
        X = snx.x[i:i+3]
        Q = snx.Q[i:i+3, i:i+3]
        
        # Update position time series
        with open('crd-clean/'+sta+'.crd', 'a') as fp:
            print(
                '{0:7.1f} {1[0]:21.14e} {1[1]:21.14e} {1[2]:21.14e} '
                '{2[0][0]:21.14e} {2[0][1]:21.14e} {2[0][2]:21.14e} '
                '{2[1][1]:21.14e} {2[1][2]:21.14e} {2[2][2]:21.14e}'
                .format(t.mjd, X, Q),
                file=fp
            )

# Sort station time series
files = np.sort(glob.glob('crd-clean/*'))

for f in files:
    os.system('sort -k1,1 {0} > tmp'.format(f))
    os.system('mv tmp {0}'.format(f))

# Fit 'without periodic' terms and plot final spectrum
for f in files:
    sta = os.path.basename(f)[:4]
    print('Read time series '+f)
    r = ts.read(
        f,
        format=('t', 'x', 'y', 'z', 'qx', 'qxy', 'qxz', 'qy', 'qyz', 'qz'),
        dtrd=1,
        rotate=True
    )

    m = model.from_solns(r, solns, code=sta, noise=['vw'])
    m.fit(quiet=True)
    m.plot_spectrum(output='fig/'+sta+'-final-spectrum.png', report=False)
