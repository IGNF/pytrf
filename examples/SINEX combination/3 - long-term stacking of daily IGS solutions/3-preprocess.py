#----------------------------------------------------------------------------------------------------------------
# This script prepares the inputs for the long-term stacking perfomed by "4-stack.py".
#
# Each of the sinex objects in the daily pickle files prepared by "1-extract-crd.py" is processed as follows:
#  - Solns are made consistent with the "master" discontinuity list, so that they don't have to be checked again
#    during the stacking.
#  - Station position outliers previously identified by "2-fit-ts.py" are reduced.
#  - Post-seismic deformation models are subtracted from station positions, so that purely piecewise linear
#    trajectory models can be adjusted during the stacking.
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

    # Invert covariance matrix to save time during stacking
    snx.N = invspd(snx.Q)

    # Overwrite pickle file with preprocessed sinex object
    snx.dump('pkl-clean/'+os.path.basename(f))



# Start of main code
#-------------------

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
