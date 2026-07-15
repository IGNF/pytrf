#----------------------------------------------------------------------------------------------------------------
# This script extracts station position estimates from the daily IGS repro3 SINEX solutions in directory "inputs"
# to form station position time series in directory "crd".
#
# By the way, the script also removes from the daily SINEX solutions the parameters that won't be used in the
# stacking, and dumps the "reduced" sinex objects in pickle files (in the "pkl" directory), so that the original
# SINEX files won't have to be read again by the next scripts.
#
# For efficiency, the script is parallelized over the daily SINEX solutions.
#
# Warning: Expect the script to take at least several minutes, and maybe much longer, depending on the number and
#          performance of available CPUs, as well as on hard disk performance. Reading SINEX files with python is
#          slow...
# 
# Warning: Depending on your system, hard disk performance could be a bottleneck for parallelization. This would
#          manifest by sub-processes waiting for disk access, hence lower than expected CPU usage. If you have
#          the choice, moving the SINEX files to a SSD rather than a HDD may solve this problem. Else, reducing
#          the number of sub-processes to less that the number of available CPUs (see variable "nproc") may also
#          help improve performance.
#
# Requirement: The script uses the GNU commands "rm", "mv" and "sort".
#----------------------------------------------------------------------------------------------------------------



# Imports
import os, glob
import multiprocessing as mp
import numpy as np
from pytrf import date, sinex



# Function to be called for every daily SINEX file
#-------------------------------------------------
def extract(f):
    print('Extract station position estimates from '+f)
    
    # Read SINEX file
    snx = sinex.read(f, dont_read=['comments', 'metadata', 'apriori'])
    
    # Solution epoch
    t = date.from_tsnx(snx.param[0].tref)

    # Delete parameters that shouldn't be stacked
    snx.del_params(['ERP', 'GC'], keep_const=True)
    
    # Only keep stations that will be used in the stacking
    # (For this long-term stacking example to run quickly, it is limited to 52 stations only.)
    snx.keep_sta(stalist)
    
    # Loop over station positions in reduced solution
    for i in snx.ix:
        sta = snx.param[i].code                         # Station 4-char code
        X = snx.x[i:i+3]                                # XYZ station position estimate
        Q = snx.Q[i:i+3,i:i+3]                          # Covariance matrix of XYZ station position estimate
            
        # Update position time series of current station
        with open('crd/'+sta+'.crd', 'a') as f:
            print('{0:7.1f} {1[0]:21.14e} {1[1]:21.14e} {1[2]:21.14e} {2[0][0]:21.14e} {2[0][1]:21.14e} {2[0][2]:21.14e} {2[1][1]:21.14e} {2[1][2]:21.14e} {2[2][2]:21.14e}'.format(t.mjd, X, Q), file=f)
    
    # Dump reduced solution into pickle file
    snx.dump('pkl/{0.week}{0.dow}.pkl'.format(t))



# Start of main code
#-------------------

# Create "crd" and "pkl" directories if they don't exist already
if not(os.path.isdir('crd')):
    os.mkdir('crd')
if not(os.path.isdir('pkl')):
    os.mkdir('pkl')

# Clean "crd" and "pkl" directories
os.system('rm crd/*')
os.system('rm pkl/*')

# List of stations to be considered
# (For this long-term stacking example to run quickly, it is limited to 52 stations only.)
stalist = np.loadtxt('gen/station-list.txt', dtype='O').tolist()

# Number of processes to run in parallel
# (By default: as many processes as there are CPUs on your computer. But you may specify some
# number instead of mp.cpu_count() if you don't want the script to use all available CPUs.)
nproc = mp.cpu_count()

# Call function "extract" in parallel over the input SINEX files
files = np.sort(glob.glob('inputs/*'))
with mp.Pool(nproc) as pool:
    pool.map(extract, files)
    
# Sort station position time series chronologically, as the parallel extraction results in unsorted time series.
print('')
print('Sort station position time series')
files = np.sort(glob.glob('crd/*'))
for f in files:
    os.system('sort -k1,1 {0} > tmp'.format(f))
    os.system('mv tmp {0}'.format(f))
