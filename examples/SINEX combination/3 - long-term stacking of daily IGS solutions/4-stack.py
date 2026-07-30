#----------------------------------------------------------------------------------------------------
# This script performs the long-term stacking of the daily sinex objects prepared by 3-preprocess.py.
#
# The script writes the long-term stacked solution in SINEX format, but also writes time series of
# the stacking residuals of each station in the directory "res".
#
# Warning: If you run this script from the same terminal as the previous scripts, do not forget to
#          delete the environment variable "OMP_NUM_THREADS" before, so that the stacking can benefit
#          of the default parallelization of numpy's linear algebra operations.
#           - On Linux and MacOS:
#              $ unset OMP_NUM_THREADS
#              $ python 4-stack.py
#           - On Windows:
#              $ set OMP_NUM_THREADS=
#              $ python 4-stack.py
#
# Requirement: The script uses the GNU command "rm".
#----------------------------------------------------------------------------------------------------



# Imports
import os
from pytrf import date, sinex
from pytrf.io import read_solns
from pytrf.utils import record
from pytrf.snxcmb import combine



# Read master discontinuity list & PSD models
# (Since the stacking example is limited to 52 stations that are all part of the IGSR3 reference frame,
# and to a time period that is covered by the IGSR3 discontinuity list and PSD models, it is fine to
# use as "master" discontinuity list and PSD file those which come with IGSR3. In a real-world stacking
# though, the "master" discontinuity list and PSD file should cover all stations included in the
# stacking, not only those in the reference frame. However, the information for reference frame stations
# in the "master" discontinuity list and PSD file should remain consistent with the information in the
# reference frame discontinuity list and PSD file, in order to ensure a proper alignment of the stacked
# solution to the reference frame.)
solns = read_solns('gen/soln_IGSR3.snx')
psd = sinex.read('gen/psd_IGSR3.snx')

# Reference epoch of the long-term stacked solution
tref = date.from_ymdhms(2018, 7, 1).tsnx()

# Prepare reference frame
ref = sinex.read('gen/IGSR3_2077.ssc', dont_read=['comments', 'metadata'])
ref.trim_solns(tref, solns)             # Only keep in the RF solns that are valid at epoch tref. That's because only one soln per station should be used for the alignment of the long-term solution to the RF.
ref.propagate(tref, keep_vel=True)      # Do not forget to keep station velocities included in the RF!

# Prepare list of inputs for the long-term stacking
inputs = []
for w in range(1930, 2086):
    for d in range(7):
        r = record()
        r.description = 'IGS repro3 daily combined solution'    # Description of input
        r.name = str(w)+str(d)                                  # Short name of input
        r.file = 'pkl-clean/'+r.name+'.pkl'                     # [File with] input sinex object
        r.params = 'RST'                                        # Helmert parameters to estimate between input solution and stacked solution (Rotations, Scale and Translations)
        
        inputs.append(r)

# Call snxcmb.combine()
combsnx = combine(inputs, tref, solns=solns, datum=ref,
                  check_solns=False,                            # No need to re-check solns in the input solutions as this was already done by 3-preprocess.py.
                  set_vel=True,                                 # Yes, station velocities should be estimated!
                  dv_sig=1e-6,                                  # Sigma [m/yr] of the equality constraints between successive station velocities that are not separated by a velocity discontinuity
                  mc_sta='RST', mc_sta_sig=1e-5,                # NNRST constraints with respect to the "full" reference frame (including possible outliers) should be applied to the stacked solution.
                  mc_vel='RST', mc_vel_sig=1e-6,                # NNRST-rate constraints with respect to the "full" reference frame (including possible outliers) should be applied to the stacked solution.
                  reduce_trans=True,                            # Helmert parameters estimated between the input solutions and stacked solution should be reduced (eliminated) from the stacked normal equation. This saves a lot of time!
                  store_inputs=False,                           # Do not store inputs in RAM!
                  clear_neq=False)                              # The unconstrained normal equation should be kept in the stacked sinex object, as it will be reused later.

# Iterative comparison with reference frame
# (The purpose of this comparison is to identify and remove outliers from the reference frame, i.e.,
# stations whose individual coordinates disagree between the stacked solution and reference frame.)
combsnx.compare_iter(ref, 'RST', weighting='identity', thr_raw=4, clean_ref=True)

# Final inversion with NNRST and NNRST-rate constraints applied wrt clean reference frame
combsnx.clear_const()                                   # Clear previously applied constraints
combsnx.add_mc('RST', 'STA', sigma=1e-5, datum=ref)     # Add NNRST constraints wrt clean RF
combsnx.add_mc('RST', 'VEL', sigma=1e-6, datum=ref)     # Add NNRST-rate constraints wrt clean RF
combsnx.add_vc(solns, sigma=1e-6)                       # Add equality constraints between successive station velocities that are not separated by a velocity discontinuity
combsnx.neqinv()                                        # Invert normal equation

# Write long-term stacked solution with and without matrices
print('Write long-term stacked solution...')
combsnx.write('stacking.snx', dont_write=['metadata'])
combsnx.write('stacking.ssc', dont_write=['metadata', 'apriori', 'matrices'])

# Create "res" directory if it doesn't exist already
if not(os.path.isdir('res')):
    os.mkdir('res')

# Clean "res" directory
os.system('rm res/*')

# Write stacking residuals
print('')
print('Write stacking residuals...')

for inp in inputs:
    snx = sinex.load(inp.file, load_mat=False)
    
    for i in snx.ix:
        sta = snx.param[i].code
        t = date.from_tsnx(snx.param[i].tref).mjd
        v = inp.v[i:i+3]
        sv = inp.sv[i:i+3]
        
        with open('res/'+sta+'.res', 'a') as f:
            print('{0:7.1f}   {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f}   {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(t, v, sv), file=f)
