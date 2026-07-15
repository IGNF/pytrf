#-------------------------------------------------------------------------------------------------------------------
# This script performs a combination of the daily IGS repro3 SINEX solutions of a given week into a weekly solution.
#-------------------------------------------------------------------------------------------------------------------



# Imports
import numpy as np
from pytrf import date, sinex
from pytrf.io import read_domes, read_solns, read_yaml
from pytrf.snxcmb import combine_iter
from pytrf.utils import record



# Generic stuff
#--------------

# Date of the combination
w = 2124                        # GPS week
t = date.from_wd(w, 3.5)

# Start, middle and end of the week in SINEX date format
tstart = date.from_wd(w, 0).tsnx()
tmid = t.tsnx()
tend = date.from_wd(w+1, 0).tsnx()

# Read DOMES number catalogue
domes = read_domes('gen/codomes_gps_coord.snx')



# Prepare reference frame
#------------------------

# Read list of discontinuities that comes with the reference frame
solns = read_solns('gen/soln_IGSR3.snx')

# Read post-seismic deformation models that come with the reference frame
psd = sinex.read('gen/psd_IGSR3.snx')

# Read reference frame into a sinex object, omitting blocks that are not needed in order to gain speed
ref = sinex.read('gen/IGSR3_2077.ssc', dont_read=['comments', 'metadata'])

# Remove from reference frame the 'solns' that are not relevant at date t
ref.trim_solns(tmid, solns)

# Compute station positions at date t from positions at t0 + velocities
ref.propagate(tmid)

# Add post-seismic deformation models
ref.add_psd(psd)



# Combine daily solutions into weekly solutions
#----------------------------------------------

# Prepare snxcmb.combine_iter() input list
print('Prepare inputs...')

inp = []
for d in range(7):
    t = date.from_wd(w, d+0.5)
    
    r = record()
    r.description = 'IGS daily combined SINEX solution'                                 # Description of input
    r.name = str(d)                                                                     # Short name of input
    f = 'inputs/IGS1R03SNX_{0.yyyy}{0.doy}0000_01D_01D_SOL.SNX.gz'.format(t)
    print(' - Reading '+f)
    r.snx = sinex.read(f, dont_read=['comments', 'metadata'])                           # Input sinex object
    r.params = 'RST'                                                                    # Helmert parameters to estimate between input solution and combined solution (Rotations, Scale and Translations)
    r.sf = 1                                                                            # A priori scaling factor (1 / sqrt(weight))
    
    inp.append(r)

# Open log file
log = open('weekly-combination.log', 'w')

# Iterative combinations
print('Iterative combination...')

combsnx = combine_iter(inp, tmid, solns=solns,
                       check_solns=True,                            # Yes, solns in the input solutions should be checked, as this was not done before.
                       stack_gc=True,                               # Yes, the daily geocenter coordinates in the input solutions should be stacked into single weekly geocenter coordinates.
                       datum=ref, mc_sta='RST', mc_sta_sig=1e-5,    # NNRST constraints with respect to the "full" reference frame (including possible outliers) should be applied to the combined solution.
                       mc_sta_thr=2,                                # Stations with "poorly" determined coordinates in the combined solution should be excluded from the application of the NNRST constraints.
                       update_sf=False,                             # No need to update the scaling factors (1 / sqrt(weight)) of the individual input solutions by VCE. A single global variance factor is enough.
                       reduce_trans=True,                           # Helmert parameters estimated between the input solutions and combined solution should be reduced (eliminated) from the combined normal equation.
                       thr_norm=5, flag_once=True,                  # At each iteration, stations with normalized residuals larger than 5 should be excluded from the corresponding input solutions. However, a given station should not be excluded from more than one input solution at a time.
                       clear_neq=False,                             # The unconstrained normal equation should be kept in the combined sinex object, as it will be reused later.
                       out=log)

# Post-process combined solution
print('Post-process combined solution...')

# Iterative Helmert comparison between the combined solution and the reference frame
# (The purpose of this comparison is to identify and remove outliers from the reference frame, i.e.,
# stations whose individual coordinates disagree between the combined solution and reference frame.)
combsnx.compare_iter(ref, 'RST', weighting='identity', norm_res='approx', thr_raw=3, clean_ref=True, out=log)

# Final inversion of the combined normal equation with NNRST constraints with respect to "clean" reference frame
combsnx.clear_const()
combsnx.add_mc('RST', 'STA', sigma=1e-5, datum=ref, thr=2)
combsnx.neqinv()

# Write combined solution with and without matrices
print('Write combined solution...')
combsnx.write('weekly-combination.snx'.format(w), dont_write=['metadata'])
combsnx.write('weekly-combination.ssc'.format(w), dont_write=['metadata', 'apriori', 'matrices'])

# Close log file
log.close()
