#-----------------------------------------------------------------------------------------------
# This script performs a combination of the daily repro3 SINEX solutions from different IGS ACs.
#-----------------------------------------------------------------------------------------------



# Imports
import os
import numpy as np
from pytrf import date, sinex
from pytrf.io import read_domes, read_solns, read_yaml
from pytrf.snxcmb import combine_iter



# Generic stuff
#--------------

# Date of the combination
w = 2124                        # GPS week
d = 0                           # Day of week
t = date.from_wd(w, d+0.5)

# Start, middle and end of the day in SINEX date format
tstart = date.from_wd(w, d).tsnx()
tmid = t.tsnx()
tend = date.from_wd(w, d+1).tsnx()

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



# Prepare inputs of the combination
#----------------------------------

print('Prepare inputs...')

# Read list of inputs
inputs = read_yaml('inputs.yml', sed=True, t=t)

# Loop over inputs
for ac in inputs:
    print(' - Pre-process {0}...'.format(ac.file))
    
    # Open log file
    log = open('log/prep-{0}.log'.format(ac.name), 'w')
    
    # Read SINEX file
    ac.snx = sinex.read(ac.file, dont_read=['comments'])
    
    # Check DOMES numbers and solns
    ac.snx.check_staid(domes, out=log)
    ac.snx.check_solns(solns, out=log)
    
    # Check parameter reference epochs & SOLUTION/EPOCHS block
    ac.snx.check_epochs(tstart, tend, tmid)
    
    # If needed, i.e., if the AC SINEX file contains a (constrained) solution,
    # recover the unconstrained normal equation.
    if (ac.snx.N is None):
        ac.snx.unconstrain()
        
    # Else, i.e., if the AC SINEX file readily contains an unconstrained normal equation,
    # just clear possible information about the constraints applied by the AC.
    else:
        ac.snx.clear_const()
        
    # Fix UT1-UTC, satellite PCOs and geocenter coordinates to their a priori values 
    ac.snx.fix_params(['UT', 'SATA', 'GC'])
    
    # Remove (eliminate) reference frame information from normal equation
    # (In this example, we do not wish, for simplicity, to combine the origin and scale information
    # from the different AC solutions. The purpose is only to obtain a combined solution that is
    # aligned in origin, scale and orientation to the reference frame. Hence, we delete the origin 'T'
    # and scale 'S' information contained in the AC normal equations. In the case of certain ACs, who
    # apply no-net-rotation constraints to their solutions, but do not report these constraints in 
    # their SINEX files, it is additionally necessary to delete the orientation 'R' information
    # contained in their normal equations. See key 'del_rot' in inputs.yml.)
    if (ac.del_rot):
        ac.snx.del_helmerts('RST', 'STA')
    else:
        ac.snx.del_helmerts('ST', 'STA')
    
    # First inversion of normal equation with no-net-rotation, -scale and -translation (NNRST) constraints
    # with respect to the "full" reference frame (including possible outliers)
    ac.snx.add_mc('RST', 'STA', sigma='auto', datum=ref, thr=2)
    ac.snx.neqinv(clear_neq=False)
    
    # Iterative Helmert comparison between AC solution and reference frame
    # (The purpose of this comparison is to identify and remove outliers from the reference frame, i.e.,
    # stations whose individual coordinates disagree between the AC solution and reference frame.)
    refc = ref.copy()
    ac.snx.compare_iter(refc, 'RST', weighting='identity', norm_res='approx', thr_raw=3, clean_ref=True, out=log)

    # Final inversion of normal equation with NNRST constraints with respect to "clean" reference frame
    ac.snx.clear_const()
    ac.snx.add_mc('RST', 'STA', sigma='auto', datum=refc, thr=2)
    ac.snx.neqinv(clear_neq=False)
    
    # Store number of RF stations used for the alignment of the AC solution
    ac.nrf = len(np.nonzero(ac.snx.v[ac.snx.ix])[0])
    
    # Assign a priori scaling factor for the combination
    # (The a priori scaling factor of each AC solution is determined so that the median of the 3D station
    # position formal errors will be 4 mm in the rescaled AC solution. This choice ensures a priori scaling
    # factors that are already close to the optimal scaling factors, hence a limited number of variance
    # component estimation (VCE) iterations during the combination.)
    senh = ac.snx.get_sigenh()                  # East, North, Up station position formal errors
    s3d = np.sqrt(np.sum(senh**2, axis=1))      # 3D station position formal errors
    ac.sf = 0.004 / np.median(s3d)              # A priori scaling factor (1 / sqrt(weight))
    
    # Close log file
    log.close()
    
    

# Combination
#------------

# Open log file
log = open('log/daily-combination.log', 'w')

# Iterative combination
print('Iterative combination...')
combsnx = combine_iter(inputs, tmid, solns=solns,
                       check_solns=False,                           # No need to re-check solns in the input solutions as this was already done before.
                       datum=ref, mc_sta='RST', mc_sta_sig=1e-5,    # NNRST constraints with respect to the "full" reference frame (including possible outliers) should be applied to the combined solution.
                       mc_sta_thr=2,                                # Stations with "poorly" determined coordinates in the combined solution should be excluded from the application of the NNRST constraints.
                       update_sf=True,                              # Yes, the scaling factors (1 / sqrt(weights)) of the input solutions should be iteratively updated by VCE until convergence.
                       thr_norm=5, flag_once=True,                  # At each iteration, stations with normalized residuals larger than 5 should be excluded from the corresponding input solutions. However, a given station should not be excluded from more than one input solution at a time.
                       clear_neq=False,                             # The unconstrained normal equation should be kept in the combined sinex object, as it will be reused later.
                       out=log)

# Post-process combined solution
print('Post-process combined solution...')

# Remove from combined sinex object the transformation parameters estimated between the input solution and the combined solution.
combsnx.del_params(['TRANS'])

# Iterative Helmert comparison between the combined solution and the reference frame
# (The purpose of this comparison is to identify and remove outliers from the reference frame, i.e.,
# stations whose individual coordinates disagree between the combined solution and reference frame.)
combsnx.compare_iter(ref, 'RST', weighting='identity', norm_res='approx', thr_raw=3, clean_ref=True, out=log)

# Final inversion of the combined normal equation with NNRST constraints with respect to "clean" reference frame
combsnx.clear_const()                                           # Clear previously applied constraints
combsnx.add_mc('RST', 'STA', sigma=1e-5, datum=ref, thr=2)      # Add NNRST constraints wrt "clean" reference frame
combsnx.neqinv()                                                # Invert normal equation

# Store number of RF stations used for the alignment of the combined solution
nrf = len(np.nonzero(combsnx.v[combsnx.ix])[0])

# Write combined solution with and without matrices
print('Write combined solution...')
combsnx.write('daily-combination.snx'.format(w, d), dont_write=['metadata'])
combsnx.write('daily-combination.ssc'.format(w, d), dont_write=['metadata', 'apriori', 'matrices'])
    
# Close log file
log.close()



# Draw maps of combination residuals
#-----------------------------------

print('Draw maps of combination residuals...')

for ac in inputs:
    ac.snx.map_res(ac.v, title='"{0} - combined" residuals'.format(ac.name), output='maps/{0}.png'.format(ac.name))
    
    
    
# Print main combination statistics
#----------------------------------

print('Print main combination statistics...')

# Open output file
with open('daily-combination.sum', 'w') as fsum:
    
    # Header
    print('------------------------------------------------------------', file=fsum)
    print('Combination of daily AC SINEX solutions for week {0}, day {1}'.format(w, d), file=fsum)
    print('------------------------------------------------------------', file=fsum)
    print('', file=fsum)
    print(' Daily AC solutions:', file=fsum)
    for ac in inputs:
        print('  - {0} = {1}'.format(ac.name, os.path.basename(ac.file)[:-3]), file=fsum)
    print('', file=fsum)
    print(' Daily combined solution:', file=fsum)
    print('  - cmb = daily-combination.snx', file=fsum)
    
    # Main statistics header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' Main combination statistics:', file=fsum)
    print(' ----------------------------', file=fsum)
    print('  - #sta   = number of stations (after rejection of outliers)', file=fsum)
    print('  - #RF    = number of usable reference frame stations', file=fsum)
    print('  - VF^0.5 = square root of estimated variance factor', file=fsum)
    print('  - WRMS   = WRMS of "AC - cmb" station position residuals', file=fsum)
    print('  - sigm   = median of station position formal errors', file=fsum)
    print('', file=fsum)
    print('                              _____________WRMS____________ _____________sigm____________', file=fsum)
    print(' AC     #sta  #RF    VF^0.5      E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm] ', file=fsum)
    print(' ----------------------------------------------------------------------------------------', file=fsum)

    # Main statistics
    for ac in inputs:
        print(' {0.name}   {1:5d} {0.nrf:5d} {0.sf:10.6f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac, len(ac.snx.sta), ac.wrms, ac.sigm), file=fsum)
    print(' cmb   {0:5d} {1:5d}'.format(len(combsnx.sta), nrf), file=fsum)
    print(' ----------------------------------------------------------------------------------------', file=fsum)
    



# Print combination residuals
#----------------------------

print('Print combination residuals...')

# Open output file
with open('daily-combination.res', 'w') as fres:

    # Header
    print('---------------------------------------------------------------------------', file=fres)
    print('Residuals from combination of daily AC SINEX solutions for week {0}, day {1}'.format(w, d), file=fres)
    print('---------------------------------------------------------------------------', file=fres)
    print('', file=fres)
    print(' Daily AC solutions:', file=fres)
    for ac in inputs:
        print('  - {0} = {1}'.format(ac.name, os.path.basename(ac.file)[:-3]), file=fres)
    print('', file=fres)
    print(' residual = AC estimate - combined estimate - 7-parameter transfo', file=fres)
    print(' sigma    = formal error of AC estimate', file=fres)
    print('', file=fres)

    # Station position residuals header
    print('', file=fres)
    print('', file=fres)
    print(' 1) Station position residuals:', file=fres)
    print(' ------------------------------', file=fres)
    print('', file=fres)
    print('            ___________residual__________ _____________sigma___________', file=fres)
    print(' sta    AC     E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm] ', file=fres)
    print(' ----------------------------------------------------------------------', file=fres)

    # Station position residuals
    for i in combsnx.ix:
        code = combsnx.param[i].code
        for ac in inputs:
            if (code in [p.code for p in [ac.snx.param[k] for k in ac.snx.ix]]):
                ix = ac.snx.ix[[p.code for p in [ac.snx.param[k] for k in ac.snx.ix]].index(code)]
                print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3]), file=fres)
        print(' ----------------------------------------------------------------------', file=fres)

    # ERP residuals header
    print('', file=fres)
    print('', file=fres)
    print('', file=fres)
    print(' 2) ERP residuals:', file=fres)
    print(' -----------------', file=fres)
    print('', file=fres)
    print(' param unit  AC   residual    sigma ', file=fres)
    print(' -----------------------------------', file=fres)

    # XPO residuals
    for ac in inputs:
        ix = ac.snx.ixpo[0]
        ac.vxpo = 1000*ac.v[ix]
        ac.svxpo = 1000*ac.sv[ix]
        print(' XPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpo, ac.svxpo), file=fres)
    print(' -----------------------------------', file=fres)

    # YPO residuals
    for ac in inputs:
        ix = ac.snx.iypo[0]
        ac.vypo = 1000*ac.v[ix]
        ac.svypo = 1000*ac.sv[ix]
        print(' YPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypo, ac.svypo), file=fres)
    print(' -----------------------------------', file=fres)

    # XPOR residuals
    for ac in inputs:
        ix = ac.snx.ixpor[0]
        ac.vxpor = 1000*ac.v[ix]
        ac.svxpor = 1000*ac.sv[ix]
        print(' XPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpor, ac.svxpor), file=fres)
    print(' -----------------------------------', file=fres)

    # YPOR residuals
    for ac in inputs:
        ix = ac.snx.iypor[0]
        ac.vypor = 1000*ac.v[ix]
        ac.svypor = 1000*ac.sv[ix]
        print(' YPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypor, ac.svypor), file=fres)
    print(' -----------------------------------', file=fres)

    # LOD residuals
    for ac in inputs:
        ix = ac.snx.ilod[0]
        ac.vlod = 1000*ac.v[ix]
        ac.svlod = 1000*ac.sv[ix]
        print(' LOD   us    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vlod, ac.svlod), file=fres)
    print(' -----------------------------------', file=fres)
