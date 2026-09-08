#-------------------------------------------------------------------------------
# Copyright (c) Institut national de l'information géographique et forestière
#
# Main authors:
#  - Paul Rebischung
#  - Julien Barnéoud
#
# This file is part of pytrf: https://github.com/IGNF/pytrf
#
# pytrf is licensed under the MIT license found in the LICENSE.md file
# in the root directory of this source tree.
#-------------------------------------------------------------------------------



"""
    pytrf SINEX combination routines
"""



# External imports
#-----------------
import os
import sys
import warnings
#import mkl
#mkl.set_num_threads(1)
import copy
import pickle
from tqdm import tqdm
import numpy as np
from scipy import sparse, linalg
from math import sqrt, pi, cos, sin
import networkx as nx
from traceback import print_exc

# Internal imports
#-----------------
from pytrf import date, sinex
from pytrf.const import mas2rad, dera_dt
from pytrf.io import read_yaml, read_solns
from pytrf.math import invspd, pinvspd, trdot, xyz2enh
from pytrf.utils import record, earlier
from pytrf.config import get_agency



# Read and pre-process input solution
#------------------------------------
def read_input(sol, tref, solns=None, check_solns=True, psd=None, stack_gc=False, stack_sc=False, load_mat=True):

    """
    Combination of SINEX solutions

    Returns
    -------
    combsnx : sinex instance
        Combined SINEX solution

    Parameters
    ----------
    sol : record instance
        One of the inputs of the combination
    tref : str
        Reference date (in SINEX format)
    solns : str or list, optional
        [File containing] discontinuity list (soln.snx). Default is None.
    check_solns : bool, optional
        Whether solution numbers should be checked in input solutions or not. Default is True.
        To save time, check solution numbers in input solutions before combination.
    psd : str or sinex object, optional
        sinex instance with post-seismic deformation models to be removed from input solutions
        before combination. Default is None.
    stack_gc : bool, optional
        Whether successive geocenter coordinates should be stacked into single
        combined geocenter coordinates. Default is False.
    stack_sc : bool, optional
        Whether successive scale factors should be stacked into a single
        combined scale factor. Default is False.
    load_mat : bool, optional
        Whether to load matrices. Default is True.

    """

    # Raise error if input solution has no name
    if not(hasattr(sol, 'name')):
        raise RuntimeError('No name specified for input solution {0}. Please set \'name\' attribute for each input solution.'.format(isol))
    
    # If sinex instance of current solution is not readily available, load it
    if not(hasattr(sol, 'snx')):
        if (hasattr(sol, 'file')):
            sol.snx = sinex.load(sol.file, load_mat)
        else:
            raise RuntimeError('No input specified for solution {0} ({1}). Please set either \'snx\' or \'file\' attribute for each input solution.'.format(isol, sol.name))
    
    # Set default scale factor if needed
    if not(hasattr(sol, 'sf')):
        sol.sf = 1
    
    # Set reference epoch of input solution
    sol.tref = date.from_mjd((date.from_tsnx(sol.snx.start).mjd + date.from_tsnx(sol.snx.end).mjd) / 2).tsnx()
    
    # Check solns if necessary
    if (solns) and (check_solns):
        sol.snx.check_solns(solns, quiet=True)
        
    # Remove PSD models if needed
    if (psd):
        sol.snx.add_psd(psd, remove=True, update_cov=False)

    # In case geocenter coordinates and scale factors should be stacked,
    # change their epochs in input solution
    if (stack_gc):
        for i in sol.snx.igc:
            sol.snx.param[i].tref = tref
            sol.snx.param[i+1].tref = tref
            sol.snx.param[i+2].tref = tref
    if (stack_sc):
        for i in sol.snx.isc:
            sol.snx.param[i].tref = tref
    
    # Delete unsupported parameters
    sol.snx.del_unknown_par()
    
    # Delete specified stations
    if hasattr(sol, 'stadel'):
        sol.snx.del_sta(sol.stadel)



# Combination of SINEX solutions
#-------------------------------
def combine(inputs, tref, solns=None, check_solns=True, psd=None, set_vel=False, set_per=[], stack_gc=False, stack_sc=False, return_neq=False,
            dv_sig=1e-6, dp_sig=1e-6, xconst=None, vconst=None, pconst=None, datum=None, crf_datum=None,
            mc_sta=None, mc_sta_sig=1e-5, mc_sta_thr=None, mc_vel=None, mc_vel_sig=1e-6, mc_vel_thr=None, mc_per=None, mc_per_sig=1e-5, mc_per_thr=None,
            ic_mean=False, ic_mean_sig=1e-5, ic_trend=False, ic_trend_sig=1e-6, ic_per=False, ic_per_sig=1e-5,
            update_sf=False, norm_res='correct', vce='correct', store_inputs=True, reduce_trans=False, clear_neq=True, quiet=False, out=sys.stdout):

    """
    Combination of SINEX solutions

    Returns
    -------
    combsnx : sinex instance
        Combined SINEX solution

    Parameters
    ----------
    inputs : str or list
        [File containing] list of input solutions
    tref : str
        Reference date (in SINEX format)
    solns : str or list, optional
        [File containing] discontinuity list (soln.snx). Default is None.
    check_solns : bool, optional
        Whether solution numbers should be checked in input solutions or not. Default is True.
        To save time, check solution numbers in input solutions before combination.
    psd : str or sinex object, optional
        sinex instance with post-seismic deformation models to be removed from input solutions
        before combination. Default is None.
    set_vel : bool, optional
        Whether velocities should be estimated for all stations. Default is False.
    set_per : list, optional
        List of periods [d] at which periodic station motions should be estimated. Default is [].
    stack_gc : bool, optional
        Whether successive geocenter coordinates should be stacked into single
        combined geocenter coordinates. Default is False.
    stack_sc : bool, optional
        Whether successive scale factors should be stacked into a single
        combined scale factor. Default is False.
    return_neq: bool, optional
        Whether to return unconstrained normal equation, without solving it.
        Default is False.
    dv_sig : float, optional
        Sigma of equality constraints to be applied between successive velocities [m/y].
        Default is 1e-6.
    dp_sig : float, optional
        List of sigmas [m] of equality constraints to be applied between successive periodic
        station motion coefficients at each period in argument "set_per". If a single value
        is provided, it is assumed to apply to ALL periods in argument "set_per". Default is 1e-6.
    xconst : str or list, optional
        [YAML file containing] station position constraints to be applied. Default is None.
    vconst : str or list, optional
        [YAML file containing] station velocity constraints to be applied. Default is None.
    pconst : list, optional
        List of [YAML files containing] periodic station motion constraints to be applied
        for every period in argument "set_per". If a single YAML file, or single set of 
        constraints is provided, it is assumed to apply to ALL periods in "set_per".
        Default is None.
    datum : str or sinex instance, optional
        [File containing] reference TRF solution. Default is None.
    crf_datum : str or sinex instance, optional
        [File containing] reference CRF solution. Default is None.
    mc_sta : str, optional
        String indicating which minimal constraints should be applied to station positions.
        It can be composed of any combination of letters 'T' (translations),
        'S' (scale) and 'R' (rotations). Default is None.
    mc_sta_sig : float or str, optional
        Sigma of minimal constraints to be applied to station positions in m. Default is 1e-5.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    mc_sta_thr : float, optional
        If set, then station positions with large uncertainties will be rejected from the set
        of station positions to which minimal constraints are applied. See sinex.add_mc() for
        detailed explanations. Default is None.
    mc_vel : str, optional
        String indicating which minimal constraints should be applied to station velocities.
        It can be composed of any combination of letters 'T' (translations),
        'S' (scale) and 'R' (rotations). Default is None.
    mc_vel_sig : float, optional
        Sigma of minimal constraints to be applied to station velocities in m/y. Default is 1e-6.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    mc_vel_thr : float, optional
        If set, then station velocities with large uncertainties will be rejected from the set
        of station velocities to which minimal constraints are applied. See sinex.add_mc() for
        detailed explanations. Default is None.
    mc_per : list or str, optional
        List of strings indicating which minimal constraints should be applied to periodic station
        motions for every period in argument "set_per". Each of the strings can be composed of any
        combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations). If a single
        string is provided, it is assumed to apply to ALL periods in argument "set_per".
        Default is None.
    mc_per_sig : list or float, optional
        List of sigmas of minimal constraints to be applied to periodic station motions (in m)
        for every period in argument "set_per". If a single value is provided, it is assumed to
        apply to ALL periods in argument "set_per". "mc_per_sig" may also be set to 'auto', in
        which case adequate sigmas will be automatically set by sinex.add_mc().
        Default is 1e-5.
    mc_per_thr : list or float, optional
        If set, then periodic station motion coefficients with large uncertainties will be rejected
        from the set of periodic station motions to which minimal constraints are applied. See
        sinex.add_mc() for detailed explanations. "mc_per_thr" may be a list of threshold values
        for every period in argument "set_per", or a single value that applied to ALL periods.
        Default is None.
    ic_mean : bool, optional
        Boolean indicating whether zero-mean constraints should be applied to the time series of
        certain types of transformation parameters. Default is False.
        If True, then every input solution in the list "inputs", that should contribute to a 
        zero-mean constraint on some type of transformation parameters, should have an attribute
        "ic_mean" assigned. This attribute may be composed of any combination of the letters 'T'
        (translations), 'S' (scale), 'R' (rotations) and 'A' (CRF rotations) indicating the types
        of transformation parameters for which the input solution should contribute to a zero-mean
        constraint. The input solutions that do not contribute to any zero-mean constraint may
        have no "ic_mean" attribute assigned, or may have an empty string or None as "ic_mean"
        attribute.
    ic_mean_sig : float, optional
        Sigma of the zero-mean constraints to be applied to the time series of transformation
        parameters, in m. Default is 1e-5.
    ic_trend : bool, optional
        Boolean indicating whether zero-trend constraints should be applied to the time series of
        certain types of transformation parameters. Default is False.
        If True, then every input solution in the list "inputs", that should contribute to a 
        zero-trend constraint on some type of transformation parameters, should have an attribute
        "ic_trend" assigned. This attribute may be composed of any combination of the letters 'T'
        (translations), 'S' (scale), 'R' (rotations) and 'A' (CRF rotations) indicating the types
        of transformation parameters for which the input solution should contribute to a zero-trend
        constraint. The input solutions that do not contribute to any zero-trend constraint may
        have no "ic_trend" attribute assigned, or may have an empty string or None as "ic_trend"
        attribute.
    ic_trend_sig : float, optional
        Sigma of the zero-trend constraints to be applied to the time series of transformation
        parameters, in m/y. Default is 1e-6.
    ic_per : list or bool, optional
        List of booleans indicating whether zero-periodic-variation constraints should be applied
        to the time series of certain types of transformation parameters, for every period in
        argument "set_per". If a single boolean value is provided, it is assumed to apply to ALL
        periods. Default is False.
        If True for any period, then every input solution in the list "inputs", that should
        contribute to a zero-periodic-motion constraint on some type of transformation parameters
        at some period, should have an attribute "ic_per" assigned. This attribute should be a
        list of strings, one for every period in argument "set_per". Each string may be composed
        of any combination of the letters 'T' (translations), 'S' (scale), 'R' (rotations) and
        'A' (CRF rotations) indicating the types of transformation parameters for which the input
        solution should contribute to a zero-periodic-variation constraint. If an input solution
        has a single string assigned as "ic_per" attribute, it is assumed to apply to ALL periods.
        The input solutions that do not contribute to any zero-periodic-variation constraint may
        have no "ic_per" attribute assigned, or may have an empty string or None as "ic_per"
        attribute.
    ic_per_sig : float, optional
        List of sigmas of the zero-periodic-variation constraints to be applied to the time series
        of transformation parameters, in m/y, for each period in argument "set_per". If a single
        value is provided, it is assumed to apply to ALL periods in "set_per". Default is 1e-5.
    update_sf : bool, optional
        Whether to update variance factors of input solutions with VCE estimates.
        Default is False.
    norm_res : str, optional
        Keyword indicating how normalized residuals should be computed.
        It can be either:
        - 'correct' in which case residuals are normalized by their own
            standard deviations, or
        - 'approx' in which case residuals are normalized by the
            standard deviations of the observations (i.e., snx.sig).
        Default is 'correct'.
    vce : str, optional
        Keyword indicating how a posteriori variance factors should be computed.
        It can be either:
        - 'correct' in which case Sillard's (1999) degree-of-freedom estimator is used, or
        - 'approx' in which case a faster approximation is used.
        Default is 'correct'.
    store_inputs : bool, optional
        If True, all input solutions will be stored in RAM simultaneously (faster option).
        If False, input solutions are successively read and deleted during the successive
        processing steps (slower, but uses less RAM).
        Default is True.
    reduce_trans : bool, optional
        Whether to reduce transformation parameters. Default is False.
        Note that if transformation parameters are reduced, the options norm_res='correct'
        and vce='correct' become unavailable.
    clear_neq : bool, optional
        Whether normal equation should be kept in combined sinex object. Default is True.
    quiet : bool, optional
        Whether not to print output messages. Default is False.
    out : file-like, optional
        Log file. Default is sys.stdout.
    
    """
    
    # Redirect progress bars to /dev/null if output is printed in a file (not sys.stdout)
    tqdm_out = out
    if (tqdm_out != sys.stdout):
        tqdm_out = open(os.devnull, 'w')
        
    # If needed, change some arguments to lists
    if (set_per):
        if np.isscalar(dp_sig):
            dp_sig = [dp_sig for p in set_per]
        if isinstance(pconst, str) or (isinstance(pconst, list) and not(isinstance(pconst[0], str))):
            pconst = [pconst for p in set_per]
        if (mc_per is None) or isinstance(mc_per, str):
            mc_per = [mc_per for p in set_per]
        if np.isscalar(mc_per_sig):
            mc_per_sig = [mc_per_sig for p in set_per]
        if (mc_per_thr is None) or  np.isscalar(mc_per_thr):
            mc_per_thr = [mc_per_thr for p in set_per]
        if (ic_per is None) or isinstance(ic_per, bool):
            ic_per = [ic_per for p in set_per]
        if np.isscalar(ic_per_sig):
            ic_per_sig = [ic_per_sig for p in set_per]  
    
    # Print header in log file
    if not(quiet):
        print('snxcomb.combine', file=out)
        print('---------------', file=out)

    # Read input file if necessary
    if not(isinstance(inputs, list)):
        inputs = read_yaml(inputs)
        
    # Set possibly missing "params", "params_rate" and "params_per" attributes of input solutions
    # and change "ic_per" attributes to lists whenever needed
    for sol in inputs:
        if not(hasattr(sol, 'params')):
            sol.params = ''
        if not(hasattr(sol, 'params_rate')):
            sol.params_rate = ''
        if not(hasattr(sol, 'params_per')):
            sol.params_per = ''
        if isinstance(sol.params_per, str):
            sol.params_per = [sol.params_per for p in set_per]
    
    # Set possibly missing "ic_mean" attributes of input solutions
    if (ic_mean):
        for sol in inputs:
            if not(hasattr(sol, 'ic_mean')):
                sol.ic_mean = ''
            elif (sol.ic_mean is None):
                sol.ic_mean = ''

    # Set possibly missing "ic_trend" attributes of input solutions
    if (ic_trend):
        for sol in inputs:
            if not(hasattr(sol, 'ic_trend')):
                sol.ic_trend = ''
            elif (sol.ic_trend is None):
                sol.ic_trend = ''
                
    # Set possibly missing "ic_per" attributes of input solutions,
    # and change "ic_per" attributes to lists whenever needed
    if (ic_per):
        for sol in inputs:
            if not(hasattr(sol, 'ic_per')):
                sol.ic_per = ['' for p in set_per]
            elif (sol.ic_per is None):
                sol.ic_per = ['' for p in set_per]
            elif isinstance(sol.ic_per, str):
                sol.ic_per = [sol.ic_per for p in set_per]
                
    # Make some checks if internal constraints should be applied to the combined solution
    if (ic_mean):
        helmerts = list(set(''.join([sol.ic_mean for sol in inputs])))
        if (len(helmerts) == 0):
            ic_mean = False
        for h in helmerts:
            if (mc_sta is not None):
                if (h in mc_sta):
                    raise RuntimeError('Conflict between "minimal" constraints and zero-mean constraint on time series of transformation parameters ({0}).'.format(h))
            for sol in inputs:
                if not(h in sol.params):
                    raise RuntimeError('Zero-mean constraint on time series of some type ({0}) of transformation parameters is not allowed when this type of transformation parameters is not estimated for EVERY input solution.'.format(h))
                
    if (ic_trend):
        helmerts = list(set(''.join([sol.ic_trend for sol in inputs])))
        if (len(helmerts) == 0):
            ic_trend = False
        for h in helmerts:
            if (mc_vel is not None):
                if (h in mc_vel):
                    raise RuntimeError('Conflict between "minimal" constraints and zero-trend constraint on time series of transformation parameters ({0}).'.format(h))
            for sol in inputs:
                if not(h in sol.params):
                    raise RuntimeError('Zero-trend constraint on time series of some type ({0}) of transformation parameters is not allowed when this type of transformation parameters is not estimated for EVERY input solution.'.format(h))
                
    for (i, p) in enumerate(set_per):
        if (ic_per[i]):
            helmerts = list(set(''.join([sol.ic_per[i] for sol in inputs])))
            if (len(helmerts) == 0):
                ic_per[i] = False
            for h in helmerts:
                if (mc_per[i] is not None):
                    if (h in mc_per[i]):
                        raise RuntimeError('Conflict between "minimal" constraints and zero-periodic-variation constraint (at {1:.3f} d) on time series of transformation parameters ({0}).'.format(h, p))
                for sol in inputs:
                    if not(h in sol.params):
                        raise RuntimeError('Zero-periodic-variation (at {1:.3f} d) constraint on time series of some type ({0}) of transformation parameters is not allowed when this type of transformation parameters is not estimated for EVERY input solution.'.format(h, p))
                
    if ((ic_mean) or (ic_trend) or np.any(ic_per)) and (reduce_trans):
        warnings.warn('Transformation parameters cannot be reduced when "internal" constraints are applied. => Parameter "reduce_trans" is forced to False.')
        reduce_trans = False
        
    # Raise warnings if "reduce_trans" is True, but correct VCE and/or correct residuals are requested.
    if (reduce_trans) and (vce == 'correct'):
        warnings.warn('"Correct" VCE is not possible when transformation parameters are reduced. Approximate VCE will be used instead.')
    if (reduce_trans) and (norm_res == 'correct'):
        warnings.warn('The computation of "correct" normalized residuals is not possible when transformation parameters are reduced. Approximate normalized residuals will be computed instead.')
        
    # Read discontinuity file if necessary
    if (solns):
        if not(isinstance(solns, list)):
            solns = read_solns(solns)

    # Read station position constraints if necessary
    if (xconst):
        if not(isinstance(xconst, list)):
            xconst = read_yaml(xconst)

    # Read station velocity constraints if necessary
    if (vconst):
        if not(isinstance(vconst, list)):
            vconst = read_yaml(vconst)
            
    # Read periodic station motion constraints if necessary
    if (pconst):
        for i in range(len(set_per)):
            if not(isinstance(pconst[i], list)):
                pconst[i] = read_yaml(pconst[i])

    # Read datum if necessary
    if (datum):
        if not(isinstance(datum, sinex)):
            try:
                datum = sinex.load(datum, load_mat=False)
            except:
                datum = sinex.read(datum, dont_read=['comments', 'metadata', 'apriori', 'matrices'])

    # Read CRF datum if necessary
    if (crf_datum):
        if not(isinstance(crf_datum, sinex)):
            try:
                crf_datum = sinex.load(crf_datum, load_mat=False)
            except:
                crf_datum = sinex.read(crf_datum, dont_read=['comments', 'metadata', 'apriori', 'matrices'])

    # Initialize combined SINEX solution
    combsnx = sinex()
    combsnx.version = '2.02'
    combsnx.agency = get_agency()
    combsnx.const = 2
    combsnx.input = []
    combsnx.sta = []
    combsnx.rs = []
    combsnx.param = []
    combsnx.x0 = []
    combsnx.codept = []
    combsnx.per = set_per
    combsnx.iper = [[] for p in set_per]

    # Other initializations
    mjd0 = date.from_tsnx(tref).mjd
    nobs = 0



    # 1 - SET UP PARAMETER LIST
    #--------------------------
    
    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Set up parameter list', file=out)



    # Loop over input solutions
    #--------------------------
    
    for isol in tqdm(range(len(inputs)), file=tqdm_out, leave=False):
        sol = inputs[isol]

        # Read input
        read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc, load_mat=store_inputs)
        
        # Shortcut for sol.snx
        snx = sol.snx
        
        # Raise an error if current input solution contains station velocities, but "set_vel" is False.
        if (len(snx.iv) > 0) and not(set_vel):
            raise RuntimeError('Input solution {0} contains station velocities, but "set_vel" is set to False.'.format(sol.file))
        
        # If current input solution includes periodic station motion coefficients,
        # check that all have the "right" reference epoch and that their periods are all in "set_per".
        if (snx.per):
            for (per, ip) in enumerate(snx.per):
                if not(per in set_per):
                    raise RuntimeError('Input solution {0} contains periodic station motions at a period that is not included in "set_per".'.format(sol.file))
                
                for p in [snx.param[i] for i in snx.iper[ip]]:
                    if (p.tref != tref):
                        raise RuntimeError('Input solution {0} contains periodic station motions with a different epoch than the reference epoch {1}.'.format(sol.file, tref))
        
        # Search keys
        snx.codept = [s.code+s.pt for s in snx.sta]

        # Update number of observations
        nobs += snx.npar

        # Update INPUT/HISTORY and INPUT/FILES blocks of combined SINEX solution
        r = record()
        r.version = snx.version
        r.agency = snx.agency
        r.t = snx.t
        r.start = snx.start
        r.end = snx.end
        r.tech = snx.tech
        r.npar = snx.npar
        r.const = snx.const
        r.content = snx.content
        r.file = '{0:<29}'.format(os.path.basename(snx.file)[:29])
        r.description = sol.description[:32]
        combsnx.input.append(r)
        
        # Get indices of common parameters
        (isnx, icmb) = snx.get_common_par(combsnx)

        # And indices of non-common parameters
        jsnx = np.setdiff1d(range(snx.npar), isnx)
        
        
        
        # Update list of stations in combined solution
        #---------------------------------------------
        
        # Loop over common solns to update their observation intervals
        for i in np.intersect1d(isnx, snx.ix):
            p = snx.param[i]
            
            # Get indices of current station
            ista = snx.codept.index(p.code+p.pt)
            icmbsta = combsnx.codept.index(p.code+p.pt)

            # Get indices of current soln
            isoln = [s.soln for s in snx.sta[ista].soln].index(p.soln)
            icmbsoln = [s.soln for s in combsnx.sta[icmbsta].soln].index(p.soln)

            # Update first observation epoch if necessary
            if (earlier(snx.sta[ista].soln[isoln].datastart, combsnx.sta[icmbsta].soln[icmbsoln].datastart)):
                combsnx.sta[icmbsta].soln[icmbsoln].datastart = snx.sta[ista].soln[isoln].datastart

            # Update last observation epoch if necessary
            if (earlier(combsnx.sta[icmbsta].soln[icmbsoln].dataend, snx.sta[ista].soln[isoln].dataend)):
                combsnx.sta[icmbsta].soln[icmbsoln].dataend = snx.sta[ista].soln[isoln].dataend

        # Loop over non-common solns to update combsnx.sta
        for i in np.intersect1d(jsnx, snx.ix):
            p = snx.param[i]
            
            # If current station is already in combsnx.sta (but not current soln)
            if (p.code+p.pt in combsnx.codept):

                # Get indices of current station
                ista = snx.codept.index(p.code+p.pt)
                icmbsta = combsnx.codept.index(p.code+p.pt)

                # Get index of current soln in input solution
                isoln = [s.soln for s in snx.sta[ista].soln].index(p.soln)
                
                # Add new soln into combined solution
                combsnx.sta[icmbsta].soln.append(copy.deepcopy(snx.sta[ista].soln[isoln]))
                
            # Else, current station in not in combsnx.sta yet
            else:
                
                # Get index of current station in input solution
                ista = snx.codept.index(p.code+p.pt)
                
                # Get index of current soln in input solution
                isoln = [s.soln for s in snx.sta[ista].soln].index(p.soln)

                # Add new station into combined solution
                combsnx.sta.append(copy.deepcopy(snx.sta[ista]))
                combsnx.sta[-1].soln = [snx.sta[ista].soln[isoln]]
                combsnx.codept.append(p.code+p.pt)



        # Update list of radiosources in combined solution
        #-------------------------------------------------
        
        # Loop over non-common radiosources to update combsnx.rs
        for i in np.intersect1d(jsnx, snx.irs):
            p = snx.param[i]

            # Get index of current radiosource in input solution
            irs = [r.iers for r in snx.rs].index(p.iers)
            
            # Add new radiosource into combined solution
            combsnx.rs.append(copy.deepcopy(snx.rs[irs]))
            combsnx.rs[-1].code = '{0:>04d}'.format(len(combsnx.rs))



        # Update list of parameters in combined solution
        #-----------------------------------------------
        
        # Loop over non-common station positions
        for i in np.intersect1d(jsnx, snx.ix):

            # Add new STAX parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].tref = tref
            combsnx.param[-1].const = '2'
            
            # Add new STAY parameter into combined solution
            combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
            combsnx.param[-1].type = 'STAY  '
            
            # Add new STAZ parameter into combined solution
            combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
            combsnx.param[-1].type = 'STAZ  '
            
            # Update combsnx.ix and combsnx.x0
            combsnx.ix.append(len(combsnx.param)-3)
            combsnx.x0.extend(snx.x[i:i+3])
            
            # If velocities need to be set up,
            if (set_vel):
                
                # Add new velocity parameters into combined solution
                combsnx.param.extend(copy.deepcopy(combsnx.param[-3:]))
                combsnx.param[-3].type = 'VELX  '
                combsnx.param[-2].type = 'VELY  '
                combsnx.param[-1].type = 'VELZ  '
                combsnx.param[-3].unit = 'm/y '
                combsnx.param[-2].unit = 'm/y '
                combsnx.param[-1].unit = 'm/y '
                
                # Update combsnx.iv and combsnx.x0
                combsnx.iv.append(len(combsnx.param)-3)
                combsnx.x0.extend([0, 0, 0])
                
            # If periodic station motions need to be set up,
            if (set_per):
                for ip in range(len(set_per)):
                
                    # Add new periodic station motion parameters into combined solution
                    combsnx.param.extend(copy.deepcopy(combsnx.param[-3:]) + copy.deepcopy(combsnx.param[-3:]))
                    combsnx.param[-6].type = 'P{0:03d}CX'.format(ip+1)
                    combsnx.param[-5].type = 'P{0:03d}SX'.format(ip+1)
                    combsnx.param[-4].type = 'P{0:03d}CY'.format(ip+1)
                    combsnx.param[-3].type = 'P{0:03d}SY'.format(ip+1)
                    combsnx.param[-2].type = 'P{0:03d}CZ'.format(ip+1)
                    combsnx.param[-1].type = 'P{0:03d}SZ'.format(ip+1)
                    combsnx.param[-6].unit = 'm   '
                    combsnx.param[-5].unit = 'm   '
                    combsnx.param[-4].unit = 'm   '
                    combsnx.param[-3].unit = 'm   '
                    combsnx.param[-2].unit = 'm   '
                    combsnx.param[-1].unit = 'm   '
                    
                    # Update combsnx.iv and combsnx.x0
                    combsnx.iper[ip].append(len(combsnx.param)-6)
                    combsnx.x0.extend([0, 0, 0, 0, 0, 0])

        # Loop over non-common radiosource coordinates
        for i in np.intersect1d(jsnx, snx.irs):

            # Index of current radiosource in combsnx.rs
            irs = [r.iers for r in combsnx.rs].index(snx.param[i].iers)

            # Add new RS_RA parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = combsnx.rs[irs].code
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '----'
            combsnx.param[-1].tref = tref
            combsnx.param[-1].const = '2'

            # Add new RS_DE parameter into combined solution
            combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
            combsnx.param[-1].type = 'RS_DE '

            # Update combsnx.irs and combsnx.x0
            combsnx.irs.append(len(combsnx.param)-2)
            combsnx.x0.extend(snx.x[i:i+2])

        # Loop over non-common X-pole coordinates
        for i in np.intersect1d(jsnx, snx.ixpo):

            # Add new XPO parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.ixpo)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.ixpo and combsnx.x0
            combsnx.ixpo.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common Y-pole coordinates
        for i in np.intersect1d(jsnx, snx.iypo):

            # Add new YPO parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.iypo)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.iypo and combsnx.x0
            combsnx.iypo.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common X-pole rates
        for i in np.intersect1d(jsnx, snx.ixpor):

            # Add new XPOR parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.ixpor)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.ixpor and combsnx.x0
            combsnx.ixpor.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common Y-pole rates
        for i in np.intersect1d(jsnx, snx.iypor):

            # Add new YPOR parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.iypor)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.iypor and combsnx.x0
            combsnx.iypor.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common UT1-UTC offsets
        for i in np.intersect1d(jsnx, snx.iut):

            # Add new UT parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.iut)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.iut and combsnx.x0
            combsnx.iut.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common LODs
        for i in np.intersect1d(jsnx, snx.ilod):

            # Add new LOD parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.ilod)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.ilod and combsnx.x0
            combsnx.ilod.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common X nutations
        for i in np.intersect1d(jsnx, snx.inutx):

            # Add new NUT_X parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.inutx)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.inutx and combsnx.x0
            combsnx.inutx.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common Y nutations
        for i in np.intersect1d(jsnx, snx.inuty):

            # Add new NUT_Y parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.inuty)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.inuty and combsnx.x0
            combsnx.inuty.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common geocenter coordinates
        for i in np.intersect1d(jsnx, snx.igc):

            # Add new XGC parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.igc)+1)
            combsnx.param[-1].const = '2'
            
            # Add new YGC parameter into combined solution
            combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
            combsnx.param[-1].type = 'YGC   '
            
            # Add new ZGC parameter into combined solution
            combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
            combsnx.param[-1].type = 'ZGC   '

            # Update combsnx.igc and combsnx.x0
            combsnx.igc.append(len(combsnx.param)-3)
            combsnx.x0.extend([0, 0, 0])

        # Loop over non-common scale factors
        for i in np.intersect1d(jsnx, snx.isc):

            # Add new DSC parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].code = '----'
            combsnx.param[-1].pt = '--'
            combsnx.param[-1].soln = '{0:>4d}'.format(len(combsnx.isc)+1)
            combsnx.param[-1].const = '2'

            # Update combsnx.isc and combsnx.x0
            combsnx.isc.append(len(combsnx.param)-1)
            combsnx.x0.append(0)

        # Loop over non-common satellite x-PCOs
        for i in np.intersect1d(jsnx, snx.isatax):

            # Add new SATA_X parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].tref = tref
            combsnx.param[-1].const = '2'

            # Update combsnx.isatax and combsnx.x0
            combsnx.isatax.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common satellite y-PCOs
        for i in np.intersect1d(jsnx, snx.isatay):

            # Add new SATA_Y parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].tref = tref
            combsnx.param[-1].const = '2'

            # Update combsnx.isatay and combsnx.x0
            combsnx.isatay.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Loop over non-common satellite z-PCOs
        for i in np.intersect1d(jsnx, snx.isataz):

            # Add new SATA_Z parameter into combined solution
            combsnx.param.append(copy.deepcopy(snx.param[i]))
            combsnx.param[-1].tref = tref
            combsnx.param[-1].const = '2'

            # Update combsnx.isataz and combsnx.x0
            combsnx.isataz.append(len(combsnx.param)-1)
            combsnx.x0.append(snx.x[i])

        # Make room if needed
        if not(store_inputs):
            del sol.snx



    # Add transfomation parameters
    #-----------------------------
    
    if not(reduce_trans):
    
        # Loop over input solutions
        for isol in range(len(inputs)):
            sol = inputs[isol]

            # Translations?
            if ('T' in sol.params):

                # Add new TX parameter into combined solution
                r = record()
                r.type = 'TX    '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'mm  '
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Add new TY parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'TY    '
                
                # Add new TZ parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'TZ    '

                # Update combsnx.x0
                combsnx.x0.extend([0, 0, 0])

            # Scale factor?
            if ('S' in sol.params):

                # Add new SC parameter into combined solution
                r = record()
                r.type = 'SC    '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'ppb '
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Update combsnx.x0
                combsnx.x0.append(0)

            # Rotations?
            if ('R' in sol.params):

                # Add new RX parameter into combined solution
                r = record()
                r.type = 'RX    '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'mas '
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Add new RY parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'RY    '
                
                # Add new RZ parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'RZ    '

                # Update combsnx.x0
                combsnx.x0.extend([0, 0, 0])

            # CRF Rotations?
            if ('A' in sol.params):

                # Add new AX parameter into combined solution
                r = record()
                r.type = 'AX    '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'mas '
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Add new AY parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'AY    '

                # Add new AZ parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'AZ    '

                # Update combsnx.x0
                combsnx.x0.extend([0, 0, 0])
                
            # Translation rates?
            if ('T' in sol.params_rate):

                # Add new dTX parameter into combined solution
                r = record()
                r.type = 'dTX   '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'mm/y'
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Add new dTY parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'dTY   '
                
                # Add new dTZ parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'dTZ   '

                # Update combsnx.x0
                combsnx.x0.extend([0, 0, 0])

            # Scale factor rate?
            if ('S' in sol.params_rate):

                # Add new dSC parameter into combined solution
                r = record()
                r.type = 'dSC   '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'pb/y'
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Update combsnx.x0
                combsnx.x0.append(0)

            # Rotation rates?
            if ('R' in sol.params_rate):

                # Add new dRX parameter into combined solution
                r = record()
                r.type = 'dRX   '
                r.code = '{0:<4}'.format(sol.name)[:4]
                r.pt = '--'
                r.soln = '{0:>4}'.format(isol+1)[-4:]
                r.tref = sol.tref
                r.unit = 'ma/y'
                r.const = 2
                r.isol = isol
                combsnx.param.append(r)

                # Add new dRY parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'dRY   '
                
                # Add new dRZ parameter into combined solution
                combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                combsnx.param[-1].type = 'dRZ   '

                # Update combsnx.x0
                combsnx.x0.extend([0, 0, 0])
                
            # Loop over periods to be included in combined solution
            for (i, p) in enumerate(set_per):
                
                # Periodic translation?
                if ('T' in sol.params_per[i]):

                    # Add new pTX parameter into combined solution
                    r = record()
                    r.type = 'pTX   '
                    r.code = '{0:<4}'.format(sol.name)[:4]
                    r.pt = '{0:>2d}'.format(i+1)[-2:]
                    r.soln = '{0:>4}'.format(isol+1)[-4:]
                    r.tref = sol.tref
                    r.unit = 'mm  '
                    r.const = 2
                    r.isol = isol
                    r.iper = i
                    combsnx.param.append(r)

                    # Add new pTY parameter into combined solution
                    combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                    combsnx.param[-1].type = 'pTY   '
                    
                    # Add new pTZ parameter into combined solution
                    combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                    combsnx.param[-1].type = 'pTZ   '

                    # Update combsnx.x0
                    combsnx.x0.extend([0, 0, 0])

                # Periodic scale factor?
                if ('S' in sol.params_per[i]):

                    # Add new pSC parameter into combined solution
                    r = record()
                    r.type = 'pSC   '
                    r.code = '{0:<4}'.format(sol.name)[:4]
                    r.pt = '{0:>2d}'.format(i+1)[-2:]
                    r.soln = '{0:>4}'.format(isol+1)[-4:]
                    r.tref = sol.tref
                    r.unit = 'ppb '
                    r.const = 2
                    r.isol = isol
                    r.iper = i
                    combsnx.param.append(r)

                    # Update combsnx.x0
                    combsnx.x0.append(0)

                # Periodic rotations?
                if ('R' in sol.params_per[i]):

                    # Add new pRX parameter into combined solution
                    r = record()
                    r.type = 'pRX   '
                    r.code = '{0:<4}'.format(sol.name)[:4]
                    r.pt = '{0:>2d}'.format(i+1)[-2:]
                    r.soln = '{0:>4}'.format(isol+1)[-4:]
                    r.tref = sol.tref
                    r.unit = 'mas '
                    r.const = 2
                    r.isol = isol
                    r.iper = i
                    combsnx.param.append(r)

                    # Add new pRY parameter into combined solution
                    combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                    combsnx.param[-1].type = 'pRY   '
                    
                    # Add new pRZ parameter into combined solution
                    combsnx.param.append(copy.deepcopy(combsnx.param[-1]))
                    combsnx.param[-1].type = 'pRZ   '

                    # Update combsnx.x0
                    combsnx.x0.extend([0, 0, 0])



    # Format combined solution
    #-------------------------
    
    # Set combsnx.start and combsnx.end
    mjd = []
    for sta in combsnx.sta:
        mjd.extend([date.from_tsnx(s.datastart).mjd for s in sta.soln])
    combsnx.start = date.from_mjd(np.min(mjd)).tsnx()

    mjd = []
    for sta in combsnx.sta:
        mjd.extend([date.from_tsnx(s.dataend).mjd for s in sta.soln])
    combsnx.end = date.from_mjd(np.max(mjd)).tsnx()

    # Set combsnx.tech
    techs = [s.tech for s in snx.sta]
    if (len(techs) > 1):
        combsnx.tech = 'C'
    else:
        combsnx.tech = techs[0]

    # Set combsnx.content
    combsnx.content = ''
    if (len(combsnx.ix+combsnx.iv) > 0):
        combsnx.content = combsnx.content + 'S '
    if (len(combsnx.ixpo+combsnx.iypo+combsnx.ixpor+combsnx.iypor+combsnx.iut+combsnx.ilod+combsnx.inutx+combsnx.inuty) > 0):
        combsnx.content = combsnx.content + 'E '
    if (len(combsnx.irs) > 0):
        combsnx.content = combsnx.content + 'C '
    if (len(combsnx.isatax+combsnx.isatay+combsnx.isataz) > 0):
        combsnx.content = combsnx.content + 'A '
    combsnx.content = combsnx.content[:-1]
    
    # Sort combsnx.sta
    ind = np.argsort([s.code+s.pt for s in combsnx.sta])
    combsnx.sta = [combsnx.sta[i] for i in ind]
    
    # Sort combsnx.sta[*].soln
    for sta in combsnx.sta:
        ind = np.argsort([int(s.soln) for s in sta.soln])
        sta.soln = [sta.soln[i] for i in ind]
    
    # Compute mid-observation epoch and observation span of each soln
    for sta in combsnx.sta:
        for soln in sta.soln:
            t1 = date.from_tsnx(soln.datastart).mjd
            t2 = date.from_tsnx(soln.dataend).mjd
            soln.datamean = date.from_mjd((t1+t2)/2).tsnx()

    # Set combsnx.npar, .x0 and .sig0
    combsnx.npar = len(combsnx.param)
    combsnx.x0 = np.array(combsnx.x0)
    combsnx.sig0 = np.zeros(combsnx.npar)

    # Sort parameters
    combsnx.sort_params()
    
    # Reset parameter indices
    combsnx.set_par_ind()

    # Get indices of transformation parameters of each input solution
    keys = np.array([p.isol for p in [combsnx.param[i] for i in combsnx.itrans]])
    for isol in range(len(inputs)):
        ind = np.nonzero(keys == isol)[0]
        inputs[isol].itrans = [combsnx.itrans[i] for i in ind]

    # Get indices of transformation parameter rates of each input solution
    keys = np.array([p.isol for p in [combsnx.param[i] for i in combsnx.idtrans]])
    for isol in range(len(inputs)):
        ind = np.nonzero(keys == isol)[0]
        inputs[isol].idtrans = [combsnx.idtrans[i] for i in ind]

    # Get indices of transformation parameter periodic variations of each input solution
    keys = np.array([p.isol for p in [combsnx.param[i] for i in combsnx.iptrans]])
    keyp = np.array([p.iper for p in [combsnx.param[i] for i in combsnx.iptrans]])
    for isol in range(len(inputs)):
        inputs[isol].iptrans = [[] for iper in range(len(set_per))]
        for iper in range(len(set_per)):
            ind = np.nonzero((keys == isol) * (keyp == iper))[0]
            inputs[isol].iptrans[iper] = [combsnx.iptrans[i] for i in ind]

    # Change a priori coordinates of reference stations
    if (datum):
        combsnx.prior2ref(datum)
        
    # Change a priori coordinates of reference radiosources
    if (crf_datum):
        combsnx.prior2ref(crf_datum)
        
    # Change a priori positions of stations with absolute position constraints, if any
    if (xconst):
        keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.ix]]

        # Loop over specified absolute position constraints
        for xc in xconst:
            if hasattr(xc, 'point'):

                # If specified point actually has an estimated position,
                tab = xc.point.split()
                sta = tab[0] + '{0:>2s}'.format(tab[1]) + '{0:4d}'.format(int(tab[2]))
                if (sta in keys):
                    i = keys.index(sta)

                    # Check that there is no conflict with datum
                    if hasattr(combsnx.param[combsnx.ix[i]], 'xref'):
                        raise RuntimeError('Absolute position constraint not allowed for datum point {0}.'.format(sta))

                    # If not, change a priori station position to the specified value
                    # and assign "reference" values to the corresponding parameters
                    else:
                        combsnx.x0[combsnx.ix[i]:combsnx.ix[i]+3] = xc.xref
                        for k in range(3):
                            combsnx.param[combsnx.ix[i]+k].xref = combsnx.x0[combsnx.ix[i]+k]

    # Change a priori velocities of stations with absolute velocity constraints, if any
    if (set_vel) and (vconst):
        keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.iv]]

        # Loop over specified absolute velocity constraints
        for vc in vconst:
            if hasattr(vc, 'point'):

                # If specified point actually has an estimated velocity,
                tab = vc.point.split()
                sta = tab[0] + '{0:>2s}'.format(tab[1]) + '{0:4d}'.format(int(tab[2]))
                if (sta in keys):
                    i = keys.index(sta)

                    # Check that there is no conflict with datum
                    if hasattr(combsnx.param[combsnx.iv[i]], 'xref'):
                        raise RuntimeError('Absolute velocity constraint not allowed for datum point {0}.'.format(sta))

                    # If not, change a priori station velocity to specified value, or zero,
                    # and assign "reference" values to the corresponding parameters
                    else:
                        if hasattr(vc, 'vref'):
                            combsnx.x0[combsnx.iv[i]:combsnx.iv[i]+3] = vc.vref
                        else:
                            combsnx.x0[combsnx.iv[i]:combsnx.iv[i]+3] = 0
                        for k in range(3):
                            combsnx.param[combsnx.iv[i]+k].xref = combsnx.x0[combsnx.iv[i]+k]
                            
    # Change a priori periodic station motion coefficients with absolute constraints, if any
    if (set_per) and (pconst):
        for ip in range(len(set_per)):
            keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.iper[ip]]]

            # Loop over specified absolute periodic station motion constraints
            for pc in pconst[ip]:
                if hasattr(pc, 'point'):

                    # If specified point actually has an estimated periodic motion at current period,
                    tab = pc.point.split()
                    sta = tab[0] + '{0:>2s}'.format(tab[1]) + '{0:4d}'.format(int(tab[2]))
                    if (sta in keys):
                        i = keys.index(sta)

                        # Check that there is no conflict with datum
                        if hasattr(combsnx.param[combsnx.iper[ip][i]], 'xref'):
                            raise RuntimeError('Absolute periodic motion constraint not allowed for datum point {0}.'.format(sta))

                        # If not, change a priori periodic station motion to specified value, or zero,
                        # and assign "reference" values to the corresponding parameters
                        else:
                            if hasattr(pc, 'pref'):
                                combsnx.x0[combsnx.iper[ip][i]:combsnx.iper[ip][i]+6] = pc.pref
                            else:
                                combsnx.x0[combsnx.iper[ip][i]:combsnx.iper[ip][i]+6] = 0
                            for k in range(6):
                                combsnx.param[combsnx.iper[ip][i]+k].xref = combsnx.x0[combsnx.iper[ip][i]+k]
        
    # If relative station position constraints are going to be applied, assign consistent a priori positions
    # to all the points within each cluster of relative position constraints
    if (xconst):

        # Build graph of relative position constraints
        Gx = combsnx.dxc_graph(xconst)
        nodes = list(Gx.nodes())

        # Loop over every connected component of the graph
        for c in nx.connected_components(Gx):
            if (len(c) > 1):

                # Get indices of the nodes in current connected component
                ind = [nodes.index(s) for s in list(c)]

                # List of reference positions of the points in current connected component
                xref = []
                for i in ind:
                    if hasattr(combsnx.param[combsnx.ix[i]], 'xref'):
                        iref = i
                        xref.append([combsnx.param[combsnx.ix[i]+k].xref for k in range(3)])
                
                # If none of the points in current connected component has a reference position,
                # choose as default reference position the a priori position of the first point
                if (len(xref) == 0):
                    iref = ind[0]
                    xref = [combsnx.x0[combsnx.ix[iref]:combsnx.ix[iref]+3]]

                # If there are more than one element in the list of reference positions, then there's a conflict.
                if (len(xref) > 1):
                    raise RuntimeError('Relative position constraint not allowed between points with different reference positions:\n{0}'.format(sorted(list(c))))
                
                # Else, if current connected component contains two points, 
                elif (len(c) == 2):

                    # Get tie vector between both points
                    dxref = np.array(Gx.get_edge_data(*c)['dxref'])
                
                    # If the second point has reference coordinates, assign as a priori position of the first point
                    # the reference position of the second point minus the tie vector.
                    if (iref == ind[1]):
                        combsnx.x0[combsnx.ix[ind[0]]:combsnx.ix[ind[0]]+3] = xref[0] - dxref
                            
                    # Else, the other way round.
                    else:
                        combsnx.x0[combsnx.ix[ind[1]]:combsnx.ix[ind[1]]+3] = xref[0] + dxref

                # Else, assign as a priori positions of all points in current connected component
                # the reference position of THE reference point (by default: the first one)
                else:
                    for i in ind:
                        combsnx.x0[combsnx.ix[i]:combsnx.ix[i]+3] = xref[0]

    # If velocities are going to be estimated, and relative station velocity constraints are going to be applied,
    # assign the same a priori velocity to all the points within each cluster of relative velocity constraints
    if (set_vel) and (((solns) and (dv_sig)) or (vconst)):

        # Build graph of relative velocity constraints
        Gv = combsnx.dvc_graph(solns, dv_sig, vconst)
        nodes = list(Gv.nodes())

        # Loop over every connected component of the graph
        for c in nx.connected_components(Gv):
            if (len(c) > 1):

                # Get indices of the nodes in current connected component
                ind = [nodes.index(s) for s in list(c)]

                # List of reference velocities of the points in current connected component
                vref = []
                for i in ind:
                    if hasattr(combsnx.param[combsnx.iv[i]], 'xref'):
                        iref = i
                        vref.append([combsnx.param[combsnx.iv[i]+k].xref for k in range(3)])
                        
                # If none of the points in current connected component has a reference velocity,
                # choose as default reference velocity the a priori velocity of the first point
                if (len(vref) == 0):
                    iref = ind[0]
                    vref = [combsnx.x0[combsnx.iv[iref]:combsnx.iv[iref]+3]]

                # If there are more than one element in the list of reference velocities, then there's a conflict.
                if (len(vref) > 1):
                    raise RuntimeError('Relative velocity constraint not allowed between points with different reference velocities:\n{0}'.format(sorted(list(c))))

                # Else, assign as a priori velocities of all points in current connected component
                # the reference velocity of THE reference point (by default: the first one)
                else:
                    for i in ind:
                        combsnx.x0[combsnx.iv[i]:combsnx.iv[i]+3] = vref[0]
                        
    # If periodic station motions are going to be estimated, and relative periodic motion constraints are going to be applied,
    # assign the same a priori periodic motion coefficients to all the points within each cluster of periodic motion constraints
    if (set_per) and ((dp_sig) or (pconst)):
        
        # Initialize graphs of relative periodic motion constraints at each period
        Gp = [None for ip in range(len(set_per))]
        
        # Loop over periods
        for ip in range(len(set_per)):

            # Build graph of relative periodic motion constraints
            Gp[ip] = combsnx.dpc_graph(ip, dp_sig[ip], pconst[ip])
            nodes = list(Gp[ip].nodes())

            # Loop over every connected component of the graph
            for c in nx.connected_components(Gp[ip]):
                if (len(c) > 1):

                    # Get indices of the nodes in current connected component
                    ind = [nodes.index(s) for s in list(c)]

                    # List of reference periodic motion coefficients of the points in current connected component
                    pref = []
                    for i in ind:
                        if hasattr(combsnx.param[combsnx.iper[ip][i]], 'xref'):
                            iref = i
                            pref.append([combsnx.param[combsnx.iper[ip][i]+k].xref for k in range(6)])
                            
                    # If none of the points in current connected component has reference periodic motion coefficients,
                    # choose as default reference coefficients the a priori periodic motion coefficients of the first point
                    if (len(pref) == 0):
                        iref = ind[0]
                        pref = [combsnx.x0[combsnx.iper[ip][iref]:combsnx.iper[ip][iref]+6]]

                    # If there are more than one element in the list of reference periodic motion coefficients, then there's a conflict.
                    if (len(pref) > 1):
                        raise RuntimeError('Relative periodic motion constraint not allowed between points with different reference periodic motion coefficients:\n{0}'.format(sorted(list(c))))

                    # Else, assign as a priori coefficients of all points in current connected component
                    # the reference coefficients of THE reference point (by default: the first one)
                    else:
                        for i in ind:
                            combsnx.x0[combsnx.iper[ip][i]:combsnx.iper[ip][i]+6] = pref[0]




    # 2 - SET UP NORMAL EQUATION
    #---------------------------
    
    # Print message
    if not(quiet):
        print('', file=out)
        print('', file=out)
        print('    '+str(date())+' : Set up normal equation', file=out)

    # Initializations
    combsnx.b = np.zeros(combsnx.npar)
    combsnx.N = np.zeros((combsnx.npar, combsnx.npar))
    combsnx.Nc = np.zeros((combsnx.npar, combsnx.npar))
    A = []
    dy = []
    lPl = 0
    


    # Loop over input solutions
    #--------------------------
    
    for isol in tqdm(range(len(inputs)), file=tqdm_out, leave=False):
        sol = inputs[isol]
        
        # Re-read input solution if needed
        if not(store_inputs):
            read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc)

        # Shortcut for sol.snx
        snx = sol.snx

        # Indices of common parameters with combined solution
        (isnx, icmb) = snx.get_common_par(combsnx)

        # Right-hand side (O-C vector)
        dyi = np.zeros(snx.npar)
        dyi[isnx] = snx.x[isnx] - combsnx.x0[icmb] 
        dy.append(dyi)
        
        # Initialize design matrix with ones for all parameters in snx
        A_rows = isnx
        A_cols = icmb
        A_vals = [1]*len(isnx)
    
        # Add position / velocity partial derivatives and update right-hand side if needed
        if (set_vel):
            keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.iv]]
            for i in snx.ix:
                p = snx.param[i]
                dt = (date.from_tsnx(p.tref).mjd - mjd0) / 365.25
                j = combsnx.iv[keys.index(p.code+p.pt+p.soln)]
                A_rows.extend([i, i+1, i+2])
                A_cols.extend([j, j+1, j+2])
                A_vals.extend([dt, dt, dt])
                dy[-1][i:i+3] -= dt * combsnx.x0[j:j+3]
                
        # Add position / periodic station motion partial derivatives and update right-hand side if needed
        # Note: These partial derivatives are relevant only if the current input solution "sol" is an
        # instantaneous solution in which periodic station motions are included in station positions.
        # We test this here by the absence of station velocities in sol.snx (len(snx.iv) == 0).
        if (set_per) and (len(snx.iv) == 0):
            for (ip, per) in enumerate(set_per):
                keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.iper[ip]]]
                for i in snx.ix:
                    p = snx.param[i]
                    dt = date.from_tsnx(p.tref).mjd - mjd0
                    c = cos(2*pi*dt/per)
                    s = sin(2*pi*dt/per)
                    j = combsnx.iper[ip][keys.index(p.code+p.pt+p.soln)]
                    A_rows.extend([i, i, i+1, i+1, i+2, i+2])
                    A_cols.extend([j, j+1, j+2, j+3, j+4, j+5])
                    A_vals.extend([c, s, c, s, c, s])
                    dy[-1][i:i+3] -= c * combsnx.x0[[j, j+2, j+4]] + s * combsnx.x0[[j+1, j+3, j+5]]

        # Add partial derivatives of transformation parameters
        if (sol.params):
            H = snx.helmert_partials(sol.params, 'STA')
            if not(reduce_trans):
                ind = np.nonzero(H)
                A_rows.extend(ind[0].tolist())
                A_cols.extend([sol.itrans[i] for i in ind[1]])
                A_vals.extend(H[ind].tolist())
        else:
            H = np.empty((snx.npar, 0))
            
        # Add partial derivatives of transformation parameter rates
        if (sol.params_rate):
            Hv = snx.helmert_partials(sol.params_rate, 'VEL')
            if not(reduce_trans):
                ind = np.nonzero(Hv)
                A_rows.extend(ind[0].tolist())
                A_cols.extend([sol.idtrans[i] for i in ind[1]])
                A_vals.extend(Hv[ind].tolist())
        else:
            Hv = np.empty((snx.npar, 0))
            
        # Add partial derivatives of transformation parameter periodic variations
        Hp = [np.empty((snx.npar, 0)) for ip in range(len(set_per))]
        if (set_per):
            for (ip, p) in enumerate(set_per):
                if (sol.params_per[ip]):
                    Hp[ip] = snx.helmert_partials(sol.params_per[ip], 'PER', period=p)
                    if not(reduce_trans):
                        ind = np.nonzero(Hp[ip])
                        A_rows.extend(ind[0].tolist())
                        A_cols.extend([sol.iptrans[ip][i] for i in ind[1]])
                        A_vals.extend(Hp[ip][ind].tolist())
        
        # Build sparse design matrix of current solution
        A.append(sparse.csc_matrix((A_vals, (A_rows, A_cols)), shape=(snx.npar, combsnx.npar)))
        ind = np.nonzero(A[isol].indptr[:-1] != A[isol].indptr[1:])[0]

        # Get weight matrix of solution isol
        if (snx.N is not None) and (snx.Nc is not None):
            P = (snx.N + snx.Nc) / sol.sf**2
        elif (snx.N is not None):
            P = snx.N / sol.sf**2
        else:
            P = invspd(snx.Q) / sol.sf**2

        # Project weight matrix if transformation parameters are reduced
        if (reduce_trans):
            sol.H = np.hstack((H, Hv, *Hp))
            HtP = np.dot(sol.H.T, P)
            HtPHi = invspd(np.dot(HtP, sol.H))
            P -= np.dot(HtP.T, np.dot(HtPHi, HtP))
        
        # Update normal equation
        AtP = A[isol][:,ind].T.dot(P)
        combsnx.N[np.ix_(ind,ind)] += A[isol][:,ind].T.dot(AtP.T)
        combsnx.b[ind] += np.dot(AtP, dy[isol])
        lPl += np.dot(dy[isol].T, np.dot(P, dy[isol]))

        # Make room if needed
        if not(store_inputs):
            del sol.snx

    # Return unconstrained normal equation if required
    if (return_neq):
        return(combsnx)



    # 3 - ADD CONSTRAINTS
    #--------------------
    
    # Print message
    if not(quiet):
        print('', file=out)
        print('', file=out)
        print('    '+str(date())+' : Add constraints', file=out)

    # Initialization
    nc = 0
        
    
    
    # Add minimal constraints
    #------------------------

    # Add minimal constraints to station positions
    if (mc_sta):
        if not(quiet):
            print('        Add NN{0} constraints to station positions'.format(mc_sta), file=out)
        nc += combsnx.add_mc(mc_sta, 'STA', sigma=mc_sta_sig, datum=datum, crf_datum=crf_datum, thr=mc_sta_thr)

    # Add minimal constraints to station velocities
    if (mc_vel):
        if not(quiet):
            print('        Add NN{0} constraints to station velocities'.format(mc_vel), file=out)
        nc += combsnx.add_mc(mc_vel, 'VEL', sigma=mc_vel_sig, datum=datum, thr=mc_vel_thr)
        
    # Add minimal constraints to periodic station motions
    if (set_per):
        for (i, p) in enumerate(set_per):
            if (mc_per[i]):
                if not(quiet):
                    print('        Add NN{0} constraints to periodic station motions at {1:7.3f} d'.format(mc_per[i], p), file=out)
                nc += combsnx.add_mc(mc_per[i], 'PER', period=p, sigma=mc_per_sig[i], datum=datum, thr=mc_per_thr[i])



    # Add "internal" constraints
    #---------------------------
    
    # Add zero-mean constraints to time series of transformation parameters
    if (ic_mean):
        if not(quiet):
            print('        Add zero-mean constraints to time series of transformation parameters', file=out)
        nc += combsnx.add_ic('mean', [sol.ic_mean for sol in inputs], sigma=ic_mean_sig, t0=tref)

    # Add zero-trend constraints to time series of transformation parameters
    if (ic_trend):
        if not(quiet):
            print('        Add zero-trend constraints to time series of transformation parameters', file=out)
        nc += combsnx.add_ic('trend', [sol.ic_trend for sol in inputs], sigma=ic_trend_sig, t0=tref)

    # Add zero-periodic-variation constraints to time series of transformation parameters
    if (set_per):
        for (i, p) in enumerate(set_per):
            if (ic_per[i]):
                if not(quiet):
                    print('        Add zero-periodic-variation constraints at {0:7.3f} d to time series of transformation parameters'.format(p), file=out)
                nc += combsnx.add_ic('periodic', [sol.ic_per[i] for sol in inputs], period=p, sigma=ic_per_sig[i], t0=tref)



    # Add station position constraints
    #---------------------------------

    if (xconst):
        if not(quiet):
            print('        Add station position constraints', file=out)
        nc += combsnx.add_xc(xconst, Gx)


        
    # Add station velocity constraints
    #---------------------------------

    if (set_vel) and (((solns) and (dv_sig)) or (vconst)):
        if not(quiet):
            print('        Add station velocity constraints', file=out)
        nc += combsnx.add_vc(solns, dv_sig, vconst, Gv)
        

        
    # Add periodic station motion constraints
    #----------------------------------------

    if (set_per) and ((dp_sig) or (pconst)):
        if not(quiet):
            print('        Add periodic station motion constraints', file=out)
        for (i, p) in enumerate(set_per):
            nc += combsnx.add_pc(p, dp_sig[i], pconst[i], Gp[i])



    # 4 - SOLVE NORMAL EQUATION
    #--------------------------

    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Solve normal equation', file=out)

    # Try to solve normal equation
    try:
        xNx = combsnx.neqinv(clear_neq=clear_neq, return_xNx=True)

    # If resolution failed, try to pinpoint problematic points
    except:
        print_exc()

        codeptsoln = np.array([p.code+p.pt+p.soln for p in combsnx.param])
        for s in combsnx.sta:
            for soln in [ss.soln for ss in s.soln]:
                ind = np.nonzero(codeptsoln == s.code+s.pt+soln)[0]
                N = combsnx.N[np.ix_(ind,ind)] + combsnx.Nc[np.ix_(ind,ind)]
                try:
                    Q = invspd(N)
                except:
                    print('It looks like combined coordinates cannot be estimated for:', s.code, s.pt, soln)

        sys.exit()



    # 5 - COMPUTE RESIDUALS AND STATISTICS
    #-------------------------------------

    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Compute residuals and statistics', file=out)

    # Initializations
    dx = combsnx.x - combsnx.x0
    vPv = lPl - xNx
    ntrans = 0
    
    
    
    # Loop over input solutions
    #--------------------------
    
    for isol in tqdm(range(len(inputs)), file=tqdm_out, leave=False):
        sol = inputs[isol]

        # Re-read input solution if needed
        if not(store_inputs):
            read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc)

        # Shortcut for sol.snx
        snx = sol.snx

        # Store number of observations
        sol.nobs = snx.npar

        # Compute residuals
        sol.v = dy[isol] - A[isol].dot(dx)
        
        # Get covariance matrix of input solution
        Q = snx.Q * sol.sf**2
        
        # Get weight matrix of input solution
        if (snx.N is not None) and (snx.Nc is not None):
            P = (snx.N + snx.Nc) / sol.sf**2
        elif (snx.N is not None):
            P = snx.N / sol.sf**2
        else:
            P = invspd(Q)
            
        # If transformation parameters were reduced, project residuals and update number of reduced transformation parameters.
        # Store, by the way, transformation parameters, their covariance matrix and formal errors.
        if (reduce_trans):
            HtP = np.dot(sol.H.T, P)
            sol.QT = invspd(np.dot(HtP, sol.H))
            sol.sT = np.sqrt(np.diag(sol.QT))
            sol.T = np.dot(sol.QT, np.dot(HtP, sol.v))
            sol.v -= np.dot(sol.H, sol.T)
            ntrans += sol.H.shape[1]
            
        # Covariance matrices of predicted observations if needed
        if not(reduce_trans) and ((norm_res == 'correct') or (vce == 'correct')):
            Ql = A[isol].dot((A[isol].dot(combsnx.Q)).T)
        
        # Compute covariance matrix of residuals if needed
        if not(reduce_trans) and (norm_res == 'correct'):
            Qv = Q - Ql
        
        # Standard deviations of residuals
        if not(reduce_trans) and (norm_res == 'correct'):
            sol.sv = np.sqrt(np.diag(Qv))
        else:
            sol.sv = np.sqrt(np.diag(Q))
        
        # Normalized residuals
        sol.vn = sol.v / sol.sv

        # Weighted squared sum of residuals
        sol.vPv = np.sum(sol.v * np.dot(P, sol.v))

        # Compute solution variance factor
        if not(reduce_trans) and (vce == 'correct'):
            sol.tr = trdot(Ql, P)
        else:
            sol.tr = 0
        sol.vf = sol.vPv / (snx.npar - sol.tr)
        
        # Rotate station position residuals to ENH frames and convert them into mm
        s2 = np.diag(Q).copy()
        for i in snx.ix:
            R = xyz2enh(snx.x[i:i+3])
            sol.v[i:i+3] = 1000 * np.dot(R, sol.v[i:i+3])
            s2[i:i+3] = np.diag(np.dot(R, np.dot(Q[i:i+3, i:i+3], R.T)))
            if not(reduce_trans) and (norm_res == 'correct'):
                sol.sv[i:i+3] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[i:i+3, i:i+3], R.T))))
            else:
                sol.sv[i:i+3] = 1000 * np.sqrt(s2[i:i+3])
            sol.vn[i:i+3] = sol.v[i:i+3] / sol.sv[i:i+3]

        # Compute WRMS of ENH station position residuals and median ENH station position formal errors
        sol.wrmsx = np.zeros(3)
        sol.sigmx = np.zeros(3)
        ix = np.array(snx.ix)
        for i in range(3):
            sol.wrmsx[i] = sqrt(np.sum(sol.v[ix+i]**2/s2[ix+i]) / np.sum(1/s2[ix+i]))
            sol.sigmx[i] = 1000 * np.median(np.sqrt(s2[ix+i]))

        # Rotate station velocity residuals to ENH frames and convert them into mm/y
        for i in snx.iv:
            R = xyz2enh(snx.x[i-3:i])
            sol.v[i:i+3] = 1000 * np.dot(R, sol.v[i:i+3])
            s2[i:i+3] = np.diag(np.dot(R, np.dot(Q[i:i+3, i:i+3], R.T)))
            if not(reduce_trans) and (norm_res == 'correct'):
                sol.sv[i:i+3] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[i:i+3, i:i+3], R.T))))
            else:
                sol.sv[i:i+3] = 1000 * np.sqrt(s2[i:i+3])
            sol.vn[i:i+3] = sol.v[i:i+3] / sol.sv[i:i+3]

        # Compute WRMS of ENH station velocity residuals and median ENH velocity formal errors
        if (len(snx.iv) > 0):
            sol.wrmsv = np.zeros(3)
            sol.sigmv = np.zeros(3)
            iv = np.array(snx.iv)
            for i in range(3):
                sol.wrmsv[i] = sqrt(np.sum(sol.v[iv+i]**2/s2[iv+i]) / np.sum(1/s2[iv+i]))
                sol.sigmv[i] = 1000 * np.median(np.sqrt(s2[iv+i]))
                
        # Loop over periods
        sol.per = snx.per
        if (snx.per):
            snx.wrmsp = [None for i in range(len(snx.per))]
            snx.sigmp = [None for i in range(len(snx.per))]
            
            for (ip, p) in enumerate(snx.per):
                
                # Rotate periodic station motion coefficient residuals to ENH frames and convert them into mm
                for i in snx.iper[ip]:
                    R = xyz2enh(snx.get_xyz([snx.param[i].code], [snx.param[i].pt], [snx.param[i].soln])[0])
                    sol.v[[i+0,i+2,i+4]] = 1000 * np.dot(R, sol.v[[i+0,i+2,i+4]])
                    sol.v[[i+1,i+3,i+5]] = 1000 * np.dot(R, sol.v[[i+1,i+3,i+5]])
                    s2[[i+0,i+2,i+4]] = np.diag(np.dot(R, np.dot(Q[np.ix_([i+0,i+2,i+4], [i+0,i+2,i+4])], R.T)))
                    s2[[i+1,i+3,i+5]] = np.diag(np.dot(R, np.dot(Q[np.ix_([i+1,i+3,i+5], [i+1,i+3,i+5])], R.T)))
                    if not(reduce_trans) and (norm_res == 'correct'):
                        sol.sv[[i+0,i+2,i+4]] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[np.ix_([i+0,i+2,i+4], [i+0,i+2,i+4])], R.T))))
                        sol.sv[[i+1,i+3,i+5]] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[np.ix_([i+1,i+3,i+5], [i+1,i+3,i+5])], R.T))))
                    else:
                        sol.sv[i:i+6] = 1000 * np.sqrt(s2[i:i+6])
                    sol.vn[i:i+6] = sol.v[i:i+6] / sol.sv[i:i+6]

                # Compute WRMS of ENH periodic station motion coefficient residuals and median ENH periodic station motion coefficient formal errors
                if (len(snx.iper[ip]) > 0):
                    sol.wrmsp[ip] = np.zeros(3)
                    sol.sigmp[ip] = np.zeros(3)
                    iper = np.array(snx.iper[ip])
                    for i in range(3):
                        sol.wrmsp[ip][i] = sqrt((np.sum(sol.v[iper+2*i]**2/s2[iper+2*i]) + np.sum(sol.v[iper+2*i+1]**2/s2[iper+2*i+1])) / (np.sum(1/s2[iper+2*i]) + np.sum(1/s2[iper+2*i+1])))
                        sol.sigmp[ip][i] = 1000 * np.median(np.sqrt(np.hstack((s2[iper+2*i], s2[iper+2*i+1]))))

        # Convert geocenter residuals into mm
        igc = snx.igc + [i+1 for i in snx.igc] + [i+2 for i in snx.igc]
        sol.v[igc] *= 1000
        sol.sv[igc] *= 1000

        # Make room if needed
        if not(store_inputs):
            del sol.snx
        
        
        
    # Compute global variance factor
    #-------------------------------
    
    # Global variance factor
    vf = vPv / (nobs + nc - combsnx.npar - ntrans)

    # Update standard devations of residuals, normalized residuals and median ENH formal errors
    # with global variance factor
    for sol in inputs:
        sol.sv *= sqrt(vf)
        sol.vn /= sqrt(vf)
        sol.sigmx *= sqrt(vf)
        if hasattr(sol, 'sigmv'):
            sol.sigmv *= sqrt(vf)
        if hasattr(sol, 'sigmp'):
            for i in range(len(sol.sigmp)):
                if (sol.sigmp[i]):
                    sol.sigmp[i] *= sqrt(vf)

    # Update covariance matrix and standard deviations of transformation parameters
    # with global variance factor if they were reduced
    if (reduce_trans):
        for sol in inputs:
            sol.QT *= vf
            sol.sT *= sqrt(vf)
        
    # Update combined solution with global variance factor
    combsnx.Nc /= vf 
    combsnx.Q *= vf
    combsnx.sig = np.sqrt(np.diag(combsnx.Q))
    
    # Set content of SOLUTION/STATISTICS block
    combsnx.stats = record()
    combsnx.stats.nobs = nobs + nc
    combsnx.stats.nunk = combsnx.npar + ntrans
    combsnx.stats.vf = vf
    


    # Print statistics
    #-----------------

    if not(quiet):
        
        # Main combination statistics
        print('', file=out)
        print('', file=out)
        print('        Combination statistics', file=out)
        print('        ----------------------', file=out)
        print('', file=out)
        print('              |                                                         |', file=out)
        print('         sol_ | nobs__ tr/npar___ vPv_______ prior_SF fact_SF_ post_SF_ |', file=out)
        print('        ------|---------------------------------------------------------|', file=out)
        for sol in inputs:
            name = '{0:4}'.format(sol.name)[:4]
            print('         {0} | {1.nobs:6d} {1.tr:10.3f} {1.vPv:10.3f} {2:8.3f} {3:8.3f} {4:8.3f} |'.format(name, sol, sol.sf, sqrt(sol.vf), sol.sf*sqrt(sol.vf)), file=out)
        print('        ------|---------------------------------------------------------|', file=out)
        print('         comb | {0:6d} {1:10.3f} {2:10.3f}          sigma0 = {3:8.3f} |'.format(nobs, combsnx.npar+ntrans, vPv, sqrt(vf)), file=out)
        print('', file=out)
        
        # Station position residual statistics
        print('        Station position residual statistics', file=out)
        print('        ------------------------------------', file=out)
        print('', file=out)
        print('              |         WRMS [mm]          |      median sigma [mm]     |', file=out)
        print('         sol_ | East____ North___ Up______ | East____ North___ Up______ |', file=out)
        print('        ------|----------------------------|----------------------------|', file=out)
        for sol in inputs:
            name = '{0:4}'.format(sol.name)[:4]
            print('         {0} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(name, sol.wrmsx, sol.sigmx), file=out)
        print('        ------|----------------------------|----------------------------|', file=out)
        print('', file=out)
        
        # Station velocity residual statistics
        b = False
        for sol in inputs:
            if hasattr(sol, 'wrmsv'):
                b = True
                
        if (b):
            print('        Station velocity residual statistics', file=out)
            print('        ------------------------------------', file=out)
            print('', file=out)
            print('              |        WRMS [mm/y]         |     median sigma [mm/y]    |', file=out)
            print('         sol_ | East____ North___ Up______ | East____ North___ Up______ |', file=out)
            print('        ------|----------------------------|----------------------------|', file=out)
            for sol in inputs:
                name = '{0:4}'.format(sol.name)[:4]
                print('         {0} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(name, sol.wrmsv, sol.sigmv), file=out)
            print('        ------|----------------------------|----------------------------|', file=out)
            print('', file=out)
            
        # Periodic station motion residuals
        if (set_per):
            for p in set_per:
        
                b = False
                for sol in inputs:
                    if (sol.per):
                        if p in sol.per:
                            b = True
                        
                if (b):
                    print('        Periodic station motion residuals at {0:7.3f} d'.format(p), file=out)
                    print('        ----------------------------------------------', file=out)
                    print('', file=out)
                    print('              |         WRMS [mm]          |      median sigma [mm]     |', file=out)
                    print('         sol_ | East____ North___ Up______ | East____ North___ Up______ |', file=out)
                    print('        ------|----------------------------|----------------------------|', file=out)
                    for sol in inputs:
                        if p in sol.per:
                            ip = sol.per.index(p)
                            name = '{0:4}'.format(sol.name)[:4]
                            print('         {0} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(name, sol.wrmsp[ip], sol.sigmp[ip]), file=out)
                    print('        ------|----------------------------|----------------------------|', file=out)
                    print('', file=out)



    # Update scale factors of input solutions if requested
    #-----------------------------------------------------
    if (update_sf):
        for sol in inputs:
            sol.sf = sol.sf * sqrt(sol.vf)



    # FINISHED!
    #----------

    # Print message
    if not(quiet):
        print('    '+str(date())+' : Finished!', file=out)
        print('', file=out)
    
    return combsnx



# Iterative combination of SINEX solutions
#-----------------------------------------
def combine_iter(inputs, tref, solns=None, check_solns=True, psd=None, set_vel=False, set_per=[], stack_gc=False, stack_sc=False, return_neq=False,
                 dv_sig=1e-6, dp_sig=1e-6, xconst=None, vconst=None, pconst=None, datum=None, crf_datum=None,
                 mc_sta=None, mc_sta_sig=1e-5, mc_sta_thr=None, mc_vel=None, mc_vel_sig=1e-6, mc_vel_thr=None, mc_per=None, mc_per_sig=1e-5, mc_per_thr=None,
                 ic_mean=False, ic_mean_sig=1e-5, ic_trend=False, ic_trend_sig=1e-6, ic_per=False, ic_per_sig=1e-5,
                 update_sf=False, norm_res='correct', vce='correct', store_inputs=True, reduce_trans=False, clear_neq=True,
                 thr_raw=None, thr_norm=None, flag_once=False, quiet=False, out=sys.stdout):

    """
    Iterative combination of SINEX solutions

    Returns
    -------
    combsnx : sinex instance
        Combined SINEX solution

    Parameters
    ----------
    inputs : str or list
        [File containing] list of input solutions
    tref : str
        Reference date (in SINEX format)
    solns : str or list, optional
        [File containing] discontinuity list (soln.snx). Default is None.
    check_solns : bool, optional
        Whether solution numbers should be checked in input solutions or not. Default is True.
        To save time, check solution numbers in input solutions before combination.
    psd : str or sinex object, optional
        sinex instance with post-seismic deformation models to be removed from input solutions
        before combination. Default is None.
    set_vel : bool, optional
        Whether velocities should be estimated for all stations. Default is False.
    set_per : list, optional
        List of periods [d] at which periodic station motions should be estimated. Default is [].
    stack_gc : bool, optional
        Whether successive geocenter coordinates should be stacked into single
        combined geocenter coordinates. Default is False.
    stack_sc : bool, optional
        Whether successive scale factors should be stacked into a single
        combined scale factor. Default is False.
    return_neq: bool, optional
        Whether to return unconstrained normal equation, without solving it.
        Default is False.
    dv_sig : float, optional
        Sigma of equality constraints to be applied between successive velocities [m/y].
        Default is 1e-6.
    dp_sig : float, optional
        List of sigmas [m] of equality constraints to be applied between successive periodic
        station motion coefficients at each period in argument "set_per". If a single value
        is provided, it is assumed to apply to ALL periods in argument "set_per". Default is 1e-6.
    xconst : str or list, optional
        [YAML file containing] station position constraints to be applied. Default is None.
    vconst : str or list, optional
        [YAML file containing] station velocity constraints to be applied. Default is None.
    pconst : list, optional
        List of [YAML files containing] periodic station motion constraints to be applied
        for every period in argument "set_per". If a single YAML file, or single set of 
        constraints is provided, it is assumed to apply to ALL periods in "set_per".
        Default is None.
    datum : str or sinex instance, optional
        [File containing] reference TRF solution. Default is None.
    crf_datum : str or sinex instance, optional
        [File containing] reference CRF solution. Default is None.
    mc_sta : str, optional
        String indicating which minimal constraints should be applied to station positions.
        It can be composed of any combination of letters 'T' (translations),
        'S' (scale) and 'R' (rotations). Default is None.
    mc_sta_sig : float or str, optional
        Sigma of minimal constraints to be applied to station positions in m. Default is 1e-5.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    mc_sta_thr : float, optional
        If set, then station positions with large uncertainties will be rejected from the set
        of station positions to which minimal constraints are applied. See sinex.add_mc() for
        detailed explanations. Default is None.
    mc_vel : str, optional
        String indicating which minimal constraints should be applied to station velocities.
        It can be composed of any combination of letters 'T' (translations),
        'S' (scale) and 'R' (rotations). Default is None.
    mc_vel_sig : float, optional
        Sigma of minimal constraints to be applied to station velocities in m/y. Default is 1e-6.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    mc_vel_thr : float, optional
        If set, then station velocities with large uncertainties will be rejected from the set
        of station velocities to which minimal constraints are applied. See sinex.add_mc() for
        detailed explanations. Default is None.
    mc_per : list or str, optional
        List of strings indicating which minimal constraints should be applied to periodic station
        motions for every period in argument "set_per". Each of the strings can be composed of any
        combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations). If a single
        string is provided, it is assumed to apply to ALL periods in argument "set_per".
        Default is None.
    mc_per_sig : list or float, optional
        List of sigmas of minimal constraints to be applied to periodic station motions (in m)
        for every period in argument "set_per". If a single value is provided, it is assumed to
        apply to ALL periods in argument "set_per". "mc_per_sig" may also be set to 'auto', in
        which case adequate sigmas will be automatically set by sinex.add_mc().
        Default is 1e-5.
    mc_per_thr : list or float, optional
        If set, then periodic station motion coefficients with large uncertainties will be rejected
        from the set of periodic station motions to which minimal constraints are applied. See
        sinex.add_mc() for detailed explanations. "mc_per_thr" may be a list of threshold values
        for every period in argument "set_per", or a single value that applied to ALL periods.
        Default is None.
    ic_mean : bool, optional
        Boolean indicating whether zero-mean constraints should be applied to the time series of
        certain types of transformation parameters. Default is False.
        If True, then every input solution in the list "inputs", that should contribute to a 
        zero-mean constraint on some type of transformation parameters, should have an attribute
        "ic_mean" assigned. This attribute may be composed of any combination of the letters 'T'
        (translations), 'S' (scale), 'R' (rotations) and 'A' (CRF rotations) indicating the types
        of transformation parameters for which the input solution should contribute to a zero-mean
        constraint. The input solutions that do not contribute to any zero-mean constraint may
        have no "ic_mean" attribute assigned, or may have an empty string or None as "ic_mean"
        attribute.
    ic_mean_sig : float, optional
        Sigma of the zero-mean constraints to be applied to the time series of transformation
        parameters, in m. Default is 1e-5.
    ic_trend : bool, optional
        Boolean indicating whether zero-trend constraints should be applied to the time series of
        certain types of transformation parameters. Default is False.
        If True, then every input solution in the list "inputs", that should contribute to a 
        zero-trend constraint on some type of transformation parameters, should have an attribute
        "ic_trend" assigned. This attribute may be composed of any combination of the letters 'T'
        (translations), 'S' (scale), 'R' (rotations) and 'A' (CRF rotations) indicating the types
        of transformation parameters for which the input solution should contribute to a zero-trend
        constraint. The input solutions that do not contribute to any zero-trend constraint may
        have no "ic_trend" attribute assigned, or may have an empty string or None as "ic_trend"
        attribute.
    ic_trend_sig : float, optional
        Sigma of the zero-trend constraints to be applied to the time series of transformation
        parameters, in m/y. Default is 1e-6.
    ic_per : list or bool, optional
        List of booleans indicating whether zero-periodic-variation constraints should be applied
        to the time series of certain types of transformation parameters, for every period in
        argument "set_per". If a single boolean value is provided, it is assumed to apply to ALL
        periods. Default is False.
        If True for any period, then every input solution in the list "inputs", that should
        contribute to a zero-periodic-motion constraint on some type of transformation parameters
        at some period, should have an attribute "ic_per" assigned. This attribute should be a
        list of strings, one for every period in argument "set_per". Each string may be composed
        of any combination of the letters 'T' (translations), 'S' (scale), 'R' (rotations) and
        'A' (CRF rotations) indicating the types of transformation parameters for which the input
        solution should contribute to a zero-periodic-variation constraint. If an input solution
        has a single string assigned as "ic_per" attribute, it is assumed to apply to ALL periods.
        The input solutions that do not contribute to any zero-periodic-variation constraint may
        have no "ic_per" attribute assigned, or may have an empty string or None as "ic_per"
        attribute.
    ic_per_sig : float, optional
        List of sigmas of the zero-periodic-variation constraints to be applied to the time series
        of transformation parameters, in m/y, for each period in argument "set_per". If a single
        value is provided, it is assumed to apply to ALL periods in "set_per". Default is 1e-5.
    update_sf : bool, optional
        Whether to update variance factors of input solutions with VCE estimates.
        Default is False.
    norm_res : str, optional
        Keyword indicating how normalized residuals should be computed.
        It can be either:
        - 'correct' in which case residuals are normalized by their own
            standard deviations, or
        - 'approx' in which case residuals are normalized by the
            standard deviations of the observations (i.e., snx.sig).
        Default is 'correct'.
    vce : str, optional
        Keyword indicating how a posteriori variance factors should be computed.
        It can be either:
        - 'correct' in which case Sillard's (1999) degree-of-freedom estimator is used, or
        - 'approx' in which case a faster approximation is used.
        Default is 'correct'.
    store_inputs : bool, optional
        If True, all input solutions will be stored in RAM simultaneously (faster option).
        If False, input solutions are successively read and deleted during the successive
        processing steps (slower, but uses less RAM).
        Default is True.
    reduce_trans : bool, optional
        Whether to reduce transformation parameters. Default is False.
        Note that if transformation parameters are reduced, the options norm_res='correct'
        and vce='correct' become unavailable.
    clear_neq : bool, optional
        Whether normal equation should be kept in combined sinex object. Default is True.
    thr_raw : float, optional
        Multiplicative factor defining thresholds for flagging stations with large residuals
        as outliers: along each ENH component, threshold = thr_raw * WRMS.
        Default is None.
    thr_norm : float, optional
        Threshold for flagging station with large normalized residuals as outliers.
    flag_once : bool, optional
        If True, then each station can be flagged as outlier in only one input solution
        (i.e. the one with the largest 3D normalized residual for that station).
    quiet : bool, optional
        Whether not to print output messages. Default is False.
    out : file-like, optional
        Log file. Default is sys.stdout.
    
    """

    # Read input file if necessary
    if not(isinstance(inputs, list)):
        inputs = read_yaml(inputs)

    # While there remains outliers,
    end = False
    while not(end):
        
        # Combine input solutions
        combsnx = combine(inputs, tref, solns, check_solns, psd, set_vel, set_per, stack_gc, stack_sc, False,
                          dv_sig, dp_sig, xconst, vconst, pconst, datum, crf_datum,
                          mc_sta, mc_sta_sig, mc_sta_thr, mc_vel, mc_vel_sig, mc_vel_thr, mc_per, mc_per_sig, mc_per_thr,
                          ic_mean, ic_mean_sig, ic_trend, ic_trend_sig, ic_per, ic_per_sig,
                          update_sf, norm_res, vce, store_inputs, reduce_trans, clear_neq, quiet, out)
        
        # First loop over input solutions to flag outliers
        for sol in inputs:
            
            # Re-read input solution if needed
            if not(store_inputs):
                read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc, load_mat=False)
                
            # Set shortcut to sol.snx
            snx = sol.snx
                
            # Indices of station coordinates
            ix = np.array([[i, i+1, i+2] for i in snx.ix])

            # Indices of outliers
            sol.iout = []
            if (thr_raw):
                for i in range(3):
                    sol.iout.extend(np.nonzero(np.abs(sol.v[ix[:,i]]) > thr_raw*sol.wrms[i])[0].tolist())
            if (thr_norm):
                for i in range(3):
                    sol.iout.extend(np.nonzero(np.abs(sol.vn[ix[:,i]]) > thr_norm)[0].tolist())
            sol.iout = list(set(sol.iout))
            
            # Residuals and normalized residuals of outliers
            sol.vout = [sol.v[ix[i,:]] for i in sol.iout]
            sol.vnout = [sol.vn[ix[i,:]] for i in sol.iout]

            # Outlying stations
            sol.codeout = [snx.param[ix[i,0]].code for i in sol.iout]
            sol.ptout = [snx.param[ix[i,0]].pt for i in sol.iout]
            sol.solnout = [snx.param[ix[i,0]].soln for i in sol.iout]
            
            # Make room if needed
            if not(store_inputs):
                del sol.snx

        # Clean list of outliers if needed
        if (flag_once):
            
            # Complete list of outlying stations
            codeptsoln = []
            for sol in inputs:
                for i in range(len(sol.codeout)):
                    codeptsoln.append(sol.codeout[i]+sol.ptout[i]+sol.solnout[i])
            codeptsoln = list(set(codeptsoln))
            
            # Loop over outlying stations
            for i in range(len(codeptsoln)):
                
                # Get indices of solutions where current station is flagged as an outlier,
                # indices of current station in the list of outliers of each of those solutions,
                # and 3D normalized residuals of current station in each of those solutions
                isol = []
                ksol = []
                vsol = []
                for j in range(len(inputs)):
                    sol = inputs[j]
                    keys = [sol.codeout[k]+sol.ptout[k]+sol.solnout[k] for k in range(len(sol.codeout))]
                    if (codeptsoln[i] in keys):
                        isol.append(j)
                        ksol.append(keys.index(codeptsoln[i]))
                        vsol.append(sqrt(np.sum(sol.vnout[ksol[-1]]**2)))
                        
                # If current station is flagged as an outlier in more than one solution,
                if (len(isol) > 1):
                    
                    # Deflag it in all solutions except the one with the largest 3D normalized residual for that station
                    for j in range(len(isol)):
                        if (vsol[j] < np.max(vsol)):
                            inputs[isol[j]].iout.pop(ksol[j])
                            inputs[isol[j]].vout.pop(ksol[j])
                            inputs[isol[j]].vnout.pop(ksol[j])
                            inputs[isol[j]].codeout.pop(ksol[j])
                            inputs[isol[j]].ptout.pop(ksol[j])
                            inputs[isol[j]].solnout.pop(ksol[j])
        
        # Print header of outliers list
        if not(quiet):
            print('snxcomb.combine_iter', file=out)
            print('--------------------', file=out)
            print('', file=out)
            print('    Station position outliers', file=out)
            print('    -------------------------', file=out)
            print('', file=out)
            print('                       |     Raw residuals [mm]     |    Normalized residuals    |', file=out)
            print('    -------------------|----------------------------|----------------------------|', file=out)
            print('     sol. code pt soln |     E        N        H    |     E        N        H    |', file=out)
            print('    -------------------|----------------------------|----------------------------|', file=out)

        # Second loop over input solutions to reject outliers
        end = True
        for sol in inputs:
            
            # If any outliers were flagged in current input solution
            if (len(sol.codeout) > 0):
                end = False
                
                # Print outliers
                if not(quiet):
                    name = '{0:4}'.format(sol.name)[:4]
                    for i in range(len(sol.codeout)):
                        print('     {0} {1} {2} {3} | {4[0]:8.3f} {4[1]:8.3f} {4[2]:8.3f} | {5[0]:8.3f} {5[1]:8.3f} {5[2]:8.3f} |'.format(name, sol.codeout[i], sol.ptout[i], sol.solnout[i], sol.vout[i], sol.vnout[i]), file=out)
                    print('    -------------------|----------------------------|----------------------------|', file=out)
                
                # Store outliers
                if not(hasattr(sol, 'staout')):
                    sol.staout = []
                    sol.resout = []
                    sol.resnout = []
                sol.staout.extend(sol.codeout)
                sol.resout.extend(sol.vout)
                sol.resnout.extend(sol.vnout)

                # Re-read input solution if needed
                if not(store_inputs):
                    read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc)
                
                # Reject outliers
                sol.snx.del_sta(sol.codeout, sol.ptout, sol.solnout)
                
                # Overwrite input file and make room if needed
                if not(store_inputs):
                    sol.snx.dump(sol.file)
                    del sol.snx
                    
        # Print blank line in log file
        if not(quiet):
            print('', file=out)
        
        # Continue to iterate if VCE has not converged yet
        if (end) and (update_sf) and (np.max(np.abs(np.log([sol.vf for sol in inputs]))) > 1e-3):
            end = False

    return combsnx
    
