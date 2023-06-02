"""
    Combination of SINEX solutions
"""

# External imports
#-----------------
import os
import sys
#import mkl
#mkl.set_num_threads(1)
import copy
import pickle
import yaml
import numpy as np
from scipy import sparse, linalg
from math import sqrt

# Internal imports
#-----------------
from pytrf import date, sinex
from pytrf.const import agency, mas2rad, dera_dt
from pytrf.io import read_yaml, read_solns, write_yaml
from pytrf.math import invspd, pinvspd, trdot, xyz2enh
from pytrf.utils import record, earlier, Period

# Generate YAML configuration file
#------------------------------------
def mkopt_file(folder="inputs", set_vel=False, per=[], default={}):
    """
    Creates options.yml file, with default combination options. This file could be edit to specify combination parameters for each solutions.

    List of possible parameters & format:
        set for each solution :
        - "description": "my center description"
        - "params":"RST"
        - "ic_mean":"RST","ic_trend":"RST"
        
        set for each frequency (if set_per != []):
        - "mc_per":"RST", "ic_per":"RST"

    Parameters
    ----------
    folder : str, optional
        Path to folder that contains SINEX files. The default is "inputs" folder.
        WARNING : this folder must contain only SINEX solution files use for combination or stacking. 
    set_vel : bool, optional
        Whether velocities should be estimated for all stations. The default is False.
    per : list of str, optional.
        List of periods if periodic signals should be estimated for all stations. A period is given with a 2 character code : type+harmonic number
            * type "A" = "annual"
            * type "D" = "draconitic"
        The default value is '[]', means no periodic signal estimated.
        Example of periods : "A1" (annual, 365.25 days),  "A2" (semi-annual, 182.625 days), "D1" (draconitic 1)
    default : dict, optional
        Provides default values that will be applied for each solutions. Specify these arguments as dict : ex. {"description": "IGS solution"}

    Returns
    -------
    None.

    """
    ### Default YAML dict
    dict_yml = {}
    
    ### Build default frequency dictionary
    if len(per)!=0: #at leat 1 period
        list_per= []
        for pe in per :
            # mc_per: minimal constraints on periodic amplitude
            # ic_per: internal constraints on periodic amplitude
            
            # period build from Period class
            dict_per = Period(pe).__dict__
            
            #default value ?
            for key in default.keys():
                if key in ["mc_per","ic_per"]: #key use for dict_freq
                    dict_per[key] = default[key]
                    
            list_per.append(dict_per)
        
        #add to global YAML file
        dict_yml["PERIODS"] = list_per
            
            
    ### Build default solution dictionary
    list_sol = [] #add freq param as 1st element
    list_files = sorted(os.listdir(folder))
    
    if len(list_files)==0:
        raise ValueError("No solution file found at '{}'".format(folder))
    for num, file in enumerate(list_files):
        sol={}
        name = file.split(".")[0]
        sol["name"] = name
        sol["file"] = os.path.join(folder,file)
        
        #default
        sol["description"] = "Solution {}".format(num)
        sol["params"] = "RST"
        sol["ic_mean"] = ""
        
        
        if set_vel: #VELOCITY will be estimate, add constraints
            sol["velocity"] = {}
            #add constraints on VEL
            sol["velocity"]["ic_trend"] = ""
        
        #default values provides ?
        for key in default.keys():
            if key in ["mc_per","ic_per"]: #key use for dict_freq
                pass
            else:
                sol[key] = default[key]
                
        ## add current sol with its attributes to global dict_sol
        list_sol.append(sol)
        
    ### Write YAML file
    dict_yml["INPUTS"] = list_sol
    #yaml format
    file_yml = yaml.dump(dict_yml, sort_keys=False) #no sorted >> name 1st key to identify solutions
    with open('options.yml', 'w') as file:
        file.write(file_yml)
        


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
        raise RuntimeError('No name specified for input solution {0}. Please set \'name\' attribute for each input solution.'.format(sol))
    
    # If sinex instance of current solution is not readily available, load it
    if not(hasattr(sol, 'snx')):
        if (hasattr(sol, 'file')):
            sol.snx = sinex.load(sol.file, load_mat)
        else:
            raise RuntimeError('No input specified for solution {0} ({1}). Please set either \'snx\' or \'file\' attribute for each input solution.'.format(sol, sol.name))
    
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
def combine(inputs, tref, solns=None, check_solns=True, psd=None, set_vel=False, periods=[], dv_sig=1e-6, stack_gc=False, stack_sc=False, datum=None,
            mc_sta=None, mc_sta_sig=1e-5, mc_sta_thr=None, mc_vel=None, mc_vel_sig=1e-6, mc_vel_thr=None, #Minimal constraints
            ic_mean=False, ic_mean_sig=1e-5, ic_trend=False, ic_trend_sig=1e-6, #Internal constraints
            update_sf=False, norm_res='correct', vce='correct', store_inputs=True, reduce_trans=False, clear_neq=True, quiet=False, out=sys.stdout,
            ):

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
    periods: list of objects (built with pytrf.utils.Period), optional
        Period of periodic signals.
    dv_sig : float, optional
        Sigma of equality constraints to be applied between successive velocities [m/y].
        Default is 1e-6.
    stack_gc : bool, optional
        Whether successive geocenter coordinates should be stacked into single
        combined geocenter coordinates. Default is False.
    stack_sc : bool, optional
        Whether successive scale factors should be stacked into a single
        combined scale factor. Default is False.
    datum : str or sinex instance, optional
        [File containing] datum. Default is None.
    
    ---------- MINIMAL constraints parameters ----------
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
        detailed explanations.
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
        detailed explanations.
    
    ---------- INTERNAL constraints parameters ----------
    ic_mean : bool, optional. Default: False
        Boolean indicating if you allow internal constraints to be applied to mean(s) of parameter(s).
        If True, you must specify the "ic_mean" parameter in the YAML file for each solution that you want apply IC.
        This param can be composed of any combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations).
        If no "ic_mean" attribute or equal to empty str ("") in YAML file, IC are not apply for this solution.
    ic_mean_sig : float or str, optional
        Sigma of internal constraints to be applied to mean(s) of parameter(s), in m. Default is 1e-5.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    ic_trend : bool, optional. Default: False
        Boolean indicating if you allow internal constraints to be applied to trend(s) of parameter(s).
        If True, you must specify the "ic_trend" parameter in the YAML file for each solution that you want apply IC.
        This param can be composed of any combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations).
        If no "ic_trend" attribute or equal to empty str ("") in YAML file, IC are not apply for this solution.
    ic_trend_sig : float, optional
        Sigma of internal constraints to be applied to trend(s) of parameter(s), in m/y. Default is 1e-6.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().    
        
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
        ! WARNING !: in case of internal constraints (i.e. ic_trend or ic_mean == True), 'reduce_trans' must be 'False'.
        In any case it is automatically reset to 'False' with internal constraints.
    clear_neq : bool, optional
        Whether normal equation should be kept in combined sinex object. Default is True.
    quiet : bool, optional
        Whether not to print output messages. Default is False.
    out : file-like, optional
        Log file. Default is sys.stdout.
    
    """
    #look at reduce_trans exception, if internal constraints > reduce_trans=False 
    if (ic_mean or ic_trend) and (reduce_trans) : #internal constraints, at least on mean or trend.
        print("WARNING : case of internal constraints, 'reduce_trans' param set to False.", file=out)
        reduce_trans=False
    
    # Print header in log file
    if not(quiet):
        print('snxcomb.combine', file=out)
        print('---------------', file=out)

    # Read input file if necessary
    if not(isinstance(inputs, list)):
        inputs = read_yaml(inputs)

    # Read discontinuity file if necessary
    if (solns):
        if not(isinstance(solns, list)):
            solns = read_solns(solns)

    # Read datum if necessary
    if (datum):
        if not(isinstance(datum, sinex)):
            try:
                datum = sinex.load(datum, load_mat=False)
            except:
                datum = sinex.read(datum, dont_read=['comments', 'metadata', 'apriori', 'matrices'])
                
    #build periods dict, to optimize Period object research
    if len(periods)>0:
        dict_periods={}
        for per in periods:
            dict_periods[per.code]=per

    # Initialize combined SINEX solution
    combsnx = sinex()
    combsnx.version = '2.02'
    combsnx.agency = agency
    combsnx.const = 2
    combsnx.input = []
    combsnx.sta = []
    combsnx.rs = []
    combsnx.param = []
    combsnx.x0 = []
    combsnx.codept = []

    # Other initializations
    mjd0 = date.from_tsnx(tref).mjd
    nobs = 0



    # 1 - SET UP PARAMETER LIST
    #--------------------------
    
    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Set up parameter list', file=out)

    # initialize internal constraints dict
    ic_helmert_mean = {}
    ic_helmert_trend = {}

    # Loop over input solutions
    #--------------------------
    
    for isol in range(len(inputs)):
        #isol use as id of solution
        
        sol = inputs[isol]

        # Print message
        if not(quiet):
            print('\x1b[2K', end='\r', file=out)
            print('        Processing input solution {0:5d}/{1} ({2})'.format(isol+1, len(inputs), sol.name), end='\r', file=out)

        # Read input
        read_input(sol, tref, solns, check_solns, psd, stack_gc, stack_sc, load_mat=store_inputs)
        
        ## Build internal constraints dict (which constraints for which solutions ?)
        # ic_helmert_mean
        if ic_mean: #allow IC on mean
            if hasattr(sol, 'ic_mean'):
                ic_helmert_mean[isol] = sol.ic_mean
            else: #this sol has not ic_mean const
                ic_helmert_mean[isol] = ''
                
        if ic_trend: #allow IC on trend
            if hasattr(sol, 'ic_trend'):
                ic_helmert_trend[isol] = sol.ic_trend
            else: #this sol has not ic_mean const
                ic_helmert_trend[isol] = ''
                
        # Shortcut for sol.snx
        snx = sol.snx
        
        # Search keys
        snx.codept = [s.code+s.pt for s in snx.sta]

        # Update number of observations
        nobs = nobs + snx.npar

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
                
            # If periodic signals estimation need to be set up,
            if (len(periods)!=0): #at leats 1 period
                
                for per in periods: #Period object unit : day.
                    # Add new periodic parameters into combined solution
                    # Add 6 params by Period (3 cos+ 3 sin), copy correct code, soln, tref
                    combsnx.param.extend(copy.deepcopy(combsnx.param[-3:]))
                    combsnx.param.extend(copy.deepcopy(combsnx.param[-3:]))
                    
                    #order period p : ApCOSX, ApSINX, ApCOSY, ApSINY, ApCOSZ, ApSINZ
                    for (nu, dim) in enumerate(['X','Y','Z']):
                        #num correspond to Period order inside list
                        combsnx.param[-6+2*nu].type = '{}COS{}'.format(per.code, dim)
                        combsnx.param[-5+2*nu].type = '{}SIN{}'.format(per.code, dim)
                        combsnx.param[-6+2*nu].unit = 'm '
                        combsnx.param[-5+2*nu].unit = 'm '
                    
                    # Update combsnx.iper and combsnx.x0
                    # 1 id by amplitude >> 6 element
                    combsnx.iper.append(len(combsnx.param)-6) # 1 id by 6 params (consistent with ix and iv param)
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
    if (len(combsnx.ixpo+combsnx.iypo+combsnx.ixpor+combsnx.iypor+combsnx.iut+combsnx.ilod) > 0):
        combsnx.content = combsnx.content + 'E '
    if (len(snx.isatax+snx.isatay+snx.isataz) > 0):
        combsnx.content = combsnx.content + 'A '
    combsnx.content = combsnx.content[:-1]

    # Set combsnx.npar, .x0, .sig0, .N, .b and .Nc
    combsnx.npar = len(combsnx.param)
    combsnx.x0 = np.array(combsnx.x0)
    combsnx.sig0 = np.zeros(combsnx.npar)
    combsnx.b = np.zeros(combsnx.npar)
    combsnx.N = np.zeros((combsnx.npar, combsnx.npar))
    combsnx.Nc = np.zeros((combsnx.npar, combsnx.npar))
    
    # Change a priori coordinates of RF stations
    if (datum):
        combsnx.prior2ref(datum)
        
    # If velocities are going to be estimated, change a priori velocities of solns of RF stations
    # that are not part of the datum
    if (datum) and (set_vel):
        keys = [p.code+p.pt for p in [datum.param[i] for i in datum.iv]]
        for i in combsnx.iv:
            if (combsnx.param[i].code+combsnx.param[i].pt in keys) and not(np.any(combsnx.x0[i:i+3])):
                j = keys.index(combsnx.param[i].code+combsnx.param[i].pt)
                combsnx.x0[i:i+3] = datum.x[datum.iv[j]:datum.iv[j]+3]
    
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

    # Sort parameters
    combsnx.sort_params()
    
    # Reset parameter indices
    combsnx.set_par_ind()

    # Get indices of transformation parameters of each input solution
    keys = np.array([p.isol for p in [combsnx.param[i] for i in combsnx.itrans]])
    for isol in range(len(inputs)):
        ind = np.nonzero(keys == isol)[0]
        inputs[isol].itrans = [combsnx.itrans[i] for i in ind]


    # 2 - SET UP NORMAL EQUATION
    #---------------------------
    
    # Print message
    if not(quiet):
        print('', file=out)
        print('', file=out)
        print('    '+str(date())+' : Set up normal equation', file=out)

    # Initializations
    A = []
    dy = []
    
    

    # Loop over input solutions
    #--------------------------
    
    for isol in range(len(inputs)):
        sol = inputs[isol]
        
        # Print message
        if not(quiet):
            print('\x1b[2K', end='\r', file=out)
            print('        Processing input solution {0:5d}/{1} ({2})'.format(isol+1, len(inputs), sol.name), end='\r', file=out)
            
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
                
        # Add periodic signals partial derivatives and update right-hand side if needed
        if (len(periods)!=0):# at leat 1 period in list
            keys = [p.code+p.pt+p.soln for p in [combsnx.param[i] for i in combsnx.iper]]
            #add periodic eq value for each station in each solution
            for i in snx.ix:
                p = snx.param[i]
                dt = (date.from_tsnx(p.tref).mjd - mjd0) / 365.25 #year conversion
                j = combsnx.iper[keys.index(p.code+p.pt+p.soln)] 
                
                for per in periods: #Period object unit : day.
                    # Add new periodic parameters into combined solution
                    # Add 6 params by Period (3 cos+ 3 sin)
          
                    #order period p : ApCOSX, ApSINX, ApCOSY, ApSINY, ApCOSZ, ApSINZ
                    A_rows.extend([i, i+1, i+2, i+3, i+4, i+5])
                    A_cols.extend([j, j+1, j+2, j+3, j+4, j+5])
                    
                    # Add seasonal term value
                    v_cos = np.cos(2*np.pi*(1/per.value)*dt)
                    v_sin = np.sin(2*np.pi*(1/per.value)*dt)
                    
                    #order period p : ApCOSX, ApSINX, ApCOSY, ApSINY, ApCOSZ, ApSINZ
                    A_vals.extend(3*[v_cos, v_sin])
                    dy[-1][i:i+6] -= dt * combsnx.x0[j:j+6]

        # Add partial derivatives of transformation parameters
        H = snx.helmert_partials(sol.params, 'STA')
        if not(reduce_trans):
            ind = np.nonzero(H)
            A_rows.extend(ind[0].tolist())
            A_cols.extend([sol.itrans[i] for i in ind[1]])
            A_vals.extend(H[ind].tolist())

        # Build sparse design matrix of current solution
        A.append(sparse.csr_matrix((A_vals, (A_rows, A_cols)), shape=(snx.npar, combsnx.npar)))
        
        # Get weight matrix of solution isol
        if (snx.N is not None) and (snx.Nc is not None):
            P = snx.N + snx.Nc
        elif (snx.N is not None):
            P = snx.N
        else:
            P = invspd(snx.Q)
            
        # Project weight matrix if transformation parameters are reduced
        if (reduce_trans):
            HtP = np.dot(H.T, P)
            HtPHi = invspd(np.dot(HtP, H))
            P = P - np.dot(HtP.T, np.dot(HtPHi, HtP))

        # Divide weight matrix by a priori variance factor
        if (sol.sf != 1):
            P = P / sol.sf**2
        
        # Update normal equation
        AtP = A[isol].T.dot(P)
        combsnx.N += A[isol].T.dot(AtP.T)
        combsnx.b += np.dot(AtP, dy[isol])
        
        # Make room if needed
        if not(store_inputs):
            del sol.snx
            

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
    
    if (datum):
            
        # Add minimal constraints to station positions
        if (mc_sta):
            if not(quiet):
                print('        Add minimal constraints to station positions', file=out)
            nc += combsnx.add_mc(mc_sta, 'STA', sigma=mc_sta_sig, datum=datum, thr=mc_sta_thr)

        # Add minimal constraints to station velocities
        if (mc_vel):
            if not(quiet):
                print('        Add minimal constraints to station velocities', file=out)
            nc += combsnx.add_mc(mc_vel, 'VEL', sigma=mc_vel_sig, datum=datum, thr=mc_vel_thr)
            
        
    # Add internal constraints
    #------------------------
             
    # Add internal constraints to MEAN
    if (ic_mean):
        if not(quiet):
            print('        Add mean internal constraints', file=out)
            print('ic_healmert_mean dict : {}'.format(ic_helmert_mean), file=out)
        nc += combsnx.add_ic(ic_helmert_mean, 'MEAN', sigma=ic_mean_sig)

    # Add internal constraints to TREND
    if (ic_trend):
        if not(quiet):
            print('        Add trend internal constraints', file=out)
            print('ic_healmert_trend dict : {}'.format(ic_helmert_trend), file=out)
        nc += combsnx.add_ic(ic_helmert_trend, 'TREND', sigma=ic_trend_sig, t0=tref)


    # Add constraints between successive station velocities
    #------------------------------------------------------
    
    if (set_vel):
        if not(quiet):
            print('        Add constraints between successive station velocities', file=out)
        nc += combsnx.add_dvc(solns, dv_sig)
        
    # Add constraints between successive station amplitudes (periodic signals)
    #---------------------------------------------------------------------------
    if (len(periods)!=0):
        if not(quiet):
            print('        Add constraints between successive station amplitudes', file=out)
        nc += combsnx.add_dpc(solns, dv_sig)




    # 4 - SOLVE NORMAL EQUATION
    #--------------------------

    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Solve normal equation', file=out)

    # Solve normal equation
    combsnx.neqinv(clear_neq=clear_neq)



    # 5 - COMPUTE RESIDUALS AND STATISTICS
    #-------------------------------------

    # Print message
    if not(quiet):
        print('', file=out)
        print('    '+str(date())+' : Compute residuals and statistics', file=out)

    # Initializations
    dx = combsnx.x - combsnx.x0
    vPv = 0
    ntrans = 0
    
    
    
    # Loop over input solutions
    #--------------------------
    
    for isol in range(len(inputs)):
        sol = inputs[isol]
        
        # Print message
        if not(quiet):
            print('\x1b[2K', end='\r', file=out)
            print('        Processing input solution {0:5d}/{1} ({2})'.format(isol+1, len(inputs), sol.name), end='\r', file=out)

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
        Q = snx.Q
        if (sol.sf != 1):
            Q = Q * sol.sf**2
        
        # Get weight matrix of input solution
        if (snx.N is not None) and (snx.Nc is not None):
            P = snx.N + snx.Nc
            if (sol.sf != 1):
                P = P / sol.sf**2
        elif (snx.N is not None):
            P = snx.N
            if (sol.sf != 1):
                P = P / sol.sf**2
        else:
            P = invspd(Q)
            
        # If transformation parameters were reduced, project residuals
        # and update number of reduced transformation parameters
        if (reduce_trans):
            H = snx.helmert_partials(sol.params, 'STA')
            HtP = np.dot(H.T, P)
            HtPHi = invspd(np.dot(HtP, H))
            sol.v = sol.v - np.dot(H, np.dot(HtPHi, np.dot(HtP, sol.v)))
            ntrans = ntrans + H.shape[1]
            
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
        
        # Update total weighted squared sum of residuals
        vPv = vPv + sol.vPv

        # Compute solution variance factor
        if not(reduce_trans) and (vce == 'correct'):
            sol.tr = trdot(Ql, P)
        else:
            sol.tr = 0
        sol.vf = sol.vPv / (snx.npar - sol.tr)
        
        # Rotate residuals into ENH frames and compute variances of ENH observations
        s2 = np.diag(Q).copy()
        for i in snx.ix:
            R = xyz2enh(snx.x[i:i+3])
            sol.v[i:i+3] = np.dot(R, sol.v[i:i+3])
            s2[i:i+3] = np.diag(np.dot(R, np.dot(Q[i:i+3,i:i+3], R.T)))
            if (norm_res == 'correct'):
                sol.sv[i:i+3] = np.sqrt(np.diag(np.dot(R, np.dot(Qv[i:i+3,i:i+3], R.T))))
            else:
                sol.sv[i:i+3] = np.sqrt(s2[i:i+3])
            sol.vn[i:i+3] = sol.v[i:i+3] / sol.sv[i:i+3]
            
        # Indices of station coordinates and geocenter coordinates
        ix = np.array([[i, i+1, i+2] for i in snx.ix])
        igc = np.array([[i, i+1, i+2] for i in snx.igc])
                
        # Compute WRMS of ENH residuals and median ENH formal errors
        sol.wrms = np.zeros(3)
        sol.sigm = np.zeros(3)
        for i in range(3):
            sol.wrms[i] = sqrt(np.sum(sol.v[ix[:,i]]**2/s2[ix[:,i]]) / np.sum(1/s2[ix[:,i]]))
            sol.sigm[i] = np.median(np.sqrt(s2[ix[:,i]]))

        # Convert residuals, WRMS and median formal errors into mm
        sol.v[ix] = 1000*sol.v[ix]
        sol.sv[ix] = 1000*sol.sv[ix]
        if (len(igc) > 0):
            sol.v[igc] = 1000*sol.v[igc]
            sol.sv[igc] = 1000*sol.sv[igc]
        sol.wrms = 1000*sol.wrms
        sol.sigm = 1000*sol.sigm

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
        sol.sv = sol.sv * sqrt(vf)
        sol.vn = sol.vn / sqrt(vf)
        sol.sigm = sol.sigm * sqrt(vf)
        
    # Update combined solution with global variance factor
    combsnx.Nc = combsnx.Nc / vf 
    combsnx.Q = combsnx.Q * vf
    combsnx.sig = np.sqrt(np.diag(combsnx.Q))
    
    # Set content of SOLUTION/STATISTICS block
    combsnx.stats = record()
    combsnx.stats.nobs = nobs + nc
    combsnx.stats.nunk = combsnx.npar + ntrans
    combsnx.stats.vf = vf
    


    # Print statistics
    #-----------------

    if not(quiet):
        print('', file=out)
        print('', file=out)
        print('        Combination statistics', file=out)
        print('        ----------------------', file=out)
        print('', file=out)
        print('              |                                                         |', file=out)
        print('         sol_ | nobs__ tr/npar___ vPv_______ prior_SF fact_SF_ post_SF_ |', file=out)
        print('        ------|---------------------------------------------------------|', file=out)
        for isol in range(len(inputs)):
            sol = inputs[isol]
            name = '{0:4}'.format(sol.name)[:4]
            print('         {0} | {1.nobs:6d} {1.tr:10.3f} {1.vPv:10.3f} {2:8.3f} {3:8.3f} {4:8.3f} |'.format(name, sol, sol.sf, sqrt(sol.vf), sol.sf*sqrt(sol.vf)), file=out)
        print('        ------|---------------------------------------------------------|', file=out)
        print('         comb | {0:6d} {1:10.3f} {2:10.3f}          sigma0 = {3:8.3f} |'.format(nobs, combsnx.npar+ntrans, vPv, sqrt(vf)), file=out)
        print('', file=out)
        print('              |         WRMS [mm]          |      median sigma [mm]     |', file=out)
        print('         sol_ | East____ North___ Up______ | East____ North___ Up______ |', file=out)
        print('        ------|----------------------------|----------------------------|', file=out)
        for isol in range(len(inputs)):
            sol = inputs[isol]
            name = '{0:4}'.format(sol.name)[:4]
            print('         {0} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(name, sol.wrms, sol.sigm), file=out)
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
def combine_iter(inputs, tref, solns=None, check_solns=True, psd=None, set_vel=False, periods=[], dv_sig=1e-6, stack_gc=False, stack_sc=False, datum=None, 
                 mc_sta=None, mc_sta_sig=1e-5, mc_sta_thr=None, mc_vel=None, mc_vel_sig=1e-6, mc_vel_thr=None,
                 ic_mean=False, ic_mean_sig=1e-5, ic_trend=False, ic_trend_sig=1e-6, #Internal constraints
                 update_sf=False, norm_res='correct', vce='correct', store_inputs=True, reduce_trans=False, clear_neq=True,
                 thr_raw=None, thr_norm=None,  thr_abs_E=None, thr_abs_N=None, thr_abs_H=None, flag_once=False, quiet=False, out=sys.stdout):

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
    periods: list of objects (built with pytrf.utils.Period), optional
        Period of periodic signals.
    dv_sig : float, optional
        Sigma of equality constraints to be applied between successive velocities [m/y].
        Default is 1e-6.
    stack_gc : bool, optional
        Whether successive geocenter coordinates should be stacked into single
        combined geocenter coordinates. Default is False.
    stack_sc : bool, optional
        Whether successive scale factors should be stacked into a single
        combined scale factor. Default is False.
    datum : str or sinex instance, optional
        [File containing] datum. Default is None.
    
    ---------- MINIMAL constraints parameters ----------
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
        detailed explanations.
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
        detailed explanations.
    
    ---------- INTERNAL constraints parameters ----------
    ic_mean : bool, optional. Default: False
        Boolean indicating if you allow internal constraints to be applied to mean(s) of parameter(s).
        If True, you must specify the "ic_mean" parameter in the YAML file for each solution that you want apply IC.
        This param can be composed of any combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations).
        If no "ic_mean" attribute or equal to empty str ("") in YAML file, IC are not apply for this solution.
    ic_mean_sig : float or str, optional
        Sigma of internal constraints to be applied to mean(s) of parameter(s), in m. Default is 1e-5.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc().
    ic_trend : bool, optional. Default: False
        Boolean indicating if you allow internal constraints to be applied to trend(s) of parameter(s).
        If True, you must specify the "ic_trend" parameter in the YAML file for each solution that you want apply IC.
        This param can be composed of any combination of letters 'T' (translations), 'S' (scale) and 'R' (rotations).
        If no "ic_trend" attribute or equal to empty str ("") in YAML file, IC are not apply for this solution.
    ic_trend_sig : float, optional
        Sigma of internal constraints to be applied to trend(s) of parameter(s), in m/y. Default is 1e-6.
        It can also be set to 'auto' in which case an adequate sigma will be automatically set
        by sinex.add_mc(). 
    
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
        ! WARNING !: in case of internal constraints (i.e. ic_trend or ic_mean == True), 'reduce_trans' must be 'False'.
        In any case it is automatically reset to 'False' with internal constraints by combine().
    clear_neq : bool, optional
        Whether normal equation should be kept in combined sinex object. Default is True.
    thr_raw : float, optional
        Multiplicative factor defining thresholds for flagging stations with large residuals
        as outliers: along each ENH component, threshold = thr_raw * WRMS.
        Default is None.
    thr_norm : float, optional
        Threshold for flagging station with large normalized residuals as outliers.
    thr_abs_E, thr_abs_N, thr_abs_H : float, optional
            Absolute threshold for respectively east, north and up positional residuals
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
        combsnx = combine(inputs=inputs, tref=tref, solns=solns, check_solns=check_solns, psd=psd, set_vel=set_vel, periods=periods, dv_sig=dv_sig, stack_gc=stack_gc, stack_sc=stack_sc, datum=datum,
                          mc_sta=mc_sta, mc_sta_sig=mc_sta_sig, mc_sta_thr=mc_sta_thr, mc_vel=mc_vel, mc_vel_sig=mc_vel_sig, mc_vel_thr=mc_vel_thr,
                          ic_mean=ic_mean, ic_mean_sig=ic_mean_sig, ic_trend=ic_trend, ic_trend_sig=ic_trend_sig,
                          update_sf=update_sf, norm_res=norm_res, vce=vce, store_inputs=store_inputs, reduce_trans=reduce_trans, clear_neq=clear_neq, quiet=quiet, out=out)
        
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
            if (thr_abs_E):
                sol.iout.extend(np.nonzero(np.abs(sol.v[ix[:,0]]) > thr_abs_E)[0].tolist())
            if (thr_abs_N):
                sol.iout.extend(np.nonzero(np.abs(sol.v[ix[:,1]]) > thr_abs_N)[0].tolist())
            if (thr_abs_H):
                sol.iout.extend(np.nonzero(np.abs(sol.v[ix[:,2]]) > thr_abs_H)[0].tolist())

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
                sol.staout.extend(sol.codeout)
                sol.resout.extend(sol.vout)
                
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
    
