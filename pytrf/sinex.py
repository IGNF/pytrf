"""
    Class for reading, writing and manipulating SINEX files
"""

# External imports
#-----------------
import os
import sys
import re
import logging
import networkx as nx
#import mkl
#mkl.set_num_threads(1)
import copy
import yaml
import json
import pickle
import pandas as pd
from math import pi, sqrt, cos, sin, acos, exp, log
import numpy as np
from scipy import sparse
import matplotlib.pyplot as pp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Internal imports
#-----------------
from pytrf import date
from pytrf.math import cart2geo, geo2cart, xyz2enh, invspd, cholesky, cholsolve, cov2corr
from pytrf.io import get_sitelog, read_sitelog, read_yaml
from pytrf.utils import record, isfloat, earlier, station_map, Period
from pytrf.const import default_domes, ae, mas2rad, ms2rad, dera_dt



# Some useful functions
#----------------------

def write_mat(M, f):
    
    '''
    Write matrix in SINEX format
    
    Parameters
    ----------
    M : array_like
        (Square) matrix
    f : file
        Output file
    '''
    
    for i in range(len(M)):
        j = 0
        while (j <= i):
            if (M[i,j] != 0):
                f.write(' {0:5} {1:5} {2:21.14e}'.format(i+1, j+1, M[i,j]))
                if ((j+2 <= i) and (M[i,j+2] != 0)):
                    f.write(' {0:21.14e}'.format(M[i,j+1]))
                    f.write(' {0:21.14e}\n'.format(M[i,j+2]))
                    j = j+3
                elif ((j+1 <= i) and (M[i,j+1] != 0)):
                    f.write(' {0:21.14e}\n'.format(M[i,j+1]))
                    j = j+2
                else: 
                    f.write('\n')
                    j = j+1
            else:
                j = j+1



# sinex class
#------------
class sinex:
  
    """
    Class for reading, writing, manipulating SINEX files

    A sinex instance is initialized one of the following ways:
    
        snx = sinex.read(sinex_file)
        snx = sinex.load(pickle_file)

    Once initialized, each sinex instance can have the following attributes:

        file     : Original file name
        version  : SINEX format version
        agency   : Agency
        t        : File creation date
        start    : Data start time
        end      : Data end time
        tech     : Technique(s) code
        npar     : Number of parameters
        const    : Constraint code
        content  : Parameter types
        ref      : Content of FILE/REFERENCE block
        comment  : Content of FILE/COMMENT block
        input    : Content of INPUT/HISTORY and INPUT/FILES blocks
        acks     : Content of INPUT/ACKNOWLEDGEMENTS block
        stats    : Content of SOLUTION/STATISTICS block
        sta      : List of stations with their metadata (content of SITE/ID, SITE/RECEIVER, SITE/ANTENNA, SITE/ECCENTRICITY and SOLUTION/EPOCHS blocks)
        rs       : List of radiosources
        param    : List of parameters
        x        : Estimated parameter values
        sig      : Estimated parameter sigmas
        x0       : A priori parameter values
        sig0     : A priori parameter sigmas
        Q        : Covariance matrix of estimated parameters
        N        : Normal matrix of observations
        b        : Right-hand side of normal equation
        Nc       : Normal matrix of constraints

    Each sinex instance has the following methods:

        clean_prior()      : Make a priori information consistent with set of estimated parameters
        clean_sta()        : Clean station list (remove stations and solns that do not correspond to parameters)
        clean_rs()         : Clean radiosource list (remove radiosources that do not correspond to parameters)
        dist_stations()    : Provides approximative distance [meter] between two stations
        sort_params()      : Sort parameters
        is_period()        : Check if a snx parameter type belongs to "periodic type"
        set_par_ind()      : Set indices of parameter categories
        write()            : Write sinex instance into SINEX file
        dump()             : Dump sinex instance into pickle file
        copy()             : Copy sinex instance
        check_staid()      : Check station PT codes and DOMES numbers
        check_solns()      : Check solution numbers (solns) in an "instantaneous" solution
        check_epochs()     : Check parameter reference epochs and SOLUTION/EPOCHS block in an "instantaneous" solution
        check_metadata()   : Check receivers, antennas and eccentricities against site logs
        get_par_ind()      : Get indices of parameters of specified types
        get_sta_ind()      : Get indices of coordinates of specified station
        get_vel_ind()      : Get indices of velocities of specified station
        get_per_ind()      : Get indices of periods  of specified station
        get_rs_ind()       : Get indices of coordinates of specified radiosources
        get_common_par()   : Get indices of common parameters between two solutions
        get_common_sta()   : Get indices of common station positions between two solutions
        get_common_vel()   : Get indices of common station velocities between two solutions
        get_common_per()   : Get indices of common stations periods between two solutions
        get_common_rs()    : Get indices of common radiosource coordinates between two solutions
        get_xyz()          : Get cartesian coordinates of specified stations
        get_plh()          : Get geographical coordinates of specified stations
        get_lonlat()       : Get longitudes and latitudes of specified stations
        get_sigenh()       : Get ENH formal errors of specified stations
        get_core_sta()     : Get list of available core RF stations
        helmert_partials() : Get partial derivative matrix of Helmert parameters
        del_ind()          : Delete (reduce) parameters with specified indices
        del_params()       : Delete (reduce) parameters of specified types
        del_unknown_par()  : Delete parameters that are not supported by snxcomb
        del_helmerts()     : Reduce origin, scale and/or orientation information in a normal equation
        del_sta()          : Delete (reduce) specified stations
        del_rs()           : Delete (reduce) specified radiosources
        del_duplicates()   : Delete (reduce) solution numbers (solns) if there are many of them in an "instantaneous" solution
        keep_sta()         : Keep specified stations - Delete (reduce) other stations
        keep_rs()          : Keep specified radiosources - Delete (reduce) other radiosources
        trim_params()      : Delete (reduce) parameters that do not belong to period of interest
        trim_solns()       : Delete (reduce) solns that are not relevant for specified date
        trim_metadata()    : Delete metadata that are not relevant for specified period
        unconstrain()      : Recover unconstrained normal equation
        clear_const()      : Clear constraints
        loose_const()      : Add loose constraints
        fix_ind()          : Fix parameters with specified indices in a normal equation
        fix_params()       : Fix parameters of specified types in a normal equation
        setup_gc()         : Set up geocenter coordinates in a normal equation
        prior2ref()        : Set a priori parameter values to reference values
        add_mc()           : Add NNR, NNT and/or NNS constraints to normal matrix of constraints
        add_ic()           : Add R, S, T internal constraints to normal matrix of constraints. Constraints possible on MEAN and/or TREND.
        add_dvc()          : Add equality constraints between successive velocities to normal matrix of constraints
        add_dac()          : Add equality constraints between successive amplitudes (periodic signals) to normal matrix of constraints
        vfconst_file()     : Generate vfconst.yml file (velocities constraints + frequencies constraints) for stations located on the same site.
        add_vfconst()      : Add constraints between stations
        neqinv()           : Invert normal equation
        compare()          : Helmert comparison between two solutions
        get_outliers()     : Get list of outliers from Helmert comparison or combination
        compare_iter()     : Iterative Helmert comparison between two solutions
        propagate()        : Propagate station positions to specified date
        get_psd()          : Compute post-seismic deformation of given station at given date
        add_psd()          : Add or remove post-seismic deformation models to a solution
        get_seas()         : Compute seasonal signal of given station at given date
        add_seas()         : Add seasonal signals to a solution, from a sinex file
        calib_lod()        : Calibrate LOD estimates wrt reference series        
        map()              : Draw station map
        map_res()          : Draw station position residual map
        print_table()      : Print table of parameters
        print_coord()      : Print table of (instantaneous) station positions
        status3()          : Print and write status3.out table (catref equivalent), with a summary of transformation parameters
    """

    # Initialize a sinex instance
    #----------------------------
    def __init__(snx):
      
        """
        Initialize a sinex instance

        Returns
        -------
        snx : sinex instance
        
        """
        
        snx.file = None
        snx.version = None
        snx.agency = None
        snx.t = None
        snx.start = None
        snx.end = None
        snx.tech = None
        snx.npar = None
        snx.const = None
        snx.content = None
        snx.ref = None
        snx.comment = None
        snx.input = None
        snx.acks = None
        snx.stats = None
        snx.sta = None
        snx.rs = None
        snx.param = None
        snx.x = None
        snx.sig = None
        snx.x0 = None
        snx.sig0 = None
        snx.Q = None
        snx.N = None
        snx.b = None
        snx.Nc = None

        snx.ix = []
        snx.iv = []
        snx.ipsd = []
        snx.iper = [] 
        snx.iper_dict = {} #dict of list for each Period
        snx.irs = []
        snx.ixpo = []
        snx.ixpor = []
        snx.iypo = []
        snx.iypor = []
        snx.iut = []
        snx.ilod = []
        snx.inutx = []
        snx.inuty = []
        snx.igc = []
        snx.isc = []
        snx.isatax = []
        snx.isatay = []
        snx.isataz = []
        snx.itrans = []
        
    # Create sinex instance from SINEX file
    #--------------------------------------
    @classmethod
    def read(self, file, dont_read=[]):
      
        """
        Create sinex instance from SINEX file

        Returns
        -------
        snx : sinex instance

        Parameters
        ----------
        file : str
            SINEX file to read
        dont_read : list
            List of keywords to indicate which blocks should not be read.
            dont_read can include the following keywords:
              - 'matrices' in order not to read matrices
              - 'comments' in order not to read all "comment" blocks
              - 'apriori'  in order not to read a priori information
              - 'metadata' in order not to read receivers, antennas...
              - 'stats' in order not to read SOLUTION/STATISTICS block
        
        """
        
        # Initialization
        snx = sinex()
        snx.file = os.path.basename(file)

        # Open input SINEX file
        f = open(file, encoding='ISO-8859-1')

        # Read 1st line
        line = f.readline()
        snx.version = line[6:10]
        snx.agency = line[11:14]
        snx.t = re.sub(' ', '0', line[15:27])
        snx.start = re.sub(' ', '0', line[32:44])
        snx.end = re.sub(' ', '0', line[45:57])
        snx.tech = line[58]
        snx.const = line[66]
        snx.content = line[68:].strip()

        # Read rest of the file to get list and addresses of blocks
        # Also get types of the matrices (COVA/CORR/INFO)
        blocks = []
        addresses = []
        while (line):
            if (line[0] == '+'):
                blocks.append(line[1:line.find(' ')].strip())
                addresses.append(f.tell())
                if (blocks[-1] == 'SOLUTION/MATRIX_ESTIMATE'):
                    type_matest = line[28:32]
                elif (blocks[-1] == 'SOLUTION/MATRIX_APRIORI'):
                    type_matapr = line[27:31]
                elif (blocks[-1] == 'SOLUTION/ESTIMATES'):
                    blocks[-1] = 'SOLUTION/ESTIMATE'
            line = f.readline()
            
        # Read FILE/REFERENCE block -> snx.ref
        if ('FILE/REFERENCE' in blocks) and not('comments' in dont_read):
            snx.ref = record()
            f.seek(addresses[blocks.index('FILE/REFERENCE')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    setattr(snx.ref, line[1:19].strip().lower(), line[20:].strip())
                line = f.readline()
            
        # Read FILE/COMMENT block -> snx.comment
        if ('FILE/COMMENT' in blocks) and not('comments' in dont_read):
            snx.comment = []
            f.seek(addresses[blocks.index('FILE/COMMENT')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    snx.comment.append(line[1:].rstrip())
                line = f.readline()

        # Read INPUT/HISTORY block -> snx.input
        if ('INPUT/HISTORY' in blocks) and not('comments' in dont_read):
            snx.input = []
            f.seek(addresses[blocks.index('INPUT/HISTORY')])
            line = f.readline()            
            while (line[0] != '-'):
                if ((line[0] != '*') and (line[1:2] == '+')):
                    r = record()
                    r.version = line[6:10]
                    r.agency = line[11:14]
                    r.t = line[15:27]
                    r.start = line[32:44]
                    r.end = line[45:57]
                    r.tech = line[58:59]
                    r.npar = int(line[60:65])
                    r.const = line[66:67]
                    r.content = line[68:].strip()
                    snx.input.append(r)
                line = f.readline()

        # Read INPUT/FILES block -> complement snx.input
        if ('INPUT/FILES' in blocks) and not('comments' in dont_read):
            f.seek(addresses[blocks.index('INPUT/FILES')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    agency = line[1:4]
                    t = line[5:17]
                    if (agency+t in [inp.agency+inp.t for inp in snx.input]):
                        i = [inp.agency+inp.t for inp in snx.input].index(agency+t)
                        snx.input[i].file = line[18:47]
                        snx.input[i].description = line[48:].strip()
                line = f.readline()

        # Read INPUT/ACKNOWLEDGEMENTS block -> snx.acks
        if ('INPUT/ACKNOWLEDGEMENTS' in blocks) and not('comments' in dont_read):
            snx.acks = []
            f.seek(addresses[blocks.index('INPUT/ACKNOWLEDGEMENTS')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    r = record()
                    r.agency = line[1:4]
                    r.description = line[5:].strip()
                    snx.acks.append(r)
                line = f.readline()

        # Read SOLUTION/STATISTICS block
        if ('SOLUTION/STATISTICS' in blocks) and not('stats' in dont_read):
            snx.stats = record()
            f.seek(addresses[blocks.index('SOLUTION/STATISTICS')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    if ('NUMBER OF OBSERVATIONS' in line[1:31]):
                        snx.stats.nobs = round(float(line[32:].replace('D', 'E')))
                    elif ('NUMBER OF UNKNOWNS' in line[1:31]):
                        snx.stats.nunk = round(float(line[32:].replace('D', 'E')))
                    elif ('SAMPLING INTERVAL' in line[1:31]):
                        snx.stats.sampling = float(line[32:].replace('D', 'E'))
                    elif ('SQUARE SUM OF RESIDUALS' in line[1:31]):
                        snx.stats.vPv = float(line[32:].replace('D', 'E'))
                    elif ('PHASE MEASUREMENTS SIGMA' in line[1:31]):
                        snx.stats.sigphase = float(line[32:].replace('D', 'E'))
                    elif ('CODE MEASUREMENTS SIGMA' in line[1:31]):
                        snx.stats.sigcode = float(line[32:].replace('D', 'E'))
                    elif ('NUMBER OF DEGREES OF FREEDOM' in line[1:31]):
                        snx.stats.dof = round(float(line[32:].replace('D', 'E')))
                    elif ('VARIANCE FACTOR' in line[1:31]):
                        snx.stats.vf = float(line[32:].replace('D', 'E'))
                    elif ('WEIGHTED SQUARE SUM OF O-C' in line[1:31]):
                        snx.stats.lPl = float(line[32:].replace('D', 'E'))
                    elif ('WRMS OF POSTFIT RESIDUALS' in line[1:31]):
                        snx.stats.wrms = float(line[32:].replace('D', 'E'))
                line = f.readline()

        # Read SITE/ID block -> snx.sta
        if ('SITE/ID' in blocks):
            snx.sta = []
            f.seek(addresses[blocks.index('SITE/ID')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    r = record()
                    r.code = line[1:5].upper()
                    r.pt = line[6:8]
                    r.domes = line[9:18]
                    r.tech = line[19:20]
                    r.description = line[21:43]
                    r.lon = line[44:55]
                    r.lat = line[56:67]
                    if (r.lat[0:4] == '  0-'):
                        r.lat = ' -0 ' + r.lat[4:]
                    r.h = line[68:75]
                    r.rec = []
                    r.ant = []
                    r.ecc = []
                    r.soln = []
                    snx.sta.append(r)
                line = f.readline()

        # Read SOURCE/ID block -> snx.rs
        if ('SOURCE/ID' in blocks):
            snx.rs = []
            f.seek(addresses[blocks.index('SOURCE/ID')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    r = record()
                    r.code = line[1:5]
                    r.iers = line[6:14]
                    r.icrf = line[15:31]
                    r.comments = line[32:].strip()
                    snx.rs.append(r)
                line = f.readline()

        # Read SITE/RECEIVER block -> snx.sta[*].rec
        if ('SITE/RECEIVER' in blocks) and not('metadata' in dont_read):
            f.seek(addresses[blocks.index('SITE/RECEIVER')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    code = line[1:5].upper()
                    pt = line[6:8]
                    
                    # PATCH: Add possibly missing station into snx.sta
                    if (not(code+pt in [s.code+s.pt for s in snx.sta])):
                        r = record()
                        r.code = code
                        r.pt = pt
                        r.domes = default_domes
                        r.tech = 'P'
                        r.description = 22*' '
                        r.lon = 11*' '
                        r.lat = 11*' '
                        r.h = 7*' '
                        r.rec = []
                        r.ant = []
                        r.ecc = []
                        r.soln = []
                        snx.sta.append(r)
                    
                    i = [s.code+s.pt for s in snx.sta].index(code+pt)
                    r = record()
                    r.start = line[16:28]
                    r.end = line[29:41]
                    r.type = line[42:62]
                    r.serie = line[63:68]
                    r.firmware = '{0:<11s}'.format(line[69:].strip())
                    snx.sta[i].rec.append(r)
                line = f.readline()

        # Read SITE/ANTENNA block -> snx.sta[*].ant
        if ('SITE/ANTENNA' in blocks) and not('metadata' in dont_read):
            f.seek(addresses[blocks.index('SITE/ANTENNA')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    code = line[1:5].upper()
                    pt = line[6:8]
                    i = [s.code+s.pt for s in snx.sta].index(code+pt)
                    r = record()
                    r.start = line[16:28]
                    r.end = line[29:41]
                    r.type = line[42:62]
                    r.serie = line[63:68]
                    if (len(line) > 72):
                        if (isfloat(line[69:])):
                            r.daz = '{0:4d}'.format(round(float(line[69:])))
                        else:
                            r.daz = '   0'                        
                    else:
                        r.daz = '   0'
                    snx.sta[i].ant.append(r)
                line = f.readline()

        # Read SITE/ECCENTRICITY block -> snx.sta[*].ecc
        if ('SITE/ECCENTRICITY' in blocks) and not('metadata' in dont_read):
            f.seek(addresses[blocks.index('SITE/ECCENTRICITY')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    code = line[1:5].upper()
                    pt = line[6:8]
                    i = [s.code+s.pt for s in snx.sta].index(code+pt)
                    r = record()
                    r.start = line[16:28]
                    r.end = line[29:41]
                    r.system = line[42:45]
                    r.dx = [0]*3
                    r.dx[0] = line[46:54]
                    r.dx[1] = line[55:63]
                    r.dx[2] = line[64:72]
                    snx.sta[i].ecc.append(r)
                line = f.readline()

        # Read SOLUTION/EPOCHS block -> snx.sta[*].soln
        if ('SOLUTION/EPOCHS' in blocks):
            f.seek(addresses[blocks.index('SOLUTION/EPOCHS')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    code = line[1:5].upper()
                    pt = line[6:8]
                    if (code+pt in [s.code+s.pt for s in snx.sta]):
                        i = [s.code+s.pt for s in snx.sta].index(code+pt)
                        r = record()
                        r.soln = line[9:13]
                        r.datastart = re.sub(' ', '0', line[16:28])
                        r.dataend = re.sub(' ', '0', line[29:41])
                        r.datamean = re.sub(' ', '0', line[42:54])
                        snx.sta[i].soln.append(r)
                line = f.readline()

        # PATCH: Add soln field for possibly missing stations in SOLUTION/EPOCHS block
        if (snx.sta):
            for s in snx.sta:
                if (len(s.soln) == 0):
                    r = record()
                    r.soln = '----'
                    r.datastart = '00:000:00000'
                    r.dataend = '00:000:00000'
                    r.datamean = '00:000:00000'
                    s.soln.append(r)
                    
        # Read SOLUTION/ESTIMATE block -> snx.param, snx.x and snx.sig
        if ('SOLUTION/ESTIMATE' in blocks):
            snx.param = []
            snx.x = []
            snx.sig = []
            f.seek(addresses[blocks.index('SOLUTION/ESTIMATE')])
            line = f.readline()
            i = -1
            while (line[0] != '-'):
                if (line[0] != '*'):
                    i = i+1
                    r = record()
                    r.type = line[7:13]
                    r.code = line[14:18].upper()
                    r.pt = line[19:21]
                    r.soln = line[22:26]
                    r.tref = re.sub(' ', '0', line[27:39])
                    r.unit = line[40:44]
                    r.const = line[45:46]
                    snx.param.append(r)
                    snx.x.append(float(line[47:68].replace('D', 'E')))
                    snx.sig.append(float(line[69:80].replace('D', 'E')))
                line = f.readline()              
            snx.x = np.array(snx.x)
            snx.sig = np.array(snx.sig)
            snx.npar = len(snx.param)
            
        # Read SOLUTION/APRIORI block -> snx.prior, snx.x0 and snx.sig0
        if ('SOLUTION/APRIORI' in blocks) and not('apriori' in dont_read):
            snx.prior = []
            snx.x0 = []
            snx.sig0 = []
            f.seek(addresses[blocks.index('SOLUTION/APRIORI')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    r = record()
                    r.type = line[7:13]
                    r.code = line[14:18].upper()
                    r.pt = line[19:21]
                    r.soln = line[22:26]
                    r.tref = re.sub(' ', '0', line[27:39])
                    r.unit = line[40:44]
                    r.const = line[45:46]
                    snx.prior.append(r)
                    snx.x0.append(float(line[47:68].replace('D', 'E')))
                    snx.sig0.append(float(line[69:80].replace('D', 'E')))
                line = f.readline()
            snx.x0 = np.array(snx.x0)
            snx.sig0 = np.array(snx.sig0)
            if (snx.npar is None):
                snx.npar = len(snx.prior)
        else:
            snx.prior = None

        # PATCH: Change "UT1" a priori parameters to "UT"
        if (snx.param):
            for p in snx.param:
                if (p.type == 'UT1   '):
                    p.type = 'UT    '
        
        if (snx.prior):
            for p in snx.prior:
                if (p.type == 'UT1   '):
                    p.type = 'UT    '
            
        # Read SOLUTION/MATRIX_ESTIMATE block -> snx.Q
        if ('SOLUTION/MATRIX_ESTIMATE' in blocks) and not('matrices' in dont_read):
            Q = np.zeros((snx.npar, snx.npar))
            f.seek(addresses[blocks.index('SOLUTION/MATRIX_ESTIMATE')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    i = int(line[1:6]) - 1
                    j = int(line[7:12]) - 1
                    if (line[13:34].strip()):
                        Q[i,j] = float(line[13:34].replace('D', 'E'))
                        Q[j,i] = Q[i,j]
                    if (line[35:56].strip()):
                        Q[i,j+1] = float(line[35:56].replace('D', 'E'))
                        Q[j+1,i] = Q[i,j+1]
                    if (line[57:78].strip()):
                        Q[i,j+2] = float(line[57:78].replace('D', 'E'))
                        Q[j+2,i] = Q[i,j+2]
                line = f.readline()
            
            # Case of a covariance matrix
            if (type_matest == 'COVA'):
                snx.Q = Q

            # Case of a correlation matrix
            elif (type_matest == 'CORR'):
                d = np.diag(Q).copy()
                Q[range(snx.npar), range(snx.npar)] = 1
                snx.Q = d*(Q*d).T
                
            # Case of a normal matrix
            elif (type_matest == 'INFO'):
                snx.Q = invspd(Q)

        # Read SOLUTION/MATRIX_APRIORI block -> snx.Nc
        if ('SOLUTION/MATRIX_APRIORI' in blocks) and not('apriori' in dont_read) and not('matrices' in dont_read):
            Q = np.zeros((snx.npar, snx.npar))
            f.seek(addresses[blocks.index('SOLUTION/MATRIX_APRIORI')])
            line = f.readline()
            while (line[0] != '-'):
                if (line[0] != '*'):
                    i = int(line[1:6]) - 1
                    j = int(line[7:12]) - 1
                    Q[i,j] = float(line[13:34].replace('D', 'E'))
                    Q[j,i] = Q[i,j]
                    if (line[35:56].strip()):
                        Q[i,j+1] = float(line[35:56].replace('D', 'E'))
                        Q[j+1,i] = Q[i,j+1]
                    if (line[57:78].strip()):
                        Q[i,j+2] = float(line[57:78].replace('D', 'E'))
                        Q[j+2,i] = Q[i,j+2]
                line = f.readline()

            # Case of a covariance matrix
            if (type_matapr == 'COVA'):

                # PATCH: if there are parameters with zero a priori variances,
                if (np.any(np.diag(Q) == 0)):
                    indc = np.nonzero(np.diag(Q))[0]
                    snx.Nc = np.zeros((snx.npar, snx.npar))
                    snx.Nc[np.ix_(indc,indc)] = invspd(Q[np.ix_(indc,indc)])

                # Else,
                else:
                    snx.Nc = invspd(Q)

            # Case of a correlation matrix
            elif (type_matapr == 'CORR'):
                d = np.diag(Q).copy()
                Q[range(snx.npar), range(snx.npar)] = 1
                snx.Nc = invspd(d*(Q*d).T)
                
            # Case of a normal matrix
            elif (type_matapr == 'INFO'):
                snx.Nc = Q

        # Read SOLUTION/NORMAL_EQUATION_VECTOR block -> snx.b
        if ('SOLUTION/NORMAL_EQUATION_VECTOR' in blocks) and not('matrices' in dont_read):
            snx.b = np.zeros(snx.npar)
            f.seek(addresses[blocks.index('SOLUTION/NORMAL_EQUATION_VECTOR')])
            line = f.readline()
            i = -1
            
            try:
                while (line[0] != '-'):
                    if (line[0] != '*'):
                        i = i+1
                        snx.b[i] = float(line[47:68].replace('D', 'E'))
                    line = f.readline()
            except:
                logging.warning("Error in SINEX {}, block 'NORMAL_EQUATION_VECTOR' >> in fact 'DECOMPOSED_NORMAL_VECTOR...'".format(file))
                while (line[0] != '-'):
                    if (line[0] != '*'):
                        i = int(line[1:6]) - 1
                        snx.b[i] = float(line[7:28].replace('D', 'E'))
                    line = f.readline()

        # Read SOLUTION/DECOMPOSED_NORMAL_VECTOR block -> snx.b
        if ('SOLUTION/DECOMPOSED_NORMAL_VECTOR' in blocks) and not('matrices' in dont_read):
            snx.b = np.zeros(snx.npar)
            f.seek(addresses[blocks.index('SOLUTION/DECOMPOSED_NORMAL_VECTOR')])
            line = f.readline()
            i = -1
            while (line[0] != '-'):
                if (line[0] != '*'):
                    i = int(line[1:6]) - 1
                    snx.b[i] = float(line[7:28].replace('D', 'E'))
                line = f.readline()

        # Read SOLUTION/NORMAL_EQUATION_MATRIX (or SOLUTION/DECOMPOSED_NORMAL_MATRIX) block -> snx.N
        if (('SOLUTION/NORMAL_EQUATION_MATRIX' in blocks) or ('SOLUTION/DECOMPOSED_NORMAL_MATRIX' in blocks)) and not('matrices' in dont_read):
            snx.N = np.zeros((snx.npar, snx.npar))
            if ('SOLUTION/NORMAL_EQUATION_MATRIX' in blocks):
                f.seek(addresses[blocks.index('SOLUTION/NORMAL_EQUATION_MATRIX')])
            else:
                f.seek(addresses[blocks.index('SOLUTION/DECOMPOSED_NORMAL_MATRIX')])
            line = f.readline()            
            while (line[0] != '-'):
                if (line[0] != '*'):
                    i = int(line[1:6]) - 1
                    j = int(line[7:12]) - 1
                    snx.N[i,j] = float(line[13:34].replace('D', 'E'))
                    snx.N[j,i] = snx.N[i,j]
                    if (line[35:56].strip()):
                        snx.N[i,j+1] = float(line[35:56].replace('D', 'E'))
                        snx.N[j+1,i] = snx.N[i,j+1]
                    if (line[57:78].strip()):
                        snx.N[i,j+2] = float(line[57:78].replace('D', 'E'))
                        snx.N[j+2,i] = snx.N[i,j+2]
                line = f.readline()
                        
        # Close input SINEX file
        f.close()
        
        # In case of a normal equation without solution, copy snx.prior into snx.param
        if (snx.prior) and (snx.param is None):
            snx.param = copy.deepcopy(snx.prior)
            snx.x = np.zeros(snx.npar)
            snx.sig = np.zeros(snx.npar)
        
        # If SINEX file contains any parameter,
        if (snx.param is not None):

            # Clean prior information
            snx.clean_prior()

            # Sort parameters
            snx.sort_params()

            # Set parameter indices
            snx.set_par_ind()
            
            # Clean station list
            snx.clean_sta()
            
            # Clean radiosource list and attribute IERS names to radiource coordinate parameters
            snx.clean_rs()
            
            # Convert radiosource coordinates into mas if needed
            if (len(snx.irs) > 0):
                if ('rad' in snx.param[snx.irs[0]].unit):
                    irs = snx.irs + [i+1 for i in snx.irs]
                    f = np.ones(snx.npar)
                    f[irs] = 1/mas2rad
                    snx.x = f*snx.x
                    snx.sig = f*snx.sig
                    if (snx.Q is not None):
                        snx.Q = (snx.Q*f).T*f
                    if (snx.x0 is not None):
                        snx.x0 = f*snx.x0
                        snx.sig0 = f*snx.sig0
                    if (snx.N is not None):
                        snx.N = (snx.N/f).T/f
                        snx.b = snx.b/f
                    if (snx.Nc is not None):
                        snx.Nc = (snx.Nc/f).T/f
                    for i in irs:
                        snx.param[i].unit = 'mas '

        return snx

    # Load sinex instance from pickle file
    #-------------------------------------
    @classmethod
    def load(self, file, load_mat=True):
        
        """
        Load sinex instance from pickle file

        Returns
        -------
        snx : sinex instance

        Parameters
        ----------
        file : str
            pickle file to read
        load_mat : bool, optional
            Whether to load matrices. Default is True.
        
        """
        
        # Load everything but matrices
        snx = pickle.load(open(file, 'rb'))
        
        # Load matrices if possible and requested
        if (os.path.isfile(file+'.mat')) and (load_mat):
            mat = pickle.load(open(file+'.mat', 'rb'))
            snx.Q = mat.Q
            snx.N = mat.N
            snx.b = mat.b
            snx.Nc = mat.Nc
        
        return snx

    # Make a priori information consistent with set of estimated parameters
    #----------------------------------------------------------------------
    def clean_prior(snx):

        """
        Make a priori information consistent with set of estimated parameters

        """
        
        # If a priori information is available, 
        if (snx.prior):
            
            # Create snx.Nc if necessary
            if (snx.Nc is None):
                snx.Nc = np.zeros((len(snx.prior), len(snx.prior)))

            # Complete diagonal of snx.Nc with 1/sig0**2 for potential a priori parameters
            # that do not appear in the MATRIX_APRIORI block
            ind = np.nonzero(np.logical_and(np.diag(snx.Nc) == 0, snx.sig0 != 0))[0]
            snx.Nc[ind,ind] = 1/snx.sig0[ind]**2

            # Get indices of matching pairs of a priori / estimated parameters
            key = [p.type+p.code+p.pt+p.soln+p.tref for p in snx.param]
            key0 = [p.type+p.code+p.pt+p.soln+p.tref for p in snx.prior]
            ind = []
            ind0 = []
            for i in range(snx.npar):
                if (key[i] in key0):
                    ind.append(i)
                    ind0.append(key0.index(key[i]))
                        
            # Delete snx.prior
            del snx.prior
            
            # New snx.x0
            x0 = snx.x.copy()
            x0[ind] = snx.x0[ind0]
            snx.x0 = x0

            # New snx.sig0
            sig0 = np.zeros(snx.npar)
            sig0[ind] = snx.sig0[ind0]
            snx.sig0 = sig0

            # New snx.Nc
            Nc = np.zeros((snx.npar, snx.npar))
            Nc[np.ix_(ind, ind)] = snx.Nc[np.ix_(ind0, ind0)]
            snx.Nc = Nc
            
        # Else, clear a priori information, just in case
        else:
            snx.x0 = None
            snx.sig0 = None
            snx.Nc = None
            
        # PATCH: remove parameters with zero variances
        if (snx.Q is not None):
            if np.any(np.diag(snx.Q) == 0):
                indk = np.nonzero(np.diag(snx.Q))[0]
                snx.Q = snx.Q[np.ix_(indk, indk)]
                if (snx.N is not None):
                    snx.N = snx.N[np.ix_(indk, indk)]
                    snx.b = snx.b[indk]
                if (snx.x0 is not None):
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                    snx.Nc = snx.Nc[np.ix_(indk, indk)]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                snx.param = [snx.param[i] for i in indk]
                snx.npar = len(indk)

    # Clean station list (remove stations and solns that do not correspond to parameters)
    #------------------------------------------------------------------------------------
    def clean_sta(snx):
        
        """
        Clean station list (remove stations and solns that do not correspond to parameters)

        """

        if (snx.sta is not None):

            # PATCH: add possibly missing stations in snx.sta
            keys = [s.code+s.pt for s in snx.sta]
            for i in snx.ix+snx.ipsd+snx.iper:
                p = snx.param[i]
                if not(p.code+p.pt in keys):
                    r = record()
                    r.code = p.code
                    r.pt = p.pt
                    r.domes = default_domes
                    r.tech = 'P'
                    r.description = 22*' '
                    r.lon = 11*' '
                    r.lat = 11*' '
                    r.h = 7*' '
                    r.rec = []
                    r.ant = []
                    r.ecc = []
                    r.soln = []
                    snx.sta.append(r)

                    r = record()
                    r.soln = '----'
                    r.datastart = '00:000:00000'
                    r.dataend = '00:000:00000'
                    r.datamean = '00:000:00000'
                    snx.sta[-1].soln.append(r)
                    
            # PATCH: Try to correct solns in SOLUTION/EPOCHS block
            if (len(snx.ix) > 0):
                keys = np.array([p.code+p.pt for p in [snx.param[i] for i in snx.ix]])
                for s in snx.sta:
                    if (len(s.soln) == 1):
                        inds = np.nonzero(keys == s.code+s.pt)[0]
                        if (len(inds) == 1):
                            if (s.soln[0].soln != snx.param[snx.ix[inds[0]]].soln):
                                s.soln[0].soln = snx.param[snx.ix[inds[0]]].soln

            # Update snx.sta
            keys = [p.code+p.pt for p in [snx.param[i] for i in snx.ix+snx.ipsd+snx.iper]]
            i = 0
            while (i < len(snx.sta)):
                if not(snx.sta[i].code+snx.sta[i].pt in keys):
                    snx.sta.pop(i)
                else:
                    i = i+1

            # Update snx.sta[*].soln
            keys = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.ix]]
            for s in snx.sta:
                i = 0
                while (i < len(s.soln)):
                    if not(s.code+s.pt+s.soln[i].soln in keys):
                        s.soln.pop(i)
                    else:
                        i = i+1
                
    # Clean radiosource list (remove radiosources that do not correspond to parameters)
    #----------------------------------------------------------------------------------
    def clean_rs(snx):
        
        """
        Clean radiosource list (remove radiosources that do not correspond to parameters)
        and attribute IERS names to radiosource coordinate parameters

        """

        if (snx.rs):

            # Update snx.rs
            keys = [p.code for p in [snx.param[i] for i in snx.irs]]
            i = 0
            while (i < len(snx.rs)):
                if not(snx.rs[i].code in keys):
                    snx.rs.pop(i)
                else:
                    i = i+1
                    
            # Attribute IERS names to radiosource coordinate parameters
            keys = [r.code for r in snx.rs]
            for i in snx.irs:
                iers = snx.rs[keys.index(snx.param[i].code)].iers
                snx.param[i].iers = iers
                snx.param[i+1].iers = iers
                
                
    def dist_stations(snx, station1, station2, type_dist="sphere"):
        """
        Provides APPROXIMATIVE distance [unit: meter] between two stations.
        Based on sinex 'SOLUTION/ESTIMATE' block (or apriori coordinates)
         
        2 types of computation are possible:
            * on IAG-GRS80 SPHERE (pytrf.const.ae) -> type_dist="sphere"
            * cartesian distance -> type_dist="cartesian" 
        
        Returns
        -------
        distance: [m], float

        Parameters
        ----------
        station1 : sinex.sta object (pytrf.utils.record)
            1st station, with 'lon' & 'lat' attributes (string: 'dd mm ss')
        station2 : sinex.sta object
            2nd station, with 'lon' & 'lat' attributes (string: 'dd mm ss')
        type_dist: str, optional
            type of distance "sphere" or "cartesian"
        """
        #get stations coordinates (rad)
        lat1, lon1, h1 = snx.get_plh([station1.code], pt=[station1.pt])
        lat2, lon2, h2 = snx.get_plh([station2.code], pt=[station2.pt])
        
        if type_dist == "sphere" : #sphere distance
            if (lon1 == lon2) and (lat1==lat2): #particular case on same lat & lon
                distance = 0
            else:
                #angular distance [rad]
                s12 = np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos((lon2-lon1)))
                #distance based in IAG-GRS80 SPHERE [m]
                distance = s12 * ae
        
        elif type_dist == "cartesian":
            x1 = geo2cart(lat1, lon1, h1) #numpy array X, Y, Z --> h='  44.8 '.split() --> ['44.8']
            x2 = geo2cart(lat2, lon2, h2)
            
            distance = np.sqrt(np.sum((x2-x1)**2))
            
        
        return distance
  

    # Sort parameters
    #----------------
    def sort_params(snx):

        """
        Sort parameters
        
        Returns
        -------
        ind : array_like
            Sort indices

        """
        
        # Set keys to sort parameters
        keys = []
        for i in range(snx.npar):
            p = snx.param[i]
            if (p.type[0:3] in ['STA', 'VEL']):
                keys.append('0'+p.code+p.pt+p.soln+p.type)
            elif snx.is_period(p.type): #period type
                logging.debug(">> period"+ p.type)
                # build Period  object
                per = Period.from_snx_param(p.type)
                keys.append('0'+p.code+p.pt+p.soln+"zperiod"+per.code+per.dim+per.cs)# ex: A001CX -> per.code: A001, per.dim: 'X', per.cs: 'COS', 
            elif (p.type[0:2] == 'RS'):
                keys.append('2'+p.code+p.pt+p.soln+p.type[::-1])
            elif (p.type[0:4] == 'SATA'):
                keys.append('3'+p.code+p.pt+p.soln+p.type)
            elif (p.type == 'XPO   '):
                keys.append('40'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'XPOR  '):
                keys.append('41'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'YPO   '):
                keys.append('42'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'YPOR  '):
                keys.append('43'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'UT    '):
                keys.append('44'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'LOD   '):
                keys.append('45'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'NUT_X '):
                keys.append('46'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'NUT_Y '):
                keys.append('47'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'XGC   '):
                keys.append('50'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'YGC   '):
                keys.append('51'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'ZGC   '):
                keys.append('52'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type == 'DSC   '):
                keys.append('53'+str(date.from_tsnx(p.tref).mjd))
            elif (p.type in ['TX    ', 'TY    ', 'TZ    ', 'SC    ', 'RX    ', 'RY    ', 'RZ    ']):
                j = ['TX    ', 'TY    ', 'TZ    ', 'SC    ', 'RX    ', 'RY    ', 'RZ    '].index(p.type)
                keys.append('6'+'{0:06d}{1}'.format(int(p.soln), j))
            else:
                keys.append('9{0:06d}'.format(i))
                
        # Sort parameters
        ind = np.argsort(keys)
        ind2 = np.ix_(ind, ind)
        snx.param = [snx.param[i] for i in ind]
        if (snx.x is not None):
            snx.x = snx.x[ind]
            snx.sig = snx.sig[ind]
        if (snx.x0 is not None):
            snx.x0 = snx.x0[ind]
            snx.sig0 = snx.sig0[ind]
        if (snx.Nc is not None):
            snx.Nc = snx.Nc[ind2]
        if (snx.N is not None):
            snx.N = snx.N[ind2]
            snx.b = snx.b[ind]
        if (snx.Q is not None):
            snx.Q = snx.Q[ind2]
            
        return ind
    
    # Check if a snx parameter type belongs to "period type"
    #------------------------------------
    def is_period(snx, param_type):
        """
        Checks if a snx.param.type belongs to "periodic type"
        Compatible with previous IRF2020 format: ['A1COSX', 'A1SINX', 'A1COSY', 'A1SINY', 'A1COSZ', 'A1SINZ']
        New format:                              ['A001CX', 'A001SX', 'A001CY', 'A001SY', 'A001CZ', 'A001SZ']
        
        Criteria of "periodic type":
            *The string must be exactly 6 characters long.
            *The first character (param_type[0]) must be either 'A', 'D', or 'P'.
            *The next three characters (param_type[1:4]) must be a numeric value between 1 and 999 (inclusive), previous format only param_type[1] is int.
            *The fifth character (param_type[4]) must be either 'C' or 'S' or 'N' ('N' due to old SIN format such as A1SINX).
            *The last character (param_type[5]) must be either 'X', 'Y', or 'Z'.

        Parameters
        ----------
        param_type : str
            snx.param.type

        Returns
        -------
        bool
            True if period type, else False.

        """
        if len(param_type) != 6:
            logging.debug("[ip] no len 6")
            return False
        if param_type[0] not in ['A', 'D', 'P']:
            logging.debug("[ip] no start ADP")
            return False
        if not param_type[1].isdigit(): # work also with new format A001CX
            logging.debug("[ip] no digit1")
            return False
        if param_type[4] not in ['C', 'S', 'N']:
            logging.debug("[ip] no CSN")
            return False
        if param_type[5] not in ['X', 'Y', 'Z']:
            logging.debug("[ip] no XYZ")
            return False
        return True

    
    # Set indices of parameter categories
    #------------------------------------
    def set_par_ind(snx):

        """
        Set indices of parameter categories
        
        """
        
        if (snx.npar > 0):
        
            # Array of parameter types
            types = np.array([p.type for p in snx.param])
            types1 = np.array([p.type[1:4] for p in snx.param])
           
            # Station positions
            snx.ix = np.nonzero(types == 'STAX  ')[0].tolist()

            # Station velocities
            snx.iv = np.nonzero(types == 'VELX  ')[0].tolist()
            
            # PSD parameters
            snx.ipsd = np.nonzero(np.in1d(types1, ['EXP', 'LOG']))[0].tolist()
            
            # Periodic terms
            snx.iper = []
            snx.iper_dict = {} #reinit to empty
            for num, p in enumerate(snx.param) :
                if snx.is_period(p.type): 
                    # belongs to 'period type' according to snx.is_period() method, built Period object
                    per = Period.from_snx_param(p.type)
                    #old vs new syntax: set period type with new 'param_type' syntax (instead of 'param_type_old'). ex 'A1COSX'<>'A001CX'
                    p.type = per.param_type
                    if per.cs == 'COS' and per.dim == 'X': #bloc of 6 params, order : COSX, SINX, COSY, SINY, COSZ, SINZ
                        snx.iper.append(num)    
                        snx.iper_dict.setdefault(per.code, []).append(num) # 1 list by per.code: {"A001":[...], "A002":[...]}
                        
            # Radiosource coordinates
            snx.irs = np.nonzero(types == 'RS_RA ')[0].tolist()

            # X-pole coordinates
            snx.ixpo = np.nonzero(types == 'XPO   ')[0].tolist()

            # X-pole rates
            snx.ixpor = np.nonzero(types == 'XPOR  ')[0].tolist()

            # Y-pole coordinates
            snx.iypo = np.nonzero(types == 'YPO   ')[0].tolist()

            # Y-pole rates
            snx.iypor = np.nonzero(types == 'YPOR  ')[0].tolist()
            
            # UT1-UTC offsets
            snx.iut = np.nonzero(types == 'UT    ')[0].tolist()

            # LODs
            snx.ilod = np.nonzero(types == 'LOD   ')[0].tolist()
            
            # case "LODR"
            lodr_id = np.nonzero(types == 'LODR  ')[0].tolist() 
            if len(lodr_id) !=0 : #at least 'LODR' in snx
                snx.ilod += lodr_id
                #rename 'LODR' to 'LOD'
                for i in lodr_id:
                    snx.param[i].type = 'LOD   '
                
            # X-nutations
            snx.inutx = np.nonzero(types == 'NUT_X ')[0].tolist()

            # Y-nutations
            snx.inuty = np.nonzero(types == 'NUT_Y ')[0].tolist()

            # Geocenter coordinates
            snx.igc = np.nonzero(types == 'XGC   ')[0].tolist()
            
            # Scale factors
            snx.isc = np.nonzero(types == 'DSC   ')[0].tolist()

            # Satellite x-PCOs
            snx.isatax = np.nonzero(types == 'SATA_X')[0].tolist()

            # Satellite y-PCOs
            snx.isatay = np.nonzero(types == 'SATA_Y')[0].tolist()

            # Satellite z-PCOs
            snx.isataz = np.nonzero(types == 'SATA_Z')[0].tolist()
            
            # Transformation parameters
            snx.itrans = np.nonzero(np.in1d(types, ['TX    ', 'TY    ', 'TZ    ', 'SC    ', 'RX    ', 'RY    ', 'RZ    ']))[0].tolist()
            
        else:
            
            snx.ix = []
            snx.iv = []
            snx.ipsd = []
            snx.iper = []
            snx.iper_dict = {}
            snx.irs = []
            snx.ixpo = []
            snx.ixpor = []
            snx.iypo = []
            snx.iypor = []
            snx.iut = []
            snx.ilod = []
            snx.inutx = []
            snx.inuty = []
            snx.igc = []
            snx.isc = []
            snx.isatax = []
            snx.isatay = []
            snx.isataz = []
            snx.itrans = []

    # Write sinex instance into SINEX file
    #-------------------------------------
    def write(snx, file, dont_write=[]):
      
        """
        Write sinex instance into SINEX file

        Parameters
        ----------
        file : str
            SINEX file to write
        dont_write : list
            List of keywords to indicate which blocks should not be written.
            dont_write can include the following keywords:
              - 'matrices' in order not to write matrices
              - 'comments' in order not to write all "comment" blocks
              - 'apriori'  in order not to write a priori information
              - 'metadata' in order not to write receivers, antennas...
              - 'epochs'   in order not to write SOLUTION/EPOCHS block
              - 'stats'    in order not to write SOLUTION/STATISTICS block
        
        """

        # Open output SINEX file
        f = open(file, 'w')

        # Set snx.t if needed and write first line
        if (snx.t is None):
            snx.t = date().tsnx()
        f.write('%=SNX 2.02 {0.agency} {0.t} {0.agency} {0.start} {0.end} {0.tech} {0.npar:>5} {0.const} {0.content}\n'.format(snx))

        # Write FILE/REFERENCE block
        if (snx.ref) and not('comments' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+FILE/REFERENCE\n')
            if (hasattr(snx.ref, 'description')):
                f.write(' {0:<18} {1}\n'.format('DESCRIPTION', snx.ref.description))
            if (hasattr(snx.ref, 'output')):
                f.write(' {0:<18} {1}\n'.format('OUTPUT', snx.ref.output))
            if (hasattr(snx.ref, 'contact')):
                f.write(' {0:<18} {1}\n'.format('CONTACT', snx.ref.contact))
            if (hasattr(snx.ref, 'software')):
                f.write(' {0:<18} {1}\n'.format('SOFTWARE', snx.ref.software))
            if (hasattr(snx.ref, 'hardware')):
                f.write(' {0:<18} {1}\n'.format('HARDWARE', snx.ref.hardware))
            if (hasattr(snx.ref, 'input')):
                f.write(' {0:<18} {1}\n'.format('INPUT', snx.ref.input))
            f.write('-FILE/REFERENCE\n')

        # Write FILE/COMMENT block
        if (snx.comment) and not('comments' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+FILE/COMMENT\n')
            for c in snx.comment:
                f.write(' {0}\n'.format(c))
            f.write('-FILE/COMMENT\n')

        # Write INPUT/ACKNOWLEDGEMENTS block
        if (snx.acks) and not('comments' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+INPUT/ACKNOWLEDGEMENTS\n')
            f.write('*AGY ______________________________FULL_DESCRIPTION_____________________________\n')
            for a in snx.acks:
                f.write(' {0} {1}\n'.format(a.agency, a.description))
            f.write('-INPUT/ACKNOWLEDGEMENTS\n')

        # Write INPUT/HISTORY block
        if (snx.input) and not('comments' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+INPUT/HISTORY\n')
            f.write('*_VERSION_ CRE __CREATION__ OWN _DATA_START_ __DATA_END__ T PARAM S ____TYPE____\n')
            for i in snx.input:
                f.write(' +SNX {0.version} {0.agency} {0.t} {0.agency} {0.start} {0.end} {0.tech} {0.npar:>5} {0.const} {0.content}\n'.format(i))
            f.write(' =SNX 2.02 {0.agency} {0.t} {0.agency} {0.start} {0.end} {0.tech} {0.npar:>5} {0.const} {0.content}\n'.format(snx))
            f.write('-INPUT/HISTORY\n')

        # Write INPUT/FILES block
        if (snx.input) and not('comments' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+INPUT/FILES\n')
            f.write('*OWN __CREATION__ ___________FILENAME__________ ___________DESCRIPTION__________\n')
            for i in snx.input:
                f.write(' {0.agency} {0.t} {0.file} {0.description}\n'.format(i))
            f.write('-INPUT/FILES\n')

        # Write SOLUTION/STATISTICS block
        if (snx.stats) and not('stats' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/STATISTICS\n')
            f.write('*____STATISTICAL_PARAMETER_____ _______VALUE(S)_______\n')
            if hasattr(snx.stats, 'nobs'):
                f.write(' NUMBER OF OBSERVATIONS         {0:22d}\n'.format(snx.stats.nobs))
            if hasattr(snx.stats, 'nunk'):
                f.write(' NUMBER OF UNKNOWNS             {0:22d}\n'.format(snx.stats.nunk))
            if hasattr(snx.stats, 'dof'):
                f.write(' NUMBER OF DEGREES OF FREEDOM   {0:22d}\n'.format(snx.stats.dof))
            if hasattr(snx.stats, 'lPl'):
                f.write(' WEIGHTED SQUARE SUM OF O-C     {0:22.16e}\n'.format(snx.stats.lPl))
            if hasattr(snx.stats, 'vPv'):
                f.write(' SQUARE SUM OF RESIDUALS (VTPV) {0:22.16e}\n'.format(snx.stats.vPv))
            if hasattr(snx.stats, 'vf'):
                f.write(' VARIANCE FACTOR                {0:22.16e}\n'.format(snx.stats.vf))
            if hasattr(snx.stats, 'sampling'):
                f.write(' SAMPLING INTERVAL (SECONDS)    {0:22f}\n'.format(snx.stats.sampling))
            if hasattr(snx.stats, 'sigphase'):
                f.write(' PHASE MEASUREMENTS SIGMA       {0:22f}\n'.format(snx.stats.sigphase))
            if hasattr(snx.stats, 'sigcode'):
                f.write(' CODE MEASUREMENTS SIGMA        {0:22f}\n'.format(snx.stats.sigcode))
            f.write('-SOLUTION/STATISTICS\n')

        # Write SITE/ID block
        if (snx.sta):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SITE/ID\n')
            f.write('*CODE PT __DOMES__ T _STATION DESCRIPTION__ _LONGITUDE_ _LATITUDE__ HEIGHT_\n')
            for s in snx.sta:
                f.write(' {0.code} {0.pt} {0.domes} {0.tech} {0.description} {0.lon} {0.lat} {0.h}\n'.format(s))
            f.write('-SITE/ID\n')

        # Write SOURCE/ID block
        if (snx.rs):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOURCE/ID\n')
            f.write('*CODE IERSNAME ___ICRF_NAME____ ___________________COMMENTS_____________________\n')
            for s in snx.rs:
                f.write(' {0.code} {0.iers} {0.icrf} {0.comments}\n'.format(s))
            f.write('-SOURCE/ID\n')

        # Write SITE/RECEIVER block
        if (snx.sta) and not('metadata' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SITE/RECEIVER\n')
            f.write('*CODE PT SOLN T _DATA START_ __DATA_END__ ___RECEIVER_TYPE____ _S/N_ _FIRMWARE__\n')
            for s in snx.sta:
                for r in s.rec:
                    f.write(' {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.firmware}\n'.format(s, r))
            f.write('-SITE/RECEIVER\n')

        # Write SITE/ANTENNA block
        if (snx.sta) and not('metadata' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SITE/ANTENNA\n')
            f.write('*CODE PT SOLN T _DATA START_ __DATA_END__ ____ANTENNA_TYPE____ _S/N_ _DAZ\n')
            for s in snx.sta:
                for a in s.ant:
                    f.write(' {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.daz}\n'.format(s, a))
            f.write('-SITE/ANTENNA\n')

        # Write SITE/ECCENTRICITY block
        if (snx.sta) and not('metadata' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SITE/ECCENTRICITY\n')
            f.write('*CODE PT SOLN T _DATA START_ __DATA_END__ REF __DX_U__ __DX_N__ __DX_E__\n')
            for s in snx.sta:
                for e in s.ecc:
                    f.write(' {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.system} {2[0]} {2[1]} {2[2]}\n'.format(s, e, e.dx))
            f.write('-SITE/ECCENTRICITY\n')

        # Write SOLUTION/EPOCHS block
        if (snx.sta) and not('epochs' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/EPOCHS\n')
            f.write('*CODE PT SOLN T _DATA_START_ __DATA_END__ _MEAN_EPOCH_\n')
            for s in snx.sta:
                for i in s.soln:
                    f.write(' {0.code} {0.pt} {1.soln} {0.tech} {1.datastart} {1.dataend} {1.datamean}\n'.format(s, i))
            f.write('-SOLUTION/EPOCHS\n')
          
        # Write SOLUTION/APRIORI block
        if (snx.param) and (snx.x0 is not None) and not('apriori' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/APRIORI\n')
            f.write('*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ____APRIORI_VALUE____ __STD_DEV__\n')
            for i in range(snx.npar):
                p = snx.param[i]
                f.write(' {0:5} {1.type} {1.code} {1.pt} {1.soln} {1.tref} {1.unit} {1.const} {2:21.14e} {3:11.5e}\n'.format(i+1, p, snx.x0[i], snx.sig0[i]))
            f.write('-SOLUTION/APRIORI\n')

        # Write SOLUTION/ESTIMATE block
        if (snx.param):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/ESTIMATE\n')
            f.write('*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ___ESTIMATED_VALUE___ __STD_DEV__\n')
            for i in range(snx.npar):
                p = snx.param[i]
                f.write(' {0:5} {1.type} {1.code} {1.pt} {1.soln} {1.tref} {1.unit} {1.const} {2:21.14e} {3:11.5e}\n'.format(i+1, p, snx.x[i], snx.sig[i]))
            f.write('-SOLUTION/ESTIMATE\n')
          
        # Write SOLUTION/MATRIX_APRIORI block
        if (snx.Nc is not None) and not('matrices' in dont_write) and not('apriori' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/MATRIX_APRIORI L INFO\n')
            f.write('*PARA1 PARA2 _______PARA2+0_______ _______PARA2+1_______ _______PARA2+2_______\n')
            write_mat(snx.Nc, f)
            f.write('-SOLUTION/MATRIX_APRIORI L INFO\n')

        # Write SOLUTION/MATRIX_ESTIMATE block
        if (snx.Q is not None) and not('matrices' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/MATRIX_ESTIMATE L COVA\n')
            f.write('*PARA1 PARA2 _______PARA2+0_______ _______PARA2+1_______ _______PARA2+2_______\n')
            write_mat(snx.Q, f)
            f.write('-SOLUTION/MATRIX_ESTIMATE L COVA\n')

        # Write SOLUTION/NORMAL_EQUATION_VECTOR block
        if (snx.b is not None) and not('matrices' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/NORMAL_EQUATION_VECTOR\n')
            f.write('*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ___ESTIMATED_VALUE___\n')
            for i in range(snx.npar):
                p =  snx.param[i]
                f.write(' {0:5} {1.type} {1.code} {1.pt} {1.soln} {1.tref} {1.unit} {1.const} {2:21.14e}\n'.format(i+1, p, snx.b[i]))
            f.write('-SOLUTION/NORMAL_EQUATION_VECTOR\n')

        # Write SOLUTION/NORMAL_EQUATION_MATRIX block
        if (snx.N is not None) and not('matrices' in dont_write):
            f.write('*-------------------------------------------------------------------------------\n')
            f.write('+SOLUTION/NORMAL_EQUATION_MATRIX\n')
            f.write('*PARA1 PARA2 _______PARA2+0_______ _______PARA2+1_______ _______PARA2+2_______\n')
            write_mat(snx.N, f)
            f.write('-SOLUTION/NORMAL_EQUATION_MATRIX\n')

        # Write last line and close output SINEX file
        f.write('%ENDSNX\n')
        f.close()

    # Dump sinex instance into pickle files
    #--------------------------------------
    def dump(snx, file):
      
        """
        Dump sinex instance into pickle files

        Parameters
        ----------
        file : str
            pickle file to write
            Note that two pickle files are actually written:
            - one with everything but matrices (file)
            - one with matrices (file.mat)

        """
        
        # New "view" of sinex instance with everything required but matrices
        pkl = sinex()
        pkl.file = snx.file
        pkl.version = snx.version
        pkl.agency = snx.agency
        pkl.t = snx.t
        pkl.start = snx.start
        pkl.end = snx.end
        pkl.tech = snx.tech
        pkl.npar = snx.npar
        pkl.const = snx.const
        pkl.content = snx.content
        pkl.ref = snx.ref
        pkl.comment = snx.comment
        pkl.input = snx.input
        pkl.acks = snx.acks
        pkl.sta = snx.sta
        pkl.rs = snx.rs
        pkl.stats = snx.stats
        pkl.param = snx.param
        pkl.x = snx.x
        pkl.sig = snx.sig
        pkl.x0 = snx.x0
        pkl.sig0 = snx.sig0
        pkl.ix = snx.ix
        pkl.iv = snx.iv
        pkl.ipsd = snx.ipsd
        pkl.iper = snx.iper
        pkl.iper_dict = snx.iper_dict
        pkl.irs = snx.irs
        pkl.ixpo = snx.ixpo
        pkl.ixpor = snx.ixpor
        pkl.iypo = snx.iypo
        pkl.iypor = snx.iypor
        pkl.iut = snx.iut
        pkl.ilod = snx.ilod
        pkl.inutx = snx.inutx
        pkl.inuty = snx.inuty
        pkl.igc = snx.igc
        pkl.isc = snx.isc
        pkl.isatax = snx.isatax
        pkl.isatay = snx.isatay
        pkl.isataz = snx.isataz
        pkl.itrans = snx.itrans

        # Write 1st pickle file
        pickle.dump(pkl, open(file, 'wb'))

        # If sinex instance has any matrices
        if (snx.Q is not None) or (snx.N is not None) or (snx.Nc is not None):

            # New view of sinex instance with matrices only
            pkl = sinex()
            pkl.Q = snx.Q
            pkl.N = snx.N
            pkl.b = snx.b
            pkl.Nc = snx.Nc

            # Write 2nd pickle file
            pickle.dump(pkl, open(file+'.mat', 'wb'))

    # Copy sinex instance
    #--------------------
    def copy(snx, dont_copy=[]):
      
        """
        Copy sinex instance

        Returns
        -------
        snx2 : sinex instance

        Parameters
        ----------
        dont_copy : list
            List of keywords to indicate which attributes should not be copied.
            dont_copy can include the following keywords:
              - 'matrices' in order not to copy matrices
              - 'apriori' in order not to copy a priori information
              - 'comments' in order not to copy all "comment" blocks
        
        """
        
        # Initialize new sinex instance
        snx2 = sinex()

        # Copy 1st line attributes
        snx2.file = snx.file
        snx2.version = snx.version
        snx2.agency = snx.agency
        snx2.t = snx.t
        snx2.start = snx.start
        snx2.end = snx.end
        snx2.tech = snx.tech
        snx2.npar = snx.npar
        snx2.const = snx.const
        snx2.content = snx.content
        
        # Copy comment blocks if needed
        if not('comments' in dont_copy):
            snx2.ref = copy.deepcopy(snx.ref)
            snx2.comment = copy.deepcopy(snx.comment)
            snx2.input = copy.deepcopy(snx.input)
            snx2.acks = copy.deepcopy(snx.acks)

        # Copy lists of stations, radiosources and parameters
        snx2.sta = copy.deepcopy(snx.sta)
        snx2.rs = copy.deepcopy(snx.rs)
        snx2.param = copy.deepcopy(snx.param)
        snx2.x = copy.deepcopy(snx.x)
        snx2.sig = copy.deepcopy(snx.sig)

        # Copy a priori information if needed
        if not('apriori' in dont_copy):
            snx2.x0 = copy.deepcopy(snx.x0)
            snx2.sig0 = copy.deepcopy(snx.sig0)
            snx2.Nc = copy.deepcopy(snx.Nc)
        
        # Copy matrices if needed
        if not('matrices' in dont_copy):
            snx2.Q = copy.deepcopy(snx.Q)
            snx2.b = copy.deepcopy(snx.b)
            snx2.N = copy.deepcopy(snx.N)

        # Copy statistics
        snx2.stats = snx.stats
        
        # Copy parameter indices
        snx2.ix = snx.ix.copy()
        snx2.iv = snx.iv.copy()
        snx2.ipsd = snx.ipsd.copy()
        snx2.iper= snx.iper.copy()
        snx2.iper_dict = snx.iper_dict.copy()
        snx2.irs = snx.irs.copy()
        snx2.ixpo = snx.ixpo.copy()
        snx2.ixpor = snx.ixpor.copy()
        snx2.iypo = snx.iypo.copy()
        snx2.iypor = snx.iypor.copy()
        snx2.iut = snx.iut.copy()
        snx2.ilod = snx.ilod.copy()
        snx2.inutx = snx.inutx.copy()
        snx2.inuty = snx.inuty.copy()
        snx2.igc = snx.igc.copy()
        snx2.isc = snx.isc.copy()
        snx2.isatax = snx.isatax.copy()
        snx2.isatay = snx.isatay.copy()
        snx2.isataz = snx.isataz.copy()
        snx2.itrans = snx.itrans.copy()
        
        return snx2
    
    # Check station PT codes and DOMES numbers
    #-----------------------------------------
    def check_staid(snx, codomes, check_pt=True, check_crd=True, quiet=False, out=sys.stdout):
      
        """
        Check station PT codes and DOMES numbers

        Parameters
        ----------
        codomes : list
            DOMES number catalogue (from ioutils.read_domes)
        check_pt : bool, optional
            Whether PT codes should be checked. Default is True.
        check_crd : bool, optional
            Whether station coordinates should be checked in case of multiple DOMES numbers. Default is True.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """
        
        # Print header in log file
        if not(quiet):
            print('sinex.check_staid', file=out)
            print('-----------------', file=out)
                
        # Loop over stations
        for i in range(len(snx.sta)):
            code = snx.sta[i].code
            pt = snx.sta[i].pt
            domes = snx.sta[i].domes
            
            # Check PT code
            if (check_pt):
                
                # PT should be A except for stations IISC and KELY
                if (code in ['IISC', 'KELY']):
                    pt2 = ' B'
                else:
                    pt2 = ' A'
                
                # If a correction is needed
                if (pt2 != pt):

                    # Correct PT in snx.sta
                    snx.sta[i].pt = pt2
                    
                    # Correct PT in snx.param
                    for j in range(snx.npar):
                        if (snx.param[j].code == code) and (snx.param[j].pt == pt):
                            snx.param[j].pt = pt2
                    
                    # Print message in log file
                    if not(quiet):
                        print('    {0} {1} {3} > {0} {2} {3}'.format(code, pt, pt2, domes), file=out)

            # Check DOMES number - 1st case: Do not check station coordinates
            if not(check_crd):
            
                # If code+domes is found in DOMES number catalogue, alright
                if (code+domes in [c.code+c.domes for c in codomes]):
                    j = [c.code+c.domes for c in codomes].index(code+domes)
                    domes2 = domes
                    desc2 = codomes[j].description
                    
                # Else,
                else:
                    
                    # If code is nevertheless found in DOMES number catalogue, a correction is needed.
                    if (code in [c.code for c in codomes]):
                        j = [c.code for c in codomes].index(code)
                        domes2 = codomes[j].domes
                        desc2 = codomes[j].description
                      
                    # If code is not found in DOMES number catalogue, set default DOMES number.
                    else:
                        domes2 = default_domes
                        desc2 = snx.sta[i].description
                    
            # Check DOMES number - 2nd case: Do not check station coordinates
            else:
              
                # Get station coordinates
                (lon, lat) = snx.get_lonlat([code])
                lam = pi/180*lon[0]
                phi = pi/180*lat[0]
                
                # Look for every occurence of code in DOMES number catalogue
                ind = np.nonzero(np.array([c.code for c in codomes]) == code)[0]
                
                # If no occurence is found, set default DOMES number if needed.
                if (len(ind) == 0):
                    domes2 = default_domes
                    desc2 = snx.sta[i].description
                    
                # Else (at least one occurence is found),
                else:

                    # Compute distances from station to every point of the DOMES number catalogue
                    d = np.zeros(len(ind))
                    for j in range(len(ind)):
                        lamj = pi/180*codomes[ind[j]].lon
                        phij = pi/180*codomes[ind[j]].lat
                        d[j] = ae * acos(sin(phi)*sin(phij) + cos(phi)*cos(phij)*cos(lam-lamj))
                      
                    # Point of the DOMES number catalogue with smallest distance to station
                    dmin = np.min(d)
                    imin = np.nonzero(d == dmin)[0][0]
                    
                    # If smallest distance is larger than 100 km, set default DOMES number if needed.
                    if (dmin > 100000):
                        domes2 = default_domes
                        desc2 = snx.sta[i].description
                        
                    # Else (point of the DOMES number catalogue with smallest distance to station is probably the right one),
                    else:
                        domes2 = codomes[ind[imin]].domes
                        desc2 = codomes[ind[imin]].description
                                      
            # Correct DOMES number in snx.sta and print message if needed
            if (domes2 != snx.sta[i].domes):
                snx.sta[i].domes = domes2
                if not(quiet):
                    print('    {0} {1} {2} > {0} {1} {3}'.format(code, pt, domes, domes2), file=out)
                    
            # Correct station description
            snx.sta[i].description = desc2

        # Print blank line in log file
        if not(quiet):
            print('', file=out)
        
    # Check solution numbers (solns) in an "instantaneous" solution
    #--------------------------------------------------------------
    def check_solns(snx, solns, quiet=False, out=sys.stdout):
        
        """
        Check solution numbers (solns) in an "instantaneous" solution

        Parameters
        ----------
        solns : list
            Reference discontinuity list (from io.read_solns)
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """

        # Print header line in log file
        if not(quiet):
            print('sinex.check_solns', file=out)
            print('-----------------', file=out)
            
        # Search keys
        codept_soln = [s.code+s.pt for s in solns]
        codept_sta = [s.code+s.pt for s in snx.sta]

        # Loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]

            # If current station is found in discontinuity list
            if (p.code+p.pt in codept_soln):
                ista = codept_soln.index(p.code+p.pt)

                # Look for appropriate soln
                isoln = 0
                while ((solns[ista].P[isoln].end != '00:000:00000') and (earlier(solns[ista].P[isoln].end, p.tref))):
                    isoln = isoln+1
                soln2 = solns[ista].P[isoln].soln

            # Else, default soln is '   1'
            else:
                soln2 = '   1'
                if not(quiet):
                    print('    {0.code} {0.pt} not found in discontinuity list'.format(p), file=out)

            # If soln has to be modified,
            if (p.soln != soln2):

                # Print message in log file
                if not(quiet):
                    print('    {0.code} {0.pt} {0.soln} > {0.code} {0.pt} {1}'.format(p, soln2), file=out)

                # Modify soln in snx.param
                snx.param[i+0].soln = soln2
                snx.param[i+1].soln = soln2
                snx.param[i+2].soln = soln2
                
                # Modify soln in snx.sta
                ista = codept_sta.index(p.code+p.pt)
                snx.sta[ista].soln[0].soln = soln2
                    
        # Print blank line in log file
        if not(quiet):
            print('', file=out)

    # Check parameter reference epochs and SOLUTION/EPOCHS block in an "instantaneous" solution
    #------------------------------------------------------------------------------------------
    def check_epochs(snx, tstart, tend, tref):
        
        """
        Check parameter reference epochs and SOLUTION/EPOCHS block in an "instantaneous" solution

        Parameters
        ----------
        tstart : str
            Start date (in SINEX date format)
        tend : str
            End date (in SINEX date format)
        tref : str
            Reference date (in SINEX date format)
        
        """
        
        # Set reference epochs of all parameters to tref
        for p in snx.param:
            p.tref = tref
                
        # Bound start and end dates in SOLUTION/EPOCHS block
        for s in snx.sta:
            if earlier(s.soln[0].datastart, tstart):
                s.soln[0].datastart = tstart
            if earlier(tend, s.soln[0].dataend):
                s.soln[0].dataend = tend
                
        # Bound start and end dates of snx itself
        if (earlier(snx.start, tstart)):
            snx.start = tstart
        if (earlier(tend, snx.end)):
            snx.end = tend
        
    # Check receivers, antennas and eccentricities against site logs
    #---------------------------------------------------------------
    def check_metadata(snx, metasnx, start=None, end=None, antlist=None, flag_daz=True, quiet=False, out=sys.stdout):

        """
        Check receivers, antennas and eccentricities against site logs

        Returns
        -------
        metaerr : list
            List of metadata inconsistencies
        nolog : list
            List of stations for which no sitelog is available
        rej : list
            List of stations which have either a wrong antenna type, an eccentricity error > 1 mm
            or an orientation error > 10°

        Parameters
        ----------
        metasnx : sinex instance
            sinex instance containing station metadata (from ioutils.sitelogs2snx)
        start : str, optional
            Start date (in SINEX date format)
        end : str, optional
            End date (in SINEX date format)
        antlist : list, optional
            List of ground antenna types in ANTEX file. Default is None.
        flag_daz : bool, optional
            Whether to flag antenna orientation errors or not. Default is True.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """
        
        # Print header line in log file
        if not(quiet):
            print('sinex.check_metadata', file=out)
            print('--------------------', file=out)
        
        # Set of antenna types without radomes
        if (antlist):
            antset = set([a[0:16] for a in antlist])
            
        # Initialize output
        metaerr = []
        nolog = []
        rej = []
        
        # Loop over stations
        for s in snx.sta:
            
            # Get coordinates of current station
            X = snx.get_xyz([s.code], [s.pt])[0]
            
            # Is there a station with the same 4-char ID less than 100 km away in metasnx.sta?
            found = False
            keys = [metas.code for metas in metasnx.sta]
            if (s.code in keys):
                ind = np.nonzero(np.array(keys) == s.code)[0]
                d = [sqrt(np.sum((metas.X-X)**2)) / 1000 for metas in [metasnx.sta[i] for i in ind]]
                if (np.min(d) < 100):
                    found = True
                    i = np.argmin(d)
                    rec = metasnx.sta[ind[i]].rec
                    ant = metasnx.sta[ind[i]].ant
                    ecc = metasnx.sta[ind[i]].ecc
                    source = metasnx.sta[ind[i]].source

            # If current station was found in metasnx.sta, 
            if (found):
                
                # Any change needed to receiver metadata?
                b = False
                if (len(s.rec) != len(rec)):
                    b = True
                else:
                    for i in range(len(s.rec)):
                        for key in s.rec[i].__dict__:
                            if (getattr(s.rec[i], key) != getattr(rec[i], key)):
                                b = True
                                
                # If any change needed,
                if (b):
                    
                    # Print messages
                    if not(quiet):
                        for i in range(len(s.rec)):
                            print('    < {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.firmware}'.format(s, s.rec[i]), file=out)
                        for i in range(len(rec)):
                            print('    > {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.firmware}'.format(s, rec[i]), file=out)
                        
                    # Any clear mistake to report about receiver type?
                    if (len(rec) == 1) and (len(s.rec) == 1):
                        if (s.rec[0].type != rec[0].type):
                            metaerr.append('{0}   receiver type         {1:<20s}   {2:<20s}   {3:<6s}'.format(s.code, s.rec[0].type, rec[0].type, source))
                            
                    # Update receiver metadata
                    s.rec = rec

                # Any change needed to antenna metadata?
                b = False
                if (len(s.ant) != len(ant)):
                    b = True
                else:
                    for i in range(len(s.ant)):
                        for key in s.ant[i].__dict__:
                            if (getattr(s.ant[i], key) != getattr(ant[i], key)):
                                b = True
                                
                # If any change needed,
                if (b):
                    
                    # Print messages
                    if not(quiet):
                        for i in range(len(s.ant)):
                            print('    < {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.daz}'.format(s, s.ant[i]), file=out)
                        for i in range(len(ant)):
                            print('    > {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.type} {1.serie} {1.daz}'.format(s, ant[i]), file=out)
                        
                    # Any clear mistake to report about antenna type?
                    if (antlist) and (len(ant) == 1) and (len(s.ant) == 1):
                        a1 = ant[0].type
                        if (a1[0:16] in antset) and not(a1 in antlist):
                            a1 = a1[0:16] + 'NONE'
                        a2 = s.ant[0].type
                        if (a2[0:16] in antset) and not(a2 in antlist):
                            a2 = a2[0:16] + 'NONE'
                        if (a2 != a1):
                            metaerr.append('{0}   antenna type          {1:<20s}   {2:<20s}   {3:<6s}'.format(s.code, s.ant[0].type, ant[0].type, source))
                            rej.append(s.code)

                    # Any clear mistake to report about antenna orientation?
                    if (len(ant) == 1) and (len(s.ant) == 1):
                        daz = (int(s.ant[0].daz) - int(ant[0].daz)) % 360
                        if (daz > 180):
                            daz = daz - 360
                        if (abs(daz) >= 1):
                            metaerr.append('{0}   antenna orientation   {1:<20d}   {2:<20d}   {3:<6s}'.format(s.code, int(s.ant[0].daz), int(ant[0].daz), source))
                        if (abs(daz) > 10):
                            rej.append(s.code)
                            
                    # Update antenna metadata
                    s.ant = ant

                # Any change needed to eccentricity metadata?
                b = False
                if (len(s.ecc) != len(ecc)):
                    b = True
                else:
                    for i in range(len(s.ecc)):
                        for key in s.ecc[i].__dict__:
                            if (getattr(s.ecc[i], key) != getattr(ecc[i], key)):
                                b = True
                                
                # If any change needed,
                if (b):
                   
                    # Print messages
                    if not(quiet):
                        for i in range(len(s.ecc)):
                            print('    < {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.system} {2[0]} {2[1]} {2[2]}'.format(s, s.ecc[i], s.ecc[i].dx), file=out)
                        for i in range(len(ecc)):
                            print('    > {0.code} {0.pt} ---- {0.tech} {1.start} {1.end} {1.system} {2[0]} {2[1]} {2[2]}'.format(s, ecc[i], ecc[i].dx), file=out)
                        
                    # Any clear mistake to report about eccentricity?
                    if (len(ecc) == 1) and (len(s.ecc) == 1):
                        if (float(s.ecc[0].dx[0]) != float(ecc[0].dx[0])):
                            metaerr.append('{0}   Up eccentricity       {1:<20s}   {2:<20s}   {3:<6s}'.format(s.code, s.ecc[0].dx[0], ecc[0].dx[0], source))
                        if (float(s.ecc[0].dx[1]) != float(ecc[0].dx[1])):
                            metaerr.append('{0}   North eccentricity    {1:<20s}   {2:<20s}   {3:<6s}'.format(s.code, s.ecc[0].dx[1], ecc[0].dx[1], source))
                        if (float(s.ecc[0].dx[2]) != float(ecc[0].dx[2])):
                            metaerr.append('{0}   East eccentricity     {1:<20s}   {2:<20s}   {3:<6s}'.format(s.code, s.ecc[0].dx[2], ecc[0].dx[2], source))
                        if (abs(float(s.ecc[0].dx[0])-float(ecc[0].dx[0])) > 0.001) or (abs(float(s.ecc[0].dx[1])-float(ecc[0].dx[1])) > 0.001) or (abs(float(s.ecc[0].dx[2])-float(ecc[0].dx[2])) > 0.001):
                            rej.append(s.code)

                    # Update eccentricity metadata
                    s.ecc = ecc
                            
            # Else, add station to list of stations without sitelogs
            else:
                nolog.append(s.code)
                
        # Print blank line in log file
        if not(quiet):
            print('', file=out)
            
        return (metaerr, nolog, rej)

    # Get indices of parameters of specified types
    #---------------------------------------------
    def get_par_ind(snx, types):

        """
        Get indices of parameters of specified types

        Returns
        -------
        ind : array_like
            Indices of parameters of specified types

        Parameters
        ----------
        types : list
            List of parameter types. It can include the following keywords:
            'XPO', 'XPOR', 'YPO', 'YPOR', 'UT', 'LOD', 'NUT_X', 'NUT_Y';
            'ERP' for all kinds of ERPs;
            'SATA_X', 'SATA_Y', 'SATA_Z'; 'SATA' for all satellite PCOs;
            'STA'; 'VEL'; 'RS'; 'GC'; 'SC' and 'TRANS' for all transformation parameters.
        
        """

        ind = []

        if ('ERP' in types):
            ind.extend(snx.ixpo+snx.ixpor+snx.iypo+snx.iypor+snx.iut+snx.ilod+snx.inutx+snx.inuty)
        else:
            if ('XPO' in types):
                ind.extend(snx.ixpo)
            if ('XPOR' in types):
                ind.extend(snx.ixpor)
            if ('YPO' in types):
                ind.extend(snx.iypo)
            if ('YPOR' in types):
                ind.extend(snx.iypor)
            if ('UT' in types):
                ind.extend(snx.iut)
            if ('LOD' in types):
                ind.extend(snx.ilod)
            if ('NUT_X' in types):
                ind.extend(snx.inutx)
            if ('NUT_Y' in types):
                ind.extend(snx.inuty)
                
        if ('SATA' in types):
            ind.extend(snx.isatax+snx.isatay+snx.isataz)
        else:
            if ('SATA_X' in types):
                ind.extend(snx.isatax)
            if ('SATA_Y' in types):
                ind.extend(snx.isatay)
            if ('SATA_Z' in types):
                ind.extend(snx.isataz)

        if ('STA' in types):
            ind.extend(snx.ix)
            ind.extend([i+1 for i in snx.ix])
            ind.extend([i+2 for i in snx.ix])
                
        if ('VEL' in types):
            ind.extend(snx.iv)
            ind.extend([i+1 for i in snx.iv])
            ind.extend([i+2 for i in snx.iv])

        if ('RS' in types):
            ind.extend(snx.irs)
            ind.extend([i+1 for i in snx.irs])

        if ('GC' in types):
            ind.extend(snx.igc)
            ind.extend([i+1 for i in snx.igc])
            ind.extend([i+2 for i in snx.igc])

        if ('SC' in types):
            ind.extend(snx.isc)
            
        if ('TRANS' in types):
            ind.extend(snx.itrans)
            
        return ind
            
    # Get indices of positions of specified stations
    #-----------------------------------------------
    def get_sta_ind(snx, code, pt=None, soln=None):

        """
        Get indices of positions of specified stations

        Returns
        -------
        ind : (...,3) array_like
            Indices of station positions in snx.param

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
        
        # Keys and holes
        if bool(pt) and bool(soln):
            keys = [code[i]+pt[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.ix]]
        elif bool(pt):
            keys = [code[i]+pt[i] for i in range(len(code))]
            holes = [p.code+p.pt for p in [snx.param[i] for i in snx.ix]]
        elif bool(soln):
            keys = [code[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.soln for p in [snx.param[i] for i in snx.ix]]
        else:
            keys = [code[i] for i in range(len(code))]
            holes = [p.code for p in [snx.param[i] for i in snx.ix]]
        
        # Initialization
        ind = -np.ones((len(code), 3), dtype='i')

        # Loop over requested stations
        for i in range(len(code)):

            # If a position is available for current station
            if (keys[i] in holes):
                
                # Get indices of its position
                j = holes.index(keys[i])
                ind[i] = range(snx.ix[j], snx.ix[j]+3)
                
        return ind

    # Get indices of velocities of specified stations
    #------------------------------------------------
    def get_vel_ind(snx, code, pt=None, soln=None):

        """
        Get indices of velocities of specified stations

        Returns
        -------
        ind : (...,3) array_like
            Indices of station velocities in snx.param

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
        
        # Keys and holes
        if (pt) and (soln):
            keys = [code[i]+pt[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iv]]
        elif (pt):
            keys = [code[i]+pt[i] for i in range(len(code))]
            holes = [p.code+p.pt for p in [snx.param[i] for i in snx.iv]]
        elif (soln):
            keys = [code[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.soln for p in [snx.param[i] for i in snx.iv]]
        else:
            keys = [code[i] for i in range(len(code))]
            holes = [p.code for p in [snx.param[i] for i in snx.iv]]
        
        # Initialization
        ind = -np.ones((len(code), 3), dtype='i')

        # Loop over requested stations
        for i in range(len(code)):

            # If a velocity is available for current station
            if (keys[i] in holes):
                
                # Get indices of its velocity
                j = holes.index(keys[i])
                ind[i] = range(snx.iv[j], snx.iv[j]+3)
                
        return ind
    
    # Get indices of periods of specified stations
    #------------------------------------------------
    def get_per_ind(snx, percode, code, pt=None, soln=None):

        """
        Get indices of periods of specified stations

        Returns
        -------
        ind : (...,3) array_like
            Indices of station periods in snx.param

        Parameters
        ----------
        percode : list
            List of 4-char period codes
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
        # Keys and holes : init on 1st period (normally all stations have same periods / same order...)
        per = percode[0]
        if (pt) and (soln):
            keys = [code[i]+pt[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iper_dict[per]]]
            
        elif (pt):
            keys = [code[i]+pt[i] for i in range(len(code))]
            holes = [p.code+p.pt for p in [snx.param[i] for i in snx.iper_dict[per]]]
           
        elif (soln):
            keys = [code[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.soln for p in [snx.param[i] for i in snx.iper_dict[per]]]
            
        else:
            keys = [code[i] for i in range(len(code))]
            holes = [p.code for p in [snx.param[i] for i in snx.iper_dict[per]]]
        
        # Initialization
        ind = -np.ones((len(code), 6), dtype='i')

        # Loop over requested stations
        for i, per in enumerate(percode):

            # If a velocity is available for current station
            if (keys[i] in holes):
                
                # Get indices of its velocity
                j = holes.index(keys[i])
                ind[i] = range(snx.iper_dict[per][j], snx.iper_dict[per][j]+6)
                
        return ind

    # Get indices of coordinates of specified radiosources
    #-----------------------------------------------------
    def get_rs_ind(snx, iers):

        """
        Get indices of coordinates of specified radiosources

        Returns
        -------
        ind : (...,2) array_like
            Indices of radiosource coordinates in snx.param

        Parameters
        ----------
        iers : list
            List of IERS radiosource names
        
        """
        
        # Search keys
        keys = [p.iers for p in [snx.param[i] for i in snx.irs]]
        
        # Initialization
        ind = -np.ones((len(iers), 2), dtype='i')

        # Loop over requested radiosources
        for i in range(len(iers)):

            # If coordinates are available for current radiosource
            if (iers[i] in keys):
                
                # Get indices of its coordinates
                j = keys.index(iers[i])
                ind[i] = range(snx.irs[j], snx.irs[j]+2)
                
        return ind

    # Get indices of common parameters between two solutions
    #-------------------------------------------------------
    def get_common_par(snx, ref):
        
        """
        Get indices of common parameters between two solutions
            
        Returns
        -------
        isnx : array_like
            Indices of parameters in snx.param that are also in ref.param
        iref : array_like
            Indices of matching parameters in ref.param

        Parameters
        ----------
        ref : sinex instance
            The other solution
            
        """
        
        # Initialization
        isnx = []
        iref = []
    
        # Common station positions
        (i, j) = snx.get_common_sta(ref)
        isnx.extend(np.array(i).flatten().tolist())
        iref.extend(np.array(j).flatten().tolist())
    
        # Common station velocities
        (i, j) = snx.get_common_vel(ref)
        isnx.extend(np.array(i).flatten().tolist())
        iref.extend(np.array(j).flatten().tolist())
        
        # Common station periods
        (i, j) = snx.get_common_per(ref)
        isnx.extend(np.array(i).flatten().tolist())
        iref.extend(np.array(j).flatten().tolist())
                
        # Common radiosource coordinates
        (i, j) = snx.get_common_rs(ref)
        isnx.extend(np.array(i).flatten().tolist())
        iref.extend(np.array(j).flatten().tolist())

        # Common X-pole coordinates
        ksnx = [p.tref for p in [snx.param[i] for i in snx.ixpo]]
        kref = [p.tref for p in [ref.param[i] for i in ref.ixpo]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.ixpo[i])
                iref.append(ref.ixpo[j])
        
        # Common X-pole rates
        ksnx = [p.tref for p in [snx.param[i] for i in snx.ixpor]]
        kref = [p.tref for p in [ref.param[i] for i in ref.ixpor]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.ixpor[i])
                iref.append(ref.ixpor[j])
                
        # Common Y-pole coordinates
        ksnx = [p.tref for p in [snx.param[i] for i in snx.iypo]]
        kref = [p.tref for p in [ref.param[i] for i in ref.iypo]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.iypo[i])
                iref.append(ref.iypo[j])
                
        # Common Y-pole rates
        ksnx = [p.tref for p in [snx.param[i] for i in snx.iypor]]
        kref = [p.tref for p in [ref.param[i] for i in ref.iypor]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.iypor[i])
                iref.append(ref.iypor[j])
                
        # Common UT1-UTC offsets
        ksnx = [p.tref for p in [snx.param[i] for i in snx.iut]]
        kref = [p.tref for p in [ref.param[i] for i in ref.iut]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.iut[i])
                iref.append(ref.iut[j])
                
        # Common LODs
        ksnx = [p.tref for p in [snx.param[i] for i in snx.ilod]]
        kref = [p.tref for p in [ref.param[i] for i in ref.ilod]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.ilod[i])
                iref.append(ref.ilod[j])
        
        # Common X-nutations
        ksnx = [p.tref for p in [snx.param[i] for i in snx.inutx]]
        kref = [p.tref for p in [ref.param[i] for i in ref.inutx]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.inutx[i])
                iref.append(ref.inutx[j])

        # Common Y-nutations
        ksnx = [p.tref for p in [snx.param[i] for i in snx.inuty]]
        kref = [p.tref for p in [ref.param[i] for i in ref.inuty]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.inuty[i])
                iref.append(ref.inuty[j])

        # Common geocenter coordinates
        ksnx = [p.tref for p in [snx.param[i] for i in snx.igc]]
        kref = [p.tref for p in [ref.param[i] for i in ref.igc]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.extend([snx.igc[i], snx.igc[i]+1, snx.igc[i]+2])
                iref.extend([ref.igc[j], ref.igc[j]+1, ref.igc[j]+2])

        # Common scale factors
        ksnx = [p.tref for p in [snx.param[i] for i in snx.isc]]
        kref = [p.tref for p in [ref.param[i] for i in ref.isc]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.isc[i])
                iref.append(ref.isc[j])
        
        # Common satellite x-PCOs
        ksnx = [p.code+p.pt for p in [snx.param[i] for i in snx.isatax]]
        kref = [p.code+p.pt for p in [ref.param[i] for i in ref.isatax]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.isatax[i])
                iref.append(ref.isatax[j])

        # Common satellite y-PCOs
        ksnx = [p.code+p.pt for p in [snx.param[i] for i in snx.isatay]]
        kref = [p.code+p.pt for p in [ref.param[i] for i in ref.isatay]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.isatay[i])
                iref.append(ref.isatay[j])

        # Common satellite z-PCOs
        ksnx = [p.code+p.pt for p in [snx.param[i] for i in snx.isataz]]
        kref = [p.code+p.pt for p in [ref.param[i] for i in ref.isataz]]
        for i in range(len(ksnx)):
            if (ksnx[i] in kref):
                j = kref.index(ksnx[i])
                isnx.append(snx.isataz[i])
                iref.append(ref.isataz[j])

        return (isnx, iref)

    # Get indices of common station positions between two solutions
    #--------------------------------------------------------------
    def get_common_sta(snx, ref):
        
        """
        Get indices of common station positions between two solutions
            
        Returns
        -------
        isnx : array_like
            Indices of station positions in snx.param that are also in ref.param
        iref : array_like
            Indices of matching station positions in ref.param

        Parameters
        ----------
        ref : sinex instance
            The other solution
            
        """
        
        # Initializations
        isnx = []
        iref = []
        
        # Get indices of common station positions
        keys = [p.code+p.pt+p.soln for p in [ref.param[i] for i in ref.ix]]
        for i in snx.ix:
            p = snx.param[i]
            if (p.code+p.pt+p.soln in keys):
                j = ref.ix[keys.index(p.code+p.pt+p.soln)]
                isnx.append([i, i+1, i+2])
                iref.append([j, j+1, j+2])
                
        return (isnx, iref)

    # Get indices of common station velocities between two solutions
    #---------------------------------------------------------------
    def get_common_vel(snx, ref):
        
        """
        Get indices of common station velocities between two solutions
            
        Returns
        -------
        isnx : array_like
            Indices of station velocities in snx.param that are also in ref.param
        iref : array_like
            Indices of matching station velocities in ref.param

        Parameters
        ----------
        ref : sinex instance
            The other solution
            
        """
        
        # Initializations
        isnx = []
        iref = []
        
        # Get indices of common station positions
        keys = [p.code+p.pt+p.soln for p in [ref.param[i] for i in ref.iv]]
        for i in snx.iv:
            p = snx.param[i]
            if (p.code+p.pt+p.soln in keys):
                j = ref.iv[keys.index(p.code+p.pt+p.soln)]
                isnx.append([i, i+1, i+2])
                iref.append([j, j+1, j+2])
                
        return (isnx, iref)
    
    
    # Get indices of common station periods between two solutions
    #---------------------------------------------------------------
    def get_common_per(snx, ref):
        
        """
        Get indices of common station seasonal periods between two solutions
            
        Returns
        -------
        isnx : array_like
            Indices of station periods in snx.param that are also in ref.param
        iref : array_like
            Indices of matching periods velocities in ref.param

        Parameters
        ----------
        ref : sinex instance
            The other solution
            
        """
        
        # Initializations
        isnx = []
        iref = []
        
        # Get indices of common station positions
        keys_per = {}
        for per in ref.iper_dict.keys():
            keys_per[per] = [p.code+p.pt+p.soln for p in [ref.param[i] for i in ref.iper_dict[per]]]
        
        for per in snx.iper_dict.keys():
            if per not in keys_per.keys(): #per not in common 
                pass
            else: #this period in snx & in ref
                for i in snx.iper_dict[per]:
                    p = snx.param[i]
                    if (p.code+p.pt+p.soln in keys_per[per]):
                        j = ref.iper_dict[per][keys_per[per].index(p.code+p.pt+p.soln)]
                        isnx.append([i, i+1, i+2, i+3, i+4, i+5])
                        iref.append([j, j+1, j+2, j+3, j+4, j+5])
                
        return (isnx, iref)

    # Get indices of common radiosource coordinates between two solutions
    #--------------------------------------------------------------------
    def get_common_rs(snx, ref):
        
        """
        Get indices of common radiosource coordinates between two solutions
            
        Returns
        -------
        isnx : array_like
            Indices of radiosource coordinates in snx.param that are also in ref.param
        iref : array_like
            Indices of matching radiosource coordinates in ref.param

        Parameters
        ----------
        ref : sinex instance
            The other solution
            
        """
        
        # Initializations
        isnx = []
        iref = []
        
        # Get indices of common radiosources
        keys = [p.iers for p in [ref.param[i] for i in ref.irs]]
        for i in snx.irs:
            p = snx.param[i]
            if (p.iers in keys):
                j = ref.irs[keys.index(p.iers)]
                isnx.append([i, i+1])
                iref.append([j, j+1])
                
        return (isnx, iref)

    # Get cartesian coordinates of specified stations
    #------------------------------------------------
    def get_xyz(snx, code, pt=None, soln=None):
      
        """
        Get cartesian coordinates of specified stations

        Returns
        -------
        X : (...,3) array_like
            Station cartesian coordinates [m]

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
        
        # Get indices of positions of specified stations
        ind = snx.get_sta_ind(code, pt, soln)
        
        # And their coordinates
        X = np.zeros((len(code), 3))
        if type(snx.x)==type(None): #particular case where snx.x = None (during combination)
            X[ind != -1] = snx.x0[ind[ind != -1]] #use APRIORI coordinates
            #logging.warning("[sinex]get_xyz(): snx.x=None, use apriori value snx.x0")
        else:
            X[ind != -1] = snx.x[ind[ind != -1]]
        
        return X
  
    # Get geographical coordinates of specified stations
    #---------------------------------------------------
    def get_plh(snx, code, pt=None, soln=None):
      
        """
        Get geographical coordinates of specified stations

        Returns
        -------
        phi : (...) array_like
            Station latitudes [rad]
        lam : (...) array_like
            Station longitudes [rad]
        h : (...) array_like
            Station heights [m]

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
      
        X = snx.get_xyz(code, pt, soln)
        
        return cart2geo(X)
    
    # Get longitudes and latitudes of specified stations
    #---------------------------------------------------
    def get_lonlat(snx, code, pt=None, soln=None):
      
        """
        Get longitudes and latitudes of specified stations

        Returns
        -------
        lon : (...) array_like
            Station longitudes [deg]
        lat : (...) array_like
            Station latitudes [deg]

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """
      
        (phi, lam, h) = snx.get_plh(code, pt, soln)
        
        return (180/pi*lam, 180/pi*phi)
    
    # Get ENH formal errors of specified stations
    #--------------------------------------------
    def get_sigenh(snx, code=None, pt=None, soln=None):

        """
        Get ENH formal errors of specified stations

        Returns
        -------
        s : (...,3) array_like
            ENH formal errors [m]

        Parameters
        ----------
        code : list
            List of 4-char station codes. Default is None,
            meaning that formal errors of all stations will be returned.
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        
        """

        # Get indices of positions of specified stations
        if (code):
            ind = snx.get_sta_ind(code, pt, soln)
        else:
            ind = [[i, i+1, i+2] for i in snx.ix]

        # Initialization
        s = np.zeros((len(ind), 3))
        
        # Loop over available specified stations
        for i in range(len(ind)):
            if (ind[i][0] != -1):

                # Compute ENH formal errors
                R = xyz2enh(snx.x[ind[i]])
                Q = snx.Q[np.ix_(ind[i], ind[i])]
                s[i] = np.sqrt(np.diag(np.dot(R, np.dot(Q, R.T))))

        return s
        
    # Get list of available core RF stations
    #---------------------------------------
    def get_core_sta(snx, file, ref, thr=None):

        """
        Get list of available core RF stations
        
        Returns
        -------
        code : list
            4-char codes of available core RF stations
        pt : list
            PT codes of available core RF stations
        soln : list
            solns of available core RF stations

        Parameters
        ----------        
        file : str
            File containing list of core RF stations (e.g., IGS14_core.txt)
        ref : sinex instance
            Datum (e.g., IGS14 propagated to the epoch of solution snx)
        thr : float, optional
            Threshold for rejection of stations whose 3D formal errors exceed
            (thr * median of 3D formal errors). Default is None (no rejection).
            
        """

        # Read list of core stations
        core = []
        f = open(file)
        line = f.readline()
        while (line):
            core.append(line.strip().split())
            line = f.readline()
        f.close()

        # Build list of core stations available in snx : loop over core clusters
        code = []
        pt = []
        soln = []
        ix = []
        for i in range(len(core)):

            # Initializations
            j = 0
            b = False

            # Look for any station of current cluster in snx and ref
            keys = [p.code for p in [snx.param[k] for k in snx.ix]]
            keyr = [p.code+p.pt+p.soln for p in [ref.param[k] for k in ref.ix]]
            while not(b) and (j < len(core[i])):
                if (core[i][j] in keys):
                    isnx = snx.ix[keys.index(core[i][j])]
                    if (core[i][j]+snx.param[isnx].pt+snx.param[isnx].soln in keyr):
                        b = True
                    else:
                        j = j+1
                else:
                    j = j+1

            # If one station of the cluster was found in snx and ref,
            if (b):
                code.append(core[i][j])
                pt.append(snx.param[isnx].pt)
                soln.append(snx.param[isnx].soln)
                ix.append(isnx)
                
        # If requested, reject stations with abnormally large formal errors
        if (thr):
        
            # Compute 3D formal errors
            sig = np.array([sqrt(np.sum(snx.sig[i:i+3]**2)) for i in ix])
            
            # Get indices of outliers
            ind = np.nonzero(sig > thr*np.median(sig))[0]
                
            # If any outlier,
            if (len(ind) > 0):
                    
                # Reject outliers
                indk = np.setdiff1d(range(len(code)), ind)
                code = [code[i] for i in indk]
                pt = [pt[i] for i in indk]
                soln = [soln[i] for i in indk]

        return (code, pt, soln)
    
    # Get partial derivative matrix of Helmert parameters
    #----------------------------------------------------
    def helmert_partials(snx, helmerts, par, units=None, select_periods='all'):

        """
        Get partial derivative matrix of Helmert parameters

        Returns
        -------
        A : array_like
            Partial derivative matrix of solution parameters wrt Helmert parameters.

        Parameters
        ----------
        helmerts : str
            Indicates which Helmert parameters should be considered.
            It can include 'T' (translations), 'S' (scale), 'R' (rotations)
            and 'A' (CRF rotations).
        par : str
            Indicates which type of parameters should be considered.
            It can be either 'STA' (station and radiosource positions)
            'VEL' (station velocities - radiosource velocities not supported yet).
            'PERIOD' (station seasonal signals 'period')
        units : str, optional
            Specifies units of Helmert parameters. It can be either None (mm, ppb, mas)
            or 'm' (m).
        select_periods: str, optional
            In case of par='PERIOD' (period), specifies on which period provid helmert partial derivative matrix.
            Default: 'all'. Else list of period code (4 chr). Ex: ['A001', 'D001']
            This list of period defined also A order construction: 20 columns by period (10 cos (TTTSRRRAAA) + 10 sin select_periods (TTTSRRRAAA))
            
        """
      
        # Initializations
        A = np.zeros((snx.npar, 10))
        if (snx.x0 is not None):
            x = snx.x0
        else:
            x = snx.x
        
        # 1st case : Helmert parameters
        if (par == 'STA'):

            # Station positions partial derivatives
            ix = np.array([[i, i+1, i+2] for i in snx.ix])
            if (len(ix) > 0):
                A[ix[:,0], 0] =  ae
                A[ix[:,1], 1] =  ae
                A[ix[:,2], 2] =  ae
                A[ix[:,0], 3] =  x[ix[:,0]]
                A[ix[:,1], 3] =  x[ix[:,1]]
                A[ix[:,2], 3] =  x[ix[:,2]]
                A[ix[:,1], 4] = -x[ix[:,2]]
                A[ix[:,2], 4] =  x[ix[:,1]]
                A[ix[:,0], 5] =  x[ix[:,2]]
                A[ix[:,2], 5] = -x[ix[:,0]]
                A[ix[:,0], 6] = -x[ix[:,1]]
                A[ix[:,1], 6] =  x[ix[:,0]]
            
            # Radiosource positions partial derivatives
            ix = np.array([[i, i+1] for i in snx.irs])
            if (len(ix) > 0):
                a = mas2rad * x[ix[:,0]]
                d = mas2rad * x[ix[:,1]]
                A[ix[:,0], 7] =  np.tan(d)*np.cos(a) / mas2rad
                A[ix[:,1], 7] = -np.sin(a)           / mas2rad
                A[ix[:,0], 8] =  np.tan(d)*np.sin(a) / mas2rad
                A[ix[:,1], 8] =  np.cos(a)           / mas2rad
                A[ix[:,0], 9] = -1                   / mas2rad
            
            # ERP partial derivatives
            A[snx.ixpo, 5]  =  1/mas2rad
            A[snx.iypo, 4]  =  1/mas2rad
            A[snx.iut, 6]   = -1/(ms2rad*dera_dt)
            A[snx.inutx, 8] =  1/mas2rad
            A[snx.inuty, 7] =  1/mas2rad
            A[snx.iut, 9]   =  1/(ms2rad*dera_dt)
            
            # Geocenter coordinate partial derivatives
            ix = np.array([[i, i+1, i+2] for i in snx.igc])
            if (len(ix) > 0):
                A[ix[:,0], 0] = ae
                A[ix[:,1], 1] = ae
                A[ix[:,2], 2] = ae

            # Scale factor partial derivatives
            if (len(snx.isc) > 0):
                A[snx.isc, 3] = -1e9
            
        # 2nd case : Helmert parameter rates
        elif (par == 'VEL'):
            
            # Station velocity partial derivatives
            iv = np.array([[i, i+1, i+2] for i in snx.iv])
            ix = np.array([[i-3, i-2, i-1] for i in snx.iv])
            A[iv[:,0], 0] =  ae
            A[iv[:,1], 1] =  ae
            A[iv[:,2], 2] =  ae
            A[iv[:,0], 3] =  x[ix[:,0]]
            A[iv[:,1], 3] =  x[ix[:,1]]
            A[iv[:,2], 3] =  x[ix[:,2]]
            A[iv[:,1], 4] = -x[ix[:,2]]
            A[iv[:,2], 4] =  x[ix[:,1]]
            A[iv[:,0], 5] =  x[ix[:,2]]
            A[iv[:,2], 5] = -x[ix[:,0]]
            A[iv[:,0], 6] = -x[ix[:,1]]
            A[iv[:,1], 6] =  x[ix[:,0]]
            
            
        # 3rd case : Helmert parameter rates, seasonal signals
        elif (par == 'PERIOD'):
            # Re-Initializations: shape according to period number
            if select_periods=="all":
                select_periods = list(snx.iper_dict.keys())
                
            else:
                per_not_found = [per for per in select_periods if per not in snx.iper_dict.keys()]
                if len(per_not_found)>0:
                    raise ValueError(f"'{per_not_found}' period not found in sinex.iper_dict ")
                    
            nper = len(select_periods) #number of periods
            A = np.zeros((snx.npar, 10*2*nper))# 10 param cos + 10 param sin, for each period
            ix = np.array([[i, i+1, i+2] for i in snx.ix])
            
            # Station periodic partial derivatives
            for num, per in enumerate(select_periods): #loop over each period
            
                ip_cos = np.array([[i, i+2, i+4] for i in snx.iper_dict[per]])
                ip_sin = np.array([[i+1, i+3, i+5] for i in snx.iper_dict[per]])
    
                A[ip_cos[:,0], 10*2*num+0] =  A[ip_sin[:,0], 10*(2*num+1)+0] = ae
                A[ip_cos[:,1], 10*2*num+1] =  A[ip_sin[:,1], 10*(2*num+1)+1] = ae
                A[ip_cos[:,2], 10*2*num+2] =  A[ip_sin[:,2], 10*(2*num+1)+2] = ae
                A[ip_cos[:,0], 10*2*num+3] =  A[ip_sin[:,0], 10*(2*num+1)+3] = x[ix[:,0]]
                A[ip_cos[:,1], 10*2*num+3] =  A[ip_sin[:,1], 10*(2*num+1)+3] = x[ix[:,1]]
                A[ip_cos[:,2], 10*2*num+3] =  A[ip_sin[:,2], 10*(2*num+1)+3] = x[ix[:,2]]
                A[ip_cos[:,1], 10*2*num+4] =  A[ip_sin[:,1], 10*(2*num+1)+4] = -x[ix[:,2]]
                A[ip_cos[:,2], 10*2*num+4] =  A[ip_sin[:,2], 10*(2*num+1)+4] = x[ix[:,1]]
                A[ip_cos[:,0], 10*2*num+5] =  A[ip_sin[:,0], 10*(2*num+1)+5] = x[ix[:,2]]
                A[ip_cos[:,2], 10*2*num+5] =  A[ip_sin[:,2], 10*(2*num+1)+5] = -x[ix[:,0]]
                A[ip_cos[:,0], 10*2*num+6] =  A[ip_sin[:,0], 10*(2*num+1)+6] = -x[ix[:,1]]
                A[ip_cos[:,1], 10*2*num+6] =  A[ip_sin[:,1], 10*(2*num+1)+6] = x[ix[:,0]]

        # Express Helmert parameters in adequate units
        if (units is None):
            unit_array = np.array([1e-3/ae, 1e-3/ae, 1e-3/ae, 1e-9, mas2rad, mas2rad, mas2rad, mas2rad, mas2rad, mas2rad])
            if par!='PERIOD':
                A = A * unit_array
            else : #PER case, duplicate dim
                A = A * np.tile(unit_array, 2*nper) #2: cos & sin parameters
        else:
            A = A / ae
        
        # Indices of relevant columns of A
        ind = []
        if par == 'PERIOD': #special dim of A
            for num, per in enumerate(select_periods): #loop over each period
                for cs in range(2):
                    if ('T' in helmerts):
                        ind.extend(list(10*(2*num+cs) + np.arange(0, 3)))
                    if ('S' in helmerts):
                        ind.append(10*(2*num+cs) + 3)
                    if ('R' in helmerts):
                        ind.extend(10*(2*num+cs) + np.arange(4, 7))
                    if ('A' in helmerts):
                        ind.extend(10*(2*num+cs) + np.arange(7, 10))
            #order ind: TSR cos per1, TSR sin per1, TSR cos per2, TSR sin per2 etc.
        else:
            if ('T' in helmerts):
                ind.extend(range(0, 3))
            if ('S' in helmerts):
                ind.append(3)
            if ('R' in helmerts):
                ind.extend(range(4, 7))
            if ('A' in helmerts):
                ind.extend(range(7, 10))
            
        return A[:,ind]
        
    # Delete (reduce) parameters with specified indices
    #--------------------------------------------------
    def del_ind(snx, ind, keep_const=False):

        """
        Delete (reduce) parameters with specified indices

        Parameters
        ----------
        ind : list
            Indices of parameters to delete
        keep_const : bool, optional
            Whether not to remove constraints before reducing parameters from a solution.
            Default is False.

        """
        
        # If any parameter to delete,
        if (len(ind) > 0):
        
            # Indices of parameters to keep
            indk = np.setdiff1d(range(snx.npar), ind)
            
            # 1st case: solution + normal equation + constraints
            if (snx.Q is not None) and (snx.N is not None) and (snx.Nc is not None):
                R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                snx.b = snx.b[indk] - np.dot(R, snx.b[ind])
                snx.Nc = snx.Nc[np.ix_(indk, indk)]
                snx.x0 = snx.x0[indk]
                snx.sig0 = snx.sig0[indk]
                snx.neqinv(clear_neq=False)
            
            # 2nd case: normal equation + constraints
            elif (snx.N is not None) and (snx.Nc is not None):
                R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                snx.b = snx.b[indk] - np.dot(R, snx.b[ind])
                snx.Nc = snx.Nc[np.ix_(indk, indk)]
                snx.x0 = snx.x0[indk]
                snx.sig0 = snx.sig0[indk]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]

            # 3rd case: solution + constraints, including constraints on reduced parameters
            elif (snx.Q is not None) and (snx.Nc is not None):
                if (np.any(snx.Nc[np.ix_(ind, ind)])) and not(keep_const):
                    snx.unconstrain(clear_const=False)
                    R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                    snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                    snx.b = snx.b[indk] - np.dot(R, snx.b[ind])
                    snx.Nc = snx.Nc[np.ix_(indk, indk)]
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                    snx.neqinv()

            # 4th case: solution + constraints, but without constraints on reduced parameters
                else:
                    snx.Q = snx.Q[np.ix_(indk, indk)]
                    snx.x = snx.x[indk]
                    snx.sig = snx.sig[indk]
                    snx.Nc = snx.Nc[np.ix_(indk, indk)]
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                    
            # 5th case: solution + normal equation
            elif (snx.Q is not None) and (snx.N is not None):
                R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                snx.Q = snx.Q[np.ix_(indk, indk)]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                    
            # 6th case: solution only
            elif (snx.Q is not None):
                snx.Q = snx.Q[np.ix_(indk, indk)]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                
            # 7th case: no matrix at all
            else:
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                if (snx.x0 is not None):
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                    
            # Update snx.npar and snx.param
            snx.npar = len(indk)
            snx.param = [snx.param[i] for i in indk]
            
            # Reset parameter indices
            snx.set_par_ind()
            
            # Update station list
            snx.clean_sta()
            
    # Delete (reduce) parameters of specified types
    #----------------------------------------------
    def del_params(snx, types, keep_const=False):

        """
        Delete (reduce) parameters of specified types

        Parameters
        ----------
        types : list
            List of keywords indicating which parameters should be deleted.
            It can include the following keywords:
            'XPO', 'XPOR', 'YPO', 'YPOR', 'UT', 'LOD', 'NUT_X', 'NUT_Y';
            'ERP' for all kinds of ERPs;
            'SATA_X', 'SATA_Y', 'SATA_Z'; 'SATA' for all satellite PCOs;
            'STA'; 'VEL'; 'RS'; 'GC'; 'SC' and 'TRANS' for all transformation parameters.
        keep_const : bool, optional
            Whether not to remove constraints before reducing parameters from a solution.
            Default is False.
        
        """
        
        # Get indices of parameters to delete
        ind = snx.get_par_ind(types)
        
        # And delete them
        snx.del_ind(ind, keep_const)

    # Delete parameters that are not supported by snxcomb
    #----------------------------------------------------
    def del_unknown_par(snx):

        """
        Delete parameters that are not supported by snxcomb
        
        """
        
        # Get indices of supported parameters
        ix = snx.ix + [i+1 for i in snx.ix] + [i+2 for i in snx.ix]
        iv = snx.iv + [i+1 for i in snx.iv] + [i+2 for i in snx.iv]
        irs = snx.irs + [i+1 for i in snx.irs]
        igc = snx.igc + [i+1 for i in snx.igc] + [i+2 for i in snx.igc]
        ind = ix + iv + irs + igc + snx.isc + snx.ixpo + snx.ixpor + snx.iypo + snx.iypor + snx.iut + snx.ilod + snx.inutx + snx.inuty + snx.isatax + snx.isatay + snx.isataz

        # And delete the others
        snx.del_ind(np.setdiff1d(range(snx.npar), ind), keep_const=True)

    # Delete unobserved parameters
    #-----------------------------
    def del_unobs_par(snx):

        """
        Delete unobserved parameters
        
        """
        
        # Get indices of unobserved parameters
        ind = np.nonzero(np.diag(snx.N) <= 0)[0].tolist()

        # Add possible missing station coordinates
        ind2 = []
        for i in ind:
            if (snx.param[i].type == 'STAX  '):
                ind2.extend([i+1, i+2])
            elif (snx.param[i].type == 'STAY  '):
                ind2.extend([i-1, i+1])
            elif (snx.param[i].type == 'STAZ  '):
                ind2.extend([i-2, i-1])
        ind = list(set(ind+ind2))
        
        # Delete unobserved parameters
        indk = np.setdiff1d(range(snx.npar), ind)
        snx.N = snx.N[np.ix_(indk, indk)]
        snx.b = snx.b[indk]
        snx.Nc = snx.Nc[np.ix_(indk, indk)]
        snx.x0 = snx.x0[indk]
        snx.sig0 = snx.sig0[indk]
        snx.x = snx.x[indk]
        snx.sig = snx.sig[indk]
        snx.param = [snx.param[i] for i in indk]
        snx.npar = len(indk)
        
        # Reset parameter indices
        snx.set_par_ind()
        
        # Update station list
        snx.clean_sta()

    # Reduce origin, scale and/or orientation information in a normal equation
    #-------------------------------------------------------------------------
    def del_helmerts(snx, helmerts, par):

        """
        Reduce origin, scale and/or orientation information in a normal equation

        Parameters
        ----------
        helmerts : str
            Indicates which Helmert parameters should be considered.
            It can include 'T' (translations), 'S' (scale) and 'R' (rotations).
        par : str
            Indicates which type of parameters should be considered.
            It can be either 'STA' (station positions) or 'VEL' (station velocities).
            
        """
        
        # Get partial derivative matrix of Helmert parameters
        A = snx.helmert_partials(helmerts, par)
        
        # Useful things
        NA = np.dot(snx.N, A)
        ANAi = invspd(np.dot(A.T, NA))
        P = np.dot(NA, ANAi)
        
        # Update snx.N and snx.b
        snx.N = snx.N - np.dot(P, NA.T)
        snx.b = snx.b - np.dot(P, np.dot(A.T, snx.b))
    
    # Delete (reduce) specified stations
    #----------------------------------
    def del_sta(snx, code, pt=None, soln=None, keep_const=False):

        """
        Delete (reduce) specified stations

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        keep_const : bool, optional
            Whether not to remove constraints before reducing specified stations.
            Default is False.
            
        """

        if (len(code) > 0):

            # Indices, keys and holes
            ixv = snx.ix+snx.iv
            par = [snx.param[i] for i in ixv]
            if (pt is not None) and (soln is not None):
                keys = [p.code+p.pt+p.soln for p in par]
                holes = [code[i]+pt[i]+soln[i] for i in range(len(code))]
            elif (pt is not None):
                keys = [p.code+p.pt for p in par]
                holes = [code[i]+pt[i] for i in range(len(code))]
            elif (soln is not None):
                keys = [p.code+p.soln for p in par]
                holes = [code[i]+soln[i] for i in range(len(code))]
            else:
                keys = [p.code for p in par]
                holes = code

            # Get indices of parameters to delete
            ind = []
            for i in range(len(ixv)):
                if (keys[i] in holes):
                    ind.extend(range(ixv[i], ixv[i]+3))
            
            # And delete them
            snx.del_ind(ind, keep_const=keep_const)

    # Delete (reduce) specified radiosources
    #---------------------------------------
    def del_rs(snx, iers):

        """
        Delete (reduce) specified radiosources

        Parameters
        ----------
        iers : list
            List of IERS radiosource names
        
        """

        # Get indices of parameters to delete
        ind = snx.get_rs_ind(iers)
        ind = ind[ind != -1].tolist()

        # And delete them
        snx.del_ind(ind)

    # Delete solution numbers (solns) if there are many of them in an "instantaneous" solution
    #------------------------------------------------------------------------------------------
    def del_duplicates(snx, quiet=False, out=sys.stdout):
        
        """
        Delete solution numbers (solns) if there are many of them in an "instantaneous" solution

        Parameters
        ----------
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """

        lst_del=[]

        for sta in snx.sta:

            #if there are many solns for the same station
            if len(sta.soln) > 1:
            
                if not(quiet):
                    print('{0.code} {0.pt} has {1} soln'.format(sta,len(sta.soln)), file=out)

                for soln in sta.soln :
                    indp = [p.code+p.pt+p.soln for p in snx.param].index(sta.code+sta.pt+soln.soln)
                    p = snx.param[indp]
                    if earlier(soln.datastart, p.tref) and earlier(p.tref,soln.dataend):
                        # if reference date is in the soln
                        if not(quiet):
                            print('{0.tref} in soln {1.soln} : {1.datastart} , {1.dataend}  '.format(p,soln), file=out)
                    
                    else:
                        if not(quiet):
                            print('Remove {0.code} {0.pt}, soln {1.soln} :{1.datastart},{1.dataend}  '.format(p,soln), file=out)

                        lst_del+=[indp,indp+1,indp+2]
            
        snx.del_ind(lst_del)

    # Keep specified stations - Delete (reduce) other stations
    #---------------------------------------------------------
    def keep_sta(snx, code, pt=None, soln=None, keep_const=False):

        """
        Keep specified stations - Delete (reduce) other stations

        Parameters
        ----------
        code : list
            List of 4-char station codes
        pt : list, optional
            List of PT codes. Default is None.
        soln : list, optional
            List of solns. Default is None.
        keep_const : bool, optional
            Whether not to remove constraints before reducing the other stations.
            Default is False.
        
        """


        # Indices, keys and holes
        ixv = snx.ix+snx.iv
        par = [snx.param[i] for i in ixv]
        if (pt is not None) and (soln is not None):
            keys = [p.code+p.pt+p.soln for p in par]
            holes = [code[i]+pt[i]+soln[i] for i in range(len(code))]
        elif (pt is not None):
            keys = [p.code+p.pt for p in par]
            holes = [code[i]+pt[i] for i in range(len(code))]
        elif (soln is not None):
            keys = [p.code+p.soln for p in par]
            holes = [code[i]+soln[i] for i in range(len(code))]
        else:
            keys = [p.code for p in par]
            holes = code

        # Get indices of parameters to delete
        ind = []
        for i in range(len(ixv)):
            if not(keys[i] in holes):
                ind.extend(range(ixv[i], ixv[i]+3))

        # And delete them
        snx.del_ind(ind, keep_const=keep_const)

    # Keep specified radiosources - Delete (reduce) other radiosources
    #-----------------------------------------------------------------
    def keep_rs(snx, iers):

        """
        Keep specified radiosources - Delete (reduce) other radiosources

        Parameters
        ----------
        iers : list
            List of IERS radiosource names
        
        """

        # Get indices of parameters to delete
        ind = np.setdiff1d(snx.irs, snx.get_rs_ind(iers)[:,0])
        ind = ind.tolist() + (ind+1).tolist()

        # And delete them
        snx.del_ind(ind)

    # Delete (reduce) parameters that do not belong to period of interest
    #--------------------------------------------------------------------
    def trim_params(snx, start, end):
        
        """
        Delete (reduce) parameters that do not belong to period of interest

        Parameters
        ----------
        start : str
            Start date (in SINEX date format)
        end : str
            End date (in SINEX date format)
        
        """
        
        # Get indices of parameters to delete
        ind = []
        for i in range(snx.npar):
            t = snx.param[i].tref
            if earlier(t, start) or earlier(end, t):
                ind.append(i)
        
        # And delete them
        snx.del_ind(ind)
        
    # Delete (reduce) solns that are not relevant for specified date
    #---------------------------------------------------------------
    def trim_solns(snx, t, solns):
        
        """
        Delete (reduce) solns that are not relevant for specified date

        Parameters
        ----------
        t : str
            Date (in SINEX date format)
        solns : list
            Reference discontinuity list (from ioutils.read_solns)
            
        """

        # List of stations in reference discontinuity list
        codept = [s.code+s.pt for s in solns]

        # Initialize list of solns to delete
        code = []
        pt = []
        soln = []
        
        # Get solns to delete : loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]
                
            # If current soln is in reference soln table,
            if (p.code+p.pt in codept):
                j = codept.index(p.code+p.pt)
                if (p.soln in [s.soln for s in solns[j].P]):
                    k = [s.soln for s in solns[j].P].index(p.soln)
                    start = solns[j].P[k].start
                    end = solns[j].P[k].end
                    
                    # And if it is not relevant for specified date, add it to the list
                    if ((end != '00:000:00000') and earlier(end, t)) or ((start != '00:000:00000') and earlier(t, start)):
                        code.append(p.code)
                        pt.append(p.pt)
                        soln.append(p.soln)
                            
        # Delete irrelevant solns
        snx.del_sta(code, pt, soln)

    # Delete metadata that are not relevant for specified period
    #-----------------------------------------------------------
    def trim_metadata(snx, start, end):

        """
        Delete metadata that are not relevant for specified period

        Parameters
        ----------
        start : str
            Start date (in SINEX date format)
        end : str
            End date (in SINEX date format)
        
        """
  
        # Loop over stations
        for s in snx.sta:

            # Loop over receivers
            i = 0
            while (i < len(s.rec)):
                r = s.rec[i]

                # If current receiver is relevant for specified period, keep it.
                if ((earlier(r.start, end)) or (r.start == '00:000:00000')) and ((earlier(start, r.end)) or (r.end   == '00:000:00000')):
                    i = i+1

                # Else, remove it.
                else:
                    s.rec.pop(i)

            # Loop over antennas
            i = 0
            while (i < len(s.ant)):
                r = s.ant[i]

                # If current antenna is relevant for specified period, keep it.
                if ((earlier(r.start, end)) or (r.start == '00:000:00000')) and ((earlier(start, r.end)) or (r.end   == '00:000:00000')):
                    i = i+1

                # Else, remove it.
                else:
                    s.ant.pop(i)

            # Loop over eccentricities
            i = 0
            while (i < len(s.ecc)):
                r = s.ecc[i]

                # If current eccentricity is relevant for specified period, keep it.
                if ((earlier(r.start, end)) or (r.start == '00:000:00000')) and ((earlier(start, r.end)) or (r.end   == '00:000:00000')):
                    i = i+1

                # Else, remove it.
                else:
                    s.ecc.pop(i)
            
    # Recover unconstrained normal equation
    #--------------------------------------
    def unconstrain(snx, clear_const=True):
        
        """
        Recover unconstrained normal equation

        Parameters
        ----------
        clear_const : bool, optional
            Whether to clear normal matrix of constraints. Default is True.
            
        """

        # If no a priori information is available, set default a priori information.
        if (snx.x0 is None):
            snx.x0 = snx.x.copy()
            snx.sig0 = np.zeros(snx.npar)
            snx.Nc = np.zeros((snx.npar, snx.npar))

        # Total normal matrix = inverse of covariance matrix
        snx.N = invspd(snx.Q)
        
        # Right-hand side of normal equation
        snx.b = np.dot(snx.N, snx.x - snx.x0)
        
        # Unconstrained normal matrix
        snx.N = snx.N - snx.Nc
                
        # Delete covariance matrix
        snx.Q = None
        
        # If necessary, delete normal matrix of constraints
        if (clear_const):
            snx.clear_const()
            
    # Clear constraints
    #------------------
    def clear_const(snx):
        
        """
        Clear constraints
        
        """
        
        snx.Nc = np.zeros((snx.npar, snx.npar))
        snx.sig0 = np.zeros(snx.npar)
        for p in snx.param:
            p.const = '2'
    
    # Add loose constraints
    #------------------
    def loose_const(snx, sigma):
        """
        Apply loose constraints on POSITION parameters (snx.ix)
        Catref equivalence: used in particular by the "premixs" and "neqloose" modules.
        
        Parameters
        ----------
        sigma : float
            sigma loose constraint value

        Returns
        -------
        Nc : Update of snx.Nc matrix values with loose contraints

        """
        ## Loose constraint, only for POSITION coord line
        val = 1/sigma**2
        #only for pos station
        isnx = [[i, i+1, i+2] for i in snx.ix]
        isnx = np.array(isnx)
        ix = isnx.flatten()
        for i in ix :
            snx.Nc[i,i] = val
            
        return snx.Nc

    # Fix parameters with specified indices in a normal equation
    #-----------------------------------------------------------
    def fix_ind(snx, ind):

        """
        Fix parameters with specified indices in a normal equation

        Parameters
        ----------
        ind : list
            Indices of parameters to fix

        """
        
        # If any parameter to fix,
        if (len(ind) > 0):
        
            # Indices of parameters to keep
            indk = np.setdiff1d(range(snx.npar), ind)
            
            # Update snx.npar, snx.param, snx.x and snx.sig
            snx.npar = len(indk)
            snx.param = [snx.param[i] for i in indk]
            snx.x = snx.x[indk]
            snx.sig = snx.sig[indk]

            # Update snx.x0, snx.sig0 and snx.Nc
            snx.x0 = snx.x0[indk]
            snx.sig0 = snx.sig0[indk]
            snx.Nc = snx.Nc[np.ix_(indk, indk)]
            
            # Update snx.N and snx.b
            snx.N = snx.N[np.ix_(indk, indk)]
            snx.b = snx.b[indk]
                                
            # Reset parameter indices
            snx.set_par_ind()
            
            # Update station list
            snx.clean_sta()
        
    # Fix parameters of specified types in a normal equation
    #-------------------------------------------------------
    def fix_params(snx, types):

        """
        Fix parameters of specified types in a normal equation

        Parameters
        ----------
        types : list
            List of keywords indicating which parameters should be deleted.
            It can include the following keywords:
            'XPO', 'XPOR', 'YPO', 'YPOR', 'UT', 'LOD', 'NUT_X', NUT_Y';
            'ERP' for all kinds of ERPs;
            'SATA_X', 'SATA_Y', 'SATA_Z'; 'SATA' for all satellite PCOs;
            'STA'; 'VEL'; 'RS', 'GC'; 'SC' and 'TRANS' for all transformation parameters.
        
        """
        
        # Get indices of parameters to fix
        ind = snx.get_par_ind(types)
        
        # And fix them
        snx.fix_ind(ind)
        
    # Set up geocenter coordinates in a normal equation
    #--------------------------------------------------
    def setup_gc(snx, tref):
      
        """
        Set up geocenter coordinates in a normal equation

        Parameters
        ----------
        tref : str
            Reference epoch in SINEX date format
            
        """

        # Add XGC parameter
        p = record()
        p.type  = 'XGC   '
        p.code  = '----'
        p.pt    = '--'
        p.soln  = '----'
        p.tref  = tref
        p.unit  = 'm   '
        p.const = '2'
        snx.param.append(p)
        
        # Add YGC parameter
        p = copy.deepcopy(p)
        p.type = 'YGC   '
        snx.param.append(p)
        
        # Add ZGC parameter
        p = copy.deepcopy(p)
        p.type = 'ZGC   '
        snx.param.append(p)
        
        # Update snx.igc, snx.x, snx.sig, snx.x0, snx.sig0 and snx.Nc
        snx.igc.append(snx.npar)
        snx.x = np.hstack((snx.x, [0, 0, 0]))
        snx.sig = np.hstack((snx.sig, [0, 0, 0]))
        snx.x0 = np.hstack((snx.x0, [0, 0, 0]))
        snx.sig0 = np.hstack((snx.sig0, [0, 0, 0]))
        snx.Nc = np.vstack((np.hstack((snx.Nc, np.zeros((snx.npar, 3)))), np.zeros((3, snx.npar+3))))

        # Initialize design matrix to identity
        A_rows = list(range(snx.npar))
        A_cols = list(range(snx.npar))
        A_vals = snx.npar * [1]
        
        # Complete design matrix with STA/GC partial derivatives
        ix = snx.ix
        iy = [i+1 for i in snx.ix]
        iz = [i+2 for i in snx.ix]
        A_rows.extend([snx.npar]*len(ix) + [snx.npar+1]*len(ix) + [snx.npar+2]*len(ix))
        A_cols.extend(ix+iy+iz)
        A_vals.extend([-1]*3*len(ix))

        # Build sparse design matrix
        A = sparse.csr_matrix((A_vals, (A_rows, A_cols)))

        # Update snx.N and snx.b
        snx.N = A.dot((A.dot(snx.N)).T)
        snx.b = A.dot(snx.b)

        # Update snx.npar
        snx.npar = snx.npar + 3
        
    # Set up scale factor in a normal equation
    #-----------------------------------------
    def setup_sc(snx, tref):
      
        """
        Set up geocenter coordinates in a normal equation

        Parameters
        ----------
        tref : str
            Reference epoch in SINEX date format
            
        """

        # Add DSC parameter
        p = record()
        p.type  = 'DSC   '
        p.code  = '----'
        p.pt    = '--'
        p.soln  = '----'
        p.tref  = tref
        p.unit  = 'ppb '
        p.const = '2'
        snx.param.append(p)
                
        # Update snx.isc, snx.x, snx.sig, snx.x0, snx.sig0 and snx.Nc
        snx.isc.append(snx.npar)
        snx.x = np.hstack((snx.x, [0]))
        snx.sig = np.hstack((snx.sig, [0]))
        snx.x0 = np.hstack((snx.x0, [0]))
        snx.sig0 = np.hstack((snx.sig0, [0]))
        snx.Nc = np.vstack((np.hstack((snx.Nc, np.zeros((snx.npar, 1)))), np.zeros((1, snx.npar+1))))

        # Initialize design matrix to identity
        A_rows = list(range(snx.npar))
        A_cols = list(range(snx.npar))
        A_vals = snx.npar * [1]
        
        # Complete design matrix with STA/SC partial derivatives
        ix = snx.ix + [i+1 for i in snx.ix] + [i+2 for i in snx.ix]
        A_rows.extend([snx.npar]*len(ix))
        A_cols.extend(ix)
        A_vals.extend((1e-9*snx.x0[ix]).tolist())

        # Build sparse design matrix
        A = sparse.csr_matrix((A_vals, (A_rows, A_cols)))

        # Update snx.N and snx.b
        snx.N = A.dot((A.dot(snx.N)).T)
        snx.b = A.dot(snx.b)

        # Update snx.npar
        snx.npar = snx.npar + 1

    # Set a priori parameter values to reference values
    #--------------------------------------------------
    def prior2ref(snx, ref):
        
        """
        Set a priori parameter values to reference values
        
        Parameters
        ----------
        ref : sinex instance
            Solution containing reference parameter values
       
        """

        # Get indices of common parameters
        (isnx, iref) = snx.get_common_par(ref)

        # And change a priori parameter values
        dx0 = np.zeros(snx.npar)
        dx0[isnx] = ref.x[iref] - snx.x0[isnx]
        if (np.any(dx0)):
            snx.x0 = snx.x0 + dx0
            if (snx.N is not None):
                snx.b = snx.b - np.dot(snx.N, dx0)
        
    # Add NNR, NNT and/or NNS constraints to normal matrix of constraints
    #--------------------------------------------------------------------
    def add_mc(snx, helmerts, par, sigma=1e-5, datum=None, crf_datum=None, thr=None, proj=True, periods=[]):
        
        """
        Add NNR, NNT and/or NNS constraints to normal matrix of constraints
        
        Returns
        -------
        nc : int
            Number of constraints added

        Parameters
        ----------
        helmerts : str
            Indicates which Helmert parameters should be constrained.
            It can include 'T' (translations), 'S' (scale), 'R' (rotations)
            and 'A' (CRF rotations).
        par : str
            Indicates to which type of parameters constraints should be applied.
            It can be either 'STA' (station and radiosource positions),
            'VEL' (station velocities - radiosource velocities not supported yet).
            'PERIOD' (station seasonal signals 'period')
        sigma : float or str, optional
            Sigma of minimal constraints in m[/y]. Default is 1e-5.
            If set to 'auto', an adequate sigma is automatically computed based on the
            median of the diagonal elements of the normal matrix that correspond to
            positions/velocities of stations to which constraints are applied:
            sigma = 0.01 / sqrt(median(N_{i,i})).
        datum : sinex instance, optional
            Reference TRF solution with respect to which constraints should be applied.
            Default is None (constraints applied with respect to snx.x0).
        crf_datum : sinex instance, optional
            Reference CRF solution with respect to which constraints should be applied.
            Default is None (constraints applied with respect to snx.x0).
        thr : float, optional
            If set, then stations with large uncertainties will be rejected from the set
            of stations to which constraints are applied. The screening is based on the
            traces of the 3x3 diagonal blocks of the normal matrix that correspond to
            positions/velocities of the candidate stations. Stations with traces
            lower than the median of traces divided by thr**2 are iteratively rejected.
        periods: list of objects (built with pytrf.utils.Period), optional
            Period of periodic signals. Must be specified if mc_type = "PERIOD". Each Period object contains Period attributes, as 'value' or 'code'
        """
        
        # If a datum is specified,
        if (datum):
            
            # Get indices of common stations
            if (par == 'STA'):
                (isnx, iref) = snx.get_common_sta(datum)
            elif (par == 'VEL'):
                (isnx, iref) = snx.get_common_vel(datum)
            elif (par == 'PERIOD'):
                (isnx, iref) = snx.get_common_per(datum)
                
            isnx = np.array(isnx)
            iref = np.array(iref)
            ix = isnx.flatten()
            ir = iref.flatten()
            
            # Modify a priori coordinates of common stations
            dx0 = np.zeros(snx.npar)
            dx0[ix] = datum.x[ir] - snx.x0[ix]
            if (np.any(dx0)):
                snx.x0 = snx.x0 + dx0
                snx.b = snx.b - np.dot(snx.N, dx0)
            
        # Else,
        else:
            
            # Get indices of all stations
            if (par == 'STA'):
                isnx = [[i, i+1, i+2] for i in snx.ix]
            elif (par == 'VEL'):
                isnx = [[i, i+1, i+2] for i in snx.iv]
            elif (par == 'PERIOD'):
                isnx = [[i, i+1, i+2, i+3, i+4, i+5] for i in snx.iper]
            isnx = np.array(isnx)
            ix = isnx.flatten()
        
        # If a CRF datum is specified,
        if (crf_datum) and (par == 'STA'):

            # Get indices of common radiosources
            (irs, iref) = snx.get_common_rs(crf_datum)
            irs = np.array(irs).flatten()
            iref = np.array(iref).flatten()
            
            # Modify a priori coordinates of common radiosources
            dx0 = np.zeros(snx.npar)
            dx0[irs] = crf_datum.x[iref] - snx.x0[irs]
            if (np.any(dx0)):
                snx.x0 = snx.x0 + dx0
                snx.b = snx.b - np.dot(snx.N, dx0)

        # Else, get indices of all radiosources,
        elif (par == 'STA'):
            irs = [[i, i+1] for i in snx.irs]
            irs = np.array(irs, dtype='int').flatten()
            
        # Else, 
        elif (par == 'VEL') or (par=='PERIOD'):
            irs = np.array([], dtype='int')
            
        
        # If a threshold is specified, reject candidate stations with large uncertainties
        if (thr):
            end = False
            while not(end):
                tr = np.array([np.sum(snx.N[i,i]) for i in isnx])
                ind = np.nonzero(tr > np.median(tr)/thr**2)[0]
                if (len(ind) == len(isnx)):
                    end = True
                else:
                    isnx = isnx[ind]
            ix = isnx.flatten()

        # If sigma of minimal constraints needs to be computed, compute it
        if (sigma == 'auto'):
            sigma = 0.01 / sqrt(np.median(snx.N[ix,ix]))
        
        # Design matrix of minimal constraints
        ix = np.hstack((ix, irs))
        
        if (par == 'STA') or (par == 'VEL'):
            A = snx.helmert_partials('RSTA', par, units='m')[ix,:7]
            
            # Indices of relevant columns of A
            ind = []
            if ('T' in helmerts):
                ind.extend(range(0, 3))
            if ('S' in helmerts):
                ind.append(3)
            if ('R' in helmerts):
                ind.extend(range(4, 7))
            if ('A' in helmerts):
                ind.extend(range(7, 10))
                
        elif (par == 'PERIOD'):
            #get helmerts partial deriv. for all periods and param anyway: RSTA
            A = snx.helmert_partials('RSTA', par, units='m', select_periods=[per.code for per in periods])[ix] #'select_periods' param: be sure of A columns order 
            #shape A: (k,20*nper) 20: 10 cos (TTTSRRRAAA) + 10 sin (TTTSRRRAAA)
            
            # Indices of relevant columns of A
            ind = []                
            for num, per in enumerate(periods): #loop over each period
                for cs in range(2): #cos & sin 
                    if ('T' in helmerts) and ('T' in per.mc_per): #ask by helmerts + depends on each period attribute: 'mc_per'
                        ind.extend(7*(2*num+cs) + np.arange(0, 3))
                    if ('S' in helmerts) and ('S' in per.mc_per):
                        ind.append(7*(2*num+cs) + 3)
                    if ('R' in helmerts) and ('R' in per.mc_per):
                        ind.extend(7*(2*num+cs) + np.arange(4, 7))
            
        # Either reduce columns of A and compute B
        if not(proj):
            A = A[:,ind]
            B = np.dot(invspd(np.dot(A.T, A)), A.T)
        
        # or compute B and reduce rows of B
        else:
            B = np.dot(invspd(np.dot(A.T, A)), A.T)
            B = B[ind,:]
        
        # Add minimal constraints to normal matrix of constraints
        ix2 = np.ix_(ix, ix)
        snx.Nc[ix2] = snx.Nc[ix2] + np.dot(B.T, B) / sigma**2
        
        # Change constraint codes of constrained parameters
        for i in ix:
            if (snx.param[i].const == '2'):
                snx.param[i].const = '1'
                
        return A.shape[1]
    
    # Add mean or trend Internal Constraints (on R,S and/or T parameters) to normal matrix of constraints.
    #--------------------------------------------------------------------
    def add_ic(snx, dict_helmert, ic_type, sigma=1e-5, t0=None, periods=[], debug=False):
    
        """
        Add R, T and/or S internal constraints to normal matrix of constraints. Available on 'MEAN' or 'TREND' constrains (par attribute).
        If you want both MEAN and TREND constraints, apply twice this method with par=MEAN and par=TREND
        You must specify parameters in the YAML file for each solution that you want apply IC: 'ic', 'ic_trend', 'ic_period'
        
        Returns
        -------
        nc : int
            Number of constraints added. Maximum 7 (RX, RY, RZ, S, TX, TY, TZ)

        Parameters
        ----------
        dict_helmert : dict of str
            Indicates which Helmert parameters should be constrained, for a particular solution (snx.param[k].isol).
            It can include 'T' (translations), 'S' (scale), 'R' (rotations)
            and 'A' (CRF rotations).
            Format example for combination of 3 soltutions : dict_helmert = {0 :'RST', 2:'RS', 3:''}. Key refer to a solution.
        ic_type : str
            Indicates to which type of constraints should be applied.
            It can be either 'MEAN' (internal constraints on MEAN) 'TREND' (internal constraints on TREND) or 'PERIOD' (internal constraints on periodic signals).
        sigma : float or str, optional
            Sigma of minimal constraints in m[/y]. Default is 1e-5.
            If set to 'auto', an adequate sigma is automatically computed based on the
            median of the diagonal elements of the normal matrix that correspond to
            positions/velocities of stations to which constraints are applied:
            sigma = 0.01 / sqrt(median(N_{i,i})).
        t0 : str
            Reference date (SINEX date format)
            Must be specified if par='TREND'
        periods: list of objects (built with pytrf.utils.Period), optional
            Period of periodic signals. Must be specified if ic_type = "PERIOD". Each Period object contains Period attributes, as 'value' or 'code'
        """
        #initialize nc : number of constraints
        nc = 0
        # get all param TRANS indices
        all_transf_id = snx.itrans
        
        # dict of id according to R, S,T (classication):
        #in each list, we keep param only according to dict_helmert from user YAML
        dict_transf = {'RX':[],
                       'RY':[],
                       'RZ':[],
                       'TX':[],
                       'TY':[],
                       'TZ':[],
                       'SC':[]}
        
        # Initialize ic_trend
        vect_ic = None
        mat_ic = None 
        if ic_type == 'TREND': #initialize tk-t0 vector
            vect_ic = np.zeros((snx.Nc.shape[0],1))
            
        if ic_type == 'PERIOD': #initialize tk vector
            # 2 dims array > line R...S...T, column according to periods (if contribute or not)
            vect_ic = np.zeros((snx.Nc.shape[0],len(periods)))
            mat_ic = np.zeros((snx.Nc.shape))
                
        # names are in snx.param. 1 object by line. We look "type" attribute.
        # filter id according code name in dict_helmert
        for (num, par) in enumerate(np.array(snx.param)[all_transf_id]):
            
            if par.isol in list(dict_helmert.keys()): #ic_ for this sol ? isol is int 0...k
                if par.type[0] in dict_helmert[par.isol] : #'par.type[0]' can be R, S or T > which one ask by user ?
                    #Here, at least 1 Internal constraint apply on this TRANSF param
                    #we are looking for if 'R...', 'SC' or 'T...' are in par.type
                    
                    #add to list inside dict_transf: par.type can also be with X, Y, Z dims
                    dict_transf[par.type[0:2]].append(all_transf_id[num]) #type[0:2] because par.type like 'RX    ' -> 'RX'
                    #we will apply same sigma on X, Y, Z for same dim, i.e. on RX, RY & RZ > id same in same key 'R'
                    
                    if ic_type == 'TREND' :#complete vect_ic
                        vect_ic[all_transf_id[num]] = (date.from_tsnx(par.tref).mjd - date.from_tsnx(t0).mjd)/365.25 #decimal year conversion
                    
                    if ic_type == 'PERIOD':
                        #time delta
                        dt = (date.from_tsnx(par.tref).mjd - date.from_tsnx(t0).mjd)/365.25 #decimal year conversion
                        for (num_per, per) in enumerate(periods):
                            if par.type[0] in per.ic_period: # yes const for this period on this para
                                vect_ic[all_transf_id[num], num_per] = dt #2 dim vect_ic : transf param sol dt according to period
                                           
                    
        ## convert sigma according to dim, giv by user in meter
        sigma_T = sigma * 1000 #mm conversion
        sigma_R = sigma * 1/(ae*mas2rad) #mas conversion
        sigma_S = sigma * 1/(1e-9*ae) #ppb conversion
        
        dict_sigma = {'T':sigma_T, 'R':sigma_R, 'S':sigma_S}
        
        if ic_type == 'MEAN' :
            ## complet Nc matrix with 1/sigma²
            for key in dict_transf.keys():
                snx.Nc[np.ix_(dict_transf[key],dict_transf[key])] += 1/(dict_sigma[key[0]]**2) #key[0] : R, S or T
                
                if len(dict_transf[key]) != 0: #we have 1 more constraint on this param "key": RX, RY, RZ, S, TX, TY, TZ
                    nc +=1
            
        elif ic_type == 'TREND' :
            #Same dim than Nc
            mat_ic = vect_ic @  vect_ic.T
            ## complet Nc matrix with 1/sigma²
            for key in dict_transf.keys():
                snx.Nc[np.ix_(dict_transf[key],dict_transf[key])] += 1/(dict_sigma[key[0]]**2) * mat_ic[np.ix_(dict_transf[key],dict_transf[key])]
                
                if len(dict_transf[key]) != 0: #we have 1 more constraint on this param "key": RX, RY, RZ, S, TX, TY, TZ
                    nc +=1
                
        elif ic_type == 'PERIOD' :
            ## complet Nc matrix with 1/sigma²
            tst_sum = np.zeros((snx.Nc.shape))
            for (num_per, per) in enumerate(periods): #sum for each period if it contributes
                wp = 2*np.pi/(per.value/365.25) #year conversion
                mat_ic[np.ix_(all_transf_id)] = np.cos(wp*(vect_ic[:,num_per].reshape(vect_ic.shape[0],1)-vect_ic[:,num_per].reshape(vect_ic.shape[0],1).T))[np.ix_(all_transf_id)]
               
                for key in dict_transf.keys(): #select correct param according to type
                    snx.Nc[np.ix_(dict_transf[key],dict_transf[key])] += 1/(dict_sigma[key[0]]**2) * mat_ic[np.ix_(dict_transf[key],dict_transf[key])]
                    tst_sum[np.ix_(dict_transf[key],dict_transf[key])] += 1/(dict_sigma[key[0]]**2) * mat_ic[np.ix_(dict_transf[key],dict_transf[key])]
                    
                    if len(dict_transf[key]) != 0: #we have 1 more constraint on this param "key": RX, RY, RZ, S, TX, TY, TZ
                        nc += 2 #sin and cos                
    
        if debug :
            return nc , dict_transf,vect_ic, mat_ic, tst_sum
        else:
            return nc
            

    # Add equality constraints between successive velocities to normal matrix of constraints
    #---------------------------------------------------------------------------------------
    def add_dvc(snx, solns, sigma=1e-6):
        
        """
        Add equality constraints between successive velocities to normal matrix of constraints
        
        Returns
        -------
        nc : int
            Number of constraints added

        Parameters
        ----------
        solns : list
            Reference discontinuity list (from ioutils.read_solns)
        sigma : float, optional
            Sigma of velocity equality constraints in m/y. Default is 1e-6.
            
        """
        
        # Initializations
        nc = 0
        keys = [s.code+s.pt for s in solns]
        keys_v = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iv]]
        
        # Loop over stations
        for sta in snx.sta:

            # Index of current station in discontinuity list
            if (sta.code+sta.pt in keys):
                isoln = keys.index(sta.code+sta.pt)

                # Loop over solns
                for i in range(len(sta.soln)-1):
                    
                    # Get end date of current soln
                    ip = [p.soln for p in solns[isoln].P].index(sta.soln[i].soln)
                    end = solns[isoln].P[ip].end
                    
                    #start next station:
                    ips = [p.soln for p in solns[isoln].P].index(sta.soln[i+1].soln)
                    start_next = solns[isoln].P[ips].start
                    
                    #vel discontinuities (decimal year)
                    list_v_dicontinuities = [date.from_tsnx(v.end).ydec() for v in solns[isoln].V if v.end !='00:000:00000']
                                    
                    if not(not(end in [v.end for v in solns[isoln].V]) and not any(date.from_tsnx(end).ydec() <= disc <= date.from_tsnx(start_next).ydec() for disc in list_v_dicontinuities)):
                        print(f"[add_dvc] DIFFERENCES SOLN VEL disc potential bug: {sta.code} --> {not(end in [v.end for v in solns[isoln].V])} {not any(date.from_tsnx(end).ydec() <= disc <= date.from_tsnx(start_next).ydec() for disc in list_v_dicontinuities)} ")
                           
                    # If current soln should be constrained with the next one,
                    #if not(end in [v.end for v in solns[isoln].V]): #si end="00:000:00000", considéré comme une disc ??
                    if not any(date.from_tsnx(end).ydec() <= disc <= date.from_tsnx(start_next).ydec() for disc in list_v_dicontinuities):
                        #if sta.code =="REZB":
                        #print(f'Apply successive VEL const: {sta.code+sta.pt+sta.soln[i].soln}, list:{[v.end for v in solns[isoln].V]}')
                        # Get indices of both velocities
                        i1 = keys_v.index(sta.code+sta.pt+sta.soln[i].soln)
                        i2 = keys_v.index(sta.code+sta.pt+sta.soln[i+1].soln) ######## --><<>>>>< pbm ou il y a 1 trou ds le prochain SOLN on a que 1 & 3, or discontinuite entre 2 et 3 ....
                        
                        # Add constraints between them
                        for k in range(3):
                            snx.Nc[snx.iv[i1]+k,snx.iv[i1]+k] += 1 / sigma**2
                            snx.Nc[snx.iv[i1]+k,snx.iv[i2]+k] -= 1 / sigma**2
                            snx.Nc[snx.iv[i2]+k,snx.iv[i1]+k] -= 1 / sigma**2
                            snx.Nc[snx.iv[i2]+k,snx.iv[i2]+k] += 1 / sigma**2
                        nc += 3
                        
        return nc
    
    # Add equality constraints between successive amplitudes (periodic signals) to normal matrix of constraints
    #---------------------------------------------------------------------------------------
    def add_dac(snx, solns, sigma=1e-6):
        
        """
        Add equality constraints between amplitudes of successive periodic signals to normal matrix of constraints
        
        Returns
        -------
        nc : int
            Number of constraints added

        Parameters
        ----------
        solns : list
            Reference discontinuity list (from io.read_solns)
        sigma : float, optional
            Sigma of velocity equality constraints in m/y. Default is 1e-6.
            
        """
        
        # Initializations
        nc = 0
        keys = [s.code+s.pt for s in solns]
        keys_per = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iper_dict[next(iter(snx.iper_dict))]] ]
       
        # Loop over stations
        for sta in snx.sta:
            
            # Index of current station in discontinuity list
            if (sta.code+sta.pt in keys):
                isoln = keys.index(sta.code+sta.pt)
                # Loop over solns
                for i in range(len(sta.soln)-1):
                    
                    # Get end date of current soln
                    ip = [p.soln for p in solns[isoln].P].index(sta.soln[i].soln)
                    end = solns[isoln].P[ip].end
                    
                    # If current soln should be constrained with the next one,
                    if not(end in [a.end for a in solns[isoln].A]): #based on code 'A' in soln
                                            
                        # Get indices of both amplitude
                        i1 = keys_per.index(sta.code+sta.pt+sta.soln[i].soln)
                        i2 = keys_per.index(sta.code+sta.pt+sta.soln[i+1].soln)
                        
                        for percode in snx.iper_dict.keys(): #for each period, according to key of iper_dict (ex:"A001", "A002", etc)
                            # Add constraints between them
                            for k in range(6):# COSX, SINX, COSY, SINY, COSZ, SINZ
                                snx.Nc[snx.iper_dict[percode][i1]+k,snx.iper_dict[percode][i1]+k] += 1 / sigma**2
                                snx.Nc[snx.iper_dict[percode][i1]+k,snx.iper_dict[percode][i2]+k] -= 1 / sigma**2
                                snx.Nc[snx.iper_dict[percode][i2]+k,snx.iper_dict[percode][i1]+k] -= 1 / sigma**2
                                snx.Nc[snx.iper_dict[percode][i2]+k,snx.iper_dict[percode][i2]+k] += 1 / sigma**2
                            nc += 6
                        
        return nc
    
    # Generate vfconst.yml file (velocities constraints + frequencies constraints) file for stations located on the same site
    #-----------------------
    def vfconst_file(snx, sigma = 1e-6, periods=[]):
        """
        Generate vfconst.yml file : velocities constraints + frequencies constraints
        Constraints for stations located on the same site (relative const) or for 1 single station (absolute const).

        Parameters
        ----------
        sigma: float, optional
            sigma constraint [m/y] or [m]
        periods: list of objects (built with pytrf.utils.Period), optional
            Period of periodic signals
        Returns
        -------
        None.

        """
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>><>>>>>>>> move to graph class                 
        #Init empty site dict. key:site id (i.e. first 5 characters of domes)
        dict_sites = {}
        dict_sites_soln = {}
        dict_const = {}
        # sort stations by site
        for sta in snx.sta:
            dict_sites.setdefault("site"+sta.domes[:5], []).append(sta) # 1 list site: {"11000":[...], "12000":[...]}
            dict_sites_soln.setdefault("site"+sta.domes[:5], []).append(sta.code + sta.pt + sta.soln[0].soln) #concat str, only on 1st soln (cf add_dvc apply on successive soln id)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
        def site_const(sta1, sta2=None, const_time=1, const_dist=10000, const_nobs_factor=1):
            """
            Determine if a constraint must be set btw 2 stations on a site (5 domes first characters), regarding time measurement, distance and number of observations (nobs).
            If 1 station, distance not considered.

            Parameters
            ----------
            sta1 : pytrf.utils.record
                DESCRIPTION.
            sta2 : pytrf.utils.record
                DESCRIPTION. The default is None.
            const_time : float, optional
                Minimal time observation [unit: year]. If under, set constraint.
            const_dist : float, optional
                Maximal distance btw 2 stations to be considered on the same site [unit: meter]. If under, set constraint.
            const_nobs_factor : float, optional
                Factor multiplied to minimal nobs number to determine a new minimal nobs value. If under minimal nobs, set constraint.
                    Default minimal nobs:
                        *VEL: 3 obs
                        *PERIOD: 6 obs/ period (ex: 2 periods: 12 obs, 3 periods 18 obs)
                 However to have a margin (questioning the quality of observations), we can set const_nobs_factor. 
                 For examples: 
                     *const_nobs_factor = 2 -> min VEL: 6 obs, PERIODS: 12 obs /period
                     *const_nobs_factor = 0 ->  no minimal obs number, set const vfconst anyway.

            Returns
            -------
            const_vel: bool
                Add const on velocity
                If True, a constraint will be applied (add to vfconst.yaml). If False, no constraint added to vfconst.yaml
                
            const_period: bool
                Add const on periods
                If True, a constraint will be applied (add to vfconst.yaml). If False, no constraint added to vfconst.yaml

            """
            #init constraints bool: default: no constraint
            const_vel = False
            const_period = False
            
            ## 1.distance if sta1 & sta2
            if sta2 != None:
                dist = snx.dist_stations(sta1, sta2, type_dist="cartesian")
                if dist > const_dist: # to far away
                    logging.warning(f"No const btw '{sta1.domes} {sta1.code+sta1.pt}' & '{sta2.domes} {sta2.code+sta2.pt}': too far away ({round(dist/1000,4)} km)")
                    return False, False
                else:
                    return True, True
            
            
            else: #if 1 sta, nobs and time range constraints? 
                # add constraints at least if 1 station has not enough observation or small time range
                # loop over soln
                nobs_tot = 0
                start = date.from_tsnx(sta1.soln[0].datastart).ydec()
                end = date.from_tsnx(sta1.soln[-1].dataend).ydec()
                
                for soln in sta1.soln:
                    if hasattr(soln, 'nobs'): #'nobs' added only in combsnx case
                        nobs_tot += soln.nobs
                    else: #simple sinex
                        nobs_tot += 1
                    #update start date and end date
                    start = min(date.from_tsnx(soln.datastart).ydec(), start)
                    end = max(date.from_tsnx(soln.dataend).ydec(), end)
                
                ## 2.nobs & delat t
                deltat = end - start #[year]
                #velocity
                if (nobs_tot < 20 * const_nobs_factor) and (deltat < const_time): # at least 20 obs velocity (catref)
                    logging.warning(f"Add const on VEL '{sta1.domes} {sta1.code+sta1.pt}': nobs={nobs_tot} & dtime={round(deltat,3)} y")
                    const_vel = True
                
                #nobs on periods
                if (len(periods)>0) and (nobs_tot < 6 * const_nobs_factor): #at least 1 period
                    logging.warning(f"Add const on PERIOD '{sta1.domes} {sta1.code+sta1.pt}': nobs={nobs_tot}")
                    const_period = True
                
                ## 3.time
                # periods
                if (len(periods)>0) and (deltat < const_time): #at least 1 period
                    logging.warning(f"Add const on PERIOD '{sta1.domes} {sta1.code+sta1.pt}': delta_time={round(deltat,3)} y")
                    const_period = True
                    
            return const_vel, const_period
                    
                
         
        # build constraint file  
        for key, values in dict_sites_soln.items():
            if len(values) > 1: #at least 2 sta on the same site
                sta1 = values[0] #1st station is the reference
                dict_const[key] = [] #init list
                for numsta2, sta2 in enumerate(values[1:]):
                    # check if we add const, regarding time range, nobs and distance
                    const_vel, const_period = site_const(dict_sites[key][0], dict_sites[key][numsta2+1])
                    if const_vel:
                        dict_const[key].append({"type": "VEL", "sta1":sta1, "sta2":sta2, "sigma":sigma}) # we make a combination between 1st station in the site and all the other
                   
                    if const_period:
                        for per in periods: #periods
                            dict_const[key].append({"type": per.code, "sta1":sta1, "sta2":sta2, "sigma":sigma}) # we make a combination between 1st station in the site and all the other
                 
                        
            elif len(values)==1: #just 1 station on this site. absolute constraint
                sta1 = values[0]
                dict_const[key] = [] #init list
                
                # check if we add const, regarding time range, nobs and distance
                const_vel, const_period = site_const(dict_sites[key][0])
                
                if const_vel:
                    dict_const[key].append({"type": "VEL", "sta1":sta1, "sigma":1e-1}) # we make a combination between 1st station in the site and all the other. sigma value from CATREF
                else:
                    #print("NO VEL vfconst",key, " sta1:", dict_sites[key][0].domes)
                    pass
                
                if const_period:
                    for per in periods: #periods
                        dict_const[key].append({"type": per.code, "sta1":sta1, "sigma":sigma}) 
                else:
                    #print("NO PERIOD vfconst",key, " sta1:", dict_sites[key][0].domes)
                    pass
        #yaml write           
        with open("vfconst.yml", 'w') as file: #this file can be open by pytrf.io.read_yaml()
            file.write("#'type': velocity 'VEL' or period code 'Axxx' (annual), 'Dxxx' (draconitic), 'Pxxx' (other)\n")
            for site in sorted(dict_const.keys()):
                if len(dict_const[site]) != 0:
                    file.write(f"#{site}\n")#comment site name
                    #1 line by site
                    for const in dict_const[site]:
                        file_yml = yaml.dump(const, sort_keys=False, default_flow_style=True)
                        file.write("- " + file_yml) #format list in YAML format
            
        return dict_sites, dict_sites_soln, dict_const
    
    
    # get linked stations according vfconst.yml file
    def get_sites_from_vfconst(snx, file):
        #open yaml site constraints
        cf = read_yaml(file)
        
        # Create a undirected graph
        graph = nx.Graph()
        
        # Loop over constrains
        for const in cf:
            # convert record to dict
            const = const.__dict__
            if "sta2" in const.keys(): #at least 2 stations on this site
                #no soln consideration
                graph.add_edge(const['sta1'], const['sta2'])
                
        #snx.graph = snx.graph.to_undirected()
        # Find connected components
        connected_components = list(nx.connected_components(graph))
        #print(graph.number_of_nodes(), graph.number_of_edges(), graph.nodes,connected_components)
        
        # Filter out single nodes (isolated vertices)
        connected_components = [component for component in connected_components if len(component) > 1]
        
        print(connected_components)
        
        # Convert connected components to lists for better representation
        linked_sta_lists = [list(component) for component in connected_components]
        
        return linked_sta_lists
                        
        
        
    
    
    # Apply vfconst.yml constraints for stations located on the same site
    #-----------------------
    def add_vfconst(snx, file="vfconst.yml", sigma = 1e-6, periods=[]):
        """
        Add constraints between stations located on the same site (velocity + period).
        Informations in 'file' (yaml file format), template generated by vfconst_file()
        If 'file' not found in repository, vfconst_file() is applied (vfconst.yml file is written)
        
        Returns
        -------
        nc : int
            Number of constraints added
        
        Parameters
        ----------
        file: str, optional (Default: "vfconst.yml")
            File name (yaml) containing site constraints. Generated by sinex.vfconst_file()
            If no 'file' found in repository, sinex.vfconst_file() applied (vfconst.yml written)
        sigma: float, optional
            sigma constraint [m/y] or [m]
        periods: list of objects (built with pytrf.utils.Period), optional
            Period of periodic signals
        """
        if not os.path.exists(file):#e no file provides by user, vfconst_file applied
            print("No file for constrains on site (velocity+periods) provides by user, generate 'vfconst.yml' automatically")
            snx.vfconst_file(sigma=sigma, periods=periods)
            file= "vfconst.yml"
        
        #open yaml site constraints
        cf = read_yaml(file) # list of "record" (from pytrf.utils)
        #record parameters: "sta1"; "type"; "sigma" (absolute const) + "s2" (relative cons
    
        # Initializations
        nc = 0
        
        #index
        keys_v = [(p.code+p.pt+p.soln).replace(" ", "") for p in [snx.param[i] for i in snx.iv]] #key without space " ", more flexible for vfconst.yml
        
        if len(periods)!=0:
            keys_per = [(p.code+p.pt+p.soln).replace(" ", "") for p in [snx.param[i] for i in snx.iper_dict[next(iter(snx.iper_dict))]] ]
        else: #no periods
            keys_per=[]
        # Loop over constrains
        for const in cf:
            # convert record to dict
            const = const.__dict__
            sigma = float(const["sigma"])
            
            ### velocity case
            if const["type"] == "VEL":
                
                try: #find param index in sinex
                    i1 = keys_v.index(const["sta1"].replace(" ", "")) #sta1
                    if "sta2" in const.keys():
                        i2 = keys_v.index(const["sta2"].replace(" ", "")) #sta2
                        
                    # Add constraints between them on 3 dims
                    for k in range(3):
                        snx.Nc[snx.iv[i1]+k,snx.iv[i1]+k] += 1 / sigma**2
                        
                        if "sta2" in const.keys(): #relative to another station "sta2" on the site
                            snx.Nc[snx.iv[i1]+k,snx.iv[i2]+k] -= 1 / sigma**2
                            snx.Nc[snx.iv[i2]+k,snx.iv[i1]+k] -= 1 / sigma**2
                            snx.Nc[snx.iv[i2]+k,snx.iv[i2]+k] += 1 / sigma**2
                    nc += 3
                    
                except Exception as e: #station not find in sinex...
                    logging.warning(f"[sinex add_vfconst] Unknown station: {e}. Check line '{const}' in {file} file.")
                    
                
            
            ### period case
            elif const["type"] in snx.iper_dict.keys():
                
                percode =  const["type"] #period code: must be key of iper_dict (ex:"A001", "A002", etc)
                   
                try: #find param index in sinex
                    i1 = keys_per.index(const["sta1"].replace(" ", "")) #sta1
                    if "sta2" in const.keys():
                        i2 = keys_per.index(const["sta2"].replace(" ", "")) #sta1
                        
                    # Add constraints between them on 3 dims COS + 3 dims SIN
                    for k in range(6):# COSX, SINX, COSY, SINY, COSZ, SINZ
                        snx.Nc[snx.iper_dict[percode][i1]+k,snx.iper_dict[percode][i1]+k] += 1 / sigma**2
                        
                        if "sta2" in const.keys(): #relative to another station "sta2" on the site
                            snx.Nc[snx.iper_dict[percode][i1]+k,snx.iper_dict[percode][i2]+k] -= 1 / sigma**2
                            snx.Nc[snx.iper_dict[percode][i2]+k,snx.iper_dict[percode][i1]+k] -= 1 / sigma**2
                            snx.Nc[snx.iper_dict[percode][i2]+k,snx.iper_dict[percode][i2]+k] += 1 / sigma**2
                    nc += 6
                    
                except Exception as e: #station not find in sinex...
                    logging.warning(f"[sinex add_vfconst] Unknown station: {e}. Check line '{const}' in {file} file.")
                
            ## unknown type    
            else: 
                raise ValueError(f"Unknown type '{const['type']}' in YAML file, at line: '{const}'.\nIn this sinex, initialized periods are {list(snx.iper_dict.keys())} or set 'type':'VEL'.")
            
        
        return nc
        

    # Invert normal equation
    #-----------------------
    def neqinv(snx, clear_neq=True):

        """
        Solve normal equation

        Parameters
        ----------        
        clear_neq : bool, optional
            Whether to clear normal equation. Default is True.

        """
        
        # Solve normal equation
        snx.Q = invspd(snx.N + snx.Nc)
        snx.x = snx.x0 + np.dot(snx.Q, snx.b)
        snx.sig = np.sqrt(np.diag(snx.Q))

        # Clear snx.N and snx.b if necessary
        if (clear_neq):
            snx.N = None
            snx.b = None
      
    # Helmert comparison between two solutions
    #-----------------------------------------
    def compare(snx, ref, helmerts, weighting='full', apply_vf=True, norm_res='approx', quiet=False, out=sys.stdout):

        """
        Helmert comparison between two solutions
        
        Returns
        -------
        T : array_like
            Estimated Helmert parameters
        Q : array_like
            Covariance matrix of estimated Helmert parameters
            
        sinex.compare additionally sets new attributes to sinex instance snx:
        
        snx.v : array_like
            Residuals from Helmert comparison
        snx.sv : array_like
            Standard deviations of residuals from Helmert comparison
        snx.vn : array_like
            Normalized residuals from Helmert comparison
        snx.wrmsx : array_like
            E/N/H WRMS of station position residuals
        snx.wrmsv : array_like
            E/N/H WRMS of station velocity residuals
        
        Parameters
        ----------        
        ref : sinex instance
            Solution with which the comparison is made
        helmerts : str
            Indicates which Helmert parameters should be estimated.
            It can include 'T' (translations), 'S' (scale), 'R' (rotations).
        weighting : str, optional
            Keyword indicating which covariance matrix should be used.
            It can take the following values :
            - 'identity' to use an identity covariance matrix
            - 'diagonal' to use a diagonal covariance matrix (diag(snx.Q))
            - 'full' to use a full covariance matrix (snx.Q)
            Default is 'full'.
        apply_vf : bool, optional
            Whether to apply unit variance factor. Default is True.
        norm_res : str, optional
            Keyword indicating how normalized residuals should be computed.
            It can be either:
            - 'correct' in which case residuals are normalized by their own
               standard deviations, or
            - 'approx' in which case residuals are normalized by the
               standard deviations of the observations (i.e., snx.sig).
            Default is 'approx'.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """

        # Get indices of common parameters between both solutions
        (isnx, iref) = snx.get_common_par(ref)
        isnx2 = np.ix_(isnx, isnx)
        
        # Design matrix
        A = snx.helmert_partials(helmerts, 'STA')[isnx]
        if (len(np.intersect1d(isnx, snx.iv)) > 0):
            A = np.hstack((A, snx.helmert_partials(helmerts, 'VEL')[isnx]))
            
        if (len(np.intersect1d(isnx, snx.iper)) > 0):
            snx_perkey = list(snx.iper_dict.keys())
            ref_perkey = list(ref.iper_dict.keys())
            per_keys = [per for per in snx_perkey if per in ref_perkey] #select only common periods . key based on code 4 chr: 'A001' etc.
            A = np.hstack((A, snx.helmert_partials(helmerts, 'PERIOD', select_periods=per_keys)[isnx]))

        # Right-hand side
        y = snx.x[isnx] - ref.x[iref]
               
        # Least-squares adjustment of Helmert parameters
        if (weighting == 'identity'):
            AtP = A.T
        elif (weighting == 'diagonal'):
            P = 1 / snx.sig[isnx]**2
            AtP = A.T * P
        elif (weighting == 'full'):
            L = cholesky(snx.Q[isnx2])
            AtP = (cholsolve(L, A)).T
        N = np.dot(AtP, A)
        b = np.dot(AtP, y)
        Qt = invspd(N)
        t = np.dot(Qt, b)
                
        # Residuals
        v = y - np.dot(A, t)
        
        # Compute unit variance factor if needed
        if (apply_vf):
            if (weighting == 'identity'):
                vPv = np.sum(v**2)
            elif (weighting == 'diagonal'):
                vPv = np.sum(v**2*P)
            elif (weighting == 'full'):
                vPv = np.sum(v*cholsolve(L, v))
            sig02 = vPv / (A.shape[0]-A.shape[1])
            
        # "Full" array of residuals
        snx.v = np.zeros(snx.npar)
        snx.v[isnx] = v
                
        # Covariance matrix of observations
        Q = np.zeros((snx.npar, snx.npar))
        if (weighting == 'identity'):
            Q[isnx2] = np.eye(len(isnx))
        elif (weighting == 'diagonal'):
            Q[isnx2] = np.diag(snx.sig[isnx]**2)
        else:
            Q[isnx2] = snx.Q[isnx2]
            
        # Compute covariance matrix of residuals if needed
        if (norm_res == 'correct'):
            Qv = np.zeros((snx.npar, snx.npar))
            Qv[isnx2] = Q[isnx2] - np.dot(A, np.dot(Qt, A.T))
            
        # Scale covariance matrices with unit variance factor if needed
        if (apply_vf):
            Q = sig02 * Q
            Qt = sig02 * Qt
            if (norm_res == 'correct'):
                Qv = sig02 * Qv
                
        # Variances of observations
        s2 = np.diag(Q).copy()
        
        # Standard deviations of residuals
        if (norm_res == 'correct'):
            snx.sv = np.sqrt(np.diag(Qv))
        else:
            snx.sv = np.sqrt(np.diag(Q))
        
        # Normalized residuals
        snx.vn = np.zeros(snx.npar)
        snx.vn[isnx] = snx.v[isnx] / snx.sv[isnx]
        
        # Rotate station position residuals to ENH frames and convert them into mm
        indx = np.nonzero(snx.v[snx.ix])[0]
        ix = np.array([snx.ix[i] for i in indx])
        for i in ix:
            R = xyz2enh(snx.x[i:i+3])
            snx.v[i:i+3] = 1000 * np.dot(R, snx.v[i:i+3])
            s2[i:i+3] = np.diag(np.dot(R, np.dot(Q[i:i+3, i:i+3], R.T)))
            if (norm_res == 'correct'):
                snx.sv[i:i+3] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[i:i+3, i:i+3], R.T))))
            else:
                snx.sv[i:i+3] = 1000 * np.sqrt(s2[i:i+3])
            snx.vn[i:i+3] = snx.v[i:i+3] / snx.sv[i:i+3]

        # Compute WRMS of ENH station position residuals
        snx.wrmsx = np.zeros(3)
        for i in range(3):
            snx.wrmsx[i] = sqrt(np.sum(snx.v[ix+i]**2/s2[ix+i]) / np.sum(1/s2[ix+i]))
            
        # Rotate station velocity residuals to ENH frames and convert them into mm
        indv = np.nonzero(snx.v[snx.iv])[0]
        iv = np.array([snx.iv[i] for i in indv])
        for i in iv:
            R = xyz2enh(snx.x[i-3:i])
            snx.v[i:i+3] = 1000 * np.dot(R, snx.v[i:i+3])
            s2[i:i+3] = np.diag(np.dot(R, np.dot(Q[i:i+3, i:i+3], R.T)))
            if (norm_res == 'correct'):
                snx.sv[i:i+3] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[i:i+3, i:i+3], R.T))))
            else:
                snx.sv[i:i+3] = 1000 * np.sqrt(s2[i:i+3])
            snx.vn[i:i+3] = snx.v[i:i+3] / snx.sv[i:i+3]
        
        # Compute WRMS of ENH station velocity residuals
        if (len(iv) > 0):
            snx.wrmsv = np.zeros(3)
            for i in range(3):
                snx.wrmsv[i] = sqrt(np.sum((snx.v[iv+i]**2/s2[iv+i])) / np.sum(1/s2[iv+i]))
                
        # Rotate station PER residuals to ENH frames and convert them into mm
        dict_all_iper = {} #dict with all common period parameters -> save to avoid other loop
        all_iper = []
        for per in snx.iper_dict.keys():
            indp = np.nonzero(snx.v[snx.iper_dict[per]])[0]
            iper = np.array([snx.iper_dict[per][i] for i in indp])
            if len(iper)!=0: #this period is common btw these 2 snx
                dict_all_iper[per] = iper #add iper in all common PER dict
                all_iper += list(iper) #add to global list
                ixsta = np.array([snx.ix[i] for i in indp]) #station position corresponding to indp
                for numsta, i in enumerate(iper):
                    R = xyz2enh(snx.x[ixsta[numsta]:ixsta[numsta]+3]) #get according STA position, same for COS and SIN
                    #i order: CosX, SinX,CosY,SinY,CosZ, SinZ
                    cos_idx = [i, i+2, i+4]
                    sin_idx = [i+1, i+3, i+5]
                    
                    sub_idx = [cos_idx, sin_idx]
                    for sub in sub_idx: # 2 loop: cos_idx & sin_idx
                        snx.v[sub] = 1000 * np.dot(R, snx.v[sub])
                        s2[sub] = np.diag(np.dot(R, np.dot(Q[np.ix_(sub, sub)], R.T)))
                        if (norm_res == 'correct'):
                            snx.sv[sub] = 1000 * np.sqrt(np.diag(np.dot(R, np.dot(Qv[np.ix_(sub, sub)], R.T))))
                        else:
                            snx.sv[sub] = 1000 * np.sqrt(s2[sub])
                        snx.vn[sub] = snx.v[sub] / snx.sv[sub]
        
        nper = len(dict_all_iper.keys()) #number of periods
        all_iper = sorted(all_iper)
        
        # Compute WRMS of ENH station periodic terms residuals
        if (nper > 0): #at least 1 period element in common, btw these 2 snx.
            snx.wrmsp = {} #init WRMS as a dict, key is period
            for per in dict_all_iper.keys():
                iper = dict_all_iper[per]
                wrmsp = np.zeros(6) #cosx, sinx, cosy, siny, cosz, sinz
                for i in range(6):
                    wrmsp[i] = sqrt(np.sum((snx.v[iper+i]**2/s2[iper+i])) / np.sum(1/s2[iper+i]))
                #add to stat dict wrms
                snx.wrmsp[per] = wrmsp
                    
        # Convert geocenter residuals into mm
        igc = snx.igc + [i+1 for i in snx.igc] + [i+2 for i in snx.igc]
        snx.v[igc] = 1000*snx.v[igc]
        snx.sv[igc] = 1000*snx.sv[igc]
        
        # Indices of radiosource coordinate residuals
        indrs = np.nonzero(snx.v[snx.irs])[0]
        irs = np.array([snx.irs[i] for i in indrs])
        
        # Indices of ERP / GC / SC residuals
        ic = ix.tolist() + [i+1 for i in ix] + [i+2 for i in ix] + iv.tolist() + [i+1 for i in iv] + [i+2 for i in iv] + irs.tolist() + [i+1 for i in irs] + sum([[i,i+1,i+2,i+3,i+4,i+5] for i in all_iper],[]) #sum trick to flat list
        ig = np.setdiff1d(isnx, ic)
    
        # Reshape array of transformation parameters and their covariance matrix
        ind = []
        if ('T' in helmerts):
            ind.extend(range(0, 3))
        if ('S' in helmerts):
            ind.append(3)
        if ('R' in helmerts):
            ind.extend(range(4, 7))
        if (len(iv) > 0):
            if ('T' in helmerts):
                ind.extend(range(7, 10))
            if ('S' in helmerts):
                ind.append(10)
            if ('R' in helmerts):
                ind.extend(range(11, 14))
                
        if (nper > 0): #at least 1 period
            startid = 14 #previously 7 sta + 7 vel
            for num, per in enumerate(dict_all_iper.keys()): #loop over each period
                for cs in range(2):
                    if ('T' in helmerts):
                        ind.extend(startid + 7*(2*num+cs) + np.arange(0, 3))
                    if ('S' in helmerts):
                        ind.append(startid + 7*(2*num+cs) + 3)
                    if ('R' in helmerts):
                        ind.extend(startid + 7*(2*num+cs) + np.arange(4, 7))
                
        T = np.zeros(14 + 14*nper) #1 per: 7 cos + 7 sin
        T[ind] = t
        QT = np.zeros((T.shape[0], T.shape[0]))
        QT[np.ix_(ind, ind)] = Qt
        sT = np.sqrt(np.diag(QT))
        
        # Print output
        if not(quiet):
            print('sinex.compare', file=out)
            print('-------------', file=out)

            # Print main options and statistics
            print('', file=out)
            print('    Main statistics', file=out)
            print('    ---------------', file=out)
            print('', file=out)
            print('    # observations      : {0}'.format(len(isnx)), file=out)
            print('    (station positions  : {0})'.format(3*len(indx)), file=out)
            print('    (station velocities : {0})'.format(3*len(indv)), file=out)
            if (nper>0):
                for per in dict_all_iper.keys():
                    print('    ({0} periods : {1})'.format(per, 6*len(indv)), file=out)
                    
            print('    (radiosource coord. : {0})'.format(2*len(indrs)), file=out)
            print('    (ERP / GC / SC      : {0})'.format(len(ig)), file=out)
            print('    # parameters        : {0}'.format(A.shape[1]), file=out)
            print('    Weighting           : {0}'.format(weighting), file=out)
            print('    WRMS East           : {0:8.3f} mm'.format(snx.wrmsx[0]), file=out)
            print('    WRMS North          : {0:8.3f} mm'.format(snx.wrmsx[1]), file=out)
            print('    WRMS Up             : {0:8.3f} mm'.format(snx.wrmsx[2]), file=out)
            if (len(iv) > 0):
                print('    WRMS vel East   : {0:8.3f} mm/y'.format(snx.wrmsv[0]), file=out)
                print('    WRMS vel North  : {0:8.3f} mm/y'.format(snx.wrmsv[1]), file=out)
                print('    WRMS vel Up     : {0:8.3f} mm/y'.format(snx.wrmsv[2]), file=out)
                
            if (nper>0):
                for per in snx.wrmsp.keys():
                    print('    WRMS {1} cos E   : {0:8.3f} mm/y'.format(snx.wrmsp[per][0], per), file=out)
                    print('    WRMS {1} sin N   : {0:8.3f} mm/y'.format(snx.wrmsp[per][1], per), file=out)
                    print('    WRMS {1} cos H   : {0:8.3f} mm/y'.format(snx.wrmsp[per][2], per), file=out)
                    print('    WRMS {1} sin E   : {0:8.3f} mm/y'.format(snx.wrmsp[per][3], per), file=out)
                    print('    WRMS {1} cos N   : {0:8.3f} mm/y'.format(snx.wrmsp[per][4], per), file=out)
                    print('    WRMS {1} sin H   : {0:8.3f} mm/y'.format(snx.wrmsp[per][5], per), file=out)
            print('', file=out)

            # Print estimated parameters and formal errors
            print('    Estimated Helmert parameters', file=out)
            print('    ----------------------------', file=out)
            print('', file=out)
            if ('T' in helmerts):
                print('    TX  : {0:8.3f} +/- {1:7.3f} mm'.format(T[0], sT[0]), file=out)
                print('    TY  : {0:8.3f} +/- {1:7.3f} mm'.format(T[1], sT[1]), file=out)
                print('    TZ  : {0:8.3f} +/- {1:7.3f} mm'.format(T[2], sT[2]), file=out)
            if ('S' in helmerts):
                print('    SC  : {0:8.3f} +/- {1:7.3f} ppb'.format(T[3], sT[3]), file=out)
            if ('R' in helmerts):
                print('    RX  : {0:8.3f} +/- {1:7.3f} mas'.format(T[4], sT[4]), file=out)
                print('    RY  : {0:8.3f} +/- {1:7.3f} mas'.format(T[5], sT[5]), file=out)
                print('    RZ  : {0:8.3f} +/- {1:7.3f} mas'.format(T[6], sT[6]), file=out)
            if (len(indv) > 0):
                if ('T' in helmerts):
                    print('    dTX : {0:8.3f} +/- {1:7.3f} mm/y'.format(T[7], sT[7]), file=out)
                    print('    dTY : {0:8.3f} +/- {1:7.3f} mm/y'.format(T[8], sT[8]), file=out)
                    print('    dTZ : {0:8.3f} +/- {1:7.3f} mm/y'.format(T[9], sT[9]), file=out)
                if ('S' in helmerts):
                    print('    dSC : {0:8.3f} +/- {1:7.3f} ppb/y'.format(T[10], sT[10]), file=out)
                if ('R' in helmerts):
                    print('    dRX : {0:8.3f} +/- {1:7.3f} mas/y'.format(T[11], sT[11]), file=out)
                    print('    dRY : {0:8.3f} +/- {1:7.3f} mas/y'.format(T[12], sT[12]), file=out)
                    print('    dRZ : {0:8.3f} +/- {1:7.3f} mas/y'.format(T[13], sT[13]), file=out)
                    
            if (nper>0): #periods
                for per in dict_all_iper.keys():
                    for cs, val in enumerate(['COS', 'SIN']):
                        print('    -- {0} {1} --'.format(per, val), file=out)
                        if ('T' in helmerts):     
                            print('    TX  : {0:8.3f} +/- {1:7.3f} mm'.format(T[startid+(2*num+cs)], sT[startid+(2*num+cs)+0]), file=out)
                            print('    TY  : {0:8.3f} +/- {1:7.3f} mm'.format(T[startid+(2*num+cs)+1], sT[startid+(2*num+cs)+1]), file=out)
                            print('    TZ  : {0:8.3f} +/- {1:7.3f} mm'.format(T[startid+(2*num+cs)+2], sT[startid+(2*num+cs)+2]), file=out)
                        if ('S' in helmerts):
                            print('    SC  : {0:8.3f} +/- {1:7.3f} ppb'.format(T[startid+(2*num+cs)+3], sT[startid+(2*num+cs)+3]), file=out)
                        if ('R' in helmerts):
                            print('    RX  : {0:8.3f} +/- {1:7.3f} mas'.format(T[startid+(2*num+cs)+4], sT[startid+(2*num+cs)+4]), file=out)
                            print('    RY  : {0:8.3f} +/- {1:7.3f} mas'.format(T[startid+(2*num+cs)+5], sT[startid+(2*num+cs)+5]), file=out)
                            print('    RZ  : {0:8.3f} +/- {1:7.3f} mas'.format(T[startid+(2*num+cs)+6], sT[startid+(2*num+cs)+6]), file=out)
                        
                    
            print('', file=out)
            
            # Print station position residuals
            print('    Station position residuals', file=out)
            print('    --------------------------', file=out)
            print('', file=out)
            print('                  |     Raw residuals [mm]     |    Normalized residuals    |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('     code pt soln |     E        N        H    |     E        N       H     |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            for i in ix:
                print('     {0.code} {0.pt} {0.soln} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[i], snx.v[i:i+3], snx.vn[i:i+3]), file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('', file=out)

            # Print station velocity residuals
            if (len(iv) > 0):
                print('    Station velocity residuals', file=out)
                print('    --------------------------', file=out)
                print('', file=out)
                print('                  |    Raw residuals [mm/y]    |    Normalized residuals    |', file=out)
                print('    --------------|----------------------------|----------------------------|', file=out)
                print('     code pt soln |     E        N        H    |     E        N        H    |', file=out)
                print('    --------------|----------------------------|----------------------------|', file=out)
                for i in iv:
                    print('     {0.code} {0.pt} {0.soln} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[i], snx.v[i:i+3], snx.vn[i:i+3]), file=out)
                print('    --------------|----------------------------|----------------------------|', file=out)
                print('', file=out)
                
            
            # Print station PERIODS residuals
            if (nper > 0):
            
                print('    Station seasonal signal residuals', file=out)
                print('    ---------------------------------', file=out)
                print('', file=out)
                print('                         |    Raw residuals [mm]      |    Normalized residuals    |', file=out)
                print('    ---------------------|----------------------------|----------------------------|', file=out)
                print('     code pt soln  type  |     E        N        H    |     E        N        H    |', file=out)
                print('    ---------------------|----------------------------|----------------------------|', file=out)
                for i in all_iper:
                    per = snx.param[i].type[:-2]
                    print('     {0.code} {0.pt} {0.soln}  {3}C | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[i], snx.v[[i, i+2, i+4]], snx.vn[[i, i+2, i+4]], per), file=out) #cos
                    print('     {0.code} {0.pt} {0.soln}  {3}S | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[i], snx.v[[i+1, i+3, i+5]], snx.vn[[i+1, i+3, i+5]], per), file=out) #sin
                print('    ---------------------|----------------------------|----------------------------|', file=out)
                print('', file=out)
                
            # Print radiosource coordinate residuals
            if (len(irs) > 0):
                print('    Radiosource coordinate residuals', file=out)
                print('    --------------------------------', file=out)
                print('', file=out)
                print('                   |   Raw res. [mas]  |  Normalized res.  |', file=out)
                print('    ---------------|-------------------|-------------------|', file=out)
                print('     code IERSname |    RA       DE    |    RA       DE    |', file=out)
                print('    ---------------|-------------------|-------------------|', file=out)
                for i in irs:
                    print('     {0.code} {0.iers} | {1[0]:8.3f} {1[1]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} |'.format(snx.param[i], snx.v[i:i+2], snx.vn[i:i+2]), file=out)
                print('    ---------------|-------------------|-------------------|', file=out)
                print('', file=out)

            # ERP/GC residuals
            if (len(ig) > 0):
                print('    ERP / geocenter / scale residuals', file=out)
                print('    ---------------------------------', file=out)
                print('', file=out)
                print('                         |      Raw      |   Norm   |', file=out)
                print('    ---------------------|---------------|----------|', file=out)
                for i in ig:
                    if (snx.param[i].type[1:3] == 'GC'):
                        print('     {0.type} {0.tref} | {1:8.3f} mm   | {2:8.3f} |'.format(snx.param[i], snx.v[i], snx.vn[i]), file=out)
                    else:
                        print('     {0.type} {0.tref} | {1:8.3f} {0.unit} | {2:8.3f} |'.format(snx.param[i], snx.v[i], snx.vn[i]), file=out)
                print('    ---------------------|---------------|----------|', file=out)
                print('', file=out)

        return (T, QT)
    
    # Get list of outliers from Helmert comparison or combination
    #------------------------------------------------------------
    def get_outliers(snx, thr_raw=None, thr_norm=None, thr_abs_E=None, thr_abs_N=None, thr_abs_H=None, reject1by1=False, ac=None, quiet=False, out=sys.stdout):

        """
        Get list of outliers from Helmert comparison or combination

        Returns
        -------
        code : list
            List of 4-char station codes
        pt : list
            List of PT codes
        soln : list
            List of solns

        Parameters
        ----------
        thr_raw : float, optional
            Multiplicative factor defining thresholds for raw residuals:
            along each component, threshold = thr_raw * WRMS
            Default is None.
        thr_norm : float, optional
            Threshold for normalized residuals
        thr_abs_E, thr_abs_N, thr_abs_H : float, optional
            Absolute threshold for respectively east, north and up positional residuals  
        reject1b1 : bool, optional
            If True, then outliers will be removed one by one.
        ac : str, optional
            AC name to be reported in outliers summary file. Default is None.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
            
        """

        # Print header in log file
        if not(quiet):
            print('sinex.get_outliers', file=out)
            print('------------------', file=out)
            print('', file=out)
            
        # Indices of station positions / velocities
        ix = np.array([[i, i+1, i+2] for i in snx.ix])
        iv = np.array([[i, i+1, i+2] for i in snx.iv])

        # Indices of station position outliers
        indx = []
        if (len(ix) > 0):
            if (thr_raw):
                for i in range(3):
                    indx.extend(np.nonzero(np.abs(snx.v[ix[:,i]]) > thr_raw*snx.wrmsx[i])[0].tolist())
            if (thr_norm):
                for i in range(3):
                    indx.extend(np.nonzero(np.abs(snx.vn[ix[:,i]]) > thr_norm)[0].tolist())
            if (thr_abs_E):
                indx.extend(np.nonzero(np.abs(snx.v[ix[:,0]]) > thr_abs_E)[0].tolist())
            if (thr_abs_N):
                indx.extend(np.nonzero(np.abs(snx.v[ix[:,1]]) > thr_abs_N)[0].tolist())
            if (thr_abs_H):
                indx.extend(np.nonzero(np.abs(snx.v[ix[:,2]]) > thr_abs_H)[0].tolist())
            indx = list(set(indx))

        if len(indx)>0:
            if reject1by1 == True:
                r3D = np.sqrt(snx.vn[ix[indx,0]]**2 + snx.vn[ix[indx,1]]**2 + snx.vn[ix[indx,2]]**2)
                indx = [indx[np.nonzero(r3D == np.max(r3D))[0][0]]]
        
        # Print station position outliers
        if (len(indx) > 0) and not(quiet):
            print('    Station position outliers', file=out)
            print('    -------------------------', file=out)
            print('', file=out)
            print('                  |     Raw residuals [mm]     |    Normalized residuals    |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('     code pt soln |     E        N        H    |     E        N       H     |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            for i in indx:
                print('     {0.code} {0.pt} {0.soln} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[ix[i,0]], snx.v[ix[i]], snx.vn[ix[i]]), file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('', file=out)
            
        # Store station position outliers
        if (len(indx) > 0):
            if not(hasattr(snx, 'staout')):
                snx.staout = []
                snx.resout = []
            snx.staout.extend([snx.param[ix[i,0]].code for i in indx])
            snx.resout.extend([snx.v[ix[i]] for i in indx])
            
        # Indices of station velocity outliers
        indv = []
        if (len(iv) > 0):
            if (thr_raw):
                for i in range(3):
                    indv.extend(np.nonzero(np.abs(snx.v[iv[:,i]]) > thr_raw*snx.wrmsv[i])[0].tolist())
            if (thr_norm):
                for i in range(3):
                    indv.extend(np.nonzero(np.abs(snx.vn[iv[:,i]]) > thr_norm)[0].tolist())
            indv = list(set(indv))

        # Print station velocity outliers
        if (len(indv) > 0) and not(quiet):
            print('    Station velocity outliers', file=out)
            print('    -------------------------', file=out)
            print('', file=out)
            print('                  |    Raw residuals [mm/y]    |    Normalized residuals    |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('     code pt soln |     E        N        H    |     E        N       H     |', file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            for i in indv:
                print('     {0.code} {0.pt} {0.soln} | {1[0]:8.3f} {1[1]:8.3f} {1[2]:8.3f} | {2[0]:8.3f} {2[1]:8.3f} {2[2]:8.3f} |'.format(snx.param[iv[i,0]], snx.v[iv[i]], snx.vn[iv[i]]), file=out)
            print('    --------------|----------------------------|----------------------------|', file=out)
            print('', file=out)

        # Outlier IDs
        code = [snx.param[snx.ix[i]].code for i in indx] + [snx.param[snx.iv[i]].code for i in indv]
        pt = [snx.param[snx.ix[i]].pt for i in indx] + [snx.param[snx.iv[i]].pt for i in indv]
        soln = [snx.param[snx.ix[i]].soln for i in indx] + [snx.param[snx.iv[i]].soln for i in indv]

        return (code, pt, soln)
    
    # Iterative Helmert comparison between two solutions
    #---------------------------------------------------
    def compare_iter(snx, ref, helmerts, weighting='full', apply_vf=True, norm_res='approx', thr_raw=None, thr_norm=None,  thr_abs_E=None, thr_abs_N=None, thr_abs_H=None, reject1by1=False, clean_ref=False, ac=None, quiet=False, out=sys.stdout):

        """
        Iterative Helmert comparison between two solutions
        
        Returns
        -------
        T : array_like
            Estimated Helmert parameters
        Q : array_like
            Covariance matrix of estimated Helmert parameters
        
        sinex.compare_iter additionally sets new attributes to sinex instance snx:
        
        snx.v : array_like
            Residuals from Helmert comparison
        snx.sv : array_like
            Standard deviations of residuals from Helmert comparison
        snx.vn : array_like
            Normalized residuals from Helmert comparison
        snx.wrmsx : array_like
            E/N/H WRMS of station position residuals
        snx.wrmsv : array_like
            E/N/H WRMS of station velocity residuals

        Besides, sinex.compare_iter rejects outlying stations from either snx or ref
        (depending on parameter clean_ref).

        Parameters
        ----------        
        ref : sinex instance
            Solution with which the comparison is made
        helmerts : str
            Indicates which Helmert parameters should be estimated.
            It can include 'T' (translations), 'S' (scale), 'R' (rotations).
        weighting : str, optional
            Keyword to indicate which weighting should be used.
            It can take the following values :
            - 'identity' to use an identity weight matrix
            - 'diagonal' to use a diagonal weight matrix
            - 'full' to use a full weight matrix (inv(snx.Q))
            Default is 'full'.
        apply_vf : bool, optional
            Whether to apply unit variance factor. Default is True.
        norm_res : str, optional
            Keyword indicating how normalized residuals should be computed.
            It can be either:
            - 'correct' in which case residuals are normalized by their own
               standard deviations, or
            - 'approx' in which case residuals are normalized by the
              standard deviations of the observations (i.e., snx.sig).
            Default is 'approx'.
        thr_raw : float, optional
            Multiplicative factor defining thresholds for raw residuals:
            along each component, threshold = thr_raw * WRMS
            Default is None.
        thr_norm : float, optional
            Threshold for normalized residuals
        thr_abs_E, thr_abs_N, thr_abs_H : float, optional
            Absolute threshold for respectively east, north and up positional residuals 
        reject1b1 : bool, optional
            If True, then outliers will be removed one by one.
        clean_ref : bool, optional
            If True, then outliers will be removed from ref instead of snx.
            Default is False.
        ac : str, optional
            AC name to be reported in outliers summary file. Default is None.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """
        
        # While there remains outliers,
        end = False
        while (not(end)):
            
            # Helmert comparison
            (T, Q) = snx.compare(ref, helmerts, weighting, apply_vf, norm_res, quiet, out)
            
            # Get outlier list
            (code, pt, soln) = snx.get_outliers(thr_raw, thr_norm, thr_abs_E, thr_abs_N, thr_abs_H, reject1by1 , ac, quiet, out)
            
            # If any outliers,
            if (len(code) > 0):
                
                # Remove them either from snx or ref
                if (clean_ref):
                    ref.del_sta(code, pt, soln)
                else:
                    snx.del_sta(code, pt, soln)
            
            # Else, we're done.
            else:
                end = True
                        
        return (T, Q)

    # Propagate station positions to specified date
    #----------------------------------------------
    def propagate(snx, tsnx, keep_vel=False):
    
        """
        Propagate station positions to specified date

        Parameters
        ----------
        tsnx : str
            Date (SINEX date format)
        keep_vel : bool, optional
            Whether to keep station velocities. Default is False.
        
        """
    
        # Propagation date
        t = date.from_tsnx(tsnx)

        # Initialize design matrix A to identity
        A_rows = list(range(snx.npar))
        A_cols = list(range(snx.npar))
        A_vals = snx.npar * [1]
        
        # Complete design matrix : loop over station velocities
        for i in snx.iv:

            # Propagation interval
            ti = date.from_tsnx(snx.param[i-3].tref)
            dti = (t.mjd - ti.mjd) / 365.25
            
            # Update design matrix
            A_rows.extend([i-3, i-2, i-1])
            A_cols.extend([i, i+1, i+2])
            A_vals.extend([dti, dti, dti])

            # Update parameter epochs
            snx.param[i-3].tref = tsnx
            snx.param[i-2].tref = tsnx
            snx.param[i-1].tref = tsnx

        # Build sparse design matrix
        A = sparse.csr_matrix((A_vals, (A_rows, A_cols)))
        
        # If velocity parameters should not be kept,
        if not(keep_vel):
            
            # Get indices of other parameters
            iv = [i for i in snx.iv] + [i+1 for i in snx.iv] + [i+2 for i in snx.iv]
            ind = np.setdiff1d(range(snx.npar), iv)
                
            # And make some cleaning
            A = A[ind]
            snx.npar = len(ind)
            snx.param = [snx.param[i] for i in ind]
            snx.set_par_ind()
        
        # Propagate coordinates
        snx.x = A.dot(snx.x)

        # Propagate covariance matrix if available
        if (snx.Q is not None):
            snx.Q = A.dot((A.dot(snx.Q)).T)
            
        # Update standard deviations if covariance matrix is available or set them to 0
        if (snx.Q is not None):
            snx.sig = np.sqrt(np.diag(snx.Q))
        else:
            snx.sig = np.zeros(snx.npar)
            
        # Delete a priori information
        snx.x0 = None
        snx.sig0 = None
        snx.Nc = None
        
    # Compute post-seismic deformation of given station at given date
    #----------------------------------------------------------------
    def get_psd(snx, code, t):

        """
        Compute post-seismic deformation of given station at given date

        Returns
        -------
        dx : array_like
            ENH deformation
        sx : array_like
            Sigma of ENH deformation

        Parameters
        ----------
        code : str
            4-char station code
        t : str
            Date (SINEX date format)
        
        """
    
        # Initializations
        dx = np.zeros(3)
        sx = np.zeros(3)
        mjd = date.from_tsnx(t).mjd
        
        # Get indices of parameters describing the East component of post-seismic deformation
        ind = []
        for i in range(snx.npar):
            p = snx.param[i]
            if ((p.code == code) and (p.type[5] == 'E') and (earlier(p.tref, t))):
                ind.append(i)
            
        # Loop over model functions
        A = np.zeros(len(ind))
        for i in range(0, len(ind), 2):
            
            # Useful things
            mjd0 = date.from_tsnx(snx.param[ind[i]].tref).mjd
            dt = (mjd - mjd0) / 365.25
            amp = snx.x[ind[i]]
            tau = snx.x[ind[i+1]]
            
            # Case of an exponential
            if (snx.param[ind[i]].type[1:4] == 'EXP'):
            
                # Compute associated deformation
                da = 1. - exp(-dt / tau)
                dx[0] = dx[0] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[0] = dx[0] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt / (1 + dt / tau) / tau**2
            
        # Compute formal error of the East component of the post-seismic deformations
        if (len(ind) > 0):
            Q = snx.Q[np.ix_(ind,ind)]
            sx[0] = sqrt(np.dot(A, np.dot(Q, A.T)))

        # Get indices of parameters describing the North component of post-seismic deformation
        ind = []
        for i in range(snx.npar):
            p = snx.param[i]
            if ((p.code == code) and (p.type[5] == 'N') and (earlier(p.tref, t))):
                ind.append(i)
            
        # Loop over model functions
        A = np.zeros(len(ind))
        for i in range(0, len(ind), 2):
            
            # Useful things
            mjd0 = date.from_tsnx(snx.param[ind[i]].tref).mjd
            dt = (mjd - mjd0) / 365.25
            amp = snx.x[ind[i]]
            tau = snx.x[ind[i+1]]
            
            # Case of an exponential
            if (snx.param[ind[i]].type[1:4] == 'EXP'):
            
                # Compute associated deformation
                da = 1. - exp(-dt / tau)
                dx[1] = dx[1] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[1] = dx[1] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt / (1 + dt / tau) / tau**2
                
        # Compute formal error of the North component of the post-seismic deformations
        if (len(ind) > 0):
            Q = snx.Q[np.ix_(ind,ind)]
            sx[1] = sqrt(np.dot(A, np.dot(Q, A.T)))

        # Get indices of parameters describing the Up component of post-seismic deformation
        ind = []
        for i in range(snx.npar):
            p = snx.param[i]
            if ((p.code == code) and (p.type[5] in 'HU') and (earlier(p.tref, t))):
                ind.append(i)
            
        # Loop over model functions
        A = np.zeros(len(ind))
        for i in range(0, len(ind), 2):
            
            # Useful things
            mjd0 = date.from_tsnx(snx.param[ind[i]].tref).mjd
            dt = (mjd - mjd0) / 365.25
            amp = snx.x[ind[i]]
            tau = snx.x[ind[i+1]]
            
            # Case of an exponential
            if (snx.param[ind[i]].type[1:4] == 'EXP'):
            
                # Compute associated deformation
                da = 1. - exp(-dt / tau)
                dx[2] = dx[2] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[2] = dx[2] + amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt / (1 + dt / tau) / tau**2
            
        # Compute formal error of the Up component of the post-seismic deformations
        if (len(ind) > 0):
            Q = snx.Q[np.ix_(ind,ind)]
            sx[2] = sqrt(np.dot(A, np.dot(Q, A.T)))

        return (dx, sx)
        
    # Add or remove post-seismic deformation models to a solution
    #------------------------------------------------------------
    def add_psd(snx, psd, remove=False, update_cov=True):
        
        """
        Add or remove post-seismic deformation models to a solution

        Parameters
        ----------
        psd : sinex instance
            sinex instance containing post-seismic deformation models
        remove : bool, optional
            Whether PSD models should be removed rather than added. Default is False.
        update_cov : bool, optional
            Whether covariance matrix of PSD models should be added to covariance
            matrix of sinex instance. Default is True.
        
        """
      
        # List of stations with post-seismic deformation models
        codept = [p.code+p.pt for p in psd.param]
        
        # Loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]

            # If current station has post-seismic deformation models,
            if (p.code+p.pt in codept):
                
                # Compute ENH post-seismic deformations
                (denh, senh) = psd.get_psd(p.code, p.tref)
                
                # Compute XYZ post-seismic deformations
                R = xyz2enh(snx.x[i:i+3])
                dxyz = np.dot(R.T, denh)
                Qxyz = np.dot(R.T, np.dot(np.diag(senh**2), R))
                
                # Add or remove post-seismic deformations
                if (remove):
                    snx.x[i:i+3] = snx.x[i:i+3] - dxyz
                else:
                    snx.x[i:i+3] = snx.x[i:i+3] + dxyz
                    
                # Update covariance matrix if required
                if (update_cov):
                    if (snx.Q is not None):
                        snx.Q[i:i+3,i:i+3] = snx.Q[i:i+3,i:i+3] + Qxyz
                        snx.sig[i:i+3] = np.sqrt(np.diag(snx.Q[i:i+3,i:i+3]))
                    else:
                        snx.sig[i:i+3] = np.sqrt(snx.sig[i:i+3]**2 + np.diag(Qxyz))

    # Compute seasonal signal of given station at given date
    #-------------------------------------------------------
    def get_seas(snx, code, pt, soln, t):

        """
        Compute seasonal signal of given station at given date

        Returns
        -------
        dx : array_like
            XYZ seasonal signal
        sx : array_like
            Sigma XYZ seasonal signal

        Parameters
        ----------
        code : str
            4-char station code
        pt : str
            PT code
        soln : str
            Solution number
        t : str
            Date (SINEX date format)
        
        """
    
        # Initializations
        dx = np.zeros(3)
        s2x = np.zeros(3)
        mjd = date.from_tsnx(t).mjd
        
        # Set snx.codeptsoln if needed
        if not(hasattr(snx, 'codeptsoln')):
            snx.codeptsoln = np.array([snx.param[i].code + snx.param[i].pt + snx.param[i].soln for i in snx.iper])
        
        # Indices of seasonal parameters of specified station
        ind = np.nonzero(snx.codeptsoln == code+pt+soln)[0]
        
        # Loop over relevant parameters
        for i in ind:
            p = snx.param[snx.iper[i]]
            
            # Component
            j = 'XYZ'.index(p.type[5])
            
            # Annual harmonic
            k = int(p.type[1])
            
            # Given date - reference date
            dt = mjd - date.from_tsnx(p.tref).mjd
            
            # Add seasonal term
            if snx.is_period(p.type): #it is a periodic param
                #built period object (compatible old and new syntax A1COSX & A001CX)
                per = Period.from_snx_param(p.type)
                if (per.cs == 'COS'):
                    c = cos(2*pi*k*dt/365.25)
                elif (per.cs == 'SIN'):
                    c = sin(2*pi*k*dt/365.25)
                dx[j] = dx[j] + c*snx.x[i]
                s2x[j] = s2x[j] + (c*snx.sig[i])**2

        return (dx, np.sqrt(s2x))
        
    # Add seasonal signals to a solution
    #-----------------------------------
    def add_seas(snx, seas):
        
        """
        Add seasonal signals to a solution

        Parameters
        ----------
        seas : sinex instance
            sinex instance containing seasonal signals
        
        """
        
        # Set useful attribute if needed
        if not(hasattr(seas, 'codeptsoln')):
            seas.codeptsoln = np.array([seas.param[i].code + seas.param[i].pt + seas.param[i].soln for i in seas.iper])
        
        # Loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]
            
            # Compute seasonal signals
            (dx, sx) = seas.get_seas(p.code, p.pt, p.soln, p.tref)
            
            # Add seasonal signals
            snx.x[i:i+3] = snx.x[i:i+3] + dx
            if (snx.Q is not None):
                snx.Q[i:i+3,i:i+3] = snx.Q[i:i+3,i:i+3] + np.diag(sx**2)
                snx.sig[i:i+3] = np.sqrt(np.diag(snx.Q[i:i+3,i:i+3]))
            else:
                snx.sig[i:i+3] = np.sqrt(snx.sig[i:i+3]**2 + sx**2)
                


    # Calibrate LOD estimates wrt reference series
    #---------------------------------------------
    def calib_lod(snx, rec, ref, quiet=False, out=sys.stdout):

        """
        Calibrate LOD estimates wrt reference series

        Parameters
        ----------
        rec : erp instance
            ERP series containing historical LOD estimates
        ref : erp instance
            Reference ERP series
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        
        """

        # Print header in log file
        print('sinex.calib_lod', file=out)
        print('---------------', file=out)
        
        # If there are any historical LOD estimates,
        if (len(rec.mjd) > 0):

            # Loop over LOD parameters in sinex instance
            for i in snx.ilod:
                p = snx.param[i]

                # Initializations
                b = 0
                n = 0
                mjdref = (date.from_tsnx(p.tref)).mjd

                # Loop over the 10 previous days
                for mjd in np.arange(mjdref-10, mjdref):
                    if (mjd in rec.mjd) and (mjd in ref.mjd):
                        irec = (np.nonzero(rec.mjd == mjd))[0][0]
                        iref = (np.nonzero(ref.mjd == mjd))[0][0]
                        b = b + ref.lod[iref] - rec.lod[irec]
                        n = n+1

                # If at least one previous day was available,
                if (n > 0):

                    # Modify normal equation
                    b = b / n
                    snx.b = snx.b + b*snx.N[:,i]

                    # Print message
                    if not(quiet):
                        print('    LOD    {0.soln} {0.tref} corrected by {1:11.4e} ms (mean over {2:2d} days)'.format(p, b, n), file=out)

                # Else, just print message
                elif not(quiet):
                    print('    LOD    {0.soln} {0.tref} not corrected'.format(p), file=out)

        # Print blank line in log file
        if not(quiet):
            print('', file=out)

    # Draw station map
    #-----------------
    def map(snx, write_codes=True, title=None, output=None):

        """
        Draw station map

        Parameters
        ----------
        write_codes : bool, optional
            Whether to print 4-char station codes on map. Default is True.
        title : str, optional
            Map title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. map shown on screen).
        
        """

        code = [s.code for s in snx.sta]
        (lon, lat) = snx.get_lonlat(code)
        station_map(lon, lat, code, write_codes, title, output)

    # Draw station position residual map
    #-----------------------------------
    def map_res(snx, v, title=None, output=None):

        """
        Draw station position residual map

        Parameters
        ----------
        v : array_like
            Array of residuals
        title : str, optional
            Map title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. map shown on screen).
        
        """

        # Draw basemap
        pp.figure(figsize=(12, 9))
        ax = pp.axes(projection=ccrs.PlateCarree())
        ax.axis([-180, 180, -90, 90])
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)

        # Add title if necessary
        if (title):
            pp.title(title)

        # Get coordinates of stations with non-zero residuals
        ix = np.intersect1d(snx.ix, np.nonzero(v)[0])
        code = [p.code for p in [snx.param[i] for i in ix]]
        pt = [p.pt for p in [snx.param[i] for i in ix]]
        soln = [p.soln for p in [snx.param[i] for i in ix]]
        (lon, lat) = snx.get_lonlat(code, pt, soln)
        
        # Append legend points
        lon = lon.tolist() + [-139, -130, -130]
        lat = lat.tolist() + [-42, -52, -52]
        
        # Plot points
        ax.plot(lon, lat, '.k', markersize=4)

        # Plot horizontal residuals
        ve = v[ix].tolist() + [5, 0, 0]
        vn = v[ix+1].tolist() + [0, 0, 0]
        ax.quiver(lon, lat, ve, vn, units='dots', width=3, scale=7e-2, color='black')

        # Plot vertical residuals
        vh = v[ix+2].tolist() + [0, -5, 5]
        for i in range(len(lon)):
            if (vh[i] > 0):
                ax.plot([lon[i], lon[i]], [lat[i], lat[i]+1.5*vh[i]], linewidth=2, color='red')
            else:
                ax.plot([lon[i], lon[i]], [lat[i], lat[i]+1.5*vh[i]], linewidth=2, color='green')
        
        # Legend box
        ax.plot([-143, -97, -97, -143, -143], [-36, -36, -62, -62, -36], linewidth=3, color='black')
        
        # Legend text
        pp.text(-100, -42, '5 mm', ha='right', va='center', fontsize=12)
        pp.text(-100, -52, r'$\pm$ 5 mm', ha='right', va='center', fontsize=12)
        
        # Tight layout
        pp.tight_layout()

        # Save figure into output file...
        if (output):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
            
        # ...or show it
        else:
            pp.show()

    # Print table of parameters
    #--------------------------
    def print_table(snx):
        
        """
        Print table of parameters
        
        """
        
        # Print table header
        print('#code pt   _________parameter_type__________   _____t0_____   _valid_from_ _valid_till_   ________value________   ___sigma___   unit')
        print('#-----------------------------------------------------------------------------------------------------------------------------------')
        
        # Loop over parameters
        for i in range(snx.npar):
            p = snx.param[i]
            
            # Station position, velocity, annual or semi-annual signal?
            if (p.type[:3] in ['STA', 'VEL']) or (snx.is_period(p.type)):
                if (p.type[:3] == 'STA'):
                    t = p.type[3]+' position'
                elif (p.type[:3] == 'VEL'):
                    t = p.type[3]+' velocity'
                    
                elif snx.is_period(p.type): #work for any periodic type value A, D or P
                    t = Period.from_snx_param(p.type).verbose
    
                # elif (p.type[:5] == 'A1COS'):
                #     t = p.type[3]+' annual cosine amplitude'
                # elif (p.type[:5] == 'A1SIN'):
                #     t = p.type[3]+' annual sine amplitude'
                # elif (p.type[:5] == 'A2COS'):
                #     t = p.type[3]+' semi-annual cosine amplitude'
                # elif (p.type[:5] == 'A2SIN'):
                #     t = p.type[3]+' semi-annual sine amplitude'
                
                ista = [s.code+s.pt for s in snx.sta].index(p.code+p.pt)
                isoln = [s.soln for s in snx.sta[ista].soln].index(p.soln)
                start = snx.sta[ista].solns[isoln].start
                end = snx.sta[ista].solns[isoln].end
                
            # PSD parameter?
            elif (p.type[1:4] in ['EXP', 'LOG']):
                if (p.type == 'AEXP_E'):
                    t = 'East exponential amplitude'
                elif (p.type == 'TEXP_E'):
                    t = 'East exponential relaxation time'
                elif (p.type == 'AEXP_N'):
                    t = 'North exponential amplitude'
                elif (p.type == 'TEXP_N'):
                    t = 'North exponential relaxation time'
                elif (p.type in ['AEXP_H', 'AEXP_U']):
                    t = 'Up exponential amplitude'
                elif (p.type in ['TEXP_H', 'TEXP_U']):
                    t = 'Up exponential relaxation time'
                elif (p.type == 'ALOG_E'):
                    t = 'East logarithm amplitude'
                elif (p.type == 'TLOG_E'):
                    t = 'East logarithm relaxation time'
                elif (p.type == 'ALOG_N'):
                    t = 'North logarithm amplitude'
                elif (p.type == 'TLOG_N'):
                    t = 'North logarithm relaxation time'
                elif (p.type in ['ALOG_H', 'ALOG_U']):
                    t = 'Up logarithm amplitude'
                elif (p.type in ['TLOG_H', 'TLOG_U']):
                    t = 'Up logarithm relaxation time'
                start = p.tref
                end = '00:000:00000'
                
            # Other parameter?
            else:
                t = p.type
                start = 12*'-'
                end = 12*'-'
                
            # Print parameter
            print(' {0.code} {0.pt}   {1:<33s}   {0.tref}   {2} {3}   {4:21.14e}   {5:11.5e}   {0.unit}'.format(p, t, start, end, snx.x[i], snx.sig[i]))
            
    # Print table of (instantaneous) station positions
    #-------------------------------------------------
    def print_coord(snx):
        
        """
        Print table of (instantaneous) station positions
        
        """
        
        # Print table header
        print('#code pt   ________X[m]_________ ________Y[m]_________ ________Z[m]_________   sigma(X)[m] sigma(Y)[m] sigma(Z)[m]   _corr(X,Y)__ _corr(X,Z)__ _corr(Y,Z)__')
        print('#----------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        # Loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]
            c = cov2corr(snx.Q[i:i+3,i:i+3])
            print(' {0.code} {0.pt}   {1[0]:21.14e} {1[1]:21.14e} {1[2]:21.14e}   {2[0]:11.5e} {2[1]:11.5e} {2[2]:11.5e}   {3[0][1]:12.5e} {3[0][2]:12.5e} {3[1][2]:12.5e}'.format(p, snx.x[i:i+3], snx.sig[i:i+3], c))


    # Print table of transformation parameters, as "status3.out" in catref software
    #-------------------------------------------------
    def status3(snx, inputs, out="status3.out", quiet=False):
        """
        Summarizes for each individual solution listed in "inputs" the transformation parameters, number of common points, WRMS
        Writes defaults status3.out file.
        The equivalent of "status3" catref module.
        
        Apply this function to sinex object built after the combination process (i.e. snxcmb.combine() or snxcmb.combine_iter())

        Parameters
        ----------
        snx : sinex object
            sinex built as a result of the combination process, from other sinex files.
        inputs : list of record() objects
            Provides list of sinex solution object, particularly with ".snx" attribute of each elements.
            Solutions read and build with read_yaml() function.
        out: str
            path of output status3 file. Default "status3.out"
        quiet : bool, optional
            Whether not to print output messages. Default is False.

        Returns
        -------
        None. Print status3 content and write table at out path (default: "status3.out")

        """
        #transform param
        list_id = snx.itrans
        
        #name in snx.param. 1 object by line. We look "type" attribute.
        codes = [record.code for record in np.array(snx.param)[list_id]]
        names = ["{}[{}]".format(record.type.split(" ")[0], record.unit.split(" ")[0]) for record in np.array(snx.param)[list_id]]
        values = [snx.x[i] for i in list_id]
        sigs = [snx.sig[i] for i in list_id]
        
        #create df objetc
        df_transf = pd.DataFrame()
        df_transf["codes"] = codes
        df_transf["names"] = names
        df_transf["values"] = values
        df_transf["sigs"] = sigs
        
        df = df_transf.pivot(index='codes', columns = 'names', values='values')
        df_sigs = df_transf.pivot(index='codes', columns = 'names', values='sigs')
        
        #reorder as catref : T,S,R
        order = ["TX[mm]" ,"TY[mm]","TZ[mm]","SC[ppb]","RX[mas]","RY[mas]","RZ[mas]"]
        df = df.reindex(columns=order)
        df_sigs = df_sigs.reindex(columns=order)
        
        centers = df.index
        dims = df.columns
        
        text = "{0:<10}".format("Solutions")
        for d in dims:
            text+="{0:>10}".format(d)
            
        text += "\n------------------------------------------------------------------------------------------"
        
        for cent in centers:
            text += "\n\n{0:<10}".format(cent)
            for d in dims:
                text+="{0:10.3f}".format(df.loc[cent,d])
            
            text += "\n{0:>10}".format("")
            for d in dims:
                text+="{0:10.3f}".format(df_sigs.loc[cent,d])
                
        ### add WRMS
        text += '\n\n__________________WRMS_________________'
        text += '\n{0:9}{1:9}{2:9}{3:9}{4:9}{5:9}'.format('AC','N','E[mm]','N[mm]','H[mm]','vf')
        text += '\n-----------------------------------------------'
        for ac in inputs:
            text+= '\n{0}{1:9}{2[0]:9.3f}{2[1]:9.3f}{2[2]:9.3f}{3:9.3f}'.format(ac.name, ac.nobs, ac.wrms, ac.vf)
        
        #global varince factor
        text+= "\n\nGlobal var factor : {}".format(np.sqrt(snx.stats.vf))
        
        if not quiet:
            print(text)
        
        with open(out, 'w') as f:
            f.write(text)
