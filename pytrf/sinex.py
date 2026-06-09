"""
    Class for reading, writing and manipulating SINEX files
"""

# External imports
#-----------------
import os
import sys
import warnings
import re
import gzip
import unlzw3
from pathlib import Path
from io import StringIO
#import mkl
#mkl.set_num_threads(1)
import copy
import pickle
from math import pi, sqrt, cos, sin, acos, exp, log
import numpy as np
from scipy import sparse
import matplotlib.pyplot as pp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx

# Internal imports
#-----------------
from pytrf import date
from pytrf.math import cart2geo, xyz2enh, invspd, cholesky, cholsolve, cov2corr, plate_rotations
from pytrf.io import get_sitelog, read_sitelog, read_yaml
from pytrf.utils import record, isfloat, earlier, station_map
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
                    j += 3
                elif ((j+1 <= i) and (M[i,j+1] != 0)):
                    f.write(' {0:21.14e}\n'.format(M[i,j+1]))
                    j += 2
                else: 
                    f.write('\n')
                    j += 1
            else:
                j += 1



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
        sort_params()      : Sort parameters
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
        get_rs_ind()       : Get indices of coordinates of specified radiosources
        get_common_par()   : Get indices of common parameters between two solutions
        get_common_sta()   : Get indices of common station positions between two solutions
        get_common_vel()   : Get indices of common station velocities between two solutions
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
        fix_ind()          : Fix parameters with specified indices in a normal equation
        fix_params()       : Fix parameters of specified types in a normal equation
        setup_gc()         : Set up geocenter coordinates in a normal equation
        prior2ref()        : Set a priori parameter values to reference values
        add_mc()           : Add NNR, NNT and/or NNS constraints to normal matrix of constraints
        add_vc()           : Add absolute and/or relative velocity constraints to normal matrix of constraints
        neqinv()           : Invert normal equation
        compare()          : Helmert comparison between two solutions
        get_outliers()     : Get list of outliers from Helmert comparison or combination
        compare_iter()     : Iterative Helmert comparison between two solutions
        propagate()        : Propagate station positions to specified date
        get_psd()          : Compute post-seismic deformation of given station at given date
        add_psd()          : Add or remove post-seismic deformation models to a solution
        get_seas()         : Compute seasonal signal of given station at given date
        add_seas()         : Add seasonal signals to a solution
        calib_lod()        : Calibrate LOD estimates wrt reference series        
        map()              : Draw station map
        map_res()          : Draw station position residual map
        print_table()      : Print table of parameters
        print_coord()      : Print table of (instantaneous) station positions
        split()            : Split sinex instance into station-specific instances
        dvc_graph()        : Build graph of relative velocity constraints
        
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
        snx.gpspco = None
        snx.galpco = None
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
        snx.iseas = []
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
        snx.iR = []
        snx.iS = []
        snx.iT = []
        snx.iA = []
        snx.itrans = []
        snx.idR = []
        snx.idS = []
        snx.idT = []
        snx.idtrans = []


    # Create sinex instance from SINEX file
    #--------------------------------------
    @classmethod
    def read(self, file, dont_read=[], gps_pco_freqs=2):
      
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
        gps_pco_freqs : int (2 or 3)
            Number of frequencies to be read in 'SITE/GPS_PHASE_CENTER' block
              - 2: two frequencies (default)
              - 3: three frequencies
        """
        
        # Initialization
        snx = sinex()
        snx.file = os.path.basename(file)

        # Open input SINEX file
        if (file[-3:] == '.gz'):
            f = gzip.open(file, 'rt', encoding='latin-1')
        elif (file[-2:] == '.Z'):
            f = StringIO(unlzw3.unlzw(Path(file)).decode())
        else:
            f = open(file, encoding='latin-1')

        with f:

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
                        r.serie = '{0:<5s}'.format(line[63:68].strip())
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

            # Read SITE/GPS_PHASE_CENTER block -> snx.gpspco
            if ('SITE/GPS_PHASE_CENTER' in blocks) and not('metadata' in dont_read):
                snx.gpspco = []
                f.seek(addresses[blocks.index('SITE/GPS_PHASE_CENTER')])
                line = f.readline()
                while (line[0] != '-'):
                    if (line.strip()) and (line[0] != '*'):
                        r = record()
                        r.type = line[1:21]
                        r.serie = line[22:27]
                        r.dx = [[0]*3 for i in range(4)]
                        r.dx[0][0] = line[28:34]
                        r.dx[0][1] = line[35:41]
                        r.dx[0][2] = line[42:48]
                        r.dx[1][0] = line[49:55]
                        r.dx[1][1] = line[56:62]
                        r.dx[1][2] = line[63:69]
                        r.model = line[70:80]
                        if (gps_pco_freqs == 3):
                            line = f.readline()
                            r.dx[2][0] = line[28:34]
                            r.dx[2][1] = line[35:41]
                            r.dx[2][2] = line[42:48]
                            r.dx[3][0] = line[49:55]
                            r.dx[3][1] = line[56:62]
                            r.dx[3][2] = line[63:69]
                            
                        snx.gpspco.append(r)
                    line = f.readline()
                        
            # Read SITE/GAL_PHASE_CENTER block -> snx.galpco
            if ('SITE/GAL_PHASE_CENTER' in blocks) and not('metadata' in dont_read):
                snx.galpco = []
                f.seek(addresses[blocks.index('SITE/GAL_PHASE_CENTER')])
                line = f.readline()
                while (line[0] != '-'):
                    if (line.strip()) and (line[0] != '*'):
                        r = record()
                        r.type = line[1:21]
                        r.serie = line[22:27]
                        r.dx = [[0]*3 for i in range(6)]
                        r.dx[0][0] = line[28:34]
                        r.dx[0][1] = line[35:41]
                        r.dx[0][2] = line[42:48]
                        r.dx[1][0] = line[49:55]
                        r.dx[1][1] = line[56:62]
                        r.dx[1][2] = line[63:69]
                        r.model = line[70:80]
                        line = f.readline()
                        r.dx[2][0] = line[28:34]
                        r.dx[2][1] = line[35:41]
                        r.dx[2][2] = line[42:48]
                        r.dx[3][0] = line[49:55]
                        r.dx[3][1] = line[56:62]
                        r.dx[3][2] = line[63:69]
                        line = f.readline()
                        r.dx[4][0] = line[28:34]
                        r.dx[4][1] = line[35:41]
                        r.dx[4][2] = line[42:48]
                        r.dx[5][0] = line[49:55]
                        r.dx[5][1] = line[56:62]
                        r.dx[5][2] = line[63:69]                        
                        snx.galpco.append(r)
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
                        i += 1
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
            else:
                snx.npar = 0

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
                if (snx.npar is None) or (snx.npar == 0):
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
                Q = np.zeros((len(snx.prior), len(snx.prior)))
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
                        snx.Nc = np.zeros((len(snx.prior), len(snx.prior)))
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
                while (line[0] != '-'):
                    if (line[0] != '*'):
                        try:
                            i = int(line[1:6]) - 1
                            snx.b[i] = float(line[47:68].replace('D', 'E'))
                        except:
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
                    snx.x *= f
                    snx.sig *= f
                    if (snx.Q is not None):
                        snx.Q = (snx.Q*f).T*f
                    if (snx.x0 is not None):
                        snx.x0 *= f
                        snx.sig0 *= f
                    if (snx.N is not None):
                        snx.N = (snx.N/f).T/f
                        snx.b /= f
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
            for i in snx.ix+snx.ipsd+snx.iseas:
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
            keys = [p.code+p.pt for p in [snx.param[i] for i in snx.ix+snx.ipsd+snx.iseas]]
            i = 0
            while (i < len(snx.sta)):
                if not(snx.sta[i].code+snx.sta[i].pt in keys):
                    snx.sta.pop(i)
                else:
                    i += 1

            # Update snx.sta[*].soln
            keys = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.ix]]
            for s in snx.sta:
                i = 0
                while (i < len(s.soln)):
                    if not(s.code+s.pt+s.soln[i].soln in keys):
                        s.soln.pop(i)
                    else:
                        i += 1
                
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
                    i += 1
                    
            # Attribute IERS names to radiosource coordinate parameters
            keys = [r.code for r in snx.rs]
            for i in snx.irs:
                iers = snx.rs[keys.index(snx.param[i].code)].iers
                snx.param[i].iers = iers
                snx.param[i+1].iers = iers

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
            elif (p.type[0:5] in ['A1COS', 'A1SIN', 'A2COS', 'A2SIN']):
                keys.append('1'+p.code+p.pt+p.soln+p.type[0:2]+p.type[5]+p.type[2:5])
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
            types2 = np.array([p.type[0:5] for p in snx.param])
            
            # Station positions
            snx.ix = np.nonzero(types == 'STAX  ')[0].tolist()

            # Station velocities
            snx.iv = np.nonzero(types == 'VELX  ')[0].tolist()
            
            # PSD parameters
            snx.ipsd = np.nonzero(np.isin(types1, ['EXP', 'LOG']))[0].tolist()
            
            # Seasonal terms
            snx.iseas = np.nonzero(np.isin(types2, ['A1COS', 'A1SIN', 'A2COS', 'A2SIN']))[0].tolist()
            
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
            snx.iR = np.nonzero(np.isin(types, ['RX    ', 'RY    ', 'RZ    ']))[0].tolist()
            snx.iS = np.nonzero(types == 'SC    ')[0].tolist()
            snx.iT = np.nonzero(np.isin(types, ['TX    ', 'TY    ', 'TZ    ']))[0].tolist()
            snx.iA = np.nonzero(np.isin(types, ['AX    ', 'AY    ', 'AZ    ']))[0].tolist()
            snx.itrans = snx.iT+snx.iS+snx.iR+snx.iA

            # Transformation parameter rates
            snx.idR = np.nonzero(np.isin(types, ['dRX   ', 'dRY   ', 'dRZ   ']))[0].tolist()
            snx.idS = np.nonzero(types == 'dSC   ')[0].tolist()
            snx.idT = np.nonzero(np.isin(types, ['dTX   ', 'dTY   ', 'dTZ   ']))[0].tolist()
            snx.idtrans = snx.idT+snx.idS+snx.idR

        else:
            
            snx.ix = []
            snx.iv = []
            snx.ipsd = []
            snx.iseas = []
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
            snx.iR = []
            snx.iS = []
            snx.iT = []
            snx.iA = []
            snx.itrans = []
            snx.idR = []
            snx.idS = []
            snx.idT = []
            snx.idtrans = []

    # Write sinex instance into SINEX file
    #-------------------------------------
    def write(snx, file, dont_write=[], gps_pco_freqs=2):
      
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
        gps_pco_freqs : int (2 or 3)
            Number of frequencies to be written in 'SITE/GPS_PHASE_CENTER' block
              - 2: two frequencies (default)
              - 3: three frequencies
        
        """

        # Open output SINEX file
        with open(file, 'w') as f:

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
                

            # Write SITE/GPS_PHASE_CENTER block
            if (snx.gpspco) and not('metadata' in dont_write):
                f.write('*-------------------------------------------------------------------------------\n')
                f.write('+SITE/GPS_PHASE_CENTER\n')
                f.write('*                           _______L1_PCO_______ _______L2_PCO_______\n')
                if (gps_pco_freqs == 3):
                    f.write('*                           _______L5_PCO_______\n')
                f.write('*____ANTENNA_TYPE____ _S/N_ __UP__ NORTH_ _EAST_ __UP__ NORTH_ _EAST_ ANT_MODEL_\n')
                for a in snx.gpspco:
                    f.write(' {0.type} {0.serie} {1[0][0]} {1[0][1]} {1[0][2]} {1[1][0]} {1[1][1]} {1[1][2]} {0.model}\n'.format(a, a.dx))
                    if (gps_pco_freqs == 3):
                        f.write(' {0.type} {0.serie} {1[2][0]} {1[2][1]} {1[2][2]} {1[3][0]} {1[3][1]} {1[3][2]} {0.model}\n'.format(a, a.dx))
                f.write('-SITE/GPS_PHASE_CENTER\n')
                
            # Write SITE/GAL_PHASE_CENTER block
            if (snx.galpco) and not('metadata' in dont_write):
                f.write('*-------------------------------------------------------------------------------\n')
                f.write('+SITE/GAL_PHASE_CENTER\n')
                f.write('*                           _______L1_PCO_______ _______L5_PCO_______\n')
                f.write('*                           _______L6_PCO_______ _______L7_PCO_______\n')
                f.write('*                           _______L8_PCO_______\n')
                f.write('*____ANTENNA_TYPE____ _S/N_ __UP__ NORTH_ _EAST_ __UP__ NORTH_ _EAST_ ANT_MODEL_\n')
                for a in snx.galpco:
                    f.write(' {0.type} {0.serie} {1[0][0]} {1[0][1]} {1[0][2]} {1[1][0]} {1[1][1]} {1[1][2]} {0.model}\n'.format(a, a.dx))
                    f.write(' {0.type} {0.serie} {1[2][0]} {1[2][1]} {1[2][2]} {1[3][0]} {1[3][1]} {1[3][2]} {0.model}\n'.format(a, a.dx))
                    f.write(' {0.type} {0.serie} {1[4][0]} {1[4][1]} {1[4][2]} {1[5][0]} {1[5][1]} {1[5][2]} {0.model}\n'.format(a, a.dx))
                f.write('-SITE/GAL_PHASE_CENTER\n')

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
        pkl.gpspco = snx.gpspco
        pkl.galpco = snx.galpco
        pkl.stats = snx.stats
        pkl.param = snx.param
        pkl.x = snx.x
        pkl.sig = snx.sig
        pkl.x0 = snx.x0
        pkl.sig0 = snx.sig0
        pkl.ix = snx.ix
        pkl.iv = snx.iv
        pkl.ipsd = snx.ipsd
        pkl.iseas = snx.iseas
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
        pkl.iR = snx.iR
        pkl.iS = snx.iS
        pkl.iT = snx.iT
        pkl.iA = snx.iA
        pkl.itrans = snx.itrans
        pkl.idR = snx.idR
        pkl.idS = snx.idS
        pkl.idT = snx.idT
        pkl.idtrans = snx.idtrans

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

        # Copy lists of stations, radiosources, ground antenna PCOs and parameters
        snx2.sta = copy.deepcopy(snx.sta)
        snx2.rs = copy.deepcopy(snx.rs)
        snx2.gpspco = copy.deepcopy(snx.gpspco)
        snx2.galpco = copy.deepcopy(snx.galpco)
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
        snx2.iseas = snx.iseas.copy()
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
        snx2.iR = snx.iR.copy()
        snx2.iS = snx.iS.copy()
        snx2.iT = snx.iT.copy()
        snx2.iA = snx.iA.copy()
        snx2.itrans = snx.itrans.copy()
        snx2.idR = snx.idR.copy()
        snx2.idS = snx.idS.copy()
        snx2.idT = snx.idT.copy()
        snx2.idtrans = snx.idtrans.copy()

        return snx2
    
    # Check station PT codes and DOMES numbers
    #-----------------------------------------
    def check_staid(snx, codomes, check_pt=True, check_crd=True, warn=False, quiet=False, out=sys.stdout):
      
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
                
                # PT should be A except for stations IISC, KELY
                if (code in ['IISC', 'KELY']):
                    pt2 = ' B'
                else:
                    pt2 = ' A'

                # Patch for station S91M
                if (code == 'S91M'):
                    pt2 = pt
                
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
                        if (warn):
                            warnings.warn('Station {0} not found in DOMES number catalogue.'.format(code))
                    
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
                    if (warn):
                        warnings.warn('Station {0} not found in DOMES number catalogue.'.format(code))

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
                        if (warn):
                            warnings.warn('Station {0} not found in DOMES number catalogue.'.format(code))

                    # Else (point of the DOMES number catalogue with smallest distance to station is probably the right one),
                    else:
                        domes2 = codomes[ind[imin]].domes
                        desc2 = codomes[ind[imin]].description

            # Patch for station S91M
            if (code == 'S91M'):
                domes2 = snx.sta[i].domes

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
            Reference discontinuity list (from ioutils.read_solns)
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

        # Set some useful indices
        ista = []
        isol = []
        for (i, ix) in enumerate(snx.ix):
            p = snx.param[ix]
            ista.append(codept_sta.index(p.code+p.pt))
            isol.append([s.soln for s in snx.sta[ista[-1]].soln].index(p.soln))

        # Loop over STAX parameters
        for (i, ix) in enumerate(snx.ix):
            p = snx.param[ix]

            # Mean observation epoch of current station position
            tref = snx.sta[ista[i]].soln[isol[i]].datamean

            # If current station is found in discontinuity list
            if (p.code+p.pt in codept_soln):
                j = codept_soln.index(p.code+p.pt)

                # Look for appropriate soln
                isoln = 0
                while ((solns[j].P[isoln].end != '00:000:00000') and (earlier(solns[j].P[isoln].end, tref))):
                    isoln += 1
                soln2 = solns[j].P[isoln].soln

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
                snx.param[ix+0].soln = soln2
                snx.param[ix+1].soln = soln2
                snx.param[ix+2].soln = soln2
                
                # Modify soln in snx.sta
                snx.sta[ista[i]].soln[isol[i]].soln = soln2
                    
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
            b = False
            if earlier(s.soln[0].datastart, tstart) or (s.soln[0].datastart == '00:000:00000'):
                s.soln[0].datastart = tstart
                b = True
            if earlier(tend, s.soln[0].dataend) or (s.soln[0].dataend == '00:000:00000'):
                s.soln[0].dataend = tend
                b = True
            if (b):
                t1 = date.from_tsnx(s.soln[0].datastart).mjd
                t2 = date.from_tsnx(s.soln[0].dataend).mjd
                s.soln[0].datamean = date.from_mjd((t1+t2)/2).tsnx()
                
        # Bound start and end dates of snx itself
        if (earlier(snx.start, tstart)):
            snx.start = tstart
        if (earlier(tend, snx.end)):
            snx.end = tend
        
    # Check receivers, antennas and eccentricities against metadata SINEX
    #--------------------------------------------------------------------
    def check_metadata(snx, metasnx, match_crd=True, check_siteid=False, add_pco=False, antlist=None, flag_daz=True, quiet=False, out=sys.stdout):

        """
        Check receivers, antennas and eccentricities against metadata SINEX

        Returns
        -------
        metaerr : list
            List of metadata inconsistencies
        nolog : list
            List of stations for which no sitelog is available
        rej : list
            List of stations which have either a wrong antenna type, an eccentricity error > 1 mm
            or an antenna orientation error > 10°

        Parameters
        ----------
        metasnx : sinex instance
            sinex instance containing station metadata (e.g., from io.sitelogs2snx)
        match_crd : bool, optional
            Whether to match station coordinates, in addition to 4-char station ID, before checking metadata.
            Default is True.
        check_siteid : bool, optional
            Whether to replace information in the SITE/ID block with that from the metadata SINEX.
            Default is False.
        add_pco : bool, optional
            Whether to add SITE/GPS_PHASE_CENTER and SITE/GAL_PHASE_CENTER block information into snx.
            Default is False.
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
            
            # Initializations
            found = False
            keys = [metas.code for metas in metasnx.sta]

            # If there are stations with the same 4-char ID in metasnx.sta,
            if (s.code in keys):
                ind = np.nonzero(np.array(keys) == s.code)[0]

                # If station coordinates should be checked,
                if (match_crd):

                    # If there are stations with the same 4-char ID within less than 100 km in metasnx.sta,
                    # select the closest one.
                    d = [sqrt(np.sum((metas.X-X)**2)) / 1000 for metas in [metasnx.sta[i] for i in ind]]
                    if (np.min(d) < 100):
                        found = True
                        i = np.argmin(d)

                # Else, just select the first station with the same 4-char ID in metasnx.sta
                else:
                    found = True
                    i = 0

            # If current station was found in metasnx.sta, 
            if (found):

                # Update station description if required
                if (check_siteid):
                    s.domes = metasnx.sta[ind[i]].domes
                    s.tech = metasnx.sta[ind[i]].tech
                    s.description = metasnx.sta[ind[i]].description
                    s.lon = metasnx.sta[ind[i]].lon
                    s.lat = metasnx.sta[ind[i]].lat
                    s.h = metasnx.sta[ind[i]].h

                # Get its lists of receivers, antennas and eccentricities
                rec = metasnx.sta[ind[i]].rec
                ant = metasnx.sta[ind[i]].ant
                ecc = metasnx.sta[ind[i]].ecc

                # Get source of the metadata information
                if (hasattr(metasnx.sta[ind[i]], 'source')):
                    source = metasnx.sta[ind[i]].source
                else:
                    source = metasnx.file

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
                            daz -= 360
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
                
        # If SITE/GPS_PHASE_CENTER and SITE/GAL_PHASE_CENTER block information should be added into snx,
        if (add_pco):
            
            # Initializations
            gps_ant = []
            gal_ant = []
            gps_ant_sn = []
            gal_ant_sn = []
            snx.gpspco = []
            snx.galpco = []
            
            # List of station antenna types + serial numbers
            ant = np.unique(sum([[a.type+a.serie for a in s.ant] for s in snx.sta], []))
            
            # Search keys
            meta_gps_ant = [a.type for a in metasnx.gpspco]
            meta_gal_ant = [a.type for a in metasnx.galpco]
            meta_gps_ant_sn = [a.type+a.serie for a in metasnx.gpspco]
            meta_gal_ant_sn = [a.type+a.serie for a in metasnx.galpco]
            
            # Loop over antennas
            for a in ant:
                
                # If current specific antenna is in metasnx.gpspco,
                if (a in meta_gps_ant_sn):
                    i = meta_gps_ant_sn.index(a)
                    gps_ant.append(a[:-5])
                    gps_ant_sn.append(a)
                    snx.gpspco.append(metasnx.gpspco[i])
                    
                # Else, if antenna type mean PCO is in metasnx.gpspco, and not yet in snx.gpspco,
                elif (a[:-5]+'-----' in meta_gps_ant_sn) and not(a[:-5]+'-----' in gps_ant_sn):
                    i = meta_gps_ant_sn.index(a[:-5]+'-----')
                    gps_ant.append(a[:-5])
                    gps_ant_sn.append(a[:-5]+'-----')
                    snx.gpspco.append(metasnx.gpspco[i])
                    
                # Else, if another antenna of the same type is in metasnx.gpspco, and not yet in snx.gpspco,
                elif (a[:-5] in meta_gps_ant) and not(a[:-5] in gps_ant):
                    i = meta_gps_ant.index(a[:-5])
                    gps_ant.append(a[:-5])
                    gps_ant_sn.append(a[:-5]+metasnx.gpspco[i].serie)
                    snx.gpspco.append(metasnx.gpspco[i])
                    
                # If current specific antenna is in metasnx.galpco,
                if (a in meta_gal_ant_sn):
                    i = meta_gal_ant_sn.index(a)
                    gal_ant.append(a[:-5])
                    gal_ant_sn.append(a)
                    snx.galpco.append(metasnx.galpco[i])
                    
                # Else, if antenna type mean PCO is in metasnx.galpco, and not yet in snx.galpco,
                elif (a[:-5]+'-----' in meta_gal_ant_sn) and not(a[:-5]+'-----' in gal_ant_sn):
                    i = meta_gal_ant_sn.index(a[:-5]+'-----')
                    gal_ant.append(a[:-5])
                    gal_ant_sn.append(a[:-5]+'-----')
                    snx.galpco.append(metasnx.galpco[i])
                    
                # Else, if another antenna of the same type is in metasnx.galpco, and not yet in snx.galpco,
                elif (a[:-5] in meta_gal_ant) and not(a[:-5] in gal_ant):
                    i = meta_gal_ant.index(a[:-5])
                    gal_ant.append(a[:-5])
                    gal_ant_sn.append(a[:-5]+metasnx.galpco[i].serie)
                    snx.galpco.append(metasnx.galpco[i])
                    
                
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
            'STA'; 'VEL'; 'RS'; 'GC'; 'SC ;
            'R', 'S', 'T', 'A' and 'TRANS' for all transformation parameters.
        
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
            ind.extend(snx.iR+snx.iS+snx.iT+snx.iA)
        else:
            if ('R' in types):
                ind.extend(snx.iR)
            if ('S' in types):
                ind.extend(snx.iS)
            if ('T' in types):
                ind.extend(snx.iT)
            if ('A' in types):
                ind.extend(snx.iA)
            
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
        if (pt) and (soln):
            keys = [code[i]+pt[i]+soln[i] for i in range(len(code))]
            holes = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.ix]]
        elif (pt):
            keys = [code[i]+pt[i] for i in range(len(code))]
            holes = [p.code+p.pt for p in [snx.param[i] for i in snx.ix]]
        elif (soln):
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
        with open(file) as f:
            line = f.readline()
            while (line):
                core.append(line.strip().split())
                line = f.readline()

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
                        j += 1
                else:
                    j += 1

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
    def helmert_partials(snx, helmerts, par, units=None):

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
            It can be either 'STA' (station and radiosource positions) or 'VEL'
            (station velocities - radiosource velocities not supported yet).
        units : str, optional
            Specifies units of Helmert parameters. It can be either None (mm, ppb, mas)
            or 'm' (m).
            
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
                A[ix[:,0], 7] = -np.tan(d)*np.cos(a) / mas2rad
                A[ix[:,1], 7] =  np.sin(a)           / mas2rad
                A[ix[:,0], 8] = -np.tan(d)*np.sin(a) / mas2rad
                A[ix[:,1], 8] = -np.cos(a)           / mas2rad
                A[ix[:,0], 9] =  1                   / mas2rad
            
            # ERP partial derivatives
            A[snx.ixpo, 5]  =  1/mas2rad
            A[snx.iypo, 4]  =  1/mas2rad
            A[snx.iut, 6]   = -1/(ms2rad*dera_dt)
            A[snx.inutx, 8] =  1/mas2rad
            A[snx.inuty, 7] = -1/mas2rad
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

        # Express Helmert parameters in adequate units
        if (units is None):
            A *= np.array([1e-3/ae, 1e-3/ae, 1e-3/ae, 1e-9, mas2rad, mas2rad, mas2rad, mas2rad, mas2rad, mas2rad])
        else:
            A /= ae
        
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
            
            # 1st case: normal equation + constraints [+solution]
            if (snx.N is not None) and (snx.Nc is not None):
                if (keep_const):
                    snx.N[np.ix_(ind, ind)] += snx.Nc[np.ix_(ind, ind)]

                R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                snx.b = snx.b[indk] - np.dot(R, snx.b[ind])
                snx.Nc = snx.Nc[np.ix_(indk, indk)]
                snx.x0 = snx.x0[indk]
                snx.sig0 = snx.sig0[indk]
                
                if (snx.Q is not None):
                    snx.neqinv(clear_neq=False)
                else:
                    snx.x = snx.x[indk]
                    snx.sig = snx.sig[indk]
            
            # 2nd case: solution + constraints
            elif (snx.Q is not None) and (snx.Nc is not None):
                if (np.any(snx.Nc[np.ix_(ind, ind)])) and not(keep_const):
                    snx.unconstrain(clear_const=False)
                    snx.del_ind(ind, keep_const=False)
                    snx.neqinv()
                    return

                else:
                    snx.Q = snx.Q[np.ix_(indk, indk)]
                    snx.x = snx.x[indk]
                    snx.sig = snx.sig[indk]
                    snx.Nc = snx.Nc[np.ix_(indk, indk)]
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                    
            # 3rd case: solution + normal equation
            elif (snx.Q is not None) and (snx.N is not None):
                R = np.dot(snx.N[np.ix_(indk, ind)], invspd(snx.N[np.ix_(ind, ind)]))
                snx.N = snx.N[np.ix_(indk, indk)] - np.dot(R, snx.N[np.ix_(ind, indk)])
                snx.Q = snx.Q[np.ix_(indk, indk)]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                    
            # 4th case: solution only
            elif (snx.Q is not None):
                snx.Q = snx.Q[np.ix_(indk, indk)]
                snx.x = snx.x[indk]
                snx.sig = snx.sig[indk]
                if (snx.x0 is not None):
                    snx.x0 = snx.x0[indk]
                    snx.sig0 = snx.sig0[indk]
                
            # 5th case: no matrix at all
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
            'STA'; 'VEL'; 'RS'; 'GC'; 'SC';
            'R', 'S', 'T' and 'TRANS' for all transformation parameters.
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
        snx.N -= np.dot(P, NA.T)
        snx.b -= np.dot(P, np.dot(A.T, snx.b))
    
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
                    i += 1

                # Else, remove it.
                else:
                    s.rec.pop(i)

            # Loop over antennas
            i = 0
            while (i < len(s.ant)):
                r = s.ant[i]

                # If current antenna is relevant for specified period, keep it.
                if ((earlier(r.start, end)) or (r.start == '00:000:00000')) and ((earlier(start, r.end)) or (r.end   == '00:000:00000')):
                    i += 1

                # Else, remove it.
                else:
                    s.ant.pop(i)

            # Loop over eccentricities
            i = 0
            while (i < len(s.ecc)):
                r = s.ecc[i]

                # If current eccentricity is relevant for specified period, keep it.
                if ((earlier(r.start, end)) or (r.start == '00:000:00000')) and ((earlier(start, r.end)) or (r.end   == '00:000:00000')):
                    i += 1

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
        snx.N -= snx.Nc
        
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
            'STA'; 'VEL'; 'RS', 'GC'; 'SC';
            'R', 'S', 'T' and 'TRANS' for all transformation parameters.
        
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
        snx.npar += 3
        
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
        snx.npar += 1

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
            snx.x0 += dx0
            if (snx.N is not None):
                snx.b -= np.dot(snx.N, dx0)
        
    # Add NNR, NNT and/or NNS constraints to normal matrix of constraints
    #--------------------------------------------------------------------
    def add_mc(snx, helmerts, par, sigma=1e-5, datum=None, crf_datum=None, thr=None, proj=True, quiet=True, out=sys.stdout):
        
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
            It can be either 'STA' (station and radiosource positions) or 'VEL'
            (station velocities - radiosource velocities not supported yet).
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
        proj : bool, optional
            Just keep the default, which is True.
        quiet : bool, optional
            Whether not to print output messages. Default is True.
        out : file-like, optional
            Log file. Default is sys.stdout.            
        """
        
        # If a datum is specified,
        if (datum):
            
            # Get indices of common stations
            if (par == 'STA'):
                (isnx, iref) = snx.get_common_sta(datum)
            elif (par == 'VEL'):
                (isnx, iref) = snx.get_common_vel(datum)
            isnx = np.array(isnx)
            iref = np.array(iref)
            ix = isnx.flatten()
            ir = iref.flatten()
            
            # Modify a priori coordinates of common stations
            dx0 = np.zeros(snx.npar)
            dx0[ix] = datum.x[ir] - snx.x0[ix]
            if (np.any(dx0)):
                snx.x0 += dx0
                snx.b -= np.dot(snx.N, dx0)
            
        # Else,
        else:
            
            # Get indices of all stations
            if (par == 'STA'):
                isnx = [[i, i+1, i+2] for i in snx.ix]
            elif (par == 'VEL'):
                isnx = [[i, i+1, i+2] for i in snx.iv]
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
                snx.x0 += dx0
                snx.b -= np.dot(snx.N, dx0)

        # Else, get indices of all radiosources,
        elif (par == 'STA'):
            irs = [[i, i+1] for i in snx.irs]
            irs = np.array(irs, dtype='int').flatten()
            
        # Else, 
        elif (par == 'VEL'):
            irs = np.array([], dtype='int')
        
        # If a threshold is specified, reject candidate stations with large position uncertainties
        if (thr):
            
            # Print header in log file
            if not(quiet):
                print('sinex.add_mc', file=out)
                print('------------', file=out)
                print('', file=out)
                print('    Stations discarded for the application of minimal constraints', file=out)
                print('    -------------------------------------------------------------', file=out)
                print('', file=out)
                print('     code pt soln |   trace(N)  <  threshold  |', file=out)
                print('    --------------|---------------------------|', file=out)
            
            # Iterative rejection of candidate stations with large position uncertainties
            end = False
            while not(end):
                tr = np.array([np.sum(snx.N[i,i]) for i in isnx])
                thrn = np.median(tr)/thr**2
                ind = np.nonzero(tr < thrn)[0]
                
                # If there are no more stations with large position uncertainties, stop iterations.
                if (len(ind) == 0):
                    end = True
                    
                # Else,
                else:
                    
                    # Print rejected stations in log file
                    if not(quiet):
                        for i in ind:
                            p = snx.param[isnx[i][0]]
                            print('     {0.code} {0.pt} {0.soln} | {1:11.5e} < {2:11.5e} |'.format(p, tr[i], thrn), file=out)
                    
                    # Reject stations with large position uncertainties
                    ind = np.setdiff1d(np.arange(len(isnx)), ind)
                    isnx = isnx[ind]
                    
            # Print end of log file
            if not(quiet):
                print('    --------------|---------------------------|', file=out)
                print('', file=out)
                    
            ix = isnx.flatten()

        # If sigma of minimal constraints needs to be computed, compute it
        if (sigma == 'auto'):
            sigma = 0.01 / sqrt(np.median(snx.N[ix,ix]))
        
        # Design matrix of minimal constraints
        if (len(irs) > 0):
            ix = np.hstack((ix, irs))
            A = snx.helmert_partials('RSTA', par, units='m')[ix]
        else:
            A = snx.helmert_partials('RST', par, units='m')[ix]

        # Indices of relevant columns of A
        ind = []
        if ('T' in helmerts):
            ind.extend(range(0, 3))
        if ('S' in helmerts):
            ind.append(3)
        if ('R' in helmerts):
            ind.extend(range(4, 7))
        if ('A' in helmerts) and (len(irs) > 0):
            ind.extend(range(7, 10))

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
        snx.Nc[ix2] += np.dot(B.T, B) / sigma**2
        
        # Change constraint codes of constrained parameters
        for i in ix:
            if (snx.param[i].const == '2'):
                snx.param[i].const = '1'
                
        return A.shape[1]

    ## Add equality constraints between successive velocities to normal matrix of constraints
    ##---------------------------------------------------------------------------------------
    #def add_dvc(snx, solns, sigma=1e-6):
        
        #"""
        #Add equality constraints between successive velocities to normal matrix of constraints
        
        #Returns
        #-------
        #nc : int
            #Number of constraints added

        #Parameters
        #----------
        #solns : list
            #Reference discontinuity list (from ioutils.read_solns)
        #sigma : float, optional
            #Sigma of velocity equality constraints in m/y. Default is 1e-6.
            
        #"""
        
        ## Initializations
        #nc = 0
        #keys = [s.code+s.pt for s in solns]
        #keys_v = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iv]]
        
        ## Loop over stations
        #for sta in snx.sta:

            ## Index of current station in discontinuity list
            #if (sta.code+sta.pt in keys):
                #isoln = keys.index(sta.code+sta.pt)

                ## Loop over solns
                #for i in range(len(sta.soln)-1):
                    
                    ## Get end date of current soln
                    #ip = [p.soln for p in solns[isoln].P].index(sta.soln[i].soln)
                    #end = solns[isoln].P[ip].end
                    
                    ## If current soln should be constrained with the next one,
                    #if not(end in [v.end for v in solns[isoln].V]):
                        
                        ## Get indices of both velocities
                        #i1 = keys_v.index(sta.code+sta.pt+sta.soln[i].soln)
                        #i2 = keys_v.index(sta.code+sta.pt+sta.soln[i+1].soln)
                        
                        ## Add constraints between them
                        #for k in range(3):
                            #snx.Nc[snx.iv[i1]+k,snx.iv[i1]+k] += 1 / sigma**2
                            #snx.Nc[snx.iv[i1]+k,snx.iv[i2]+k] -= 1 / sigma**2
                            #snx.Nc[snx.iv[i2]+k,snx.iv[i1]+k] -= 1 / sigma**2
                            #snx.Nc[snx.iv[i2]+k,snx.iv[i2]+k] += 1 / sigma**2
                        #nc += 3
                        
        #return nc

    # Add absolute and/or relative velocity constraints to normal matrix of constraints
    #----------------------------------------------------------------------------------
    def add_vc(snx, solns=None, sigma=1e-6, vconst=None, G=None):

        """
        Add absolute and/or relative velocity constraints to normal matrix of constraints

        Returns
        -------
        nc : Number of constraints added

        Parameters
        ----------
        solns : list, optional
            Discontinuity list (from io.read_solns). Default is None.
        sigma : float, optional
            Sigma of velocity equality constraints between successive solns of individual stations [m/y].
            Default is 1e-6.
        vconst : str or list, optional
            [YAML file containing] velocity constraints to be applied. Default is None.
        G : networkx Graph instance, optional
            Graph of relative velocity constraints constructed by sinex.dvc_graph(). May be provided here
            to save time if the graph was computed beforehand.

        """

        # Initializations
        nc = 0
        keys = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iv]]

        # Read custom velocity constraints if necessary
        if (vconst):
            if not(isinstance(vconst, list)):
                vconst = read_yaml(vconst)

        # 1 - Absolute velocity constraints
        #----------------------------------

        # Loop over absolute velocity constraints, if any
        if (vconst):
            for vc in vconst:
                if hasattr(vc, 'point'):

                    # If specified point actually has an estimated velocity,
                    tab = vc.point.split()
                    sta = tab[0] + '{0:>2s}'.format(tab[1]) + '{0:4d}'.format(int(tab[2]))
                    if (sta in keys):

                        # Apply constraint
                        i = keys.index(sta)
                        for k in range(3):
                            snx.Nc[snx.iv[i]+k,snx.iv[i]+k] += 1 / vc.sigma**2
                            snx.param[snx.iv[i]+k].const = '0'
                        nc += 3

        # 2 - Relative velocity constraints
        #----------------------------------

        # Compute graph of relative velocity constraints if needed
        if not(G):
            G = snx.dvc_graph(solns, sigma, vconst)

        # Loop over edges of the graph
        for e in G.edges:

            # Get indices of both points
            i1 = keys.index(e[0])
            i2 = keys.index(e[1])

            # Get weight of the constraint
            w = G.get_edge_data(*e)['weight']

            # Apply constraint
            for k in range(3):
                snx.Nc[snx.iv[i1]+k,snx.iv[i1]+k] += w
                snx.Nc[snx.iv[i1]+k,snx.iv[i2]+k] -= w
                snx.Nc[snx.iv[i2]+k,snx.iv[i1]+k] -= w
                snx.Nc[snx.iv[i2]+k,snx.iv[i2]+k] += w
            nc += 3

        return nc
        
    # Invert normal equation
    #-----------------------
    def neqinv(snx, clear_neq=True, return_xNx=False):

        """
        Solve normal equation

        Parameters
        ----------        
        clear_neq : bool, optional
            Whether to clear normal equation. Default is True.
        return_xNx : bool, optional
            Whether to return (x-x0)^T * N * (x-x0)

        """
        
        # Solve normal equation
        snx.Q = invspd(snx.N + snx.Nc)
        dx = np.dot(snx.Q, snx.b)
        snx.x = snx.x0 + dx
        snx.sig = np.sqrt(np.diag(snx.Q))
        xNx = np.dot(snx.b.T, dx)

        # Clear snx.N and snx.b if necessary
        if (clear_neq):
            snx.N = None
            snx.b = None

        # Return xNx if requested
        if (return_xNx):
            return xNx
      
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
            It can include 'T' (translations), 'S' (scale), 'R' (rotations) and 'A' (CRF rotations).
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

        # If both solutions are identical,
        if np.array_equal(snx.x[isnx], ref.x[iref]):

            # Print message
            if not(quiet):
                print('sinex.compare', file=out)
                print('-------------', file=out)
                print('', file=out)
                print('Both solutions are identical!', file=out)

            return (None, None)

        # Else,
        else:

            # Design matrix
            A = snx.helmert_partials(helmerts, 'STA')[isnx]
            if (len(np.intersect1d(isnx, snx.iv)) > 0):
                A = np.hstack((A, snx.helmert_partials(helmerts, 'VEL')[isnx]))

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
                Q *= sig02
                Qt *= sig02
                if (norm_res == 'correct'):
                    Qv *= sig02

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

            # Convert geocenter residuals into mm
            igc = snx.igc + [i+1 for i in snx.igc] + [i+2 for i in snx.igc]
            snx.v[igc] *= 1000
            snx.sv[igc] *= 1000

            # Indices of radiosource coordinate residuals
            indrs = np.nonzero(snx.v[snx.irs])[0]
            irs = np.array([snx.irs[i] for i in indrs])

            # Indices of ERP / GC / SC residuals
            ic = ix.tolist() + [i+1 for i in ix] + [i+2 for i in ix] + iv.tolist() + [i+1 for i in iv] + [i+2 for i in iv] + irs.tolist() + [i+1 for i in irs]
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
            if ('A' in helmerts):
                ind.extend(range(14, 17))

            T = np.zeros(17)
            T[ind] = t
            QT = np.zeros((17, 17))
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
                if ('A' in helmerts):
                    print('    AX  : {0:8.3f} +/- {1:7.3f} mas'.format(T[14], sT[14]), file=out)
                    print('    AY  : {0:8.3f} +/- {1:7.3f} mas'.format(T[15], sT[15]), file=out)
                    print('    AZ  : {0:8.3f} +/- {1:7.3f} mas'.format(T[16], sT[16]), file=out)
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
                dx[0] += amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[0] += amp * da
                
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
                dx[1] += amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[1] += amp * da
                
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
                dx[2] += amp * da
                
                # Compute partial derivatives of the model parameters
                A[i] = da
                A[i+1] = -amp * dt * (1-da) / tau**2
            
            # Case of a logarithm
            elif (snx.param[ind[i]].type[1:4] == 'LOG'):
            
                # Compute associated deformation
                da = log(1 + dt / tau)
                dx[2] += amp * da
                
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
                    snx.x[i:i+3] -= dxyz
                else:
                    snx.x[i:i+3] += dxyz
                    
                # Update covariance matrix if required
                if (update_cov):
                    if (snx.Q is not None):
                        snx.Q[i:i+3,i:i+3] += Qxyz
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
            snx.codeptsoln = np.array([snx.param[i].code + snx.param[i].pt + snx.param[i].soln for i in snx.iseas])
        
        # Indices of seasonal parameters of specified station
        ind = np.nonzero(snx.codeptsoln == code+pt+soln)[0]
        
        # Loop over relevant parameters
        for i in ind:
            p = snx.param[snx.iseas[i]]
            
            # Component
            j = 'XYZ'.index(p.type[5])
            
            # Annual harmonic
            k = int(p.type[1])
            
            # Given date - reference date
            dt = mjd - date.from_tsnx(p.tref).mjd
            
            # Add seasonal term
            if (p.type[2:5] == 'COS'):
                c = cos(2*pi*k*dt/365.25)
            elif (p.type[2:5] == 'SIN'):
                c = sin(2*pi*k*dt/365.25)
            dx[j] += c*snx.x[i]
            s2x[j] += (c*snx.sig[i])**2

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
            seas.codeptsoln = np.array([seas.param[i].code + seas.param[i].pt + seas.param[i].soln for i in seas.iseas])
        
        # Loop over STAX parameters
        for i in snx.ix:
            p = snx.param[i]
            
            # Compute seasonal signals
            (dx, sx) = seas.get_seas(p.code, p.pt, p.soln, p.tref)
            
            # Add seasonal signals
            snx.x[i:i+3] += dx
            if (snx.Q is not None):
                snx.Q[i:i+3,i:i+3] += np.diag(sx**2)
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
                        b += ref.lod[iref] - rec.lod[irec]
                        n += 1

                # If at least one previous day was available,
                if (n > 0):

                    # Modify normal equation
                    b /= n
                    snx.b += b*snx.N[:,i]

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

    # Draw station position or velocity residual map
    #-----------------------------------------------
    def map_res(snx, v, unit='mm', scale=14, legend=5, title=None, output=None):

        """
        Draw station position or velocity residual map

        Parameters
        ----------
        v : array_like
            Array of residuals
        unit : str, optional
            Unit to be printed in the legend. Default is 'mm'.
        scale : float, optional
            Controls the length of the arrows in the map. Default is 14.
        legend : float, optional
            Value of the residuals to be drawn in the legend. Default is 5.
        title : str, optional
            Map title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. map shown on screen).
        
        """

        # Draw basemap
        pp.figure(figsize=(12, 6.5))
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
        ve = v[ix].tolist() + [legend, 0, 0]
        vn = v[ix+1].tolist() + [0, 0, 0]
        ax.quiver(lon, lat, ve, vn, units='dots', width=3, scale=1/scale, color='black')

        # Plot vertical residuals
        vh = v[ix+2].tolist() + [0, -legend, legend]
        for i in range(len(lon)):
            if (vh[i] > 0):
                ax.plot([lon[i], lon[i]], [lat[i], lat[i]+1.5*scale/14*vh[i]], linewidth=2, color='red')
            else:
                ax.plot([lon[i], lon[i]], [lat[i], lat[i]+1.5*scale/14*vh[i]], linewidth=2, color='green')
        
        # Legend box
        ax.plot([-143, -86, -86, -143, -143], [-36, -36, -62, -62, -36], linewidth=3, color='black')
        
        # Legend text
        pp.text(-89, -42, str(legend)+' '+unit, ha='right', va='center', fontsize=12)
        pp.text(-89, -52, r'$\pm$ '+str(legend)+' '+unit, ha='right', va='center', fontsize=12)
        
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
            if (p.type[:3] in ['STA', 'VEL']) or (p.type[:5] in ['A1COS', 'A1SIN', 'A2COS', 'A2SIN']):
                if (p.type[:3] == 'STA'):
                    t = p.type[3]+' position'
                elif (p.type[:3] == 'VEL'):
                    t = p.type[3]+' velocity'
                elif (p.type[:5] == 'A1COS'):
                    t = p.type[3]+' annual cosine amplitude'
                elif (p.type[:5] == 'A1SIN'):
                    t = p.type[3]+' annual sine amplitude'
                elif (p.type[:5] == 'A2COS'):
                    t = p.type[3]+' semi-annual cosine amplitude'
                elif (p.type[:5] == 'A2SIN'):
                    t = p.type[3]+' semi-annual sine amplitude'
                ista = [s.code+s.pt for s in snx.sta].index(p.code+p.pt)
                isoln = [s.soln for s in snx.sta[ista].solns].index(p.soln)
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
    
    # Split sinex instance into station-specific instances
    #-----------------------------------------------------
    def split(snx, solns, dir):

        """
        Split sinex instance into station-specific instances

        Parameters
        ----------
        solns : list
            Discontinuity list (from io.read_solns)
        dir : str
            Directory where to dump station-specific sinex instances
        """
        
        # Loop over stations
        for s in snx.sta:
        
            # Create station-specific sinex object
            stasnx = sinex()
            stasnx.file = snx.file
            stasnx.version = '2.02'
            stasnx.agency = 'IGN'
            stasnx.start = snx.start
            stasnx.end = snx.end
            stasnx.tech = 'P'
            stasnx.const = '2'
            stasnx.content = 'X V'
            stasnx.sta = [s]
            
            ind = np.nonzero(np.array([p.code for p in snx.param]) == s.code)[0]      
            stasnx.npar = len(ind)
            stasnx.param = [snx.param[i] for i in ind]
            stasnx.x = snx.x[ind]
            stasnx.sig = snx.sig[ind]
            stasnx.Q = snx.Q[np.ix_(ind, ind)]
            
            # Add soln information
            if (s.code in [ss.code for ss in solns]):
                ind = [ss.code for ss in solns].index(s.code)
                stasnx.sta[0].solns = solns[ind].P
            else:
                r = record()
                r.soln = '   1'
                r.start = '00:000:00000'
                r.end = '00:000:00000'
                stasnx.sta[0].solns = [r]
            
            # Dump sinex object
            stasnx.set_par_ind()
            stasnx.dump(dir+'/'+s.code+'.snx')

    # Build graph of relative velocity constraints
    #---------------------------------------------
    def dvc_graph(snx, solns=None, sigma=1e-6, vconst=None):

        """
        Build graph of relative velocity constraints from a reference discontinuity list
        and/or a YAML file

        Returns
        -------
        G : networkx Graph instance

        Parameters
        ----------
        solns : list, optional
            Discontinuity list (from io.read_solns). Default is None.
        sigma : float, optional
            Sigma of velocity equality constraints between successive solns of individual stations [m/y].
            Default is 1e-6.
        vconst : str or list, optional
            [YAML file containing] velocity constraints to be applied. Default is None.

        """

        # Initializations
        nodes = [p.code+p.pt+p.soln for p in [snx.param[i] for i in snx.iv]]
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # If a discontinuity list is provided
        if (solns):
            keys = [s.code+s.pt for s in solns]

            # Loop over stations
            for sta in snx.sta:

                # Index of current station in discontinuity list
                if (sta.code+sta.pt in keys):
                    isoln = keys.index(sta.code+sta.pt)

                    # Dates of velocity discontinuities
                    tdv = [date.from_tsnx(v.end).mjd for v in solns[isoln].V[:-1]]

                    # Loop over solns
                    for i in range(len(sta.soln)-1):

                        # Get end date of current soln and start date of next soln
                        ip = [p.soln for p in solns[isoln].P].index(sta.soln[i].soln)
                        t1 = date.from_tsnx(solns[isoln].P[ip].end).mjd
                        ip = [p.soln for p in solns[isoln].P].index(sta.soln[i+1].soln)
                        t2 = date.from_tsnx(solns[isoln].P[ip].start).mjd

                        # Check whether there is a velocity discontinuity between both dates,
                        b = False
                        for t in tdv:
                            if (t1 <= t) and (t2 >= t):
                                b = True

                        # If not, add an edge to the graph
                        if not(b):
                            G.add_edge(sta.code+sta.pt+sta.soln[i].soln, sta.code+sta.pt+sta.soln[i+1].soln, weight=1/sigma**2)

        # If additional velocity constraints are provided,
        if (vconst):

            # Read them if necessary
            if not(isinstance(vconst, list)):
                vconst = read_yaml(vconst)

            # Loop over clusters of relative velocity constraints
            for vc in vconst:
                if hasattr(vc, 'points'):

                    # Get points IDs
                    sta = []
                    for i in range(len(vc.points)):
                        tab = vc.points[i].split()
                        sta.append(tab[0] + '{0:>2s}'.format(tab[1]) + '{0:4d}'.format(int(tab[2])))

                    # Remove points that are not part of the solution
                    i = 0
                    while (i < len(sta)):
                        if (sta[i] in nodes):
                            i += 1
                        else:
                            sta.pop(i)

                    # Loop over points of the clusters
                    for i in range(len(sta)-1):

                        # If an edge from current point to next point is already present in the graph,
                        if G.has_edge(sta[i], sta[i+1]):

                            # Overwrite edge weight
                            G[sta[i]][sta[i+1]]['weight'] = 1/vc.sigma**2

                        # Else, add an edge to the graph
                        else:
                            G.add_edge(sta[i], sta[i+1], weight=1/vc.sigma**2)

        return G

    # Estimate tectonic plate rotation vectors from station velocities
    #-----------------------------------------------------------------
    def plate_rotations(snx, plate_geom, weighting='full', set_dT=True, quiet=False, out=sys.stdout):

        """
        Estimate tectonic plate rotation vectors from station velocities

        Returns
        -------
        stats : record
            stats.nplates : Number of plates with an estimated rotation vector
            stats.nsta    : Overall number of stations used in the estimation
            stats.vf      : Overall variance factor
            stats.wrms    : WRMS of [East, North] velocity residuals (mm/yr)
        plates : list of records
            plate[i].name      : Plate name
            plate[i].nsta      : Number of stations on the plate
            plate[i].wrms      : WRMS of [East, North] velocity residuals (mm/yr)
            plate[i].omega     : XYZ components of estimated plate rotation vector (mas/yr)
            plate[i].var_omega : Their covariance matrix (mas²/yr²)
            plate[i].pole      : Longitude, latitude and rotation speed (deg, deg, mas/yr)
            plate[i].var_pole  : Corresponding covariance matrix
        dT : (3,) array_like or None
            XYZ components of estimated translation rate (mm/yr)
        var_dT : (3,) array_like or None
            Their covariance matrix (mm²/yr²)
        x : (3*p,) or (3*p+3,) array_like
            Vector of all estimated parameters (mas/yr and mm/yr)
            (plate rotation vectors possibly followed by translation rate)
        Qx : (3*p,3*p) or (3*p+3,3*p+3) array_like
            Covariance matrix of estimated parameters (mas²/yr² and mm²/yr²)
        code : list
            4-char IDs of stations effectively used in the estimation
        v : (n,2)
            Station velocity residuals (along the East and North directions, in mm/yr)
        vn : (n,2)
            Station velocity normalized residuals

        Parameters
        ----------
        plate_geom : str
            Path to a GeoJSON file containing plate boundaries. Should be of type "FeatureCollection",
            with each feature having a "PlateName" and a geometry of type "Polygon".
            You can use for instance this file, which contains the plate boundaries from Bird (2003):
            https://github.com/fraxen/tectonicplates/blob/master/GeoJSON/PB2002_plates.json
        weighting : str, optional
            Keyword indicating which covariance matrix should be used for station velocities in the estimation.
            It can take the following values :
            - 'identity' to use an identity covariance matrix
            - 'diagonal' to use a diagonal covariance matrix (diag(snx.Q))
            - 'full' to use a full covariance matrix (snx.Q)
            Default is 'full'.
        set_dT : bool
            Whether or not to estimate a global translation rate together with the plate rotation vectors.
            Default is True.
        quiet : bool, optional
            Whether not to print output messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.
        """

        # Number, IDs and coordinates of stations with velocity estimates
        n = len(snx.iv)
        code = [snx.param[i].code for i in snx.iv]
        X = snx.get_xyz(code)

        # Compute East/North velocities and their covariance matrix or vector
        R = xyz2enh(X)
        V = np.zeros((n, 2))
        for i in range(n):
            V[i] = np.dot(R[i,:2], snx.x[snx.iv[i]:snx.iv[i]+3]) * 1000

        if (weighting == 'identity'):
            Q = None
        elif (weighting == 'diagonal'):
            Q = np.zeros(2*n)
            for i in range(n):
                Q[2*i:2*i+2] = np.diag(np.dot(R[i,:2], np.dot(snx.Q[snx.iv[i]:snx.iv[i]+3,snx.iv[i]:snx.iv[i]+3], R[i,:2].T))) * 1e6
        else:
            Q = np.zeros((2*n, 2*n))
            for i in range(n):
                for j in range(n):
                    Q[2*i:2*i+2,2*j:2*j+2] = np.dot(R[i,:2], np.dot(snx.Q[snx.iv[i]:snx.iv[i]+3,snx.iv[j]:snx.iv[j]+3], R[j,:2].T)) * 1e6

        # Call math.plate_rotations
        return plate_rotations(code, X, V, plate_geom, Q=Q, set_dT=set_dT, quiet=quiet, out=out)
