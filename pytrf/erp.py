"""
    Class for reading, writing and manipulating ERP files
"""

# External imports
#-----------------
import os
import numpy as np

# Internal imports
#-----------------
from pytrf import date
from pytrf.utils import temp_file, leapsec



# erp class
#----------
class erp:
  
    """
    Class for reading, writing, manipulating ERP files

    An erp instance can be initialized in one of the following ways:
    
        r = erp.read_igs(file)
        r = erp.read_bua(file)
        r = erp.from_sinex(snx)
        
    Once initialized, each erp instance has the following attributes:

        mjd   : array of MJDs
        xpo   : array of X-pole coordinates
        xpor  : array of X-pole rates
        ypo   : array of Y-pole coordinates
        ypor  : array of Y-pole rates
        ut1   : array of UT1-UTC offsets
        lod   : array of LODs
        dX    : array of dX nutation offsets
        dXr   : array of dX nutation rates
        dY    : array of dY nutation offsets
        dYr   : array of dY nutation rates
        sxpo  : array of X-pole coordinate sigmas
        sxpor : array of X-pole rate sigmas
        sypo  : array of Y-pole coordinate sigmas
        sypor : array of Y-pole rate sigmas
        sut1  : array of UT1-UTC offset sigmas
        slod  : array of LOD sigmas
        sdX   : array of dX nutation offset sigmas
        sdXr  : array of dX nutation rate sigmas
        sdY   : array of dY nutation offset sigmas
        sdYr  : array of dY nutation rate sigmas
        
    Each erp instance has the following methods:

        append()    : Append record from other erp instance
        trim()      : Delete records that do not belong to period of interest
        interp()    : Linear interpolation at specified dates
        write_igs() : Write erp instance into IGS ERP file
        
    """

    # Initialize an erp instance with default attributes
    #---------------------------------------------------
    def __init__(r):
        
        """
        Initialize an erp instance with default attributes

        Returns
        -------
        r : erp instance
        
        """

        r.mjd   = None
        r.xpo   = None
        r.xpor  = None
        r.ypo   = None
        r.ypor  = None
        r.ut1   = None
        r.lod   = None
        r.dX    = None
        r.dXr   = None
        r.dY    = None
        r.dYr   = None
        r.sxpo  = None
        r.sxpor = None
        r.sypo  = None
        r.sypor = None
        r.sut1  = None
        r.slod  = None
        r.sdX   = None
        r.sdXr  = None
        r.sdY   = None
        r.sdYr  = None

    # Read IGS ERP file
    #------------------
    @classmethod
    def read_igs(self, file):

        """
        Read IGS ERP file

        Returns
        -------
        r : erp instance
        
        Parameters
        ----------
        file : str
            IGS ERP file to read

        """

        # Initialization
        r = erp()
        
        # Write temporary file containing valid lines
        tmp = temp_file()
        os.system('grep -P "[0-9]{{5}}\\.[0-9]{{2}}" {0} > {1}'.format(file, tmp))
        
        # Read temporary file
        (r.mjd, r.xpo, r.ypo, r.ut1, r.lod, r.sxpo, r.sypo, r.sut1, r.slod, r.xpor, r.ypor, r.sxpor, r.sypor) = np.loadtxt(tmp, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15), unpack=True)

        # Convert EOPs into mas and ms
        r.xpo   *= 1e-3
        r.ypo   *= 1e-3
        r.ut1   *= 1e-4
        r.xpor  *= 1e-3
        r.ypor  *= 1e-3
        r.lod   *= 1e-4
        r.sxpo  *= 1e-3
        r.sypo  *= 1e-3
        r.sut1  *= 1e-4
        r.sxpor *= 1e-3
        r.sypor *= 1e-3
        r.slod  *= 1e-4
        
        # Remove temporary file
        os.system('rm {0}'.format(tmp))

        return r

    # Read Bulletin A file
    #---------------------
    @classmethod
    def read_bua(self, file):

        """
        Read Bulletin A file

        Returns
        -------
        r : erp instance
        
        Parameters
        ----------
        file : str
            Bulletin A file to read

        """

        # Initialization
        r = erp()
        
        # Read input ERP file
        lines = open(file).readlines()
        
        # Keep only lines with nutation offsets
        ind = np.nonzero([len(line.strip()) >= 134  for line in lines])[0]
        lines = [lines[i] for i in ind]
        
        # Keep only lines with LOD values
        ind = np.nonzero([line[78:93] != 15*' ' for line in lines])[0]
        lines = [lines[i] for i in ind]
        
        # Fill fields of erp object
        r.mjd  = np.array([float(line[7:15])    for line in lines])
        r.xpo  = np.array([float(line[18:27])   for line in lines]) * 1e3
        r.sxpo = np.array([float(line[27:36])   for line in lines]) * 1e3
        r.ypo  = np.array([float(line[37:46])   for line in lines]) * 1e3
        r.sypo = np.array([float(line[46:55])   for line in lines]) * 1e3
        r.ut1  = np.array([float(line[58:68])   for line in lines]) * 1e3
        r.sut1 = np.array([float(line[68:78])   for line in lines]) * 1e3
        r.lod  = np.array([float(line[79:86])   for line in lines])
        r.slod = np.array([float(line[86:93])   for line in lines])
        r.dX   = np.array([float(line[97:106])  for line in lines])
        r.sdX  = np.array([float(line[106:115]) for line in lines])
        r.dY   = np.array([float(line[116:125]) for line in lines])
        r.sdY  = np.array([float(line[125:134]) for line in lines])

        return r

    # Create erp instance from sinex instance
    #----------------------------------------
    @classmethod
    def from_sinex(self, snx):

        """
        Create erp instance from sinex instance

        Returns
        -------
        r : erp instance
        
        Parameters
        ----------
        snx : sinex instance

        """

        r = erp()
        r.mjd = np.array([date.from_tsnx(p.tref).mjd for p in [snx.param[i] for i in snx.ixpo]])
        r.xpo = snx.x[snx.ixpo]
        r.sxpo = snx.sig[snx.ixpo]
        r.ypo = snx.x[snx.iypo]
        r.sypo = snx.sig[snx.iypo]
        r.xpor = snx.x[snx.ixpor]
        r.sxpor = snx.sig[snx.ixpor]
        r.ypor = snx.x[snx.iypor]
        r.sypor = snx.sig[snx.iypor]
        r.ut = snx.x[snx.iut]
        r.sut = snx.sig[snx.iut]
        r.lod = snx.x[snx.ilod]
        r.slod = snx.sig[snx.ilod]

        return r

    # Append records from other ERP instance
    #---------------------------------------
    def append(r, s):
        
        """
        Append records from other ERP instance

        Parameters
        ----------
        s : erp instance
            The one to append
        
        """

        # If r is empty, return s
        if (r.mjd is None):
            for key in vars(s):
                setattr(r, key, getattr(s, key))
            
        # Else,
        else:
            r.mjd = np.hstack((r.mjd, s.mjd))
            if (r.xpo is not None):
                r.xpo = np.hstack((r.xpo, s.xpo))
            if (r.xpor is not None):
                r.xpor = np.hstack((r.xpor, s.xpor))
            if (r.ypo is not None):
                r.ypo = np.hstack((r.ypo, s.ypo))
            if (r.ypor is not None):
                r.ypor = np.hstack((r.ypor, s.ypor))
            if (r.ut is not None):
                r.ut = np.hstack((r.ut, s.ut))
            if (r.lod is not None):
                r.lod = np.hstack((r.lod, s.lod))
            if (r.dX is not None):
                r.dX = np.hstack((r.dX, s.dX))
            if (r.dXr is not None):
                r.dXr = np.hstack((r.dXr, s.dXr))
            if (r.dY is not None):
                r.dY = np.hstack((r.dY, s.dY))
            if (r.dYr is not None):
                r.dYr = np.hstack((r.dYr, s.dYr))
            if (r.sxpo is not None):
                r.sxpo = np.hstack((r.sxpo, s.sxpo))
            if (r.sxpor is not None):
                r.sxpor = np.hstack((r.sxpor, s.sxpor))
            if (r.sypo is not None):
                r.sypo = np.hstack((r.sypo, s.sypo))
            if (r.sypor is not None):
                r.sypor = np.hstack((r.sypor, s.sypor))
            if (r.sut is not None):
                r.sut = np.hstack((r.sut, s.sut))
            if (r.slod is not None):
                r.slod = np.hstack((r.slod, s.slod))
            if (r.sdX is not None):
                r.sdX = np.hstack((r.sdX, s.sdX))
            if (r.sdXr is not None):
                r.sdXr = np.hstack((r.sdXr, s.sdXr))
            if (r.sdY is not None):
                r.sdY = np.hstack((r.sdY, s.sdY))
            if (r.sdYr is not None):
                r.sdYr = np.hstack((r.sdYr, s.sdYr))
        
    # Delete records that do not belong to period of interest
    #--------------------------------------------------------
    def trim(r, start, end):
        
        """
        Delete records that do not belong to period of interest

        Parameters
        ----------
        start : float
            Start date (MJD)
        end : float
            End date (MJD)
        
        """

        # Indices of rords that should be kept
        ind = np.nonzero(np.logical_and(r.mjd >= start, r.mjd <= end))[0]

        # Restrict erp object
        for key in r.__dict__:
            if (getattr(r, key) is not None):
                setattr(r, key, getattr(r, key)[ind])

    # Linear interpolation at specified dates
    #----------------------------------------
    def interp(r, mjd):

        """
        Linear interpolation at specified dates

        Returns
        -------
        ri : erp instance
        
        Parameters
        ----------
        mjd : array_like
            List of MJDs

        """

        # Initialize riput erp object
        ri = erp()
        ri.mjd = mjd

        # Get indices of input dates just before and just after each output date
        ind1 = -np.ones(len(mjd), dtype='int')
        ind2 = -np.ones(len(mjd), dtype='int')
        for i in range(len(mjd)):
            ind1[i] = np.nonzero(r.mjd <= mjd[i])[0][-1]
            ind2[i] = np.nonzero(r.mjd >  mjd[i])[0][0]

        # Get lengths of intervals between each output date and the input dates just before and just after
        dt1 = mjd - r.mjd[ind1]
        dt2 = r.mjd[ind2] - mjd
        dt = dt1 + dt2

        # Remove leap seconds from input UT1
        ut1 = r.ut1 - 1000*leapsec(r.mjd)

        # Compute ERP rates
        ri.xpor =  (r.xpo[ind2] - r.xpo[ind1]) / dt
        ri.ypor =  (r.ypo[ind2] - r.ypo[ind1]) / dt
        ri.lod  = -(  ut1[ind2] -   ut1[ind1]) / dt
        ri.dXr  =  (r.dX [ind2] - r.dX [ind1]) / dt
        ri.dYr  =  (r.dY [ind2] - r.dY [ind1]) / dt

        # Linear interpolations of ERPs
        ri.xpo = (dt2*r.xpo[ind1] + dt1*r.xpo[ind2]) / dt
        ri.ypo = (dt2*r.ypo[ind1] + dt1*r.ypo[ind2]) / dt
        ri.ut1 = (dt2*  ut1[ind1] + dt1*  ut1[ind2]) / dt
        ri.dX  = (dt2*r.dX [ind1] + dt1*r.dX [ind2]) / dt
        ri.dY  = (dt2*r.dY [ind1] + dt1*r.dY [ind2]) / dt

        # Add leap seconds to output UT1
        ri.ut1 = ri.ut1 + 1000*leapsec(ri.mjd)

        return ri

    # Write erp instance into IGS ERP file
    #-------------------------------------
    def write_igs(r, file):
      
        """
        Write erp instance into IGS ERP file

        Parameters
        ----------
        file : file-like
            IGS ERP file to write
        
        """

        # Fill missing fields with zeros if necessary
        if (r.xpo is None):
            r.xpo  = np.zeros(len(r.mjd))
        if (r.sxpo is None):
            r.sxpo = np.zeros(len(r.mjd))
        if (r.ypo is None):
            r.ypo  = np.zeros(len(r.mjd))
        if (r.sypo is None):
            r.sypo = np.zeros(len(r.mjd))
        if (r.xpor is None):
            r.xpor  = np.zeros(len(r.mjd))
        if (r.sxpor is None):
            r.sxpor = np.zeros(len(r.mjd))
        if (r.ypor is None):
            r.ypor  = np.zeros(len(r.mjd))
        if (r.sypor is None):
            r.sypor = np.zeros(len(r.mjd))
        if (r.ut1 is None):
            r.ut1  = np.zeros(len(r.mjd))
        if (r.sut1 is None):
            r.sut1 = np.zeros(len(r.mjd))
        if (r.lod is None):
            r.lod  = np.zeros(len(r.mjd))
        if (r.slod is None):
            r.slod = np.zeros(len(r.mjd))

        # Print header
        print('version 2', file=file)
        print('EOP  SOLUTION', file=file)
        print('  MJD         X        Y     UT1-UTC    LOD   Xsig   Ysig   UTsig LODsig  Nr Nf Nt     Xrt    Yrt  Xrtsig Yrtsig   dpsi    deps', file=file)
        print('               10**-6"        .1us    .1us/d    10**-6"     .1us  .1us/d                10**-6"/d    10**-6"/d        10**-6', file=file)

        # Print records
        for i in range(len(r.mjd)):
            print('{0:8.2f} {1:8.0f} {2:8.0f} {3:9.0f} {4:6.0f} {5:6.0f} {6:6.0f} {7:7.0f} {8:6.0f}   0  0  0  {9:6.0f} {10:6.0f}  {11:6.0f} {12:6.0f}      0       0'.format(r.mjd[i], 1e3*r.xpo[i], 1e3*r.ypo[i], 1e4*r.ut1[i], 1e4*r.lod[i], 1e3*r.sxpo[i], 1e3*r.sypo[i], 1e4*r.sut1[i], 1e4*r.slod[i], 1e3*r.xpor[i], 1e3*r.ypor[i], 1e3*r.sxpor[i], 1e3*r.sypor[i]), file=file)
