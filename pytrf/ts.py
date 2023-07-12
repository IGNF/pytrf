"""
pytrf time series utilities

This subpackage contains various useful classes for modeling time series.

"""

# External imports
#-----------------
import sys
eps = sys.float_info.epsilon
import copy
import pickle
from math import pi, sqrt, exp, log, ceil, factorial
import numpy as np
from scipy import linalg, optimize, special, signal, sparse
from scipy.stats import median_abs_deviation as mad
import matplotlib.pyplot as pp
pp.rcParams['font.family'] = 'monospace'
pp.rcParams['font.size'] = 12
from traceback import print_exc
from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum

# Internal imports
#-----------------
from pytrf import date
from pytrf.math import xyz2enh, trend, invspd, cholesky, cholsolve, lombscargle, trdot



# ts class
#---------
class ts:
  
    """
    pytrf time series class

    A ts instance is initialized in one of the following ways:
    
        dat = ts()
        dat = ts.read(text_file)
        dat = ts.load(pickle_file)

    Once initialized, each ts instance has the following attributes:

        nd     : Number of components (e.g., 3 for station position time series)
        n      : Length
        t      : Dates (n,)
        T      : Integration interval(s)
        y      : Values (n, nd)
        Q      : Formal covariance matrices (n, nd, nd)
        tunit  : Time unit
        yunit  : Series unit
        dims   : Component names (e.g., ['East', 'North', 'Up'])
        R      : XYZ -> ENH rotation matrix (in case of a series given in
                 geocentric coordinates and rotated into topocentric coordinates)
        dtrd   : Degree of detrending polynomial
        ctrd   : Coefficients of detrending polynomial (dtrd+1, nd)
        t0     : Origin of time used for detrending
        ndel   : Number of detected outliers
        tdel   : Dates of detected outliers (ndel,)
        Tdel   : Integration intervals of detected outliers (ndel,)
        ydel   : Values of detected outliers (ndel, nd)
        Qdel   : Covariance matrices of detected outliers (ndel, nd, nd)
        
    Each ts instance has the following methods:

        __len__()       : Get number of components of ts instance
        __getitem__()   : Get specific component of ts instance
        dump()          : Dump ts instance into pickle file
        trim()          : Trim ts instance down to period of interest
        detrend()       : (Re-)detrend ts instance
        del_points()    : Flag outliers
        clean_sigmas()  : Flag observations with large formal errors as outliers
        plot()          : Plot time series
        
    """

    # Initialize a ts instance
    #-------------------------
    def __init__(r, *args, t=None, T=None, y=None, Q=None, tunit='d', yunit='m', dims=None, rotate=False, t0=None, dtrd=None):
      
        """
        Initialize a ts instance

        Returns
        -------
        r : ts instance
        
        Parameters
        ----------
        1st non-keyword parameter : int, tuple or array, optional
            Specifies either the length of the time series (n),
            the length and number of components of the time series ((n, nd)),
            or the time series values (y).
        t : array, optional
            Dates. Default is None, in which case t is set to np.arange(n).
        T : float or array, optional
            Integration interval(s). Default is None, in which case constant
            integration intervals of length min(t[1:]-t[:-1]) are assumed.
        y : array, optional
            Time series values ((n,) or (n, nd))
        Q : array, optional
            Formal covariance matrices (n, nd, nd). Default is None.
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.
        dims : str or list, optional
            Component names. Default is None.
        rotate : bool, optional
            If True and time series is 3-dimensional, it is assumed to be given in geocentric coordinates,
            and is rotated into topocentric coordinates. Defaut is False.
        t0 : float, optional
            The origin of time used for detrending. Default is None (automatically set).
        dtrd : int, optional
            Degree of detrending polynomial.
            - If 0, then an average is removed from each component of the time series (and stored in r.ctrd).
            - If 1, then a linear trend is removed from each component of the time series (and its coefficients stored in r.ctrd).
            - If anything else, the time series is not detrended.
            Default is None.

        """

        # Initialize all attributes to None
        r.nd = None
        r.n = None
        r.t = None
        r.T = None
        r.y = None
        r.Q = None
        r.tunit = None
        r.yunit = None
        r.dims = None
        r.R = None
        r.dtrd = None
        r.ctrd = None
        r.t0 = None
        r.ndel = None
        r.tdel = None
        r.Tdel = None
        r.ydel = None
        r.Qdel = None
        
        # Set time series values (y) in function of a possible non-keyword argument
        # or of argument t
        if (len(args) == 1) and (y is None):
            if isinstance(args[0], (int, tuple)):
                y = np.zeros(args[0])
            elif isinstance(args[0], np.ndarray):
                y = args[0]
        elif (t is not None) and (y is None):
            y = np.zeros(len(t))
        
        # If any values can be attributed to the time series
        if (y is not None):
        
            # Set number of components
            if (y.ndim == 2):
                r.nd = y.shape[1]
            else:
                r.nd = 1
            
            # Set time series length
            r.n = y.shape[0]
            
            # Set array of dates
            if (t is not None):
                r.t = t.copy()
            else:
                r.t = np.arange(r.n)
            
            # Set array of time series values
            r.y = y.copy()
            
            # Set covariance matrices of time series values
            if (Q is not None):
                r.Q = Q.copy()
            else:
                r.Q = None
            
            # Set integration intervals
            if (T is not None):
                if np.isscalar(T):
                    r.T = T
                else:
                    r.T = T.copy()
            elif (len(r.t) > 1):
                r.T = np.min(r.t[1:]-r.t[:-1])
            else:
                r.T = 0

            # Set time and series units
            r.tunit = tunit
            r.yunit = yunit

            # Set component names
            r.dims = dims

            # Rotate time series to topocentric frame if needed
            if (r.nd == 3) and (rotate):
                ym = np.mean(r.y, axis=0)
                r.R = xyz2enh(ym)
                for i in range(r.n):
                    r.y[i] = np.dot(r.R, r.y[i])
                    if (r.Q is not None):
                        r.Q[i] = np.dot(r.R, np.dot(r.Q[i], r.R.T))
                    
                # Also set default component names
                if (r.dims is None):
                    r.dims = ['East', 'North', 'Up']
            
            # Set origin of time
            if (t0 is not None):
                r.t0 = t0
            else:
                r.t0 = np.mean(r.t)

            # Detrend time series if needed
            if (dtrd == 0):
                r.dtrd = 0
                r.ctrd = np.zeros((1, r.nd))
                for j in range(r.nd):
                    r.ctrd[0,j] = np.mean(r.y[:,j])
                    r.y[:,j] = r.y[:,j] - r.ctrd[0,j]
                    
            elif (dtrd == 1):
                r.dtrd = 1
                r.ctrd = np.zeros((2, r.nd))
                dt = r.t - r.t0 
                for j in range(r.nd):
                    r.ctrd[:,j] = trend(dt, r.y[:,j])
                    r.y[:,j] = r.y[:,j] - r.ctrd[0,j] - r.ctrd[1,j]*dt

            # Initialize outliers
            r.ndel = 0
            r.tdel = np.array([])
            if not(np.isscalar(r.T)):
                r.Tdel = np.array([])
            r.ydel = np.empty((0, r.nd))
            if (r.Q is not None):
                r.Qdel = np.empty((0, r.nd, r.nd))

            # Squeeze unnecessary components
            if (r.nd == 1):
                if (r.dtrd is not None):
                    r.ctrd.resize((r.dtrd+1,))
                r.y.resize((r.n,))
                if (r.Q is not None):
                    r.Q.resize((r.n,))
                r.ydel.resize((0,))
                if (r.Q is not None):
                    r.Qdel.resize((0,))

    # Create ts instance from text file
    #----------------------------------
    @classmethod
    def read(self, file, format, usecols=None, skiprows=0, comments='#', delimiter=None, T=None, tunit='d', yunit='m', dims=None, rotate=False, t0=None, dtrd=None):

        """
        Create ts instance from text file

        Returns
        -------
        r : ts instance

        Parameters
        ----------
        file : str
            Text file to read
        format : sequence of str
            Strings indicating the content of the columns to be read.
            It can include the following keywords:
            - t    : for a column containing the time series dates
            - T    : for a column containing the time series integration intervals
            - x    : for a column containing the 1st component of the time series
            - y    : for a column containing the 2nd component of the time series
            - z    : for a column containing the 3rd component of the time series
            - sx   : for a column containing the formal errors of the 1st component of the time series
            - sy   : for a column containing the formal errors of the 2nd component of the time series
            - sz   : for a column containing the formal errors of the 3rd component of the time series
            - qx   : for a column containing the formal variances of the 1st component of the time series
            - qy   : for a column containing the formal variances of the 2nd component of the time series
            - qz   : for a column containing the formal variances of the 3rd component of the time series
            - cxy  : for a column containing the formal correlations between the 1st and 2nd components of the time series
            - cxz  : for a column containing the formal correlations between the 1st and 3rd components of the time series
            - cyz  : for a column containing the formal correlations between the 2nd and 3rd components of the time series
            - qxy  : for a column containing the formal covariances between the 1st and 2nd components of the time series
            - qxz  : for a column containing the formal covariances between the 1st and 3rd components of the time series
            - qyz  : for a column containing the formal covariances between the 2nd and 3rd components of the time series
        usecols : int or sequence, optional
            Indices of the columns to be read among the columns of the text file (see numpy.loadtxt).
            Default is None: all columns are read.
        skiprows : int, optional
            Number of header lines to skip (see numpy.loadtxt). Default is 0.
        comments : str or sequence of str, optional
            Characters or list of characters used to indicate the start of a comment (see numpy.loadtxt). Default is None.
        delimiter : str, optional
            String used to separate columns (see numpy.loadtxt). Default is None (i.e., white space).
        T : float or array, optional
            Integration interval(s) (if not given in input file). Default is None, in which case constant integration
            intervals of length min(t[1:]-t[:-1]) are assumed.
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.
        dims : str or list, optional
            Component names. Default is None.
        rotate : bool, optional
            If True and time series is 3-dimensional, it is assumed to be given in geocentric coordinates,
            and is rotated into topocentric coordinates. Defaut is False.
        t0 : float, optional
            The origin of time used for detrending. Default is None (automatically set).
        dtrd : int, optional
            Degree of detrending polynomial.
            - If 0, then an average is removed from each component of the time series (and stored in r.ctrd).
            - If 1, then a linear trend is removed from each component of the time series (and its coefficients stored in r.ctrd).
            - If anything else, the time series is not detrended.
            Default is None.
            
        """
        
        # Read text file
        dat = np.loadtxt(file, usecols=usecols, skiprows=skiprows, comments=comments, delimiter=delimiter, ndmin=2)
        
        # Get array of dates
        j = format.index('t')
        t = dat[:,j]
        
        # Get array of time series values
        j = [format.index('x')]
        if ('y' in format):
            j.append(format.index('y'))
        if ('z' in format):
            j.append(format.index('z'))
        y = dat[:,j]
        (n, nd) = y.shape
        
        # Get covariance matrices of time series values
        Q = None
        if ('sx' in format) or ('qx' in format):
            Q = np.zeros((n, nd, nd))
            
            if ('sx' in format):
                j = format.index('sx')
                Q[:,0,0] = dat[:,j]**2
            elif ('qx' in format):
                j = format.index('qx')
                Q[:,0,0] = dat[:,j]

            if (nd > 1):
                if ('sy' in format):
                    j = format.index('sy')
                    Q[:,1,1] = dat[:,j]**2
                elif ('qy' in format):
                    j = format.index('qy')
                    Q[:,1,1] = dat[:,j]
                if ('cxy' in format):
                    j = format.index('cxy')
                    Q[:,0,1] = dat[:,j] * np.sqrt(Q[:,0,0]*Q[:,1,1])
                    Q[:,1,0] = Q[:,0,1]
                elif ('qxy' in format):
                    j = format.index('qxy')
                    Q[:,0,1] = dat[:,j]
                    Q[:,1,0] = Q[:,0,1]

            if (nd > 2):
                if ('sz' in format):
                    j = format.index('sz')
                    Q[:,2,2] = dat[:,j]**2
                elif ('qz' in format):
                    j = format.index('qz')
                    Q[:,2,2] = dat[:,j]
                if ('cxz' in format):
                    j = format.index('cxz')
                    Q[:,0,2] = dat[:,j] * np.sqrt(Q[:,0,0]*Q[:,2,2])
                    Q[:,2,0] = Q[:,0,2]
                elif ('qxz' in format):
                    j = format.index('qxz')
                    Q[:,0,2] = dat[:,j]
                    Q[:,2,0] = Q[:,0,2]
                if ('cyz' in format):
                    j = format.index('cyz')
                    Q[:,1,2] = dat[:,j] * np.sqrt(Q[:,1,1]*Q[:,2,2])
                    Q[:,2,1] = Q[:,1,2]
                elif ('qyz' in format):
                    j = format.index('qyz')
                    Q[:,1,2] = dat[:,j]
                    Q[:,2,1] = Q[:,1,2]
            
        # Get integration intervals
        T = None
        if ('T' in format):
            j = format.index('T')
            T = dat[:,j]
            
        # Create ts instance
        r = ts(y, t=t, T=T, Q=Q, tunit=tunit, yunit=yunit, dims=dims, rotate=rotate, t0=t0, dtrd=dtrd)
            
        return r
    
    # Load ts instance from pickle file
    #----------------------------------
    @classmethod
    def load(self, file):

        """
        Load ts instance from pickle file

        Returns
        -------
        r : ts instance

        Parameters
        ----------
        file : str
            Pickle file to load

        """
    
        return pickle.load(open(file, 'rb'))
    
    # Get number of components of ts instance
    #----------------------------------------
    def __len__(r):
      
        """
        Get number of components of ts instance

        """

        return r.nd

    # Get specific component of ts instance
    #--------------------------------------
    def __getitem__(r, i):
      
        """
        Extract specific component of ts instance

        Returns
        -------
        ri : ts instance

        Parameters
        ----------
        i : int
            Component index

        """
        
        if (r.nd == 1):
            return r

        else:
            ri = ts()
            ri.nd = 1
            ri.n = r.n
            ri.t = r.t
            ri.T = r.T
            ri.y = r.y[:,i]
            if (r.Q is not None):
                ri.Q = r.Q[:,i,i]
            ri.tunit = r.tunit
            ri.yunit = r.yunit
            ri.dims = None
            if (r.dims is not None):
                ri.dims = r.dims[i]
            ri.R = r.R
            ri.dtrd = r.dtrd
            if (r.ctrd is not None):
                ri.ctrd = r.ctrd[:,i]
            ri.t0 = r.t0
            ri.ndel = r.ndel
            ri.tdel = r.tdel
            ri.Tdel = r.Tdel
            ri.ydel = r.ydel[:,i]
            if (r.Q is not None):
                ri.Qdel = r.Qdel[:,i,i]
            
            return ri

    # Dump ts instance into pickle file
    #----------------------------------
    def dump(r, file):
      
        """
        Dump ts instance into pickle file

        Parameters
        ----------
        file : str
            Pickle file to write

        """

        pickle.dump(r, open(file, 'wb'))

    # Trim ts instance down to period of interest
    #--------------------------------------------
    def trim(r, start, end):
        
        """
        Trim ts instance down to period of interest

        Parameters
        ----------
        start : float
            Start date
        end : float
            End date
        
        """

        # Indices of observations that should be kept
        ind = np.nonzero(np.logical_and(r.t >= start, r.t <= end))[0]

        # Update ts instance
        r.n = len(ind)
        r.t = r.t[ind]
        if not(np.isscalar(r.T)):
            r.T = r.T[ind]
        r.y = r.y[ind]
        if (r.Q is not None):
            r.Q = r.Q[ind]

        # Indices of outliers that should be kept
        ind = np.nonzero(np.logical_and(r.tdel >= start, r.tdel <= end))[0]

        # Update ts instance
        r.ndel = len(ind)
        r.tdel = r.tdel[ind]
        if not(np.isscalar(r.T)):
            r.Tdel = r.Tdel[ind]
        r.ydel = r.ydel[ind]
        if (r.Q is not None):
            r.Qdel = r.Qdel[ind]
        
    # (Re-)detrend ts instance
    #-------------------------
    def detrend(r, dtrd=1, t0=None):

        """
        (Re-)detrend ts instance

        Parameters
        ----------
        dtrd : int, optional
            Degree of detrending polynomial.
            - If 0, then an average is removed from each component of the time series (and stored in r.ctrd).
            - If 1, then a linear trend is removed from each component of the time series (and its coefficients stored in r.ctrd).
            - If anything else, the time series is not detrended.
            Default is 1.
        t0 : float, optional
            The origin of time used for detrending. Default is None (automatically set).
            
        """

        if (t0 is None):
            t0 = np.mean(r.t)
    
        if (dtrd == 0):
            if (r.dtrd is None):
                r.dtrd = 0
                r.ctrd = np.zeros((1, r.nd))
                
            for j in range(r.nd):
                tr = np.mean(r.y[:,j])
                r.ctrd[0,j] = r.ctrd[0,j] + tr
                r.y[:,j] = r.y[:,j] - tr
                
        elif (dtrd == 1):
            if (r.dtrd is None):
                r.dtrd = 1
                r.ctrd = np.zeros((2, r.nd))
            elif (r.dtrd == 0):
                r.dtrd = 1
                r.ctrd = np.vstack((r.ctrd, np.zeros((1, r.nd))))
            elif (r.dtrd == 1) and (r.t0 != t0):
                r.ctrd[:,0] = r.ctrd[:,0] + r.ctrd[:,1]*(t0-r.t0)
            
            r.t0 = t0
            t = r.t - r.t0

            if (r.nd == 1):
                tr = trend(t, r.y)
                r.ctrd = r.ctrd + tr
                r.y = r.y - tr[0] - tr[1]*t 
            else:
                for j in range(r.nd):
                    tr = trend(t, r.y[:,j])
                    r.ctrd[:,j] = r.ctrd[:,j] + tr
                    r.y[:,j] = r.y[:,j] - tr[0] - tr[1]*t 

    # Flag outliers
    #--------------
    def del_points(r, ind):

        """
        Flag outliers

        Parameters
        ----------
        ind : list
            Indices of outliers
            
        """
        
        # Update list of outliers
        r.ndel = r.ndel + len(ind)
        r.tdel = np.hstack((r.tdel, r.t[ind]))
        if not(np.isscalar(r.T)):
            r.Tdel = np.hstack((r.Tdel, r.T[ind]))
        if (r.nd == 1):
            r.ydel = np.hstack((r.ydel, r.y[ind]))
            if (r.Q is not None):
                r.Qdel = np.hstack((r.Qdel, r.Q[ind]))
        else:
            r.ydel = np.vstack((r.ydel, r.y[ind]))
            if (r.Q is not None):
                r.Qdel = np.vstack((r.Qdel, r.Q[ind]))
        
        # Clean time series
        ind = np.setdiff1d(np.arange(r.n), ind)
        r.n = len(ind)
        r.t = r.t[ind]
        if not(np.isscalar(r.T)):
            r.T = r.T[ind]
        r.y = r.y[ind]
        if (r.Q is not None):
            r.Q = r.Q[ind]

    # Flag observations with large formal errors as outliers
    #-------------------------------------------------------
    def clean_sigmas(r, thr=5):

        """
        Flag observations with large formal errors as outliers

        Parameters
        ----------
        thr : float, optional
            Threshold. All observations with formal sigmas larger than thr * median of formal sigmas
            along any component are flagged as outliers. Default is 5.
            
        """

        # While there remains outliers,
        end = False
        while not(end):
                        
            # Get indices of outliers
            ind = []
            for i in range(r.nd):
                if (r.nd == 1):
                    s = np.sqrt(r.Q)
                else:
                    s = np.sqrt(r.Q[:,i,i])
                sm = np.median(s)
                ind = ind + np.nonzero(s > thr*sm)[0].tolist()
            ind = list(set(ind))
            
            # If any outlier remains, delete them
            if (len(ind) > 0):
                r.del_points(ind)
            
            # Else, we're done.
            else:
                end = True

    # Plot time series
    #-----------------
    def plot(r, figsize=None, tunit=None, dims=None, title=None, output=None, return_fig=False, show=True):

        """
        Plot time series

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (see matplotlib.pyplot.figure). Default is None (automatically set).
        tunit : str, optional
            Time unit for the plot. Default is None (i.e., time unit of the time series).
        dims : str or list, optional
            Component names. Default is None.
        title : str, optional
            Figure title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. figure shown on screen).
        return_fig : bool, optional
            Whether to return matplotlib figure object insted of showing it. Default is False.
        show : bool, optional
            Whether to show figure. Default is True.
            
        """

        # Figure size
        if (figsize is None):
            if (r.nd == 1):
                figsize = (10, 4)
            elif (r.nd == 2):
                figsize = (10, 7)
            else:
                figsize = (10, 10)

        # Time
        if (tunit is None) or (tunit == r.tunit):
            tunit = r.tunit
            t = r.t
        elif (r.tunit == 'd') and (tunit == 'y'):
            t = [date.from_mjd(d).ydec() for d in r.t]
        else:
            tunit = r.tunit
            t = r.t

        # Component names
        if (dims is None):
            dims = r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(r.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Create new figure
        fig = pp.figure(figsize=figsize, tight_layout=True)
        
        # Loop over components
        for d in range(r.nd):
            if (r.nd == 1):
                y = r.y
                if (r.Q is not None):
                    s = np.sqrt(r.Q)
            else:
                y = r.y[:,d]
                if (r.Q is not None):
                    s = np.sqrt(r.Q[:,d,d])
            ax = fig.add_subplot(r.nd, 1, d+1)
            ax.margins(0.01, 0.01)
            ax.grid(zorder=0)
            ax.set_ylabel(dims[d]+' ['+r.yunit+']')
            if (r.Q is not None):
                ax.errorbar(t, y, yerr=s, fmt='.k', ecolor='gray', zorder=3)
            else:
                ax.plot(t, y, '.k', zorder=3)
        ax.set_xlabel('Time ['+tunit+']')
        
        # Return, save or show figure
        if (return_fig):
            return (fig, t)
        elif (output is not None):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
        elif (show):
            pp.show()



# param class
#------------
class param:
  
    """
    Generic class for deterministic/noise parameters adjusted to time series

    A param instance is initialized by:
    
        p = param()

    Once initialized, each param instance has the following attributes:

        type  : Description of parameter type (str)
        t     : Start of validity (date instance or None)
        x     : Value (float or None)
        fixed : Fixed or estimated parameter? (bool)
        sig   : Formal error (float or None)
        unit  : Parameter unit (str)
        xc    : Reference value of constraint (float or None)
        sigc  : Sigma of constraint (float or None)
        
    """

    # Initialize a param instance
    #----------------------------
    def __init__(p, type, t=-np.inf, x=None, fixed=False, unit='', xc=None, sigc=None):
      
        """
        Initialize a param instance

        Returns
        -------
        p : param instance
        
        Parameters
        ----------
        type : str
            Parameter type
        t : date instance, optional
            Start of validity
        x : float, optional
            Parameter value
        fixed : bool, optional
            Fixed or estimated parameter?
        unit : str, optional
            Parameter unit. Default is ''.
        xc : float, optional
            Reference value of constraint
        sigc : float, optional
            Sigma of constraint
        """

        p.type = type
        p.t = t
        p.x = x
        p.fixed = fixed
        if (p.fixed):
            p.sig = 0
        else:
            p.sig = None
        p.unit = unit
        p.xc = xc
        p.sigc = sigc



# scale_param class
#------------------
class scale_param(param):
  
    """
    Sub-class of the param class for scale parameters (variance factors, decay times)
    whose logarithms are internally estimated

    A scale_param instance is initialized by:
    
        p = scale_param()

    A scale_param instance inherits the attributes from a param instance.
        
    Each scale_param instance additionally has the following methods:

        x2xr()   : Compute reparameterized value (xr=log(x)) from original value
        xr2x()   : Compute original value (x=exp(xr)) from reparameterized value
        dx_dxr() : Compute partial derivative of original value wrt reparameterized value
        
    """

    # Initialize a scale_param instance
    #----------------------------------
    def __init__(p, type, t=-np.inf, x=None, fixed=False, unit='', xc=None, sigc=None):
      
        """
        Initialize a scale_param instance

        Returns
        -------
        p : scale_param instance
        
        Parameters
        ----------
        type : str
            Parameter type
        t : date instance, optional
            Start of validity
        x : float, optional
            Parameter value
        fixed : bool, optional
            Fixed or estimated parameter?
        unit : str, optional
            Parameter unit. Default is ''.
        xc : float, optional
            Reference value of constraint
        sigc : float, optional
            Sigma of constraint
            
        """

        super().__init__(type=type, t=t, x=x, fixed=fixed, unit=unit, xc=xc, sigc=sigc)
        
    # Compute reparameterized value (xr=log(x)) from original value
    #--------------------------------------------------------------
    def x2xr(p, x):
        
        """
        Compute reparameterized value (xr=log(x)) from original value

        Returns
        -------
        xr : float
            Reparameterized value (xr=log(x))
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return log(x)

    # Compute original value (x=exp(xr)) from reparameterized value
    #--------------------------------------------------------------
    def xr2x(p, xr):
        
        """
        Compute original value (x=exp(xr)) from reparameterized value

        Returns
        -------
        x : float
            Original value (x=exp(xr))
        
        Parameter
        ---------
        xr : float
            Reparameterized value
            
        """
        
        return exp(xr)
    
    # Compute partial derivative of original value wrt reparameterized value
    #-----------------------------------------------------------------------
    def dx_dxr(p, x):
        
        """
        Compute partial derivative of original value wrt reparameterized value

        Returns
        -------
        dx : float
            Partial derivative of original value wrt reparameterized value
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return x



# tanh_param class
#-----------------
class tanh_param(param):
  
    """
    Sub-class of the param class for parameters in ]-1,1[ (e.g., MA(1) coefficients)
    whose inverse hyperbolic tangents are internally estimated

    A tanh_param instance is initialized by:
    
        p = tanh_param()

    A tanh_param instance inherits the attributes from a param instance.
        
    Each tanh_param instance additionally has the following methods:

        x2xr()   : Compute reparameterized value (xr=atanh(x)) from original value
        xr2x()   : Compute original value (x=tanh(xr)) from reparameterized value
        dx_dxr() : Compute partial derivative of original value wrt reparameterized value
        
    """

    # Initialize a tanh_param instance
    #---------------------------------
    def __init__(p, type, t=-np.inf, x=None, fixed=False, unit='', xc=None, sigc=None):
      
        """
        Initialize an tanh_param instance

        Returns
        -------
        p : tanh_param instance
        
        Parameters
        ----------
        type : str
            Parameter type
        t : date instance, optional
            Start of validity
        x : float, optional
            Parameter value
        fixed : bool, optional
            Fixed or estimated parameter?
        unit : str, optional
            Parameter unit. Default is ''.
        xc : float, optional
            Reference value of constraint
        sigc : float, optional
            Sigma of constraint
            
        """

        super().__init__(type=type, t=t, x=x, fixed=fixed, unit=unit, xc=xc, sigc=sigc)
        
    # Compute reparameterized value (xr=atanh(x)) from original value
    #----------------------------------------------------------------
    def x2xr(p, x):
        
        """
        Compute reparameterized value (xr=atanh(x)) from original value

        Returns
        -------
        xr : float
            Reparameterized value (xr=atanh(x))
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return atanh(x)

    # Compute original value (x=tanh(xr)) from reparameterized value
    #---------------------------------------------------------------
    def xr2x(p, xr):
        
        """
        Compute original value (x=tanh(xr)) from reparameterized value

        Returns
        -------
        x : float
            Original value (x=tanh(xr))
        
        Parameter
        ---------
        xr : float
            Reparameterized value
            
        """
        
        return tanh(xr)
    
    # Compute partial derivative of original value wrt reparameterized value
    #-----------------------------------------------------------------------
    def dx_dxr(p, x):
        
        """
        Compute partial derivative of original value wrt reparameterized value

        Returns
        -------
        dx : float
            Partial derivative of original value wrt reparameterized value
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return 1-x**2



# pl_index class
#---------------
class pl_index(param):
  
    """
    Sub-class of the param class for for spectral indices of power-law noise,
    which are internally reparameterized as xr = -log(exp(3-x)-1)

    A pl_index instance is initialized by:
    
        p = pl_index()

    A pl_index instance inherits the attributes from a param instance.
        
    Each pl_index instance additionally has the following methods:

        x2xr()   : Compute reparameterized value (xr = -log(exp(3-x)-1)) from original value
        xr2x()   : Compute original value (x = 3-log(1+exp(-xr))) from reparameterized value
        dx_dxr() : Compute partial derivative of original value wrt reparameterized value
        
    """

    # Initialize a pl_index instance
    #-------------------------------
    def __init__(p, type, t=-np.inf, x=None, fixed=False, unit='', xc=None, sigc=None):
      
        """
        Initialize a pl_index instance

        Returns
        -------
        p : pl_index instance
        
        Parameters
        ----------
        type : str
            Parameter type
        t : date instance, optional
            Start of validity
        end : date instance, optional
            End of validity
        x : float, optional
            Parameter value
        fixed : bool, optional
            Fixed or estimated parameter?
        unit : str, optional
            Parameter unit. Default is ''.
        xc : float, optional
            Reference value of constraint
        sigc : float, optional
            Sigma of constraint
            
        """

        super().__init__(type=type, t=t, x=x, fixed=fixed, unit=unit, xc=xc, sigc=sigc)
        
    # Compute reparameterized value (xr=-log(exp(3-x)-1)) from original value
    #------------------------------------------------------------------------
    def x2xr(p, x):
        
        """
        Compute reparameterized value (xr=-log(exp(3-x)-1)) from original value

        Returns
        -------
        xr : float
            Reparameterized value (xr=-log(exp(3-x)-1))
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return -log(exp(3-x)-1)

    # Compute original value (x=3-log(1+exp(-xr))) from reparameterized value
    #------------------------------------------------------------------------
    def xr2x(p, xr):
        
        """
        Compute original value (x=3-log(1+exp(-xr))) from reparameterized value

        Returns
        -------
        x : float
            Original value (x=3-log(1+exp(-xr)))
        
        Parameter
        ---------
        xr : float
            Reparameterized value
            
        """
        
        return 3-log(1+exp(-xr))
    
    # Compute partial derivative of original value wrt reparameterized value
    #-----------------------------------------------------------------------
    def dx_dxr(p, x):
        
        """
        Compute partial derivative of original value wrt reparameterized value

        Returns
        -------
        dx : float
            Partial derivative of original value wrt reparameterized value
        
        Parameter
        ---------
        x : float
            Original value
            
        """
        
        return 1-exp(x-3)



# function class
#---------------
class function:
  
    """
    Generic class for deterministic functions adjusted to time series

    A function instance is initialized by:
    
        f = function()

    Once initialized, each function instance has the following attributes:

        par : List of param instances (initialized to [])
        yc  : Computed observations (initialized to None)
        A   : Design matrix (initialized to None)
        
    """

    # Initialize a function instance
    #-------------------------------
    def __init__(f):
      
        """
        Initialize a function instance

        Returns
        -------
        f : function instance
        
        """

        f.par = []
        f.yc = None
        f.A = None



# polynom class
#--------------
class polynom(function):

    """
    Sub-class of the function class for polynomial functions

    A polynom instance is initialized by:
    
        f = polynom()

    A polynom instance inherits the attributes from a function instance.
        
    Each polynom instance additionally has the following attributes:

        deg : Polynomial degree
        t   : List of jump dates
        
    Each polynom instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_oeq() : Compute predicted observations and design matrix

    """

    # Initialize a polynom instance
    #------------------------------
    def __init__(f, deg, t=[], x=None, fix_x=False, tunit='d', yunit='m'):

        """
        Initialize a polynom instance

        Returns
        -------
        f : polynom instance
        
        Parameters
        ----------
        deg : int
            Polynomial degree
        t : list, optional
            List of jump dates
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori)
            Default is False.
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """

        super().__init__()
        f.deg = deg
        f.t = t

        t = [-np.inf] + t
        
        if (x is None):
            x = len(t) * [None]
        elif np.isscalar(x):
            x = [x]
        
        if isinstance(fix_x, bool):
            fix_x = len(t) * [fix_x]
        
        if (deg == 0):
            unit = yunit
        elif (deg == 1):
            unit = yunit + '/' + tunit
        elif (deg >= 2):
            unit = yunit + '/' + tunit + '^' + str(deg)
        
        f.par.append(param(type='polynomial coefficient', x=x[0], fixed=fix_x[0], unit=unit))
        for i in range(1, len(t)):
            f.par.append(param(type='polynomial coefficient jump', t=t[i], x=x[i], fixed=fix_x[i], unit=unit))

    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(f, m):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        for p in f.par:
            if (p.x is None):
                if (p.xc is not None):
                    p.x = p.xc
                else:
                    p.x = 0

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(f, m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets attributes yc and A of the polynom instance.

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Initializations
        t = m.r.t
        f.yc = np.zeros(len(t))
        f.A = []
        
        # Loop over parameters
        for p in f.par:
            t0 = p.t
            if (t0 < t[0]):
                t0 = t[0]
            b = (t >= t0)
            a = b * (t-t0)**f.deg/factorial(f.deg)
            f.yc = f.yc + p.x*a
            if not(p.fixed):
                f.A.append(a)



# sine class
#-----------
class sine(function):

    """
    Sub-class of the function class for sine wave functions

    A sine instance is initialized by:
    
        f = sine()

    A sine instance inherits the attributes from a function instance.
        
    Each sine instance additionally has the following attributes:

        per : Period
        t   : List of jump dates
        
    Each sine instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_oeq() : Compute predicted observations and design matrix

    """

    # Initialize a sine instance
    #---------------------------
    def __init__(f, per, t=[], x=None, fix_x=False, yunit='m'):

        """
        Initialize a sine instance

        Returns
        -------
        f : sine instance
        
        Parameters
        ----------
        per : float
            Period
        t : list, optional
            List of jump dates
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori)
            Default is False.
        yunit : str, optional
            Time series unit. Default is 'm'.
            
        """

        super().__init__()
        f.per = per
        f.t = t

        t = [-np.inf] + t
        
        if (x is None):
            x = 2*len(t) * [None]
        
        if isinstance(fix_x, bool):
            fix_x = 2*len(t) * [fix_x]
            
        f.par.append(param(type='cos amplitude', x=x[0], fixed=fix_x[0], unit=yunit))
        f.par.append(param(type='sin amplitude', x=x[1], fixed=fix_x[1], unit=yunit))
        for i in range(1, len(t)):
            f.par.append(param(type='cos amplitude jump', t=t[i], x=x[2*i], fixed=fix_x[2*i], unit=yunit))
            f.par.append(param(type='sin amplitude jump', t=t[i], x=x[2*i+1], fixed=fix_x[2*i+1], unit=yunit))
            
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(f, m):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        for p in f.par:
            if (p.x is None):
                if (p.xc is not None):
                    p.x = p.xc
                else:
                    p.x = 0

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(f, m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets attributes yc and A of the sine instance.

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Initializations
        t = m.r.t
        dt = t - m.t0
        f.yc = np.zeros(len(t))
        f.A = []
        
        # Loop over pairs of cos/sin parameters
        for (pc, ps) in zip(f.par[::2], f.par[1::2]):
            b = (t >= pc.t)
            Ac = b * np.cos(2*pi*dt/f.per)
            As = b * np.sin(2*pi*dt/f.per)
            f.yc = f.yc + pc.x*Ac + ps.x*As
            if not(pc.fixed):
                f.A.append(Ac)
            if not(ps.fixed):
                f.A.append(As)



# poisson class
#--------------
class poisson(function):

    """
    Sub-class of the function class for Poisson functions

    A poisson instance is initialized by:
    
        f = poisson()

    A poisson instance inherits the attributes from a function instance.
        
    Each poisson instance additionally has the following attributes:

        per : Period
        deg : Polynomial degree
        
    Each poisson instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_oeq() : Compute predicted observations and design matrix

    """

    # Initialize a poisson instance
    #---------------------------
    def __init__(f, per, deg, x=None, fix_x=False, tunit='d', yunit='m'):

        """
        Initialize a poisson instance

        Returns
        -------
        f : poisson instance
        
        Parameters
        ----------
        per : float
            Period
        deg : int
            Polynomial degree
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori).
            Default is False.
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.
            
        """

        super().__init__()
        f.per = per
        f.deg = deg

        if (x is None):
            x = 2*(deg+1) * [None]
        
        if isinstance(fix_x, bool):
            fix_x = 2*(deg+1) * [fix_x]        
        
        for i in range(deg+1):
            if (i == 0):
                unit = yunit
            elif (i == 1):
                unit = yunit + '/' + tunit
            elif (i == 2):
                unit = yunit + '/' + tunit + '^' + str(deg)
            f.par.append(param(type='cos amplitude deg'+str(i), x=x[2*i], fixed=fix_x[2*i], unit=unit))
            f.par.append(param(type='sin amplitude deg'+str(i), x=x[2*i+1], fixed=fix_x[2*i+1], unit=unit))
            
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(f, m):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        for p in f.par:
            if (p.x is None):
                p.x = 0

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(f, m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets attributes yc and A of the poisson instance.

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Initializations
        t = m.r.t
        dt = t - m.t0
        f.yc = np.zeros(len(t))
        f.A = []

        # Loop over pairs of cos/sin parameters
        deg = 0
        for (pc, ps) in zip(f.par[::2], f.par[1::2]):

            Ac = dt**deg * np.cos(2*pi*dt/f.per)
            As = dt**deg * np.sin(2*pi*dt/f.per)

            f.yc = f.yc + pc.x*Ac + ps.x*As

            if not(pc.fixed):
                f.A.append(Ac)
            if not(ps.fixed):
                f.A.append(As)
                
            deg += 1



# fexp class
#-----------
class fexp(function):

    """
    Sub-class of the function class for exponential functions

    An fexp instance is initialized by:
    
        f = fexp()

    An fexp instance inherits the attributes from a function instance.
        
    Each fexp instance additionally has the following attribute:

        t0 : Start date
        
    Each fexp instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_oeq() : Compute predicted observations and design matrix

    """

    # Initialize an fexp instance
    #----------------------------
    def __init__(f, t0, amp=None, tau=None, fix_amp=False, fix_tau=False, tunit='d', yunit='m'):

        """
        Initialize an fexp instance

        Returns
        -------
        f : fexp instance
        
        Parameters
        ----------
        t0 : float
            Start date
        amp : float, optional
            Amplitude
        tau : float, optional
            Relaxation time
        fix_amp : bool, optional
            Whether the provided amplitude should be fixed (or only used as a priori)
        fix_tau : bool, optional
            Whether the provided relaxation time should be fixed (or only used as a priori)
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """

        super().__init__()
        f.t0 = t0

        f.par.append(param(type='exp amplitude', t=t0, x=amp, fixed=fix_amp, unit=yunit))
        f.par.append(scale_param(type='exp relaxation time', t=t0, x=tau, fixed=fix_tau, unit=tunit))
            
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(f, m):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
            
        """
        
        # Set a priori amplitude if needed
        if (f.par[0].x is None):
            if (f.par[0].xc is not None):
                f.par[0].x = f.par[0].xc
            else:
                f.par[0].x = 1

        # If [a priori] relaxation time is not already set,
        if (f.par[1].x is None):

            # If relaxation time is constrained, set a priori value to reference value
            if (f.par[1].xc is not None):
                f.par[1].x = f.par[1].xc
            
            # Else,
            else:
                
                # Look for other exp functions starting at the same date
                # and list their [a priori] relaxation times
                tau = []
                for ff in m.f:
                    if isinstance(ff, fexp):
                        if (ff.par[1].x is not None):
                            tau.append(ff.par[1].x)
                
                # Set a priori relaxation time of current function
                if (len(tau) == 0):
                    f.par[1].x = 100
                else:
                    f.par[1].x = np.min(tau) / 10

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(f, m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets attributes yc and A of the fexp instance.

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Initializations
        dt = np.abs(m.r.t - f.t0)
        b = (m.r.t >= f.t0)
        amp = f.par[0].x
        tau = f.par[1].x
        da = 1. - np.exp(-dt / tau)
        
        # Predicted observations
        f.yc = b * amp * da
        
        # Design matrix
        f.A = []
        if not(f.par[0].fixed):
            f.A.append(b * da)
        if not(f.par[1].fixed):
            f.A.append(b * -amp * dt * (1-da) / tau**2)



# flog class
#-----------
class flog(function):

    """
    Sub-class of the function class for logarithmic functions

    An flog instance is initialized by:
    
        f = flog()

    An flog instance inherits the attributes from a function instance.
        
    Each flog instance additionally has the following attribute:

        t0 : Start date
        
    Each flog instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_oeq() : Compute predicted observations and design matrix

    """

    # Initialize an flog instance
    #----------------------------
    def __init__(f, t0, amp=None, tau=None, fix_amp=False, fix_tau=False, tunit='d', yunit='m'):

        """
        Initialize an flog instance

        Returns
        -------
        f : flog instance
        
        Parameters
        ----------
        t0 : float
            Start date
        amp : float, optional
            Amplitude
        tau : float, optional
            Relaxation time
        fix_amp : bool, optional
            Whether the provided amplitude should be fixed (or only used as a priori)
        fix_tau : bool, optional
            Whether the provided relaxation time should be fixed (or only used as a priori)
        tunit : str, optional
            Time unit. Default is 'd'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """

        super().__init__()
        f.t0 = t0

        f.par.append(param(type='log amplitude', t=t0, x=amp, fixed=fix_amp, unit=yunit))
        f.par.append(scale_param(type='log relaxation time', t=t0, x=tau, fixed=fix_tau, unit=tunit))
            
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(f, m):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
            
        """
        
        # Set a priori amplitude if needed
        if (f.par[0].x is None):
            if (f.par[0].xc is not None):
                f.par[0].x = f.par[0].xc
            else:
                f.par[0].x = 1

        # If [a priori] relaxation time is not already set,
        if (f.par[1].x is None):

            # If relaxation time is constrained, set a priori value to reference value
            if (f.par[1].xc is not None):
                f.par[1].x = f.par[1].xc
            
            # Else,
            else:

                # Look for other log functions starting at the same date
                # and list their [a priori] relaxation times
                tau = []
                for ff in m.f:
                    if isinstance(ff, flog):
                        if (ff.par[1].x is not None):
                            tau.append(ff.par[1].x)
                
                # Set a priori relaxation time of current function
                if (len(tau) == 0):
                    f.par[1].x = 100
                else:
                    f.par[1].x = np.min(tau) / 10

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(f, m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets attributes yc and A of the flog instance.

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Initializations
        dt = np.abs(m.r.t - f.t0)
        b = (m.r.t >= f.t0)
        amp = f.par[0].x
        tau = f.par[1].x
        da = np.log(1 + dt / tau)
        
        # Predicted observations
        f.yc = b * amp * da
        
        # Design matrix
        f.A = []
        if not(f.par[0].fixed):
            f.A.append(b * da)
        if not(f.par[1].fixed):
            f.A.append(b * -amp * dt / (1 + dt/tau) / tau**2)



# noise class
#------------
class noise:
  
    """
    Generic class for noise components adjusted to time series

    A noise instance is initialized by:
    
        n = noise()

    Once initialized, each noise instance has the following attributes:

        dt   : Original noise sampling
        per  : Period of possible modulating sine wave
        flow : Keyword indicating how the noise is assumed to propagate
        tn   : Dates of the noise's beginning and end (first and final epochs if set to None)
        par  : List of param instances (initialized to [])
        c    : Covariance vector (in case of stationary noise - initialized to None)
        dc   : Partial derivatives of c wrt noise parameters (initialized to None)
        h    : Coefficients of MA representation (in case of non-stationary noise - initialized to None)
        dh   : Partial derivatives of h wrt noise parameters (initialized to None)
        Q    : Covariance matrix or its diagonal (initialized to None)
        dQ   : Partial derivatives of Q wrt noise parameters (initialized to None)
        P    : Power spectral density (initialized to None)
        dP   : Partial derivatives of P wrt noise parameters (initialized to None)
        xi   : Estimated noise values (initialized to None)
        Qxi  : Covariance matrix of estimated noise values (initialized to None)
        sxi  : Formal errors of estimated noise values (initialized to None)
        
    Each noise instance has the following methods:
    
        get_dt()    : Get original noise sampling
        get_dates() : Get dates of original noise (covering the whole time series at original noise sampling)
        set_cov()   : Set covariance matrix [and its partial derivatives]
        set_psd()   : Set power spectral density [and its partial derivatives]
        
    """

    # Initialize a noise instance
    #----------------------------
    def __init__(n, dt=None, per=None, flow=None, tn=None):
      
        """
        Initialize a noise instance

        Returns
        -------
        n : noise instance
        
        Parameters
        ----------
        dt : float, optional
            Original noise sampling
        
        """

        n.dt = dt
        n.per = per
        n.flow = flow
        n.tn = tn
        n.par = []
        n.c = None
        n.dc = None
        n.h = None
        n.dh = None
        n.Q = None
        n.dQ = None
        n.P = None
        n.dP = None
        n.xi = None
        n.Qxi = None
        n.sxi = None

    # Get original noise sampling
    #----------------------------
    def get_dt(n, m):

        """
        Get original noise sampling

        Returns
        -------
        dt : float
            Original noise sampling

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        if (n.dt is not None):
            return n.dt
        else:
            return np.min(m.r.T)

    # Get dates of original noise (covering the whole time series at original noise sampling)
    #----------------------------------------------------------------------------------------
    def get_dates(n, m):

        """
        Get dates of original noise (covering the whole time series at original noise sampling)

        Returns
        -------
        tf : array
            Dates of original noise

        Parameters
        ----------
        m : model instance
            The parent model
            
        """

        # Shortcuts to time series dates and integration interval[s]
        t = m.r.t
        T = m.r.T
        
        # Noise sampling
        dt = n.get_dt(m)
        
        # Date of the noise's beginning and end
        if n.tn is None :
            n.tn = [t[0],t[-1]]
        if np.isscalar(n.tn) :
            n.tn = [n.tn,t[-1]]
        elif len(n.tn) == 1 :
            n.tn = [n.tn[0],t[-1]]
                
        # Dates of original noise
        if np.isscalar(T):
            tf = np.arange(n.tn[0]-(T-dt)/2, n.tn[-1]+T/2, dt)
        else:
            tf = np.arange(n.tn[0]-T[0]/2+dt/2, n.tn[-1]+T[-1]/2, dt)
        
        return tf

    # Set covariance matrix [and its partial derivatives]
    #----------------------------------------------------
    def set_cov(n, m, set_dcov=False):

        """
        Set covariance matrix [and its partial derivatives]

        set_cov() does not return anything, but sets attributes Q [and dQ] of the noise instance.
        
        Parameters
        ----------
        m : model instance
            The parent model
        set_dcov : bool, optional
            Whether to compute partial derivatives of Q wrt unknown noise parameters
            
        """  
        
        # Shortcuts to time series dates and integration interval[s]
        t = m.r.t
        T = m.r.T
        
        # Original noise sampling and dates
        dt = n.get_dt(m)
        tf = n.get_dates(m)
        nf = len(tf)
        
        # Set either covariance vector or MA coefficients of original noise
        if (n.flow == 'stationary'):
            n.set_c(m, set_dc=set_dcov)                        
        else:
            n.set_h(m, set_dh=set_dcov)

        # Initialize partial derivatives of covariance matrix
        if (set_dcov):
            n.dQ = []
        else:
            n.dQ = None
        
        # Simple case : T is a constant and a multiple of dt
        #---------------------------------------------------
        if np.isscalar(T) and (T/dt).is_integer():
            
            # Dates of averaged noise
            td = np.arange(n.tn[0],n.tn[-1]+T, T)
            nd = len(td)

            # Averaging factor
            d = round(T/dt)
            
            # If averaging of original noise is needed
            if (d > 1):
                
                # Case of stationary noise
                if (n.flow == 'stationary'):
                    
                    # Covariance vector of averaged noise
                    w = signal.triang(2*d-1) / d
                    cd = signal.convolve(np.hstack((n.c[d-1:0:-1], n.c)), w, mode='valid')[0::d]                        
                    
                    # And its partial derivatives
                    if (set_dcov):
                        dcd = []
                        for k in range(len(n.dc)):
                            dcd.append(signal.convolve(np.hstack((n.dc[k][d-1:0:-1], n.dc[k])), w, mode='valid')[0::d])
                        
                    # Covariance matrix of averaged noise
                    n.Q = linalg.toeplitz(cd)
                    
                    # And its partial derivatives
                    if (set_dcov):
                        for k in range(len(dcd)):
                            n.dQ.append(linalg.toeplitz(dcd[k]))

                # Case of 1-way or 2-way noise
                else:
                    
                    # MA coefficients of averaged noise
                    hd = signal.convolve(n.h[::-1], np.ones(d))[d-1:] / d

                    # And their partial derivatives
                    if (set_dcov):
                        dhd = []
                        for k in range(len(n.dh)):
                            dhd.append(signal.convolve(n.dh[k][::-1], np.ones(d))[d-1:] / d)
                    
                    # Covariance matrix of averaged noise
                    L = np.zeros((nd, nf))
                    for i in range(nd):
                        L[i,:(i+1)*d] = hd[-(i+1)*d:]
                    n.Q = np.dot(L, L.T)
                    if (n.flow == '2-way'):
                        n.Q = (n.Q + n.Q[::-1,::-1]) / 2
                    
                    # And its partial derivatives
                    if (set_dcov):
                        for k in range(len(dhd)):
                            dL = np.zeros((nd, nf))
                            for i in range(nd):
                                dL[i,:(i+1)*d] = dhd[k][-(i+1)*d:]
                            dLLt = np.dot(dL, L.T)
                            n.dQ.append(dLLt + dLLt.T)
                            if (n.flow == '2-way'):
                                n.dQ[-1] = (n.dQ[-1] + n.dQ[-1][::-1,::-1]) / 2
                            
            # Else (no averaging needed),
            else:
                
                # Case of stationary noise
                if (n.flow == 'stationary'):
                    
                    # Covariance matrix
                    n.Q = linalg.toeplitz(n.c)
                    
                    # And its partial derivatives
                    if (set_dcov):
                        for k in range(len(n.dc)):
                            n.dQ.append(linalg.toeplitz(n.dc[k]))

                # Case of 1-way or 2-way noise
                else:
                    
                    # Covariance matrix
                    L = linalg.toeplitz(n.h, np.zeros(nf))
                    n.Q = np.dot(L, L.T)
                    if (n.flow == '2-way'):
                        n.Q = (n.Q + n.Q[::-1,::-1]) / 2
                    
                    # And its partial derivatives
                    if (set_dcov):
                        for k in range(len(n.dh)):
                            dL = linalg.toeplitz(n.dh[k], np.zeros(nf))
                            dLLt = np.dot(dL, L.T)
                            n.dQ.append(dLLt + dLLt.T)                
                            if (n.flow == '2-way'):
                                n.dQ[-1] = (n.dQ[-1] + n.dQ[-1][::-1,::-1]) / 2

            # Modulate covariance matrix and partial derivatives with sine wave if needed
            if (n.per is not None):
                C = linalg.toeplitz(np.cos(2*pi*np.arange(nd)*T/n.per))
                n.Q = n.Q * C
                if (n.dQ is not None):
                    for k in range(len(n.dQ)):
                        n.dQ[k] = n.dQ[k] * C

            # If there are gaps in the series,
            if (m.r.n < nd):
            
                # Indices of observation dates within averaged noise dates
                ind = []
                j = 0
                for i in range(len(t)):
                    while (td[j] < t[i]):
                        j = j+1
                    ind.append(j)

                # Covariance matrix of averaged noise at observation dates
                n.Q = n.Q[np.ix_(ind,ind)]
                
                # And its partial derivatives
                if (n.dQ is not None):
                    for k in range(len(n.dQ)):
                        n.dQ[k] = n.dQ[k][np.ix_(ind,ind)]

        # Other cases
        #------------
        else:

            # Case of stationary noise
            if (n.flow == 'stationary'):

                # Covariance matrix of original noise
                n.Q = linalg.toeplitz(n.c)
                
                # And its partial derivatives
                if (set_dcov):
                    for k in range(len(n.dc)):
                        n.dQ.append(linalg.toeplitz(n.dc[k]))
                        
            # Case of 1-way or 2-way noise
            else:
                
                # Covariance matrix of original noise
                L = linalg.toeplitz(n.h, np.zeros(nf))
                n.Q = np.dot(L, L.T)
                if (n.flow == '2-way'):
                    n.Q = (n.Q + n.Q[::-1,::-1]) / 2
                
                # And its partial derivatives
                if (set_dcov):
                    for k in range(len(n.dh)):
                        dL = linalg.toeplitz(n.dh[k], np.zeros(nf))
                        dLLt = np.dot(dL, L.T)
                        n.dQ.append(dLLt + dLLt.T)
                        if (n.flow == '2-way'):
                            n.dQ[-1] = (n.dQ[-1] + n.dQ[-1][::-1,::-1]) / 2
            
            # Modulate covariance matrix and partial derivatives with sine wave if needed
            if (n.per is not None):
                C = linalg.toeplitz(np.cos(2*pi*np.arange(nf)*dt/n.per))
                n.Q = n.Q * C
                if (n.dQ is not None):
                    for k in range(len(n.dQ)):
                        n.dQ[k] = n.dQ[k] * C

            # Set original noise dates -> observation dates matrix
            A_rows = []
            A_cols = []
            A_vals = []
            j = 0
            for i in range(len(t)):
                while (tf[j] < t[i]-T[i]/2):
                    j = j+1
                k = j
                end = False
                while not(end):
                    if (k >= len(tf)):
                        end = True
                    elif (tf[k] > t[i]+T[i]/2):
                        end = True
                    else:
                        k = k+1
                ind = np.arange(j, k)
                A_rows.extend(len(ind)*[i])
                A_cols.extend(range(j, k))
                A_vals.extend(np.ones(len(ind))/len(ind))
                j = k
                
            A = sparse.csr_matrix((A_vals, (A_rows, A_cols)))
                
            # Covariance matrix of averaged noise
            n.Q = A.dot((A.dot(n.Q)).T)
            
            # And its partial derivatives
            if (n.dQ is not None):
                for k in range(len(n.dQ)):
                    n.dQ[k] = A.dot((A.dot(n.dQ[k])).T)

    # Set power spectral density [and its partial derivatives]
    #---------------------------------------------------------
    def set_psd(n, m, fr, set_dpsd=False, from_cov=False):

        """
        Set power spectral density [and its partial derivatives]

        set_psd() does not return anything, but sets attributes P [and dP] of the noise instance.

        Warning: If from_cov is True, then set_cov() must be called before calling set_psd().
        If the partial derivatives of the power spectral density additionally need to be computed
        (set_dpsd=True), then the prior call to set_cov() must be made with argument set_dcov=True.

        Parameters
        ----------
        m : model instance
            The parent model
        fr : array
            Frequencies at which to compute power
        set_dpsd : bool, optional
            Whether to compute and set partial derivatives of power spectral density
            with respect to unknown noise parameters. Default is False.
        from_cov : bool, optional
            Whether PSD should be computed from the noise covariance matrix, i.e.,
            accounting for possible irregular integration intervals and gaps in the series
            
        """
        
        # If PSD should be computed from the covariance matrix,
        if (from_cov):

            # Array of lags
            dt = n.get_dt(m)
            tau = np.arange(0, m.r.t[-1]-m.r.t[0]+dt, dt)
            
            # Number of points used to compute total covariance
            nf = m.r.n

            # Average integration factor
            f = 1
            
            # Compute total covariance [and its partial derivatives] at each lag
            q = np.zeros(len(tau))
            if (set_dpsd):
                dq = [np.zeros(len(tau)) for k in range(len(n.dQ))]
                
            for i in range(m.r.n):
                ind = ((m.r.t[i] - m.r.t[:i+1]) / dt).astype(int)
                q[ind] = q[ind] + n.Q[i,:i+1]
                if (set_dpsd):
                    for k in range(len(n.dQ)):
                        dq[k][ind] = dq[k][ind] + n.dQ[k][i,:i+1]
                        
            q[1:] = 2*q[1:]
            if (set_dpsd):
                for k in range(len(n.dQ)):
                    dq[k][1:] = 2*dq[k][1:]

        # Else (PSD is computed from the covariance vector or MA representation of original noise,
        # assuming constant integration intervals and ignoring gaps in the series),
        else:

            # Noise sampling
            dt = n.get_dt(m)

            # Array of lags
            tau = n.get_dates(m)
            tau = tau - tau[0]

            # Number of points used to compute total covariance
            nf = len(tau)

            # Average integration factor
            if np.isscalar(m.r.T):
                f = m.r.T / dt
            else:
                f = np.mean(m.r.T) / dt
                
            # Set either covariance vector or MA coefficients of original noise
            if (n.flow == 'stationary'):
                n.set_c(m, set_dc=set_dpsd)                        
            else:
                n.set_h(m, set_dh=set_dpsd)
            
            # Compute total covariance [and its partial derivatives] at each lag
            if (n.flow == 'stationary'):
                q = n.c * np.arange(nf, 0, -1)
                if (set_dpsd):
                    dq = n.dc * np.arange(nf, 0, -1)
                    
            else:
                q = signal.convolve(n.h, np.arange(1, nf+1)*n.h[::-1])[nf-1::-1]
                if (set_dpsd):
                    dq = []
                    for k in range(len(n.dh)):
                        dq.append(signal.convolve(n.dh[k], np.arange(1, nf+1)*n.h[::-1])[nf-1::-1] + signal.convolve(n.h, np.arange(1, nf+1)*n.dh[k][::-1])[nf-1::-1])
            
            q[1:] = 2*q[1:]
            if (set_dpsd):
                for k in range(len(dq)):
                    dq[k][1:] = 2*dq[k][1:]

            # Modulate total covariance with sine wave if needed
            if (n.per is not None):
                q = q * np.cos(2*pi*tau/n.per)
                if (set_dpsd):
                    for k in range(len(dq)):
                        dq[k] = dq[k] * np.cos(2*pi*tau/n.per)                

        # Compute PSD
        n.P = trig_sum(tau, q/(f*nf), fr[1]-fr[0], len(fr), f0=fr[0], Mfft=24)[1]
        
        # And its partial derivatives
        if (set_dpsd):
            n.dP = []
            for k in range(len(dq)):
                n.dP.append(trig_sum(tau, dq[k]/nf, fr[1]-fr[0], len(fr), f0=fr[0], Mfft=24)[1])
        else:
            n.dP = None

# wn class
#---------
class wn(noise):

    """
    Sub-class of the noise class for homogeneous white noise

    A wn instance is initialized by:
    
        n = wn()

    A wn instance inherits the attributes and methods from a noise instance.
    
    Each wn instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_cov() : Set variance vector [and its partial derivatives]
                    (overrides default noise.set_cov() method)
        set_psd() : Set power spectral density [and its partial derivatives]
                    (overrides default noise.set_psd() method)

    """

    # Initialize a wn instance
    #-------------------------
    def __init__(n, dt=None, s2=None, fix_s2=False, yunit='m'):

        """
        Initialize a wn instance

        Returns
        -------
        n : wn instance
        
        Parameters
        ----------
        dt : float, optional
            Noise sampling
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        yunit : str, optional
            Time series unit. Default is 'm'.
        """
        
        super().__init__(dt=dt)
        n.par.append(scale_param(type='WN variance factor', x=s2, fixed=fix_s2, unit=yunit+'^2'))

    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(n, m, v0):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
        v0 : float
            A priori variance
            
        """
        
        # Set a priori variance factor if needed
        if (n.par[0].x is None):
            n.par[0].x = 1
            n.set_cov(m)
            n.par[0].x = v0 / np.mean(n.Q)

    # Set variance vector [and its partial derivatives]
    #--------------------------------------------------
    def set_cov(n, m, set_dcov=False):

        """
        Set variance vector [and its partial derivatives]
        This method overrides the default noise.set_cov() method.

        set_cov() does not return anything, but sets attributes Q [and dQ] of the wn instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dcov : bool, optional
            Whether to compute partial derivatives of Q wrt unknown parameters
            
        """
        
        # Noise sampling
        dt = n.get_dt(m)
        
        # Variance factor
        s2 = n.par[0].x
        
        # Variance vector
        if np.isscalar(m.r.T):
            n.Q = s2 * dt/m.r.T * np.ones(m.r.n)
        else:
            n.Q = s2 * dt/m.r.T
        
        # Initialize n.dQ
        if (set_dcov):
            n.dQ = []
        else:
            n.dQ = None

        # If needed, set partial derivative of variance vector wrt variance factor
        if (set_dcov) and not(n.par[0].fixed):
            n.dQ.append(n.Q/s2)

    # Set power spectral density [and its partial derivatives]
    #---------------------------------------------------------
    def set_psd(n, m, fr, set_dpsd=False, from_cov=False):

        """
        Set power spectral density [and its partial derivatives]
        This method overrides the default noise.set_psd() method.

        set_cov() does not return anything, but sets attributes P [and dP] of the wn instance.

        Parameters
        ----------
        m : model instance
            The parent model
        fr : array
            Frequencies at which to compute power
        set_dpsd : bool, optional
            Whether to compute partial derivatives of P wrt unknown parameters
        from_cov : bool, optional
            Dummy argument necessary for consistency with noise.set_psd() but not used
            
        """

        # Noise sampling
        dt = n.get_dt(m)
        
        # Variance factor
        s2 = n.par[0].x

        # Power spectral density
        n.P = np.mean(s2 * dt/m.r.T) * np.ones(len(m.fr))

        # Initialize n.dP
        if (set_dpsd):
            n.dP = []
        else:
            n.dP = None

        # If needed, set partial derivative of PSD wrt variance factor
        if (set_dpsd) and not(n.par[0].fixed):
            n.dP.append(n.P/s2)



# vw class
#---------
class vw(noise):

    """
    Sub-class of the noise class for variable white noise

    A vw instance is initialized by:
    
        n = vw()

    A vw instance inherits the attributes and methods from a noise instance.
    
    Each vw instance additionally has the following methods:

        set_x0()  : Set default a priori values for unknown parameters
        set_cov() : Set variance vector [and its partial derivatives]
                    (overrides default noise.set_cov() method)
        set_psd() : Set power spectral density [and its partial derivatives]
                    (overrides default noise.set_psd() method)

    """

    # Initialize a vw instance
    #-------------------------
    def __init__(n, s2=None, fix_s2=False):

        """
        Initialize a vw instance

        Returns
        -------
        n : vw instance
        
        Parameters
        ----------
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.

        """
        
        super().__init__()
        n.par.append(scale_param(type='VW variance factor', x=s2, fixed=fix_s2))

    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(n, m, v0):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
        v0 : float
            A priori variance
            
        """
        
        # Set a priori variance factor if needed
        if (n.par[0].x is None):
            n.par[0].x = v0 / np.mean(m.r.Q)

    # Compute variance vector [and its partial derivatives]
    #------------------------------------------------------
    def set_cov(n, m, set_dcov=False):

        """
        Compute variance vector [and its partial derivatives]

        set_cov() does not return anything, but sets attributes Q [and dQ] of the vw instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dcov : bool, optional
            Whether to compute partial derivatives of Q wrt unknown parameters
            
        """
        
        # Variance factor
        s2 = n.par[0].x
        
        # Covariance vector
        n.Q = s2 * m.r.Q
        
        # Initialize n.dQ
        if (set_dcov):
            n.dQ = []
        else:
            n.dQ = None

        # If needed, set partial derivative of variance vector wrt variance factor
        if (set_dcov) and not(n.par[0].fixed):
            n.dQ.append(m.r.Q)

    # Set power spectral density [and its partial derivatives]
    #---------------------------------------------------------
    def set_psd(n, m, fr, set_dpsd=False, from_cov=False):

        """
        Set power spectral density [and its partial derivatives]
        This method overrides the default noise.set_psd() method.

        set_cov() does not return anything, but sets attributes P [and dP] of the vw instance.

        Parameters
        ----------
        m : model instance
            The parent model
        fr : array
            Frequencies at which to compute power
        set_dpsd : bool, optional
            Whether to compute partial derivatives of P wrt unknown parameters
        from_cov : bool, optional
            Dummy argument necessary for consistency with noise.set_psd() but not used
            
        """
        
        # Variance factor
        s2 = n.par[0].x

        # Power spectral density
        n.P = np.mean(s2 * m.r.Q) * np.ones(len(m.fr))

        # Initialize n.dP
        if (set_dpsd):
            n.dP = []
        else:
            n.dP = None

        # If needed, set partial derivative of PSD wrt variance factor
        if (set_dpsd) and not(n.par[0].fixed):
            n.dP.append(n.P/s2)



# ar1 class
#----------
class ar1(noise):

    """
    Sub-class of the noise class for AR(1) processes

    An ar1 instance is initialized by:
    
        n = ar1()

    An ar1 instance inherits the attributes and methods from a noise instance.
    
    Each ar1 instance additionally has the following methods:

        set_x0() : Set default a priori values for unknown parameters
        set_c()  : Compute covariance vector [and its partial derivatives] (in case of stationary noise)
        set_h()  : Compute coefficients of MA representation [and their partial derivatives] (in case of stationary noise)

    """

    # Initialize an ar1 instance
    #---------------------------
    def __init__(n, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, tau=None, fix_tau=False, tunit='d', yunit='m'):

        """
        Initialize an ar1 instance

        Returns
        -------
        n : ar1 instance
        
        Parameters
        ----------
        dt : float, optional
            Noise sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            - 'stationary' means that the noise is assumed to have started infinitely long ago.
            Default if '2-way'.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        tau : float, optional
            [A priori] correlation time
        fix_tau : bool, optional
            Whether provided correlation time should be fixed (or only used as a priori).
            Default is False.
        tunit : str, optional
            Time unit. Default is 'm'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """
        
        super().__init__(dt=dt, per=per, flow=flow)
        n.par.append(scale_param(type='AR(1) variance factor', x=s2, fixed=fix_s2, unit=yunit+'^2'))
        n.par.append(scale_param(type='AR(1) correlation time', x=tau, fixed=fix_tau, unit=tunit))
        
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(n, m, v0):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
        v0 : float
            A priori variance
            
        """
        
        # Noise sampling
        dt = n.get_dt(m)
        
        # Set a priori correlation time if needed
        if (n.par[1].x is None):
            if (n.per is None):
                n.par[1].x = -dt/log(0.9)
            else:
                n.par[1].x = 5*n.per
        
        # Set a priori variance factor if needed
        if (n.par[0].x is None):
            n.par[0].x = 1
            n.set_cov(m)
            v = (np.trace(n.Q)-np.sum(n.Q)/m.r.n) / (m.r.n-1)
            n.par[0].x = v0 / v
            
    # Compute covariance vector [and its partial derivatives]
    #--------------------------------------------------------
    def set_c(n, m, set_dc=False):

        """
        Compute covariance vector [and its partial derivatives]

        set_c() does not return anything, but sets attributes c [and dc] of the ar1 instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dc : bool, optional
            Whether to compute partial derivatives of c wrt unknown parameters
            
        """

        # Original noise sampling and dates
        dt = n.get_dt(m)
        tf = n.get_dates(m)
        nf = len(tf)
        
        # Get noise parameters
        s2 = n.par[0].x
        tau = n.par[1].x
        phi = exp(-dt/tau)
        phik = phi**np.arange(nf)
        phi2 = np.abs(phik[2])
        
        # Covariance vector
        n.c = s2 * phik / (1 - phi2)

        # Initialize partial derivatives
        if (set_dc):
            n.dc = []
        else:
            n.dc = None
            
        # Partial derivative wrt variance factor
        if (set_dc) and not(n.par[0].fixed):
            n.dc.append(n.c/s2)
            
        # Partial derivative wrt correlation time
        if (set_dc) and not(n.par[1].fixed):
            n.dc.append(dt/tau**2 * (np.arange(nf) + 2*phi2/(1-phi2)) * n.c)

    # Compute coefficients of MA representation [and their partial derivatives]
    #-------------------------------------------------------------------------
    def set_h(n, m, set_dh=False):

        """
        Compute coefficients of MA representation [and their partial derivatives]

        set_h() does not return anything, but sets attributes h [and dh] of the ar1 instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dh : bool, optional
            Whether to compute partial derivatives of h wrt unknown parameters
            
        """

        # Original noise sampling and dates
        dt = n.get_dt(m)
        tf = n.get_dates(m)
        nf = len(tf)
        
        # Get noise parameters
        s2 = n.par[0].x
        tau = n.par[1].x
        phi = exp(-dt/tau)
        phik = phi**np.arange(nf)
        # phi2 = np.abs(phik[2])
        
        # Coefficients of MA representation
        n.h = sqrt(s2) * phik

        # Initialize partial derivatives
        if (set_dh):
            n.dh = []
        else:
            n.dh = None
            
        # Partial derivative wrt variance factor
        if (set_dh) and not(n.par[0].fixed):
            n.dh.append(n.h / (2*s2))
            
        # Partial derivative wrt correlation time
        if (set_dh) and not(n.par[1].fixed):
            n.dh.append(dt/tau**2 * np.arange(nf)*n.h)



## ma1 class
##----------
#class ma1(noise):

    #"""
    #Sub-class of the noise class for MA(1) processes

    #An ma1 instance is initialized by:
    
        #n = ma1()

    #An ma1 instance inherits the attributes and methods from a noise instance.
    
    #Each ma1 instance additionally has the following methods:

        #set_x0() : Set default a priori values for unknown parameters
        #set_c()  : Compute covariance vector [and its partial derivatives] (in case of stationary noise)
        #set_h()  : Compute coefficients of MA representation [and their partial derivatives] (in case of stationary noise)

    #"""

    ## Initialize an ma1 instance
    ##---------------------------
    #def __init__(n, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, theta=None, fix_theta=False, tunit='d', yunit='m'):

        #"""
        #Initialize an ma1 instance

        #Returns
        #-------
        #n : ma1 instance
        
        #Parameters
        #----------
        #dt : float, optional
            #Noise sampling
        #per : float, optional
            #Period of possible modulating sine wave
        #flow : str, optional
            #Keyword indicating how the noise is assumed to propagate:
            #- '1-way' means that the noise is assumed to start at the beginning of the series
              #and to propagate forward.
            #- '2-way' means that half of the noise is assumed to start at the beginning of the series
              #and to propagate forward, and half of the noise is assumed to start at the end
              #of the series and to propagate backward.
            #- 'stationary' means that the noise is assumed to have started infinitely long ago.
            #Default if '2-way'.
        #s2 : float, optional
            #[A priori] variance factor
        #fix_s2 : bool, optional
            #Whether provided variance factor should be fixed (or only used as a priori).
            #Default is False.
        #theta : float, optional
            #[A priori] MA coefficient
        #fix_theta : bool, optional
            #Whether provided MA coefficient should be fixed (or only used as a priori).
            #Default is False.
        #tunit : str, optional
            #Time unit. Default is 'm'.
        #yunit : str, optional
            #Time series unit. Default is 'm'.

        #"""
        
        #super().__init__(dt=dt, per=per, flow=flow)
        #n.par.append(scale_param(type='MA(1) variance factor', x=s2, fixed=fix_s2, unit=yunit+'^2'))
        #n.par.append(tanh_param(type='MA(1) coefficient', x=theta, fixed=fix_theta, unit=tunit))
        
    ## Set default a priori values for unknown parameters
    ##---------------------------------------------------
    #def set_x0(n, m, v0):

        #"""
        #Set default a priori values for unknown parameters

        #Parameters
        #----------
        #m : model instance
            #The parent model
        #v0 : float
            #A priori variance
            
        #"""
        
        ## Noise sampling
        #dt = n.get_dt(m)
        
        ## Set a priori MA coefficient if needed
        ## Here's where I stopped modifications on Feb 17, 2023.
        #if (n.par[1].x is None):
            #if (n.per is None):
                #n.par[1].x = -dt/log(0.9)
            #else:
                #n.par[1].x = 5*n.per
        
        ## Set a priori variance factor if needed
        #if (n.par[0].x is None):
            #n.par[0].x = 1
            #n.set_cov(m)
            #v = (np.trace(n.Q)-np.sum(n.Q)/m.r.n) / (m.r.n-1)
            #n.par[0].x = v0 / v
            
    ## Compute covariance vector [and its partial derivatives]
    ##--------------------------------------------------------
    #def set_c(n, m, set_dc=False):

        #"""
        #Compute covariance vector [and its partial derivatives]

        #set_c() does not return anything, but sets attributes c [and dc] of the ar1 instance.

        #Parameters
        #----------
        #m : model instance
            #The parent model
        #set_dc : bool, optional
            #Whether to compute partial derivatives of c wrt unknown parameters
            
        #"""

        ## Original noise sampling and dates
        #dt = n.get_dt(m)
        #tf = n.get_dates(m)
        #nf = len(tf)
        
        ## Get noise parameters
        #s2 = n.par[0].x
        #tau = n.par[1].x
        #phi = exp(-dt/tau)
        #phik = phi**np.arange(nf)
        #phi2 = np.abs(phik[2])
        
        ## Covariance vector
        #n.c = s2 * phik / (1 - phi2)

        ## Initialize partial derivatives
        #if (set_dc):
            #n.dc = []
        #else:
            #n.dc = None
            
        ## Partial derivative wrt variance factor
        #if (set_dc) and not(n.par[0].fixed):
            #n.dc.append(n.c/s2)
            
        ## Partial derivative wrt correlation time
        #if (set_dc) and not(n.par[1].fixed):
            #n.dc.append(dt/tau**2 * (np.arange(nf) + 2*phi2/(1-phi2)) * n.c)

    ## Compute coefficients of MA representation [and their partial derivatives]
    ##-------------------------------------------------------------------------
    #def set_h(n, m, set_dh=False):

        #"""
        #Compute coefficients of MA representation [and their partial derivatives]

        #set_h() does not return anything, but sets attributes h [and dh] of the ar1 instance.

        #Parameters
        #----------
        #m : model instance
            #The parent model
        #set_dh : bool, optional
            #Whether to compute partial derivatives of h wrt unknown parameters
            
        #"""

        ## Original noise sampling and dates
        #dt = n.get_dt(m)
        #tf = n.get_dates(m)
        #nf = len(tf)
        
        ## Get noise parameters
        #s2 = n.par[0].x
        #tau = n.par[1].x
        #phi = exp(-dt/tau)
        #phik = phi**np.arange(nf)
        ## phi2 = np.abs(phik[2])
        
        ## Coefficients of MA representation
        #n.h = sqrt(s2) * phik

        ## Initialize partial derivatives
        #if (set_dh):
            #n.dh = []
        #else:
            #n.dh = None
            
        ## Partial derivative wrt variance factor
        #if (set_dh) and not(n.par[0].fixed):
            #n.dh.append(n.h / (2*s2))
            
        ## Partial derivative wrt correlation time
        #if (set_dh) and not(n.par[1].fixed):
            #n.dh.append(dt/tau**2 * np.arange(nf)*n.h)



# pl class
#---------
class pl(noise):

    """
    Sub-class of the noise class for power-law noise

    A pl instance is initialized by:
    
        n = pl()

    A pl instance inherits the attributes and methods from a noise instance.
    
    Each pl instance additionally has the following methods:

        set_x0() : Set default a priori values for unknown parameters
        set_h()  : Compute coefficients of MA representation [and their partial derivatives]

    """

    # Initialize a pl instance
    #-------------------------
    def __init__(n, dt=None, per=None, flow='2-way', tn=None, s2=None, fix_s2=False, a=None, fix_a=False, tunit='d', yunit='m'):

        """
        Initialize a pl instance

        Returns
        -------
        n : pl instance
        
        Parameters
        ----------
        dt : float, optional
            Noise sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            - 'stationary' means that the noise is assumed to have started infinitely long ago.
            Default if '2-way'.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.        
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        a : float, optional
            [A priori] spectral index
        fix_a : bool, optional
            Whether provided spectral index should be fixed (or only used as a priori).
            Default is False.
        tunit : str, optional
            Time unit. Default is 'm'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """
        
        super().__init__(dt=dt, per=per, flow=flow, tn=tn)
        n.par.append(scale_param(type='PL variance factor', x=s2, fixed=fix_s2, unit=yunit+'^2'))
        n.par.append(pl_index(type='PL spectral index', x=a, fixed=fix_a))
        
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(n, m, v0):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
        v0 : float
            A priori variance
            
        """
        
        # Noise sampling
        dt = n.get_dt(m)
        
        # Set a priori spectral index if needed
        if (n.par[1].x is None):
            n.par[1].x = 1
        
        # Set a priori variance factor if needed
        if (n.par[0].x is None):
            n.par[0].x = 1
            n.set_cov(m)
            v = (np.trace(n.Q)-np.sum(n.Q)/m.r.n) / (m.r.n-1)
            n.par[0].x = v0 / v

    # Compute coefficients of MA representation [and their partial derivatives]
    #-------------------------------------------------------------------------
    def set_h(n, m, set_dh=False):

        """
        Compute coefficients of MA representation [and their partial derivatives]

        set_h() does not return anything, but sets attributes h [and dh] of the pl instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dh : bool, optional
            Whether to compute partial derivatives of h wrt unknown parameters
            
        """

        # Original noise sampling and dates
        dt = n.get_dt(m)
        tf = n.get_dates(m)
        nf = len(tf)
        
        # Get noise parameters
        s2 = n.par[0].x
        a = n.par[1].x
        
        # Coefficients of MA representation
        n.h = np.zeros(nf)
        n.h[0] = sqrt(s2)
        for i in range(1, nf):
            n.h[i] = (a/2+i-1)/i * n.h[i-1]

        # Initialize partial derivatives
        if (set_dh):
            n.dh = []
        else:
            n.dh = None
            
        # Partial derivative wrt variance factor
        if (set_dh) and not(n.par[0].fixed):
            n.dh.append(n.h / (2*s2))
            
        # Partial derivative wrt a
        if (set_dh) and not(n.par[1].fixed):
            n.dh.append(np.zeros(nf))
            for i in range(1, nf):
                n.dh[-1][i] = (a/2+i-1)/i * n.dh[-1][i-1] + n.h[i-1] / (2*i)



# ggm class
#----------
class ggm(noise):

    """
    Sub-class of the noise class for GGM processes

    A ggm instance is initialized by:
    
        n = ggm()

    A ggm instance inherits the attributes and methods from a noise instance.
    
    Each ggm instance additionally has the following methods:

        set_x0() : Set default a priori values for unknown parameters
        set_h()  : Compute coefficients of MA representation [and their partial derivatives]

    """

    # Initialize a ggm instance
    #-------------------------
    def __init__(n, dt=None, per=None, flow='2-way', tn=None, s2=None, fix_s2=False, a=None, fix_a=False, tau=None, fix_tau=False, tunit='d', yunit='m'):

        """
        Initialize a ggm instance

        Returns
        -------
        n : ggm instance
        
        Parameters
        ----------
        dt : float, optional
            Noise sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            - 'stationary' means that the noise is assumed to have started infinitely long ago.
            Default if '2-way'.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        a : float, optional
            [A priori] spectral index
        fix_a : bool, optional
            Whether provided spectral index should be fixed (or only used as a priori).
            Default is False.
        tau : float, optional
            [A priori] correlation time
        fix_tau : bool, optional
            Whether provided correlation time should be fixed (or only used as a priori).
            Default is False.
        tunit : str, optional
            Time unit. Default is 'm'.
        yunit : str, optional
            Time series unit. Default is 'm'.

        """
        
        super().__init__(dt=dt, per=per, flow=flow, tn=tn)
        n.par.append(scale_param(type='GGM variance factor', x=s2, fixed=fix_s2, unit=yunit+'^2'))
        n.par.append(pl_index(type='GGM spectral index', x=a, fixed=fix_a))
        n.par.append(scale_param(type='GGM correlation time', x=tau, fixed=fix_tau, unit=tunit))
        
    # Set default a priori values for unknown parameters
    #---------------------------------------------------
    def set_x0(n, m, v0):

        """
        Set default a priori values for unknown parameters

        Parameters
        ----------
        m : model instance
            The parent model
        v0 : float
            A priori variance
            
        """
        
        # Noise sampling
        dt = n.get_dt(m)
        
        # Set a priori spectral index if needed
        if (n.par[1].x is None):
            n.par[1].x = 1
        
        # Set a priori correlation time if needed
        if (n.par[2].x is None):
            if (n.per is None):
                n.par[2].x = -dt/log(0.9)
            else:
                n.par[2].x = 5*n.per

        # Set a priori variance factor if needed
        if (n.par[0].x is None):
            n.par[0].x = 1
            n.set_cov(m)
            v = (np.trace(n.Q)-np.sum(n.Q)/m.r.n) / (m.r.n-1)
            n.par[0].x = v0 / v

    # Compute coefficients of MA representation [and their partial derivatives]
    #-------------------------------------------------------------------------
    def set_h(n, m, set_dh=False):

        """
        Compute coefficients of MA representation [and their partial derivatives]

        set_h() does not return anything, but sets attributes h [and dh] of the ggm instance.

        Parameters
        ----------
        m : model instance
            The parent model
        set_dh : bool, optional
            Whether to compute partial derivatives of h wrt unknown parameters
            
        """

        # Original noise sampling and dates
        dt = n.get_dt(m)
        tf = n.get_dates(m)
        nf = len(tf)
        
        # Get noise parameters
        s2 = n.par[0].x
        a = n.par[1].x
        tau = n.par[2].x
        phi = exp(-dt/tau)
        
        # Coefficients of MA representation
        n.h = np.zeros(nf)
        n.h[0] = sqrt(s2)
        for i in range(1, nf):
            n.h[i] = phi * (a/2+i-1)/i * n.h[i-1]

        # Initialize partial derivatives
        if (set_dh):
            n.dh = []
        else:
            n.dh = None
            
        # Partial derivative wrt variance factor
        if (set_dh) and not(n.par[0].fixed):
            n.dh.append(n.h / (2*s2))
            
        # Partial derivative wrt a
        if (set_dh) and not(n.par[1].fixed):
            n.dh.append(np.zeros(nf))
            for i in range(1, nf):
                n.dh[-1][i] = phi * ((a/2+i-1)/i * n.dh[-1][i-1] + n.h[i-1] / (2*i))

        # Partial derivative wrt tau
        if (set_dh) and not(n.par[2].fixed):
            n.dh.append(np.zeros(nf))
            for i in range(1, nf):
                n.dh[-1][i] = (a/2+i-1)/i * (phi*n.dh[-1][i-1] + n.h[i-1])
            n.dh[-1] = dt/tau**2*phi * n.dh[-1]



# model class
#------------
class model:
  
    """
    Class for deterministic+noise models adjusted to time series

    A model instance is initialized in one of the following ways:
    
        m = model()
        m = model.from_solns()
        m = model.load()
        
    Once initialized, each model instance has the following attribute:

        nd : Number of dimensions
        
    If nd > 1, then the model instance has the following attributes
    
        r  : The corresponding time series
        t0 : The origin of time
        md : List of 1-dimensional model instances
        
    A 1-dimensional model instance (either m if nd = 1, or any m.md[i] if nd > 1)
    has the following attributes

        r  : The corresponding time series
        f  : List of functions
        n  : List of noise components
        
    Once (a priori or estimated) values have been assigned to the unknown
    deterministic parameters of a 1-dimensional model, its following attributes
    (initialized to None) may be set by calling set_oeq():

        yc : Computed observations
        A  : Design matrix

    Once (a priori or estimated) values have been assigned to the unknown
    noise parameters of a 1-dimensional model, its following attributes
    (initialized to None) may be set by calling set_cov():

        Q  : Covariance matrix
        L  : Cholesky factorization of Q
        P  : Inverse of Q
        dQ : Log-determinant of Q
        
    Once (a priori or estimated) values have been assigned to the unknown
    noise parameters of a 1-dimensional model, its following attributes
    (initialized to None) may be set by calling set_psd():

        fr  : Frequency range
        pn  : Theoretical power spectral density of noise model
        Qpn : Covariance matrix of noise model PSD
        spn : Formal errors of noise model PSD
        pv  : Power spectral density of residuals

    The following attributes (initialized to None) of a 1-dimensional model instance 
    may be set by calling fit():
    
        nx    : Number of deterministic parameters
        nb    : Number of noise parameters
        dQ    : Log-determinant of Q
        dN    : Log-determinant of normal matrix of deterministic parameters
        dH    : Log-determinant of normal matrix of noise parameters
        s2    : Global variance factor
        y2x   : Weighted Least-Squares Estimator (WLSE) transition matrix
        Qx    : Covariance matrix of deterministic parameters
        Qb    : Covariance matrix of noise parameters
        Qc    : Covariance matrix of predicted observations
        sc    : Formal errors of predicted observations
        v     : Residuals
        Pv    : Product of observation weight matrix with residuals
        Qv    : Covariance matrix of residuals
        sv    : Formal errors of residuals
        vn    : Normalized residuals
        wrms  : WRMS of residuals
        logl  : Log-likelihood
        loglr : Restricted log-likelihood
        bic   : -BIC/2
        bicr  : -(restricted BIC)/2
        E     : Evidence
        Er    : Restriced evidence
        
    Each model instance has the following methods:
    
        add_function() : Add custom function to model
        add_polynom()  : Add polynomial function to model
        add_sine()     : Add sine wave function to model
        add_exp()      : Add exponential function to model
        add_log()      : Add logarithmic function to model
        add_psd()      : Add exp and log functions to model based on a SINEX file containing post-seismic deformation models
        add_noise()    : Add custom noise to model
        add_wn()       : Add homogeneous white noise to model
        add_vw()       : Add variable white noise to model
        add_ar1()      : Add AR(1) process to model
        add_pl()       : Add power-law noise to model
        add_fn()       : Add flicker noise to model
        add_rw()       : Add random walk to model
        add_ggm()      : Add GGM process to model
        add_jumps()    : Add jumps to specified polynomial and/or sine wave functions of model
        set_x0()       : Set default a priori values for unknown deterministic parameters
        set_x()        : Set values of unknown deterministic parameters
        get_x()        : Get values of unknown deterministic parameters
        set_xr()       : Set values of unknown deterministic parameters given reparameterized parameters
        get_xr()       : Get values of reparameterized unknown deterministic parameters
        set_sigx()     : Set formal errors of deterministic parameters
        dx_dxr()       : Compute partial derivatives of deterministic parameters wrt reparameterized deterministic parameters
        set_b0()       : Set default a priori values for unknown noise parameters
        set_b()        : Set values of unknown noise parameters
        get_b()        : Get values of unknown noise parameters
        set_br()       : Set values of unknown noise parameters given reparameterized parameters
        get_br()       : Get values of reparameterized unknown noise parameters
        set_sigb()     : Set formal errors of noise parameters
        db_dbr()       : Compute partial derivatives of noise parameters wrt reparameterized noise parameters
        set_oeq()      : Compute predicted observations and design matrix
        set_cov()      : Compute covariance matrix
        set_psd()      : Compute power spectral density of noise model and of residuals
        set_xi()       : Estimate individual noise components
        simulate()     : Simulate time series values
        fitx()         : Fit deterministic model with fixed covariance matrix
        fit()          : Fit deterministic + noise model
        fit_iter()     : Fit deterministic + noise model and iteratively remove outliers
        plot_fit()     : Plot time series + deterministic model
        plot_res()     : Plot fit residuals
        plot_normres() : Plot normalized residuals
        plot_psd()     : Plot PSD of residuals and of noise model
        plot_all()     : plot_fit(), plot_res(), plot_normres() & plot_psd()
        __str__()      : Print fit statistics and parameters
        dump()         : Dump model instance into pickle file
        
    """

    # Initialize a model instance
    #----------------------------
    def __init__(m, r, t0=None, deg=None, per=None, noise=None, tn=None):
      
        """
        Initialize a model instance

        Parameters
        ----------
        r : ts instance
            The corresponding time series
        t0 : float, optional
            The origin of time. Set by default to mean(r.t).
        deg : list, optional
            List of polynomial degrees
        per : list, optional
            List of sine wave periods
        noise : list, optional
            List of noise types
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
        Returns
        -------
        m : model instance
                    
        """
        
        # Set time series, origin of time and number of dimensions
        m.r = r
        if (t0 is not None):
            m.t0 = t0
        else:
            m.t0 = np.mean(r.t)
        m.nd = r.nd
        
        # If time series and model are 1-dimensional,
        if (m.nd == 1):

            # Initialize all other attributes
            m.f = []
            m.n = []
            
            m.yc = None
            m.A = None
            
            m.Q = None
            m.L = None
            m.P = None
            m.dQ = None
            
            m.fr = None
            m.pn = None
            m.Qpn = None
            m.spn = None
            m.pv = None            
            m.nx = None
            m.nb = None
            m.dN = None
            m.dH = None
            m.s2 = None
            m.y2x = None
            m.Qx = None
            m.Qb = None
            m.Qc = None
            m.sc = None
            m.v = None
            m.Pv = None
            m.sv = None
            m.Qv = None
            m.vn = None
            m.wrms = None
            m.logl = None
            m.loglr = None
            m.bic = None
            m.bicr = None
            m.E = None
            m.Er = None
            
        # Else (time series and model have more than one dimension),
        else:
            
            # Fill in m.md with a list of 1-dimensional models
            m.md = [model(r[d], t0=m.t0) for d in range(m.nd)]

        # Add polynomials
        if (deg is not None):
            for d in deg:
                m.add_polynom(d)

        # Add sine waves
        if (per is not None):
            for p in per:
                m.add_sine(p)
        
        # Add noise components
        if (noise is not None):
            for n in noise:
                if (n == 'wn'):
                    m.add_wn()
                elif (n == 'vw'):
                    m.add_vw()
                elif (n == 'ar1'):
                    m.add_ar1()
                elif (n == 'pl'):
                    m.add_pl(tn=tn)
                elif (n == 'fn'):
                    m.add_fn(tn=tn)
                elif (n == 'rw'):
                    m.add_rw(tn=tn)
                elif (n == 'ggm'):
                    m.add_ggm(tn=tn)
    
    # Load model instance from pickle file
    #-------------------------------------
    @classmethod
    def load(self, file):

        """
        Load model instance from pickle file

        Returns
        -------
        m : model instance

        Parameters
        ----------
        file : str
            Pickle file to load

        """
    
        return pickle.load(open(file, 'rb'))

    # Get specific component of model instance
    #-----------------------------------------
    def __getitem__(m, i):
      
        """
        Get specific component of model instance

        Returns
        -------
        mi : model instance

        Parameters
        ----------
        i : int
            Component index

        """
        
        if (m.nd == 1):
            return m
        else:
            return m.md[i]
        
    # Initialize model instance from discontinuity list in (pseudo-)SINEX format
    #---------------------------------------------------------------------------
    @classmethod
    def from_solns(m, r, solns, code, pt=None, t0=None, per=None, noise=None, psd=None, fix_tau=False, fix_amp=False, dims='ENU'):
        
        """
        Create model instance from discontinuity list in (pseudo-)SINEX format
        and optionally SINEX file containing post-seismic deformation models
        
        In case the specified station is not found in the discontinuity list,
        a model instance with "constant position" and "constant velocity" is returned.        
        
        Warning: Dates in the returned model are MJDs. The dates of the time series r
        must therefore also be MJDs.

        Warning: If a SINEX file containing post-seismic deformation models is provided,
        the amplitudes of the exp/log functions in the returned model are in m.
        The time series r must therefore also be expressed in m.
        
        Parameters
        ----------
        r : ts instance
            The corresponding time series
        solns : list
            Discontinuity list (see pytrf.io.read_solns)
        code : str
            4-char station ID
        pt : str, optional
            Station PT code. Default is None.
        t0 : float, optional
            The origin of time. Set by default to mean(r.t).
        per : list, optional
            List of sine wave periods
        noise : list, optional
            List of noise types
        psd : sinex instance, optional
            sinex instance containing post-seismic deformation models. Default is None.
        fix_tau : bool, optional
            Whether relaxation times should be considered fixed. Default is False.
        fix_amp : bool, optional
            Whether amplitudes should be considered fixed. Default is False.
        dims: str, optional
            If a sinex instance with post-seismic deformation models is provided, then
            this keyword is needed to know the order of the ENH component(s) in the time
            series r. dims must thus be a combination of the letters 'E', 'N' and/or 'H',
            in the same order as those components are stored in r. Default is 'ENH'.
            
        """

        # Raise an error if a SINEX file with post-seismic deformation models is provided,
        # but there is an obvious problem with argument dims.
        if (psd is not None) and (len(dims) != r.nd):
            raise RuntimeError('Provided argument dims=\''+dims+'\' does not match time series dimension.')
        
        # Time series start and end dates
        tmin = r.t[0]
        tmax = r.t[-1]
        
        # Get index of station in discontinuity list
        i = None
        if (pt is not None):
            keys = [s.code+s.pt for s in solns]
            if (code+pt in keys):
                i = keys.index(code+pt)
        else:
            keys = [s.code for s in solns]
            if (code in keys):
                i = keys.index(code)
            
        # Initialize model instance
        m = model(r, t0=t0, deg=[0, 1], per=per, noise=noise)
        
        # If station was found in discontinuity list,
        if (i is not None):
            
            # Add position discontinuities
            for p in solns[i].P:
                if (p.end != '00:000:00000'):
                    t = date.from_tsnx(p.end).mjd
                    if (t > tmin) and (t < tmax):
                        m.add_jumps([t], deg=[0])

            # Add velocity discontinuities
            for p in solns[i].V:
                if (p.end != '00:000:00000'):
                    t = date.from_tsnx(p.end).mjd
                    if (t > tmin) and (t < tmax):
                        m.add_jumps([t], deg=[1])
        
        # If a PSD file is provided, add PSD functions to model
        if (psd is not None):
            m.add_psd(psd, code=code, pt=pt, fix_tau=fix_tau, fix_amp=fix_amp, dims=dims)
            
        return m
        
    # Add custom function to model
    #-----------------------------
    def add_function(m, f):

        """
        Add custom function to model

        Parameters
        ----------
            f : function instance
            
        """
        
        for d in range(m.nd):
            m[d].f.append(f)

    # Add polynomial function to model
    #---------------------------------
    def add_polynom(m, deg, t=[], x=None, fix_x=False):

        """
        Add polynomial function to model

        Parameters
        ----------
        deg : int
            Polynomial degree
        t : list, optional
            List of dates (dates of jumps)
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori)
            Default is False.
            
        """
        
        for d in range(m.nd):
            m[d].f.append(polynom(deg, t, x, fix_x, m.r.tunit, m.r.yunit))
        
    # Add sine wave function to model
    #--------------------------------
    def add_sine(m, per, t=[], x=None, fix_x=False):

        """
        Add sine wave function to model

        Parameters
        ----------
        per : float
            Period in days
        t : list, optional
            List of dates (dates of jumps)
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori)
            Default is False.
            
        """
        
        for d in range(m.nd):
            m[d].f.append(sine(per, t, x, fix_x, m.r.yunit))
        
    # Add Poisson function to model
    #------------------------------
    def add_poisson(m, per, deg, t=[], x=None, fix_x=False):

        """
        Add Poisson function to model

        Parameters
        ----------
        per : float
            Period in days
        deg : int
            Polynoms degree
        x : array, optional
            Parameter values. Default is None.
        fix_x : bool or array of bool, optional
            Whether the provided parameter values should be fixed (or only used as a priori)
            Default is False.            
        """

        for d in range(m.nd):
            m[d].f.append(poisson(per, deg, x, fix_x, m.r.tunit, m.r.yunit))

    # Add exponential function to model
    #----------------------------------
    def add_exp(m, t0, amp=None, tau=None, fix_amp=False, fix_tau=False):

        """
        Add exponential function to model

        Parameters
        ----------
        t0 : float
            Start date
        tau : float, optional
            Fixed relaxation time
        amp : float, optional
            Fixed amplitude
        fix_tau : bool, optional
            Whether the provided relaxation time should be fixed (or only used as a priori)
        fix_amp : bool, optional
            Whether the provided amplitude should be fixed (or only used as a priori)
            
        """
        
        for d in range(m.nd):
            m[d].f.append(fexp(t0, amp, tau, fix_amp, fix_tau, m.r.tunit, m.r.yunit))
    
    # Add logarithmic function to model
    #----------------------------------
    def add_log(m, t0, amp=None, tau=None, fix_amp=False, fix_tau=False):

        """
        Add logarithmic function to model

        Parameters
        ----------
        t0 : float
            Start date
        tau : float, optional
            Fixed relaxation time
        amp : float, optional
            Fixed amplitude
        fix_tau : bool, optional
            Whether the provided relaxation time should be fixed (or only used as a priori)
        fix_amp : bool, optional
            Whether the provided amplitude should be fixed (or only used as a priori)
            
        """
        
        for d in range(m.nd):
            m[d].f.append(flog(t0, amp, tau, fix_amp, fix_tau, m.r.tunit, m.r.yunit))
        
    # Add exp and log functions to model based on a SINEX file containing post-seismic deformation models
    #----------------------------------------------------------------------------------------------------
    def add_psd(m, psd, code, pt=None, fix_amp=False, fix_tau=False, dims='ENU'):

        """
        Add exp and log functions to model based on a SINEX file containing post-seismic deformation models

        Warning: Dates of the added functions are MJDs. The dates of the time series m.r
        must therefore also be MJDs.

        Warning: Amplitudes of the added functions are in m. The time series m.r
        must therefore also be expressed in m.

        Parameters
        ----------
        psd : sinex instance
            sinex instance containing post-seismic deformation models
        code : str
            4-char station ID
        pt : str, optional
            Station PT code. Default is None.
        fix_amp : bool, optional
            Whether amplitudes should be considered fixed. Default is False.
        fix_tau : bool, optional
            Whether relaxation times should be considered fixed. Default is False.
        dims: str, optional
            This keyword is needed to know the order of the ENH component(s) in the time
            series m.r. dims must thus be a combination of the letters 'E', 'N' and/or 'U',
            in the same order as those components are stored in m.r. Default is 'ENU'.
            
        """
        
        # Raise an error if there is an obvious problem with argument dims
        if (len(dims) != m.nd):
            raise RuntimeError('Provided argument dims=\''+dims+'\' does not match model dimension.')

        # Loop over SINEX parameters
        for i in range(psd.npar):
            p = psd.param[i]
            
            # Is this the right station?
            b = False
            if (pt is not None):
                if (p.code == code) and (p.pt == pt) and (p.type[0] == 'A'):
                    b = True
            elif (p.code == code) and (p.type[0] == 'A'):
                b = True
                
            # If yes, is this a relevant component?
            if (b):
                if (p.type[5] in dims):
                    d = dims.index(p.type[5])
                else:
                    b = False
                
            # If yes, then add corresponding function to model
            if (b):
                t = date.from_tsnx(p.tref).mjd
                amp = psd.x[i]
                tau = psd.x[i+1] * 365.25
                if (p.type[1:4] == 'EXP'):
                    m[d].add_exp(t, amp, tau, fix_amp, fix_tau)
                elif (p.type[1:4] == 'LOG'):
                    m[d].add_log(t, amp, tau, fix_amp, fix_tau)
        
    # Add custom noise to model
    #-----------------------------
    def add_noise(m, n):

        """
        Add custom noise to model

        Parameters
        ----------
            n : noise instance
            
        """
        
        for d in range(m.nd):
            m[d].n.append(n)

    # Add homogeneous white noise to model
    #-------------------------------------
    def add_wn(m, dt=None, s2=None, fix_s2=False):

        """
        Add homogeneous white noise to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
            
        """

        for d in range(m.nd):
            m[d].n.append(wn(dt=dt, s2=s2, fix_s2=fix_s2, yunit=m.r.yunit))

    # Add variable white noise to model
    #----------------------------------
    def add_vw(m, s2=None, fix_s2=False):

        """
        Add variable white noise to model

        Parameters
        ----------
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(vw(s2=s2, fix_s2=fix_s2))

    # Add AR(1) process to model
    #---------------------------
    def add_ar1(m, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, tau=None, fix_tau=False):

        """
        Add AR(1) process to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            - 'stationary' means that the noise is assumed to have started infinitely long ago.
            Default if '2-way'.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        tau : float, optional
            [A priori] correlation time
        fix_tau : bool, optional
            Whether provided correlation time should be fixed (or only used as a priori).
            Default is False.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(ar1(dt=dt, per=per, flow=flow, s2=s2, fix_s2=fix_s2, tau=tau, fix_tau=fix_tau, tunit=m.r.tunit, yunit=m.r.yunit))

    # Add power-law noise to model
    #-----------------------------
    def add_pl(m, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, a=None, fix_a=False, tn=None):

        """
        Add power-law noise to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            Default if '2-way'.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        a : float, optional
            [A priori] spectral index
        fix_a : bool, optional
            Whether provided spectral index should be fixed (or only used as a priori).
            Default is False.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(pl(dt=dt, per=per, flow=flow, s2=s2, fix_s2=fix_s2, a=a, fix_a=fix_a, tn=tn, yunit=m.r.yunit))

    # Add flicker noise to model
    #---------------------------
    def add_fn(m, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, tn=None):

        """
        Add flicker noise to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            Default if '2-way'.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(pl(dt=dt, per=per, flow=flow, s2=s2, fix_s2=fix_s2, tn=tn, a=1, fix_a=True, yunit=m.r.yunit))

    # Add random walk to model
    #-------------------------
    def add_rw(m, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, tn=None):

        """
        Add random walk to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        per : float, optional
            Period of possible modulating sine wave
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            Default if '2-way'.
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(pl(dt=dt, per=per, flow=flow, s2=s2, fix_s2=fix_s2, tn=tn, a=2, fix_a=True, yunit=m.r.yunit))

    # Add GGM process to model
    #-------------------------
    def add_ggm(m, dt=None, per=None, flow='2-way', s2=None, fix_s2=False, a=None, fix_a=False, tau=None, fix_tau=False, tn=None):

        """
        Add GGM process to model

        Parameters
        ----------
        dt : float, optional
            Sampling
        s2 : float, optional
            [A priori] variance factor
        fix_s2 : bool, optional
            Whether provided variance factor should be fixed (or only used as a priori).
            Default is False.
        a : float, optional
            [A priori] spectral index
        fix_a : bool, optional
            Whether provided spectral index should be fixed (or only used as a priori).
            Default is False.
        tau : float, optional
            [A priori] correlation time
        fix_tau : bool, optional
            Whether provided correlation time should be fixed (or only used as a priori).
            Default is False.
        flow : str, optional
            Keyword indicating how the noise is assumed to propagate:
            - '1-way' means that the noise is assumed to start at the beginning of the series
              and to propagate forward.
            - '2-way' means that half of the noise is assumed to start at the beginning of the series
              and to propagate forward, and half of the noise is assumed to start at the end
              of the series and to propagate backward.
            Default if '2-way'.
        tn : float or list, optionnal
            Float : date of the beginning of the noise
            List : dates of the beginning and the end of the noise.
            If missing, the date(s) will correspond to the beginning or the end of the time series.
            Default is None.
            
        """
        
        for d in range(m.nd):
            m[d].n.append(ggm(dt=dt, per=per, flow=flow, s2=s2, fix_s2=fix_s2, a=a, fix_a=fix_a, tau=tau, fix_tau=fix_tau, tn=tn, tunit=m.r.tunit, yunit=m.r.yunit))

    # Add jumps to specified polynomial and/or sine wave functions of model
    #----------------------------------------------------------------------
    def add_jumps(m, t, deg=[], per=[]):
      
        """
        Add jumps to specified polynomial and/or sine wave functions of model

        Parameters
        ----------
        t : list
            Dates of jumps
        deg : list, optional
            List of polynomial degrees for which the jump should be introduced
        per : list, optional
            List of sine wave periods for which the jump should be introduced
        
        """
        
        # Loop over dimensions and functions
        for d in range(m.nd):
            for i in range(len(m[d].f)):
                f = m[d].f[i]
                if isinstance(f, polynom):
                    if (f.deg in deg):
                        m[d].f[i] = polynom(f.deg, sorted(f.t+t))
                elif isinstance(f, sine):
                    if (f.per in per):
                        m[d].f[i] = sine(f.per, sorted(f.t+t))

    # Set default a priori values for unknown deterministic parameters
    #----------------------------------------------------------------
    def set_x0(m):
    
        """
        Set default a priori values for unknown deterministic parameters
        
        Warning: Only for 1-dimensional model.
            
        """

        # Reset some model attributes
        m.yc = None
        m.A = None

        # Set default a priori values for unknown parameters of each function
        for f in m.f:
            f.set_x0(m)

    # Set values of deterministic parameters
    #---------------------------------------
    def set_x(m, x):
    
        """
        Set values of deterministic parameters

        Warning: Only for 1-dimensional model.

        Parameters
        ----------
        x : array_like
            Values of unknown deterministic parameters
            
        """
        
        # Reset some model attributes
        m.yc = None
        m.A = None

        # Initialization
        i = -1
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    i = i+1
                    p.x = x[i]

    # Get values of unknown deterministic parameters
    #-----------------------------------------------
    def get_x(m):
    
        """
        Get values of unknown deterministic parameters

        Warning: Only for 1-dimensional model.

        Returns
        -------
        x : array_like
            Values of unknown deterministic parameters
        """
        
        # Initialization
        x = []
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    x.append(p.x)
                    
        return np.array(x)

    # Set values of unknown deterministic parameters given reparameterized parameters
    #--------------------------------------------------------------------------------
    def set_xr(m, xr):
    
        """
        Set values of unknown deterministic parameters given reparameterized parameters

        Warning: Only for 1-dimensional model.

        Parameters
        ----------
        xr : array_like
            Values of unknown reparameterized deterministic parameters
            
        """
        
        # Reset some model attributes
        m.yc = None
        m.A = None

        # Initialization
        i = -1
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    i = i+1
                    if hasattr(p, 'xr2x'):
                        p.x = p.xr2x(xr[i])
                    else:
                        p.x = xr[i]

    # Get values of unknown reparameterized deterministic parameters
    #---------------------------------------------------------------
    def get_xr(m):
    
        """
        Get values of unknown reparameterized deterministic parameters

        Warning: Only for 1-dimensional model.

        Returns
        -------
        xr : array_like
            Values of reparameterized deterministic parameters
        """
        
        # Initialization
        xr = []
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    if hasattr(p, 'x2xr'):
                        xr.append(p.x2xr(p.x))
                    else:
                        xr.append(p.x)
                    
        return np.array(xr)

    # Set formal errors of deterministic parameters
    #----------------------------------------------
    def set_sigx(m):
    
        """
        Set formal errors of deterministic parameters
            
        Warning: Only for 1-dimensional model.

        """
        
        # Initialization
        i = -1
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    i = i+1
                    p.sig = sqrt(m.Qx[i,i])
                    
    # Compute partial derivatives of deterministic parameters wrt reparameterized parameters
    #---------------------------------------------------------------------------------------
    def dx_dxr(m):
    
        """
        Compute partial derivatives of deterministic parameters wrt reparameterized parameters
            
        Warning: Only for 1-dimensional model.

        Returns
        -------
        dx : array_like
            Partial derivatives of deterministic parameters wrt reparameterized parameters
            
        """
        
        # Initialization
        dx = []
        
        # Loop over unknown deterministic parameters
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    if hasattr(p, 'dx_dxr'):
                        dx.append(p.dx_dxr(p.x))
                    else:
                        dx.append(1)
        
        return np.array(dx)

    # Set default a priori values for unknown noise parameters
    #---------------------------------------------------------
    def set_b0(m, v0):
    
        """
        Set default a priori values for unknown noise parameters
        
        Warning: Only for 1-dimensional model.
        
        Parameters
        ----------
        v0 : float
            A priori variance of each noise component
            
        """

        # Reset some model attributes
        m.Q = None
        m.L = None
        m.P = None
        m.dQ = None

        # Set default a priori values for unknown parameters of each noise component
        for n in m.n:
            n.set_x0(m, v0)

    # Set values of noise parameters
    #-------------------------------
    def set_b(m, x):
    
        """
        Set values of noise parameters

        Warning: Only for 1-dimensional model.

        Parameters
        ----------
        x : array_like
            Values of unknown noise parameters
            
        """
        
        # Reset some model attributes
        m.Q = None
        m.L = None
        m.P = None
        m.dQ = None

        # Initialization
        i = -1
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    i = i+1
                    p.x = x[i]

    # Get values of unknown noise parameters
    #---------------------------------------
    def get_b(m):
    
        """
        Get values of unknown noise parameters

        Warning: Only for 1-dimensional model.

        Returns
        -------
        x : array_like
            Values of unknown noise parameters
            
        """
        
        # Initialization
        x = []
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    x.append(p.x)
                    
        return np.array(x)

    # Set values of noise parameters given reparameterized parameters
    #----------------------------------------------------------------
    def set_br(m, xr):
    
        """
        Set values of noise parameters given reparameterized parameters

        Warning: Only for 1-dimensional model.

        Parameters
        ----------
        xr : array_like
            Values of unknown reparameterized noise parameters
            
        """
        
        # Reset some model attributes
        m.Q = None
        m.L = None
        m.P = None
        m.dQ = None

        # Initialization
        i = -1
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    i = i+1
                    if hasattr(p, 'xr2x'):
                        p.x = p.xr2x(xr[i])
                    else:
                        p.x = xr[i]

    # Get values of unknown reparameterized noise parameters
    #-------------------------------------------------------
    def get_br(m):
    
        """
        Get values of unknown reparameterized noise parameters

        Warning: Only for 1-dimensional model.

        Returns
        -------
        xr : array_like
            Values of reparameterized noise parameters
        """
        
        # Initialization
        xr = []
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    if hasattr(p, 'x2xr'):
                        xr.append(p.x2xr(p.x))
                    else:
                        xr.append(p.x)
                    
        return np.array(xr)

    # Set formal errors of noise parameters
    #--------------------------------------
    def set_sigb(m):
    
        """
        Set formal errors of noise parameters
            
        Warning: Only for 1-dimensional model.

        """
        
        # Initialization
        i = -1
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    i = i+1
                    p.sig = sqrt(m.Qb[i,i])
                    
    # Compute partial derivatives of noise parameters wrt reparameterized parameters
    #-------------------------------------------------------------------------------
    def db_dbr(m):
    
        """
        Compute partial derivatives of noise parameters wrt reparameterized parameters
            
        Warning: Only for 1-dimensional model.

        Returns
        -------
        dx : array_like
            Partial derivatives of noise parameters wrt reparameterized parameters
            
        """
        
        # Initialization
        dx = []
        
        # Loop over unknown noise parameters
        for n in m.n:
            for p in n.par:
                if not(p.fixed):
                    if hasattr(p, 'dx_dxr'):
                        dx.append(p.dx_dxr(p.x))
                    else:
                        dx.append(1)
        
        return np.array(dx)

    # Compute predicted observations and design matrix
    #-------------------------------------------------
    def set_oeq(m):

        """
        Compute predicted observations and design matrix

        set_oeq() does not return anything, but sets the attributes yc and A
        of the model instance.

        Warning: Only for 1-dimensional model.

        """
        
        # Initializations
        m.yc = np.zeros(len(m.r.t))
        m.A = []
        
        # Loop over model functions
        for f in m.f:
            f.set_oeq(m)
            m.yc = m.yc + f.yc
            m.A.extend(f.A)
            
        # Finalize design matrix
        m.A = np.array(m.A).T
        #m.A = m.A * m.dx_dxr()

    # Compute covariance matrix
    #--------------------------
    def set_cov(m, chol=False, inv=False, set_dcov=False):

        """
        Compute covariance matrix

        set_cov() does not return anything, but sets the attributes Q, [L], [P] and [dQ]
        of the model instance.

        Warning: Only for 1-dimensional model.
            
        Parameters
        ----------
        chol : bool, optional
            Whether the Cholesky factorization of m.Q should be computed and stored in m.L.
            Default is False.
        inv : bool, optional
            Whether inverse and determinant of m.Q should be computed and stored in m.P
            and m.d. Default is False.
        set_dcov : bool, optional
            Whether partial derivatives of m.Q wrt noise parameters should be computed.
            Default is False.

        """
        
        # Reset some model attributes
        m.Q = None
        m.L = None
        m.P = None
        m.dQ = None

        # Set covariance matrix of each noise component
        for n in m.n:
            n.set_cov(m, set_dcov=set_dcov)
            
        # Should we set a covariance vector or matrix for the model?
        nd = np.max([n.Q.ndim for n in m.n])
        
        # Initialize total covariance matrix
        if (nd == 1):
            m.Q = np.zeros(m.r.n)
        else:
            m.Q = np.zeros((m.r.n, m.r.n))

        # Compute total covariance matrix
        for n in m.n:
            if (nd == 2) and (n.Q.ndim == 1):
                m.Q = m.Q + np.diag(n.Q)
            else:
                m.Q = m.Q + n.Q
                
        # Factorize covariance matrix if requested
        if (chol) and (nd == 2):
            m.L = cholesky(m.Q)
            
        # Invert covariance matrix if requested
        if (inv) and (nd == 2):
            (m.P, m.dQ) = invspd(m.Q, return_det=True)
        elif (inv) and (nd == 1):
            m.P = 1/m.Q
            m.dQ = np.sum(np.log(m.Q))

    # Compute power spectral density of noise model and of residuals
    #---------------------------------------------------------------
    def set_psd(m, sf=4, fr=None, from_cov=False, set_dpsd=False, set_spsd=False):

        """
        Compute power spectral density of noise model and of residuals

        set_psd() does not return anything, but sets the attributes fr, pn, [Qpn], [spn] and [pv]
        of the model instance.

        Warning: Only for 1-dimensional model.
            
        Parameters
        ----------
        sf : int, optional
            Oversampling factor. Default is 4.
        fr : array, optional
            Frequency range. Automatically set by default.
        from_cov : bool, optional
            Whether PSD of noise model should be computed from its covariance matrix,
            i.e., accounting for possible irregular integration intervals and gaps in the series.
        set_dpsd : bool, optional
            Whether partial derivatives of power spectral density wrt noise parameters
            should be computed. Default is False.
        set_spsd : bool, optional
            Whether covariance matrix of formal errors of noise model PSD should be
            computed. Default is False.

        """
        
        # Reset some model attributes
        m.fr = None
        m.pn = None
        m.Qpn = None
        m.spn = None
        m.pv = None

        # Set frequency range
        if (fr is not None):
            m.fr = fr
        else:
            T = m.r.t[-1] - m.r.t[0]
            dt = np.min(m.r.t[1:]-m.r.t[:-1])
            if not(np.isscalar(m.r.T)):
                Tmax = np.max(m.r.T)
                if (dt < Tmax):
                    dt = Tmax
            f0 = 1/T
            fc = 1/(2*dt)
            df = f0/sf
            m.fr = np.arange(f0, fc+df/2, df)
            
        # If covariance matrix of noise model PSD should be computed,
        # then we need the partial derivatives
        if (set_spsd):
            set_dpsd = True
            
        # Compute PSD of noise components [and their partial derivatives]
        for n in m.n:
            n.set_psd(m, fr=m.fr, set_dpsd=set_dpsd, from_cov=from_cov)
            
        # Compute total PSD of noise model
        m.pn = np.zeros(len(m.fr))
        for n in m.n:
            m.pn = m.pn + n.P
            
        # Compute covariance matrix and formal errors of noise model PSD if requested
        if (set_spsd):

            # Design matrix
            A = []
            for n in m.n:
                A.extend(n.dP)
            A = np.array(A).T
            
            # Covariance matrix of noise model PSD
            m.Qpn = np.dot(A, np.dot(m.Qb, A.T))
            
            # Standard deviations of noise model PSD
            m.spn = np.sqrt(np.diag(m.Qpn))
            
        # Compute PSD of residuals if available
        if (m.v is not None):
            m.pv = lombscargle(m.r.t, m.v, f=m.fr, dtrd=None)[1]
    
    # Estimate individual noise components
    #-------------------------------------
    def set_xi(m):

        """
        Estimate individual noise components

        filter_noise does not return anything but sets attributes xi, sxi and Qxi
        of the model noise components.
        
        """

        # Loop over dimensions
        for d in range(m.nd):

            # If there's only one noise component,
            if (len(m[d].n) == 1):
                m[d].n[0].xi = m[d].v
                m[d].n[0].Qxi = m[d].Qv
                m[d].n[0].sxi = m[d].sv
                
            # Else, loop over noise components
            else:
                for i in range(len(m[d].n)):

                    # Residuals -> noise estimates projector
                    Q = m[d].n[i].Q
                    if (Q.ndim == 1):
                        QP = (m[d].P*Q).T
                    else:
                        QP = np.dot(Q, m[d].P)
                    
                    # Noise estimates, their covariance matrix and formal errors
                    m[d].n[i].xi = np.dot(QP, m[d].v)
                    m[d].n[i].Qxi = np.dot(QP, np.dot(m[d].Qv, QP.T))
                    m[d].n[i].sxi = np.sqrt(np.diag(m[d].n[i].Qxi))

    # Simulate time series values
    #----------------------------
    def simulate(m):
    
        """
        Simulate time series values

        simulate() does not return anything, but sets the attribute y of the model's time series.
        
        """
        
        # Loop over dimensions
        for d in range(m.nd):

            # Set observation equation and covariance matrix
            m[d].set_oeq()
            m[d].set_cov()
            
            # Initialize time series values with deterministic model
            if (m.nd > 1):
                m.r.y[:,d] = m[d].yc
            else:
                m.r.y = m[d].yc
            
            # Loop over noise components
            for n in m[d].n:
            
                # Generate random noise
                if (n.Q.ndim == 2):
                    (f, L) = cholesky(n.Q)
                    n.xi = np.dot((L/f).T, np.random.randn(m[d].r.n))
                else:
                    n.xi = np.sqrt(n.Q) * np.random.randn(m[d].r.n)

                # Add noise to time series values
                if (m.nd > 1):
                    m.r.y[:,d] = m.r.y[:,d] + n.xi
                else:
                    m.r.y = m.r.y + n.xi

    # Fit deterministic model with fixed covariance matrix
    #-----------------------------------------------------
    def fitx(m, vf=False, estimator='reml'):
    
        """
        Fit deterministic model with fixed covariance matrix

        fitx() does not return anything, but updates the values and formal errors
        of the model parameters, and also sets attributes of the model instance m:
        
        nx    : Number of deterministic parameters
        dQ    : Log-determinant of covariance matrix
        dN    : Log-determinant of normal matrix of deterministic parameters
        s2    : Global variance factor
        Qx    : Covariance matrix of deterministic parameters
        v     : Residuals
        Pv    : Product of observation weight matrix with residuals
        logl  : Log-likelihood
        loglr : Restricted log-likelihood
        
        Warning: Only for 1-dimensional model.

        Parameters
        ----------
        vf : bool, optional
            Whether m.Q is known only up to an unknown variance factor which should be estimated.
            Default is False.
        estimator : str, optional
            Specifies how global variance factor should be estimated: either 'ml' or 'reml'.
            Default is 'reml'.
        
        """
        
        # Raise an error if covariance matrix is not set
        if (m.Q is None):
            raise RuntimeError('Covariance matrix of model should be set before calling fitx().')
        
        # If necessary, compute Cholesky factorization of m.Q
        if (m.Q.ndim == 2) and (m.L is None) and (m.P is None):
            m.L = cholesky(m.Q)

        # Set possibly unset a priori parameters
        m.set_x0()
        
        # Initializations
        xr = m.get_xr()
        m.nx = len(xr)
        dx = np.ones(m.nx)
        
        # If any parameter to estimate,
        if (m.nx > 0):

            # Get a priori log-relaxation times and "fix" them
            # (This should be generalized to any kind of non-linear parameters.)
            ltau = []
            for f in m.f:
                if isinstance(f, fexp) or isinstance(f, flog):
                    if not(f.par[1].fixed):
                        ltau.append(f.par[1].x2xr(f.par[1].x))
                        f.par[1].fixed = True
                        f.par[1].estim = True
                        
            # If any relaxation times to estimate,
            if (len(ltau) > 0):
                
                # Inner function: ltau -> -logl
                def logl(ltau):
                    i = -1
                    for f in m.f:
                        if isinstance(f, fexp) or isinstance(f, flog):
                            if (f.par[1].estim):
                                i = i+1
                                f.par[1].x = f.par[1].xr2x(ltau[i])
                    m.fitx(vf=vf, estimator=estimator)
                    if (estimator == 'reml'):
                        return -m.loglr
                    else:
                        return -m.logl
                
                # Find optimal log-relaxation times
                ltau = optimize.minimize(logl, ltau, method='Nelder-Mead', tol=1e-5).x
                
                # Set optimal log-relaxation times and unfix them
                i = -1
                for f in m.f:
                    if isinstance(f, fexp) or isinstance(f, flog):
                        if (f.par[1].estim):
                            i = i+1
                            f.par[1].x = f.par[1].xr2x(ltau[i])
            
            # Set predicted observations and design matrix
            m.set_oeq()
            A = m.A * m.dx_dxr()
            
            # Set up normal equation
            if (m.Q.ndim == 2) and (m.P is not None):
                AtP = np.dot(A.T, m.P)
            elif (m.Q.ndim == 2):
                AtP = (cholsolve(m.L, A)).T
            else:
                AtP = A.T/m.Q
            N = np.dot(AtP, A)
            b = np.dot(AtP, m.r.y-m.yc)

            # Add constraints if any,
            i = -1
            for f in m.f:
                for p in f.par:
                    if not(p.fixed):
                        i = i+1
                    if (p.sigc is not None):
                        if hasattr(p, 'x2xr'):
                            x0 = p.x2xr(p.x)
                            xc = p.x2xr(p.xc)
                            sc = p.sigc / p.dx_dxr(p.xc)
                        else:
                            x0 = p.x
                            xc = p.xc
                            sc = p.sigc
                        N[i,i] = N[i,i] + 1/sc**2
                        b[i] = b[i] + (xc-x0)/sc**2
            
            # Solve normal equation
            (m.Qx, m.dN) = invspd(N, return_det=True)
            dx = np.dot(m.Qx, b)

            # Update deterministic parameters
            m.set_xr(m.get_xr()+dx)

            # If there were relaxation times to estimate,
            if (len(ltau) > 0):

                # Unfix them
                i = -1
                for f in m.f:
                    if isinstance(f, fexp) or isinstance(f, flog):
                        if (f.par[1].estim):
                            f.par[1].fixed = False
                            
                # Update parameter covariance matrix
                m.set_oeq()
                A = m.A * m.dx_dxr()
                if (m.Q.ndim == 2) and (m.P is not None):
                    AtP = np.dot(A.T, m.P)
                elif (m.Q.ndim == 2):
                    AtP = (cholsolve(m.L, A)).T
                else:
                    AtP = A.T/m.Q
                N = np.dot(AtP, A)
                i = -1
                for f in m.f:
                    for p in f.par:
                        if not(p.fixed):
                            i = i+1
                        if (p.sigc is not None):
                            if hasattr(p, 'x2xr'):
                                x0 = p.x2xr(p.x)
                                xc = p.x2xr(p.xc)
                                sc = p.sigc / p.dx_dxr(p.xc)
                            else:
                                x0 = p.x
                                xc = p.xc
                                sc = p.sigc
                            N[i,i] = N[i,i] + 1/sc**2
                (m.Qx, m.dN) = invspd(N, return_det=True)
                
            # Else, just update observation equations
            else:
                m.set_oeq()
                
            # WLSQE transition matrix
            m.y2x = m.Qx@AtP   

        # Else (no parameter to estimate),
        else:
            m.Qx = np.array([[]])
            m.dN = 0
            
        
        # Compute residuals
        m.set_oeq()
        m.v = m.r.y - m.yc
        
        # Weight matrix * residuals
        if (m.Q.ndim == 2) and (m.P is not None):
            m.Pv = np.dot(m.P, m.v)
        elif (m.Q.ndim == 2):
            m.Pv = cholsolve(m.L, m.v)
        else:
            m.Pv = m.v/m.Q
        
        # Weighted-square sum of residuals
        vPv = np.sum(m.v*m.Pv)
        
        # Log-determinant of covariance matrix
        if (m.dQ is None):
            if (m.Q.ndim == 2):
                m.dQ = 2 * (np.sum(np.log(np.diag(m.L[1]))) - np.sum(np.log(m.L[0])))
            else:
                m.dQ = np.sum(np.log(m.Q))
                
        # Compute number of constraints, constraint residuals
        # and half-log determinant of constraint covariance matrix
        c = 0
        dQc = 0
        vcPcvc = 0
        i = -1
        for f in m.f:
            for p in f.par:
                if not(p.fixed):
                    i = i+1
                if (p.sigc is not None):
                    if hasattr(p, 'x2xr'):
                        x = p.x2xr(p.x)
                        xc = p.x2xr(p.xc)
                        sc = p.sigc / p.dx_dxr(p.xc)
                    else:
                        x = p.x
                        xc = p.xc
                        sc = p.sigc
                    c = c + 1
                    dQc = dQc + log(sc**2)
                    vcPcvc = vcPcvc + ((xc-x)/sc)**2
        
        # Global variance factor
        if (vf):
            if (estimator == 'ml'):
                m.s2 = vPv / m.r.n
            elif (estimator == 'reml'):
                m.s2 = vPv / (m.r.n-m.nx)
        else:
            m.s2 = 1
            
        # Log-likelihood
        m.logl = -(m.r.n+c)/2*log(2*pi*m.s2) - m.dQ/2 - dQc/2 - (vPv+vcPcvc)/(2*m.s2)
        
        # Restricted log-likelihood
        m.loglr = m.logl + m.nx/2*log(2*pi*m.s2) - m.dN/2 + dQc/2
        
        # Finalize parameter covariance matrix
        m.Qx = m.s2 * m.Qx
        dx = m.dx_dxr()
        m.Qx = dx*(m.Qx*dx).T
        m.set_sigx()
    
    # Fit deterministic + noise model
    #--------------------------------
    def fit(m, estimator='reml', method='BFGS', prefit_x=True, prefit_b=False, hessian='expected', fr=None, finalize=True, quiet=False, verbose=False, out=sys.stdout):
    
        """
        Fit deterministic + noise model

        fit does not return anything, but updates the values and formal errors of the
        model parameters, and also sets attributes of the model instance m:

        nx    : Number of deterministic parameters
        nb    : Number of noise parameters
        dQ    : Log-determinant of covariance matrix
        dN    : Log-determinant of normal matrix of deterministic parameters
        dH    : Log-determinant of normal matrix of noise parameters
        s2    : Global variance factor
        Qx    : Covariance matrix of deterministic parameters
        Qb    : Covariance matrix of noise parameters
        Qc    : Covariance matrix of predicted observations
        sc    : Formal errors of predicted observations
        v     : Residuals
        Pv    : Product of observation weight matrix with residuals
        sv    : Formal errors of residuals
        Qv    : Covariance matrix of residuals
        vn    : Normalized residuals
        wrms  : WRMS of residuals
        logl  : Log-likelihood
        loglr : Restricted log-likelihood
        bic   : -BIC/2
        bicr  : -(restricted BIC)/2
        E     : Evidence
        Er    : Restriced evidence
        
        Parameters
        ----------
        estimator : str, optional
            Specified which estimator should be used for the noise parameters:
            either 'ml', 'reml' or 'ls'. Default is 'reml'.
        method : str, optional
            Numerical maximization method to be used in case estimator is either
            'ml' or 'reml': either 'Nelder-Mead', 'Powell', 'CG', 'BFGS' or 'Newton'.
            Default is 'Newton'.
        prefit_x : bool, optional
            Should we start with a first fit of the deterministic parameters
            assuming white noise only?
        prefit_b : bool, optional
            In case estimator is either 'ml' or 'reml', should we start with a first
            LS fit of the noise parameters? Default is True.
        hessian : str, optional
            Specifies how the hessian matrix of noise parameters should be computed:
            either 'expected' or 'numeric'. Default is 'expected'.
        fr : array, optional
            Frequency range used to compute PSD of noise model and residuals.
            Default is None (automatically set).
        finalize : bool, optional
            If False, then fit() will stop right after the optimal noise parameters
            are found and set, but most other attributes of the model will not be set.
            Default is True.
        quiet : bool, optional
            Whether to hide messages. Default is False.
        verbose : bool, optional
            Whether to show detailed messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.

        """
        
        # If log file is actually a file, not sys.stdout, set verbose=True.
        if (out != sys.stdout):
            verbose = True
        
        # Raise error if model includes VW noise, but time series has no formal errors
        b = False
        for d in range(m.nd):
            for n in m[d].n:
                if isinstance(n, vw):
                    b = True
        if (b) and (m.r.Q is None):
            raise RuntimeError('Cannot fit VW noise to time series without formal errors.')

        try:
            
            # Print header in log file
            if not(quiet):
                print('', file=out)
                print('ts.model.fit', file=out)
                print('------------', file=out)
            
            # Loop over dimensions
            #---------------------
            for d in range(m.nd):
            
                # Dimension name
                dim = m[d].r.dims
                if (dim is None):
                    dim = 'Component '+str(d+1)
            
                # Print message
                if not(quiet):
                    print('', file=out)
                    if (m.nd > 1):
                        print('    '+dim, file=out)
                        print('    '+'-'*len(dim), file=out)
                        print('', file=out)

                # First fit with white noise only to get good a priori deterministic parameters
                if (prefit_x):
                    if not(quiet):
                        print('    '+str(date())+' : Initial fit', file=out)
                    if (m[d].r.Q is not None):
                        m[d].Q = m[d].r.Q
                    else:
                        m[d].Q = np.ones(m[d].r.n)
                    m[d].fitx()
                
                # Raise error if initial fit of deterministic parameters failed
                if (m[d].v is None):
                    raise RuntimeError('Initial fit of deterministic parameters failed.')

                # Set possibly unset a priori noise parameters
                m[d].set_b0(np.sum(m[d].v**2)/m[d].r.n/len(m[d].n))
                
                # In case there's just one noise component, with just one unknown variance factor, and no constraint on any parameter,
                # de-activate initial LS fit of noise parameters and switch to Nelder-Mead method,
                # so that thanks to Williams' trick, no iterations are required.
                b = True
                if (len(m[d].n) > 1):
                    b = False
                if (b):
                    n = m[d].n[0]
                    if (n.par[0].fixed):
                        b = False
                    else:
                        for i in range(1, len(n.par)):
                            if not(n.par[i].fixed):
                                b = False
                if (b):
                    for f in m[d].f:
                        for p in f.par:
                            if not(p.fixed) and (p.sigc is not None):
                                b = False
                if (b):
                    prefit_b = False
                    method = 'Nelder-Mead'
                    
                # 1st case : [RE]ML estimation of noise parameters
                #-------------------------------------------------
                if (estimator in ['ml', 'reml']):
                    
                    # Initial LS fit of noise parameters
                    if (prefit_b):
                        m[d].fit(estimator='ls', finalize=False, quiet=True)
                        
                        # Raise error if initial LS fit of noise parameters failed
                        if (m[d].v is None):
                            raise RuntimeError('Initial LS fit of noise parameters failed.')
                
                    # 1st sub-case : Nelder-Mead or Powell maximization
                    #--------------------------------------------------
                    if (method in ['Nelder-Mead', 'Powell']):

                        # Does the model support Williams' trick?
                        w = True
                        for n in m[d].n:
                            if (n.par[0].fixed):
                                w = False

                        # If yes, then temporarily fix variance factor of 1st noise component
                        if (w):
                            m[d].n[0].par[0].fixed = True

                        # Inner function: reparameterized noise parameters -> minus [restricted] log-likelihood
                        def loglr(br):
                            
                            # Fit
                            m[d].set_br(br)
                            m[d].set_cov(chol=True)
                            m[d].fitx(vf=w)
                            
                            # Message
                            if not(quiet):
                                if not(verbose):
                                    print('\x1b[2K', end='\r', file=out)
                                    end = '\r'
                                else:
                                    end = '\n'
                                if (estimator == 'ml'):
                                    print('        Current noise parameters and likelihood :', m[d].get_b(), m[d].logl, end=end, file=out)
                                else:
                                    print('        Current noise parameters and restricted likelihood :', m[d].get_b(), m[d].loglr, end=end, file=out)
                            
                            # Return minus [restricted] log-likelihood
                            if (estimator == 'ml'):
                                return -m[d].logl
                            elif (estimator == 'reml'):
                                return -m[d].loglr
                        
                        # Search optimal reparameterized noise parameters...
                        if (len(m[d].get_br()) > 0):
                            if not(quiet):
                                print('    '+str(date())+' : Search optimal noise parameters', file=out)
                            br = optimize.minimize(loglr, m[d].get_br(), method=method, tol=1e-5).x
                            m[d].set_br(br)
                        
                        #... or just fit deterministic parameters
                        else:
                            m[d].set_cov(chol=True)
                            m[d].fitx(vf=w)
                        
                        # If Williams' trick was used,
                        if (w):

                            # Set 1st variance factor free again
                            m[d].n[0].par[0].fixed = False
                            
                            # Multiply all variance factors by estimated global variance factor
                            for n in m[d].n:
                                n.par[0].x = m[d].s2 * n.par[0].x
                        
                    # 2nd sub-case : conjugate gradient maximization
                    #-----------------------------------------------
                    elif (method in ['CG', 'BFGS']):
                        
                        # Inner function: reparameterized noise parameters -> minus [restricted] log-likelihood
                        def loglr(br):
                            
                            # Fit
                            m[d].set_br(br)
                            m[d].set_cov(inv=True, set_dcov=True)
                            m[d].fitx()
                            
                            # Message
                            if not(quiet):
                                if not(verbose):
                                    print('\x1b[2K', end='\r', file=out)
                                    end = '\r'
                                else:
                                    end = '\n'
                                if (estimator == 'ml'):
                                    print('        Current noise parameters and likelihood :', m[d].get_b(), m[d].logl, end=end, file=out)
                                else:
                                    print('        Current noise parameters and restricted likelihood :', m[d].get_b(), m[d].loglr, end=end, file=out)
                            
                            # Return minus [restricted] log-likelihood
                            if (estimator == 'ml'):
                                return -m[d].logl
                            elif (estimator == 'reml'):
                                return -m[d].loglr

                        # Inner function: reparameterized noise parameters -> partial derivatives of minus [restricted] log-likelihood wrt reparameterized noise parameters
                        def dloglr(br):
                            
                            # Fit if necessary
                            if (m[d].P is None) or not(np.allclose(br, m[d].get_br())):
                                m[d].set_br(br)
                                m[d].set_cov(inv=True, set_dcov=True)
                                m[d].fitx()
                            
                            # Set "weight matrix"
                            if (estimator == 'ml') or (m[d].nx == 0):
                                P = m[d].P
                            else:
                                if (m[d].P.ndim == 2):
                                    AtP = np.dot(m[d].A.T, m[d].P)
                                    P = m[d].P - np.dot(AtP.T, np.dot(m[d].Qx, AtP))
                                else:
                                    AtP = m[d].A.T * m[d].P
                                    P = np.diag(m[d].P) - np.dot(AtP.T, np.dot(m[d].Qx, AtP))

                            # Partial derivatives of minus [restricted] log-likelihood wrt reparameterized noise parameters
                            J = []
                            for n in m[d].n:
                                for i in range(len(n.dQ)):
                                    if (P.ndim == 2) and (n.dQ[i].ndim == 2):
                                        J.append(trdot(n.dQ[i], P) - np.sum(m[d].Pv * np.dot(n.dQ[i], m[d].Pv)))
                                    elif (P.ndim == 2) and (n.dQ[i].ndim == 1):
                                        J.append(np.sum(n.dQ[i]*np.diag(P)) - np.sum(m[d].Pv*n.dQ[i]*m[d].Pv))
                                    else:
                                        J.append(np.sum(n.dQ[i]*P) - np.sum(m[d].Pv*n.dQ[i]*m[d].Pv))
                                    
                            return np.array(J)/2 * m[d].db_dbr()

                        # Search optimal reparameterized noise parameters
                        if not(quiet):
                            print('    '+str(date())+' : Search optimal noise parameters', file=out)
                        br = optimize.minimize(loglr, m[d].get_br(), method=method, jac=dloglr, options={'gtol': 1e-3}).x

                        # Set optimal noise parameters
                        m[d].set_br(br)                    

                    # 3rd sub-case : Newton maximization
                    #-----------------------------------
                    elif (method == 'Newton'):

                        if not(quiet):
                            print('    '+str(date())+' : Search optimal noise parameters', file=out)

                        # Initializations
                        br = m[d].get_br()
                        nb = len(br)
                        db = np.ones(nb)
                        d2b = np.zeros(nb)
                        # dbp = None
                        
                        # Iterations until reparameterized noise parameters have converged
                        while (np.max(np.abs(db+d2b)) > 1e-5):
                            
                            # Fit with current noise parameters
                            m[d].set_cov(inv=True, set_dcov=True)
                            m[d].fitx()
                            db_dbr = m[d].db_dbr()
                            
                            # Message
                            if not(quiet):
                                if not(verbose):
                                    print('\x1b[2K', end='\r', file=out)
                                    end = '\r'
                                else:
                                    end = '\n'
                                if (estimator == 'ml'):
                                    print('        Current noise parameters and likelihood :', m[d].get_b(), m[d].logl, end=end, file=out)
                                else:
                                    print('        Current noise parameters and restricted likelihood :', m[d].get_b(), m[d].loglr, end=end, file=out)

                            # Set "weight matrix"
                            if (estimator == 'ml') or (m[d].nx == 0):
                                P = m[d].P
                            else:
                                if (m[d].P.ndim == 2):
                                    AtP = np.dot(m[d].A.T, m[d].P)
                                    P = m[d].P - np.dot(AtP.T, np.dot(m[d].Qx, AtP))
                                else:
                                    AtP = m[d].A.T * m[d].P
                                    P = np.diag(m[d].P) - np.dot(AtP.T, np.dot(m[d].Qx, AtP))

                            # Products of partial derivatives of covariance matrix wrt reparameterized noise parameters
                            # with weight matrix, and with residuals
                            dQP = []
                            dQPv = []
                            for n in m[d].n:
                                for i in range(len(n.dQ)):
                                    if (P.ndim == 2) and (n.dQ[i].ndim == 2):
                                        dQP.append(np.dot(n.dQ[i], P))
                                        dQPv.append(np.dot(dQP[-1], m[d].v))
                                    elif (P.ndim == 2) and (n.dQ[i].ndim == 1):
                                        dQP.append((P*n.dQ[i]).T)
                                        dQPv.append(np.dot(dQP[-1], m[d].v))
                                    else:
                                        dQP.append(n.dQ[i]*P)
                                        dQPv.append(dQP[-1]*m[d].v)
                            
                            # Fill in normal matrix
                            N = np.zeros((nb, nb))
                            for i in range(nb):
                                for j in range(i+1):
                                    if (dQP[i].ndim == 2) and (dQP[j].ndim == 2):
                                        N[i,j] = trdot(dQP[i], dQP[j]) / 2
                                        #N[i,j] = -trdot(dQP[i], dQP[j]) / 2 + np.sum(dQPv[i]*np.dot(P, dQPv[j]))
                                    elif (dQP[i].ndim == 1) and (dQP[j].ndim == 2):
                                        N[i,j] = np.sum(dQP[i]*np.diag(dQP[j])) / 2
                                    elif (dQP[i].ndim == 2) and (dQP[j].ndim == 1):
                                        N[i,j] = np.sum(dQP[j]*np.diag(dQP[i])) / 2                                    
                                    else:
                                        N[i,j] = np.sum(dQP[i]*dQP[j]) / 2
                                    N[j,i] = N[i,j]
                            N = db_dbr*(N*db_dbr).T
                            
                            # Fill in right-hand side
                            b = np.zeros(nb)
                            for i in range(nb):
                                if (dQP[i].ndim == 2):
                                    b[i] = (np.sum(m[d].Pv*dQPv[i]) - np.trace(dQP[i])) / 2
                                else:
                                    b[i] = (np.sum(m[d].Pv*dQPv[i]) - np.sum(dQP[i])) / 2
                            b = db_dbr*b
                            
                            # Solve normal equation
                            db = 0.5*linalg.solve(N, b)
                            
                            ## Compute correction to increment based on previous increment
                            #if (dbp is not None):
                                #c = np.sum(db*dbp)
                                #z = np.sum(dbp**2)
                                #d2b = c**2 / z / (z-c) * dbp
                            #else:
                            d2b = 0
                            
                            # Update noise parameters
                            br = br + db + d2b
                            m[d].set_br(br)
                            
                            # Store increment
                            # dbp = db

                    # Final fit + compute covariance matrix of noise parameters
                    #----------------------------------------------------------

                    # Message
                    if not(quiet) and not(verbose):
                        print('\x1b[2K', end='\r', file=out)
                    if not(quiet):
                        print('    '+str(date())+' : Final fit', file=out)
                    
                    # Get optimal noise parameters
                    b = m[d].get_b()
                    m[d].nb = len(b)
                    for n in m[d].n:
                        for p in n.par:
                            if not(p.fixed):
                                p.sig = np.nan
                    
                    # Set observation covariance matrix
                    if not(finalize):
                        m[d].set_cov()
                    elif (hessian == 'expected'):
                        m[d].set_cov(inv=True, set_dcov=True)
                    else:
                        m[d].set_cov(inv=True)                    
                    
                    # Final fit
                    m[d].fitx()
                    
                    # If model should be finalized,
                    if (finalize):

                        # Message
                        if not(quiet):
                            print('    '+str(date())+' : Compute covariance matrix of noise parameters', file=out)
                        
                        # If expected hessian matrix should be computed,
                        if (hessian == 'expected'):
                            
                            # Set "weight matrix"
                            if (estimator == 'ml') or (m[d].nx == 0):
                                P = m[d].P
                            else:
                                AtP = np.dot(m[d].A.T, m[d].P)
                                P = m[d].P - np.dot(AtP.T, np.dot(m[d].Qx, AtP))

                            # Products of partial derivatives of covariance matrix wrt reparameterized noise parameters
                            # with weight matrix
                            dQP = []
                            for n in m[d].n:
                                for i in range(len(n.dQ)):
                                    if (P.ndim == 2) and (n.dQ[i].ndim == 2):
                                        dQP.append(np.dot(n.dQ[i], P))
                                    elif (P.ndim == 2) and (n.dQ[i].ndim == 1):
                                        dQP.append((P*n.dQ[i]).T)
                                    else:
                                        dQP.append(n.dQ[i]*P)
                            
                            # Fill in hessian matrix
                            H = np.zeros((m[d].nb, m[d].nb))
                            for i in range(m[d].nb):
                                for j in range(i+1):
                                    if (dQP[i].ndim == 2) and (dQP[j].ndim == 2):
                                        H[i,j] = -trdot(dQP[i], dQP[j]) / 2
                                    elif (dQP[i].ndim == 1) and (dQP[j].ndim == 2):
                                        H[i,j] = -np.sum(dQP[i]*np.diag(dQP[j])) / 2
                                    elif (dQP[i].ndim == 2) and (dQP[j].ndim == 1):
                                        H[i,j] = -np.sum(dQP[j]*np.diag(dQP[i])) / 2                                    
                                    else:
                                        H[i,j] = -np.sum(dQP[i]*dQP[j]) / 2
                                    H[j,i] = H[i,j]
                        
                        # Else (hessian matrix should be computed numerically),
                        else:
                            
                            # Model copy
                            mc = copy.deepcopy(m[d])

                            # Inner function: noise parameters -> log-likelihood
                            def logl(b):
                                mc.set_b(b)
                                mc.set_cov(chol=True)
                                mc.fitx()
                                return mc.logl

                            # Fill in hessian matrix
                            H = np.zeros((m[d].nb, m[d].nb))
                            for i in range(m[d].nb):
                                dpi = np.zeros(m[d].nb)
                                dpi[i] = 1e-4*b[i]
                                for j in range(i+1):
                                    dpj = np.zeros(m[d].nb)
                                    dpj[j] = 1e-4*b[j]
                                    H[i,j] = (logl(b+dpi+dpj) - logl(b+dpi-dpj) - logl(b-dpi+dpj) + logl(b-dpi-dpj)) / (4e-8*b[i]*b[j])
                                    H[j,i] = H[i,j]
                        
                        # Covariance matrix of noise parameters
                        if (m[d].nb > 0):
                            (m[d].Qb, m[d].dH) = invspd(-H, return_det=True)
                        else:
                            m[d].Qb = np.empty((0, 0))
                        m[d].set_sigb()
                
                # 2nd case : LS estimation of noise parameters
                #---------------------------------------------
                elif (estimator == 'ls'):

                    if not(quiet):
                        print('    '+str(date())+' : Search optimal noise parameters', file=out)

                    # Initializations
                    br = m[d].get_br()
                    nb = len(br)
                    db = np.ones(nb)
                    
                    # Iterations until reparameterized noise parameters have converged
                    niter = 0
                    while (np.max(np.abs(db)) > 1e-5):
                        
                        # Raise error if we're at more than 200 iterations
                        niter = niter + 1
                        if (niter > 200):
                            raise RuntimeError('Maximum number of iterations exceeded.')
                        
                        # Message
                        if not(quiet):
                            if not(verbose):
                                print('\x1b[2K', end='\r', file=out)
                                end = '\r'
                            else:
                                end = '\n'
                            print('        Current noise parameters :', m[d].get_b(), end=end, file=out)
                        
                        # Compute PSD of residuals + theoretical PSD of noise model and its partial derivatives
                        m[d].set_psd(set_dpsd=True, fr=fr)
                        
                        ###pp.loglog(m[d].fr, m[d].pv, 'k')
                        ###pp.loglog(m[d].fr, m[d].pn, 'r', linewidth=2)
                        ###pp.show()
                        
                        # Design matrix
                        A = []
                        for n in m[d].n:
                            A.extend(n.dP)
                        A = np.array(A).T * m[d].db_dbr()
                        
                        # Build and solve normal equation
                        AtP = A.T / m[d].pn**2
                        N = np.dot(AtP, A)
                        
                        (ll, vv) = linalg.eig(N)
                        N = N + np.min(np.real(ll))*np.eye(len(N))
                        
                        b = np.dot(AtP, m[d].pv - m[d].pn)
                        db = linalg.solve(N, b)

                        # Update noise parameters
                        br = br + db
                        m[d].set_br(br)

                    # Message
                    if not(quiet) and not(verbose):
                        print('\x1b[2K', end='\r', file=out)
                    if not(quiet):
                        print('    '+str(date())+' : Final fit', file=out)

                    # Get optimal noise parameters
                    b = m[d].get_b()
                    m[d].nb = len(b)
                    
                    # Final fit
                    if not(finalize):
                        m[d].set_cov()
                    else:
                        m[d].set_cov(inv=True)
                    m[d].fitx()

                    # Set covariance matrix of noise parameters
                    f = 1 / m[d].db_dbr()
                    N = f*(N*f).T
                    (m[d].Qb, m[d].dH) = invspd(N, return_det=True)
                    m[d].set_sigb()
                    
                # Set some final attributes of the model
                #---------------------------------------

                if (finalize):

                    # Covariance matrix and formal errors of predicted observations
                    m[d].Qc = np.zeros((m[d].r.n, m[d].r.n))
                    m[d].sc = np.zeros((m[d].r.n))
                    if (m[d].nx > 0):
                        m[d].Qc = np.dot(m[d].A, np.dot(m[d].Qx, m[d].A.T))
                        m[d].sc = np.sqrt(np.diag(m[d].Qc))

                    # Covariance matrix and formal errors of residuals
                    if (m[d].nx > 0):
                        if (m[d].Q.ndim == 1):
                            m[d].Qv = np.diag(m[d].Q) - m[d].Qc
                        else:
                            m[d].Qv = m[d].Q - m[d].Qc
                        m[d].sv = np.sqrt(np.diag(m[d].Qv))
                    else:
                        m[d].Qv = m[d].Q
                        if (m[d].Q.ndim == 1):
                            m[d].sv = np.sqrt(m[d].Qv)
                        else:
                            m[d].sv = np.sqrt(np.diag(m[d].Qv))
                    
                    # Normalized residuals
                    m[d].vn = m[d].v / m[d].sv
                    
                    # WRMS of residuals
                    m[d].wrms = sqrt(np.sum((m[d].v/m[d].sv)**2) / np.sum(1/m[d].sv**2))
                    
                    # Set -BIC/2 and evidence if noise parameters were estimated by ML
                    if (estimator == 'ml'):
                        m[d].bic = m[d].logl - (m[d].nx+m[d].nb)*log(m[d].r.n)/2
                        m[d].E = m[d].logl + ((m[d].nx+m[d].nb)*log(2*pi) - m[d].dN - m[d].dH) / 2

                    # Set -(restricted BIC)/2 and restricted evidence if noise parameters were estimated by REML
                    elif (estimator == 'reml'):
                        m[d].bicr = m[d].loglr - m[d].nb*log(m[d].r.n-m[d].nx)/2
                        m[d].Er = m[d].loglr + (m[d].nb*log(2*pi) - m[d].dH) / 2

                    # Compute PSD of noise model and of residuals
                    m[d].set_psd(set_spsd=True, fr=fr)
                    
                    # Estimate individual noise components
                    m[d].set_xi()

                # Print end message
                if not(quiet):
                    print('    '+str(date())+' : Done!', file=out)
                    print('', file=out)

        except:

            # Print end message
            if not(quiet) and not(verbose):
                print('\x1b[2K', end='\r', file=out)
            if not(quiet):
                print('    '+str(date())+' : Error...', file=out)
                print('', file=out)
                
            # Print error
            if not(quiet):
                print_exc()
            
            # Reset output model attributes
            for d in range(m.nd):
                for f in m[d].f:
                    f.dx = None
                    f.sdx = None
                    for p in f.par:
                        if not(p.fixed):
                            p.x = None
                for n in m[d].n:
                    n.xi = None
                    n.Qxi = None
                    n.sxi = None
                    for p in n.par:
                        if not(p.fixed):
                            p.x = None

                m[d].fr = None
                m[d].pn = None
                m[d].Qpn = None
                m[d].spn = None
                m[d].pv = None

                m[d].nx = None
                m[d].nb = None
                m[d].dN = None
                m[d].dH = None
                m[d].s2 = None
                m[d].Qx = None
                m[d].Qb = None
                m[d].Qc = None
                m[d].sc = None
                m[d].v = None
                m[d].Pv = None
                m[d].sv = None
                m[d].Qv = None
                m[d].vn = None
                m[d].wrms = None
                m[d].logl = None
                m[d].loglr = None
                m[d].bic = None
                m[d].bicr = None
                m[d].E = None
                m[d].Er = None

    # Fit deterministic + noise model and iteratively remove outliers
    #----------------------------------------------------------------
    def fit_iter(m, estimator='reml', method='Newton', prefit_x=True, prefit_b=True, hessian='expected', fr=None, thr_raw=None, thr_norm=None, thr_mad=None, win_mad=None, finalize=True, quiet=False, verbose=False, out=sys.stdout):
    
        """
        Fit deterministic + noise model and iteratively remove outliers

        fit does not return anything, but updates the values and formal errors of the
        model parameters, and also sets attributes of the model instance m:

        nx    : Number of deterministic parameters
        nb    : Number of noise parameters
        dQ    : Log-determinant of covariance matrix
        dN    : Log-determinant of normal matrix of deterministic parameters
        dH    : Log-determinant of normal matrix of noise parameters
        s2    : Global variance factor
        Qx    : Covariance matrix of deterministic parameters
        Qb    : Covariance matrix of noise parameters
        Qc    : Covariance matrix of predicted observations
        sc    : Formal errors of predicted observations
        v     : Residuals
        Pv    : Product of observation weight matrix with residuals
        sv    : Formal errors of residuals
        Qv    : Covariance matrix of residuals
        vn    : Normalized residuals
        wrms  : WRMS of residuals
        logl  : Log-likelihood
        loglr : Restricted log-likelihood
        bic   : -BIC/2
        bicr  : -(restricted BIC)/2
        E     : Evidence
        Er    : Restriced evidence
        
        Parameters
        ----------
        estimator : str, optional
            Specified which estimator should be used for the noise parameters:
            either 'ml', 'reml' or 'ls'. Default is 'reml'.
        method : str, optional
            Numerical maximization method to be used in case estimator is either
            'ml' or 'reml': either 'Nelder-Mead' or 'Newton'. Default is 'Newton'.
        prefit_x : bool, optional
            Should we start with a first fit of the deterministic parameters
            assuming white noise only?
        prefit_b : bool, optional
            In case estimator is either 'ml' or 'reml', should we start with a quick
            LS fit of the noise parameters? Default is True.
        hessian : str, optional
            Specifies how the hessian matrix of noise parameters should be computed:
            either 'expected' or 'numeric'. Default is 'expected'.
        fr : array, optional
            Frequency range used to compute PSD of noise model and residuals.
            Default is None (automatically set).
        thr_raw : float, optional
            Multiplicative factor defining thresholds for raw residuals:
            along each component, threshold = thr_raw * WRMS. Default is 5.
        thr_norm : float, optional
            Threshold for normalized residuals. Default is 5.
        thr_mad : float, optional
            Another threshold for raw residuals: along each component, points outside
            a "running median +/- thr_mad * running MAD" limit will be rejected.
        win_mad : float, optional
            Length of the window used to compute running median and MAD
        finalize : bool, optional
            If False, then fit_iter() will stop right after the optimal noise parameters
            are found and set, but most other attributes of the model will not be set.
            Default is True.
        quiet : bool, optional
            Whether to hide messages. Default is False.
        verbose : bool, optional
            Whether to show detailed messages. Default is False.
        out : file-like, optional
            Log file. Default is sys.stdout.

        """
        
        try:
        
            # While there remains outliers,
            i = 0
            end = False
            while (not(end)):
                i = i+1
                
                # De-activate initial LS fit of noise parameters if we're after the 1st iteration
                if (i > 1):
                    prefit_b = False
                
                # Fit model
                m.fit(estimator=estimator, method=method, prefit_x=prefit_x, prefit_b=prefit_b, hessian=hessian, fr=fr, finalize=finalize, quiet=quiet, verbose=verbose, out=out)
                
                # If necessary, compute approximate normalized residuals and WRMS assuming VW only
                # (This should be removed!)
                if not(finalize):
                    for d in range(m.nd):
                        m[d].sv = np.sqrt(m[d].s2*m[d].Q)
                        m[d].vn = m[d].v / m[d].sv
                        m[d].wrms = sqrt(np.sum((m[d].v/m[d].sv)**2) / np.sum(1/m[d].sv**2))
                        m[d].bic = m[d].logl - (m[d].nx+m[d].nb)/2*log(m.r.n)

                # If necessary, compute running median and MAD of residuals
                if (thr_mad is not None):
                    vmed = np.nan * np.ones((m.r.n, m.nd))
                    vmad = np.nan * np.ones((m.r.n, m.nd))
                    for i in range(m.r.n):
                        ind = np.nonzero(np.abs(m.r.t-m.r.t[i]) <= (win_mad-1)/2)[0]
                        for d in range(m.nd):
                            vmed[i,d] = np.median(m[d].v[ind])
                            vmad[i,d] = mad(m[d].v[ind])
                
                # Get outlier indices
                ind = []
                if (thr_raw is not None):
                    for d in range(m.nd):
                        ind = ind + np.nonzero(np.abs(m[d].v) > thr_raw*m[d].wrms)[0].tolist()
                if (thr_norm is not None):
                    for d in range(m.nd):
                        ind = ind + np.nonzero(np.abs(m[d].vn) > thr_norm)[0].tolist()
                if (thr_mad is not None):
                    for d in range(m.nd):
                        ind = ind + np.nonzero(np.abs(m[d].v - vmed[:,d]) > thr_mad*vmad[:,d])[0].tolist()
                ind = list(set(ind))
                
                # Clean outliers
                if (len(ind) > 0):
                    m.r.del_points(ind)
                    for d in range(m.nd):
                        m[d].r = m.r[d]
                    
                # Or exit
                else:
                    end = True
                    
        except:
            pass

    # Plot time series + deterministic model
    #---------------------------------------
    def plot_fit(m, figsize=None, tunit=None, dims=None, title=None, output=None, show=True):

        """
        Plot time series + deterministic model

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (see matplotlib.pyplot.figure). Default is None (automatically set).
        tunit : str, optional
            Time unit for the plot. Default is None (i.e., time unit of the time series).
        dims : str or list, optional
            Component names. Default is None.
        title : str, optional
            Figure title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. figure shown on screen).
        show : bool, optional
            Whether to show figure. Default is True.
            
        """
        
        # Plot time series
        (fig, t) = m.r.plot(figsize=figsize, tunit=tunit, dims=dims, title=title, return_fig=True)

        # Loop over components
        for d in range(m.nd):
            
            # Plot predicted observations
            fig.axes[d].plot(t, m[d].yc, 'r', linewidth=2, zorder=4)
            if (m[d].sc is not None):
                fig.axes[d].fill_between(t, m[d].yc-m[d].sc, m[d].yc+m[d].sc, color='r', alpha=0.6, zorder=4)
            
        # Save or show figure
        if (output is not None):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
        elif (show):
            pp.show()

    # Plot fit residuals
    #-------------------
    def plot_res(m, thr_raw=None, figsize=None, tunit=None, dims=None, title=None, output=None, show=True):

        """
        Plot fit residuals

        Parameters
        ----------
        thr_raw : float, optional
            If specified, then residuals larger than thr_raw*WRMS are plotted in red.
        figsize : tuple, optional
            Figure size (see matplotlib.pyplot.figure). Default is None (automatically set).
        tunit : str, optional
            Time unit for the plot. Default is None (i.e., time unit of the time series).
        dims : str or list, optional
            Component names. Default is None.
        title : str, optional
            Figure title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. figure shown on screen).
        show : bool, optional
            Whether to show figure. Default is True.
            
        """
        
        # Figure size
        if (figsize is None):
            if (m.nd == 1):
                figsize = (10, 4)
            elif (m.nd == 2):
                figsize = (10, 7)
            else:
                figsize = (10, 10)

        # Time
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            t = m.r.t
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            t = np.array([date.from_mjd(d).ydec() for d in m.r.t])
        else:
            tunit = m.r.tunit
            t = m.r.t

        # Component names
        if (dims is None):
            dims = m.r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(m.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Create new figure
        fig = pp.figure(figsize=figsize, tight_layout=True)
        
        # Loop over components
        for d in range(m.nd):
            ax = fig.add_subplot(m.nd, 1, d+1)
            ax.margins(0.01, 0.01)
            ax.grid(zorder=0)
            ax.set_ylabel(dims[d]+' residuals ['+m.r.yunit+']')
            ax.errorbar(t, m[d].v, yerr=m[d].sv, fmt='.k', ecolor='gray', zorder=3)
            if (thr_raw is not None):
                thr = thr_raw*m[d].wrms
                ind = np.nonzero(np.abs(m[d].v) > thr)[0]
                if (len(ind) > 0):
                    ax.errorbar(t[ind], m[d].v[ind], yerr=m[d].sv[ind], fmt='.r', ecolor='orange', zorder=4)
                ax.plot([t[0], t[-1]], [thr, thr], '--r', linewidth=2)
                ax.plot([t[0], t[-1]], [-thr, -thr], '--r', linewidth=2)
        ax.set_xlabel('Time ['+tunit+']')
        
        # Save or show figure
        if (output is not None):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
        elif (show):
            pp.show()

    # Plot normalized residuals
    #--------------------------
    def plot_normres(m, thr_norm=None, figsize=None, tunit=None, dims=None, title=None, output=None, show=True):

        """
        Plot fit residuals

        Parameters
        ----------
        thr_norm : float, optional
            If specified, then residuals larger than thr_norm are plotted in red.
        figsize : tuple, optional
            Figure size (see matplotlib.pyplot.figure). Default is None (automatically set).
        tunit : str, optional
            Time unit for the plot. Default is None (i.e., time unit of the time series).
        dims : str or list, optional
            Component names. Default is None.
        title : str, optional
            Figure title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. figure shown on screen).
        show : bool, optional
            Whether to show figure. Default is True.
            
        """
        
        # Figure size
        if (figsize is None):
            if (m.nd == 1):
                figsize = (10, 4)
            elif (m.nd == 2):
                figsize = (10, 7)
            else:
                figsize = (10, 10)

        # Time
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            t = m.r.t
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            t = np.array([date.from_mjd(d).ydec() for d in m.r.t])
        else:
            tunit = m.r.tunit
            t = m.r.t

        # Component names
        if (dims is None):
            dims = m.r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(m.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Create new figure
        fig = pp.figure(figsize=figsize, tight_layout=True)
        
        # Loop over components
        for d in range(m.nd):
            ax = fig.add_subplot(m.nd, 1, d+1)
            ax.margins(0.01, 0.01)
            ax.grid(zorder=0)
            ax.set_ylabel(dims[d] + ' norm. res.')
            ax.plot(t, m[d].vn, '.k', zorder=3)
            if (thr_norm is not None):
                ind = np.nonzero(np.abs(m[d].vn) > thr_norm)[0]
                if (len(ind) > 0):
                    ax.plot(t[ind], m[d].vn[ind], '.r', zorder=4)
                ax.plot([t[0], t[-1]], [thr_norm, thr_norm], '--r', linewidth=2)
                ax.plot([t[0], t[-1]], [-thr_norm, -thr_norm], '--r', linewidth=2)
        ax.set_xlabel('Time ['+tunit+']')
        
        # Save or show figure
        if (output is not None):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
        elif (show):
            pp.show()

    # Plot PSD of fit residuals and of noise model
    #---------------------------------------------
    def plot_psd(m, smooth=1, figsize=None, tunit=None, dims=None, title=None, output=None, show=True):

        """
        Plot PSD of fit residuals and of noise model

        Parameters
        ----------
        smooth : int, optional
            If specified, then PSD of residuals will be smoothed with a gaussian filter of
            standard deviation "smooth" before it is plotted. Default is 1.
        figsize : tuple, optional
            Figure size (see matplotlib.pyplot.figure). Default is None (automatically set).
        tunit : str, optional
            Time unit for the plot. Default is None (i.e., time unit of the time series).
        dims : str or list, optional
            Component names. Default is None.
        title : str, optional
            Figure title. Default is None.
        output : str, optional
            Output file. Default is None (i.e. figure shown on screen).
        show : bool, optional
            Whether to show figure. Default is True.
            
        """

        # Figure size
        if (figsize is None):
            if (m.nd == 1):
                figsize = (10, 4)
            elif (m.nd == 2):
                figsize = (10, 7)
            else:
                figsize = (10, 10)

        # Time unit and frequencies
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            fr = m[0].fr
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            fr = 365.25 * m[0].fr
        else:
            tunit = m.r.tunit
            fr = m[0].fr
    
        # Frequency unit
        if (tunit == 'd'):
            funit = 'cpd'
        else:
            funit = 'cp'+tunit

        # Component names
        if (dims is None):
            dims = m.r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(m.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Create new figure
        fig = pp.figure(figsize=figsize, tight_layout=True)
        
        # Loop over components
        for d in range(m.nd):
            ax = fig.add_subplot(m.nd, 1, d+1)
            ax.margins(0.01, 0.01)
            ax.set_ylabel(dims[d]+' res. PSD ['+m.r.yunit+'^2/'+funit+']')

            # Smoothed PSD of residuals
            pv = m[d].pv
            if (smooth is not None):
                w = signal.gaussian(2*ceil(3*smooth)+1, smooth)
                pv = signal.convolve(pv, w/np.sum(w), mode='same')

            # Useful things
            fm = exp((log(fr[0])+log(fr[-1]))/2)
            pm = np.mean(m[d].pv)
            pmax = np.max([np.max(pv), np.max(m[d].pn)])
            pmin = np.min([np.min(pv), np.min(m[d].pn)])
            
            # WN, FN and RW
            ax.loglog(fr, pm*np.ones(len(fr)), color='#b0b0b0', linewidth=0.8, zorder=0)
            p = pm * (fm/fr)
            p[p > pmax] = np.inf
            p[p < pmin] = -np.inf
            ax.loglog(fr, p, color='#b0b0b0', linewidth=0.8, zorder=0)
            p = pm * (fm/fr)**2
            p[p > pmax] = np.inf
            p[p < pmin] = -np.inf
            ax.loglog(fr, p, color='#b0b0b0', linewidth=0.8, zorder=0)
            
            # Residuals and noise model spectra
            ax.loglog(fr, pv, 'k', zorder=3)
            ax.loglog(fr, m[d].pn, 'r', linewidth=2, zorder=4)
            if (m[d].spn is not None):
                ax.fill_between(fr, m[d].pn-m[d].spn, m[d].pn+m[d].spn, color='r', alpha=0.6, zorder=4)
            
            ## Debug
            #for n in m[d].n:
                #ax.loglog(fr, n.P, '--r', zorder=4)
            #for f in m[d].f:
                #if isinstance(f, sine):
                    #ax.loglog([365.25/f.per, 365.25/f.per], [pmin, pmax], '--r', zorder=0)
            
        ax.set_xlabel('Frequency ['+funit+']')
        
        # Save or show figure
        if (output is not None):
            pp.savefig(output, bbox_inches='tight')
            pp.close()
        elif (show):
            pp.show()

    # plot_fit(), plot_res(), plot_normres() & plot_psd()
    #----------------------------------------------------
    def plot_all(m, tunit=None):

        """
        plot_fit(), plot_res(), plot_normres() & plot_psd()

        Parameters
        ----------
        tunit : str, optional
            Time unit for the plots. Default is None (i.e., time unit of the time series).
            
        """
        m.plot_fit(tunit=tunit, show=False)
        m.plot_res(tunit=tunit, show=False)
        m.plot_normres(tunit=tunit, show=False)
        m.plot_psd(tunit=tunit)

    # Print fit statistics and parameters
    #------------------------------------
    def __str__(m, tformat='10.4f', out=sys.stdout):

        """
        Print fit statistics and parameters

        Returns
        -------
        txt : str
            YAML description of model

        Parameters
        ----------
        tformat : str, optional
            Print format for dates. It can either be a float print format (e.g., '8.3f')
            or one of the following keywords: 'snx', 'iso'. Default is '10.4f'.
        out : file-like, optional
            Log file. Default is sys.stdout.
            
        """
        
        ## Restore trend
        #if (r.dtrd == 1):
            #for d in range(m.nd):
                #for f in m[d].f:
                    #if (f.type == 'poly') and (f.deg == 1):
                        #for p in f.par:
                            #p.x = p.x + r.ctrd[1,d]
        
        # Time unit
        if (m.r.tunit == 'd'):
            tunit = 'd'
        else:
            tunit = m.r.tunit
        
        # Date printing function
        if (m.r.tunit == 'd') and (tformat == 'snx'):
            dateformat = 'snx'
            def print_date(t):
                if (np.isinf(t)):
                    return '00:000:00000'
                else:
                    return date.from_mjd(t).tsnx()
        elif (m.r.tunit == 'd') and (tformat == 'iso'):
            dateformat = 'iso'
            def print_date(t):
                if (np.isinf(t)):
                    return '0000-00-00T00:00:00'
                else:
                    return date.from_mjd(t).tiso()
        else:
            dateformat = 'decimal '+tunit
            i = tformat.index('.')
            nt = int(tformat[:i])
            def print_date(t):
                if np.isinf(t) and (t < 0):
                    return ('{0:<'+str(nt)+'s}').format('-.Inf')
                elif np.isinf(t):
                    return ('{0:<'+str(nt)+'s}').format('.Inf')
                else:
                    return ('{0:'+tformat+'}').format(t)

        # Initialize output text
        txt = ''

        # Print "metadata"
        txt = txt + 'ndim: '+str(m.nd) + '\n'
        if (m.nd > 1) and (m.r.dims is not None):
            s = 'dims: ['
            for d in range(m.nd):
                s = s + m.r[d].dims + ', '
            s = s[:-2] + ']'
            txt = txt + s + '\n'
        txt = txt + 'nobs: '+str(m.r.n) + '\n'
        txt = txt + 'time_unit: '+tunit + '\n'
        txt = txt + 'series_unit: '+m.r.yunit + '\n'
        txt = txt + 'date_format: '+dateformat + '\n'
        txt = txt + 't0: '+print_date(m.t0) + '\n'
        txt = txt + '\n'
        
        # Print statistics
        if (m.nd == 1):
            txt = txt + 'npar:  {0}'.format(m[0].nx+m[0].nb) + '\n'
            txt = txt + 'wrms:  {0:13.6e}'.format(m[0].wrms) + '\n'
            txt = txt + 'logl:  {0:13.6e}'.format(m[0].logl) + '\n'
            txt = txt + 'loglr: {0:13.6e}'.format(m[0].loglr) + '\n'
            if (m[0].bic is not None):
                txt = txt + 'bic:   {0:13.6e}'.format(m[0].bic) + '\n'
            if (m[0].bicr is not None):
                txt = txt + 'bicr:  {0:13.6e}'.format(m[0].bicr) + '\n'
            if (m[0].E is not None):
                txt = txt + 'E:     {0:13.6e}'.format(m[0].E) + '\n'
            if (m[0].Er is not None):
                txt = txt + 'Er:    {0:13.6e}'.format(m[0].Er) + '\n'
        else:
            s = 'npar:  ['
            for d in range(m.nd):
                s = s + '{0}, '.format(m[d].nx+m[d].nb)
            s = s[:-2] + ']'
            txt = txt + s + '\n'
            
            if (m[d].wrms is not None):
                s = 'wrms:  ['
                for d in range(m.nd):
                    s = s + '{0:13.6e}, '.format(m[d].wrms)
                s = s[:-2] + ']'
                txt = txt + s + '\n'
            
            s = 'logl:  ['
            for d in range(m.nd):
                s = s + '{0:13.6e}, '.format(m[d].logl)
            s = s[:-2] + ']'
            txt = txt + s + '\n'
            
            s = 'loglr: ['
            for d in range(m.nd):
                s = s + '{0:13.6e}, '.format(m[d].loglr)
            s = s[:-2] + ']'
            txt = txt + s + '\n'

            if (m[d].bic is not None):
                s = 'bic:   ['
                for d in range(m.nd):
                    s = s + '{0:13.6e}, '.format(m[d].bic)
                s = s[:-2] + ']'
                txt = txt + s + '\n'

            if (m[d].bicr is not None):
                s = 'bicr:  ['
                for d in range(m.nd):
                    s = s + '{0:13.6e}, '.format(m[d].bicr)
                s = s[:-2] + ']'
                txt = txt + s + '\n'

            if (m[d].E is not None):
                s = 'E:     ['
                for d in range(m.nd):
                    s = s + '{0:13.6e}, '.format(m[d].E)
                s = s[:-2] + ']'
                txt = txt + s + '\n'

            if (m[d].Er is not None):
                s = 'Er:    ['
                for d in range(m.nd):
                    s = s + '{0:13.6e}, '.format(m[d].Er)
                s = s[:-2] + ']'
                txt = txt + s + '\n'

        txt = txt + '\n'
        
        # Loop over noise parameters
        txt = txt + 'noise_params:\n'
        for d in range(m.nd):
            for n in m[d].n:
                for p in n.par:
                    
                    # Print current parameter
                    if (n.per is not None):
                        txt = txt + '    - {{idim: {0}, type: {1:<27s}, period: {2:7.3f}, start: {3}, value: {4:13.6e}, sigma: {5:12.6e}, unit: {6}}}'.format(d, p.type, n.per, print_date(p.t), p.x, p.sig, p.unit) + '\n'
                    else:
                        txt = txt + '    - {{idim: {0}, type: {1:<44s}, start: {2}, value: {3:13.6e}, sigma: {4:12.6e}, unit: {5}}}'.format(d, p.type, print_date(p.t), p.x, p.sig, p.unit) + '\n'

        txt = txt + '\n'
        
        # Loop over deterministic parameters
        txt = txt + 'params:\n'
        for d in range(m.nd):
            for f in m[d].f:
                for p in f.par:
                    
                    # Print current parameter
                    if isinstance(f, polynom):
                        txt = txt + '    - {{idim: {0}, type: {1:<27s}, degree: {2:<7d}, start: {3}, value: {4:13.6e}, sigma: {5:12.6e}, unit: {6}}}'.format(d, p.type, f.deg, print_date(p.t), p.x, p.sig, p.unit) + '\n'
                    elif isinstance(f, sine):
                        txt = txt + '    - {{idim: {0}, type: {1:<27s}, period: {2:7.3f}, start: {3}, value: {4:13.6e}, sigma: {5:12.6e}, unit: {6}}}'.format(d, p.type, f.per, print_date(p.t), p.x, p.sig, p.unit) + '\n'
                    else:
                        txt = txt + '    - {{idim: {0}, type: {1:<44s}, start: {2}, value: {3:13.6e}, sigma: {4:12.6e}, unit: {5}}}'.format(d, p.type, print_date(p.t), p.x, p.sig, p.unit) + '\n'

        txt = txt + '\n'
        
        return txt

    # Clean matrix attributes
    #------------------------
    def clean(m):
      
        """
        Clean matrix attributes

        """

        # Loop over dimensions
        for d in range(m.nd):
            m[d].Q = None
            m[d].L = None
            m[d].P = None
            m[d].Qpn = None
            m[d].Qc = None
            m[d].Qv = None
            
            # Loop over noise components
            for n in m[d].n:
                n.Q = None
                n.dQ = None
                n.P = None
                n.dP = None
                n.Qxi = None
                
    # Dump model instance into pickle file
    #-------------------------------------
    def dump(m, file):
      
        """
        Dump model instance into pickle file

        Parameters
        ----------
        file : str
            Pickle file to write

        """

        pickle.dump(m, open(file, 'wb'))

    # Likelihood ratio test for outliers
    #-----------------------------------
    def glr_outlier(m):

        """
        Likelihood ratio test for outliers
        
        Returns
        -------
        T : array
            T-statistics for possible outliers at each date of the time series
            
        """

        # Initialization
        T = np.zeros(m.r.n)

        # Loop over dimensions
        for d in range(m.nd):

            # Useful stuff
            if (m[d].P.ndim == 2):
                CtPC = np.diag(m[d].P)
                if (m[d].nx > 0):
                    CtPA = np.dot(m[d].P, m[d].A)
                CtPv = m[d].Pv
            else:
                CtPC = m[d].P
                if (m[d].nx > 0):
                    CtPA = (m[d].A.T*m[d].P).T
                CtPv = m[d].Pv

            # Update T-statistics
            for i in range(m.r.n):
                Nc = CtPC[i]
                if (m[d].nx > 0):
                    Nc = Nc - np.dot(CtPA[i], np.dot(m[d].Qx, CtPA[i].T))
                T[i] = T[i] + np.sum(CtPv[i]**2/Nc)
                
        return T

    # Likelihood ratio test for mean changes (position discontinuities)
    #------------------------------------------------------------------
    def glr_mean(m):

        """
        Likelihood ratio test for mean changes (position discontinuities)
        
        Returns
        -------
        T : array
            T-statistics for possible mean changes at each date of the time series
            
        """

        # Initialization
        T = np.zeros(m.r.n)

        # Loop over dimensions
        for d in range(m.nd):

            # Useful stuff
            if (m[d].P.ndim == 2):
                CtP = np.cumsum(m[d].P, axis=0)
                CtPC = np.sum(np.tril(CtP), axis=1)
                if (m[d].nx > 0):
                    CtPA = np.dot(CtP, m[d].A)
                CtPv = np.dot(CtP, m[d].v)
            else:
                CtPC = np.cumsum(m[d].P)
                if (m[d].nx > 0):
                    CtPA = np.cumsum((m[d].A.T*m[d].P).T, axis=0)
                CtPv = np.cumsum(m[d].Pv)

            # Update T-statistics
            for i in range(m.r.n-1):
                Nc = CtPC[i]
                if (m[d].nx > 0):
                    Nc = Nc - np.dot(CtPA[i], np.dot(m[d].Qx, CtPA[i].T))
                if (Nc > 0):
                    T[i+1] = T[i+1] + np.sum(CtPv[i]**2/Nc)
                
        return T

    # Likelihood ratio test for trend changes (velocity discontinuities)
    #-------------------------------------------------------------------
    def glr_trend(m):

        """
        Likelihood ratio test for trend changes (velocity discontinuities)
        
        Returns
        -------
        T : array
            T-statistics for possible trend changes at each date of the time series

        """

        # Initializations
        T = np.zeros(m.r.n)
        t = m.r.t - m.t0
        
        # Loop over dimensions
        for d in range(m.nd):

            # Useful stuff
            if (m[d].P.ndim == 2):
                CtP = np.cumsum((m[d].P*t).T, axis=0) - (np.cumsum(m[d].P, axis=1)*t).T
                CtPC = np.sum(np.tril(CtP)*t, axis=1) - np.sum(np.tril(CtP), axis=1)*t
                if (m[d].nx > 0):
                    CtPA = np.dot(CtP, m[d].A)
                CtPv = np.dot(CtP, m[d].v)
            else:                
                CtPC = np.cumsum(m[d].P*t**2) - 2*t*np.cumsum(m[d].P*t) + t**2*np.cumsum(m[d].P)
                if (m[d].nx > 0):
                    CtPA = np.cumsum((m[d].A.T*m[d].P*t).T, axis=0) - (np.cumsum((m[d].A.T*m[d].P), axis=1)*t).T
                CtPv = np.cumsum(m[d].Pv*t) - t*np.cumsum(m[d].Pv)
                
            # Update T-statistics
            for i in range(m.r.n):
                Nc = CtPC[i]
                if (m[d].nx > 0):
                    Nc = Nc - np.dot(CtPA[i], np.dot(m[d].Qx, CtPA[i].T))
                if (Nc > 0):
                    T[i] = T[i] + np.sum(CtPv[i]**2/Nc)
                
        return T

    # Likelihood ratio test for mean+trend changes (position+velocity discontinuities)
    #---------------------------------------------------------------------------------
    def glr_mean_trend(m):

        """
        Likelihood ratio test for mean+trend changes (position+velocity discontinuities)
        
        Returns
        -------
        T : array
            T-statistics for possible mean+trend changes at each date of the time series
        """

        # Initializations
        T = np.zeros(m.r.n)
        t = m.r.t - m.t0
        
        # Loop over dimensions
        for d in range(m.nd):

            # Useful stuff
            if (m[d].P.ndim == 2):
                C1tP = np.cumsum(m[d].P, axis=0)
                C2tP = np.cumsum((m[d].P*t).T, axis=0)
                C1tPC1 = np.sum(np.tril(C1tP), axis=1)
                C1tPC2 = np.sum(np.tril(C2tP), axis=1)
                C2tPC2 = np.sum(np.tril(C2tP)*t, axis=1)
                C1tPA = np.dot(C1tP, m[d].A)
                C2tPA = np.dot(C2tP, m[d].A)
                C1tPv = np.dot(C1tP, m[d].v)
                C2tPv = np.dot(C2tP, m[d].v)
            else:
                C1tPC1 = np.cumsum(m[d].P)
                C1tPC2 = np.cumsum(m[d].P*t)
                C2tPC2 = np.cumsum(m[d].P*t**2)
                C1tPA = np.cumsum((m[d].A.T*m[d].P).T, axis=0)
                C2tPA = np.cumsum((m[d].A.T*m[d].P*t).T, axis=0)
                C1tPv = np.cumsum(m[d].Pv)
                C2tPv = np.cumsum(m[d].Pv*t)
                
            # Update T-statistics
            for i in range(1, m[d].r.n-2):
                CtPC = np.array([[C1tPC1[i], C1tPC2[i]], [C1tPC2[i], C2tPC2[i]]])
                CtPA = np.array([C1tPA[i], C2tPA[i]])
                CtPv = np.array([C1tPv[i], C2tPv[i]])
                Nc = CtPC - np.dot(CtPA, np.dot(m[d].Qx, CtPA.T))
                if (np.linalg.matrix_rank(Nc) == 2):
                    xc = linalg.solve(Nc, CtPv)
                    T[i+1] = np.dot(xc.T, CtPv)
                
        return T

    # Likelihood ratio test for periodic signals
    #-------------------------------------------
    def glr_sine(m, use_fft=True):

        """
        Likelihood ratio test for periodic signals
        
        Returns
        -------
        T : array
            T-statistics for possible periodic signals at each frequency of m.fr
        use_fft : bool
            Whether trigonometric sums should be computed using fast but approximate method
            of Press & Rybicki (1989). Default is True.

        """

        # Initializations
        T = np.zeros(len(m.fr))
        t = m.r.t - m.t0                

        # If series is regularly sampled (m.r.T is constant),
        if np.isscalar(m.r.T):
            
            # Full array of dates
            tf = np.arange(t[0], t[-1]+m.r.T, m.r.T)

            # Indices of observed dates
            ind = []
            j = 0
            for i in range(m.r.n):
                while (tf[j] < t[i]):
                    j = j+1
                ind.append(j)

            # Loop over dimensions
            for d in range(m.nd):
                
                # Full weight matrix
                Pf = np.zeros((len(tf), len(tf)))
                if (m[d].P.ndim == 2):
                    Pf[np.ix_(ind,ind)] = m[d].P
                else:
                    Pf[ind,ind] = m[d].P

                # Compute sum(sin(2*pi*f*(ti-tj)*Pij)) and sum(cos(2*pi*f*(ti-tj)*Pij))
                Pc = np.zeros(len(tf))
                Pc[0] = np.sum(np.diag(Pf))
                for i in range(1, len(tf)):
                    Pc[i] = 2*np.sum(np.diag(Pf, i))
                (S0, C0) = trig_sum(m.r.T*np.arange(len(tf)), Pc, m.fr[1]-m.fr[0], len(m.fr), f0=m.fr[0], use_fft=use_fft, Mfft=24)

                # Compute sum(sin(2*pi*f*(ti+tj)*Pij)) and sum(cos(2*pi*f*(ti+tj)*Pij))
                Pf = np.fliplr(Pf)
                Pc = np.zeros(2*len(tf)-1)
                for i in range(2*len(tf)-1):
                    Pc[i] = np.sum(np.diag(Pf, len(tf)-1-i))
                (S1, C1) = trig_sum(2*tf[0]+m.r.T*np.arange(2*len(tf)-1), Pc, m.fr[1]-m.fr[0], len(m.fr), f0=m.fr[0], use_fft=use_fft, Mfft=24)

                # Compute C^T*P*C at all frequencies
                CtPC = np.zeros((len(m.fr), 2, 2))
                CtPC[:,0,0] = (C0+C1)/2
                CtPC[:,0,1] = S1/2
                CtPC[:,1,0] = CtPC[:,0,1]
                CtPC[:,1,1] = (C0-C1)/2

                # Compute C^T*P*A at all frequencies
                if (m[d].nx > 0):
                    PA = np.zeros((len(tf), m[d].nx))
                    if (m[d].P.ndim == 2):
                        PA[ind,:] = np.dot(m[d].P, m[d].A)
                    else:
                        PA[ind,:] = (m[d].A.T * m[d].P).T
                    CtPA = np.zeros((len(m.fr), 2, m[d].nx))
                    for i in range(m[d].nx):
                        (CtPA[:,1,i], CtPA[:,0,i]) = trig_sum(tf, PA[:,i], m.fr[1]-m.fr[0], len(m.fr), f0=m.fr[0], use_fft=use_fft, Mfft=24)

                # Compute C^T*P*v at all frequencies
                Pv = np.zeros(len(tf))
                Pv[ind] = m[d].Pv
                CtPv = np.zeros((len(m.fr), 2))
                (CtPv[:,1], CtPv[:,0]) = trig_sum(tf, Pv, m.fr[1]-m.fr[0], len(m.fr), f0=m.fr[0], use_fft=True, Mfft=24)

                # Update T-statistics
                for i in range(len(m.fr)):    
                    Nc = CtPC[i]
                    if (m[d].nx > 0):
                        Nc2 = Nc - np.dot(CtPA[i], np.dot(m[d].Qx, CtPA[i].T))
                    if (np.linalg.det(Nc2)/np.linalg.det(Nc) > 1e-12):
                        xc = linalg.solve(Nc2, CtPv[i])
                        T[i] = T[i] + np.dot(xc.T, CtPv[i])
            
        # Else (irregularly sampled series),
        else:

            # Loop over frequencies
            for i in range(len(m.fr)):    
                C = np.zeros((m.r.n, 2))
                C[:,0] = np.cos(2*pi*m.fr[i]*t)
                C[:,1] = np.sin(2*pi*m.fr[i]*t)
                
                # Loop over dimensions
                for d in range(m.nd):
                    if (m[d].P.ndim == 2):
                        CtP = np.dot(C.T, m[d].P)
                    else:
                        CtP = C.T * m[d].P
                    CtPC = np.dot(CtP, C)
                    CtPA = np.dot(CtP, m[d].A)
                    CtPv = np.dot(CtP, m[d].v)
                        
                    Nc = CtPC - np.dot(CtPA, np.dot(m[d].Qx, CtPA.T))
                    if (np.linalg.matrix_rank(Nc) == 2):
                        xc = linalg.solve(Nc, CtPv)
                        T[i] = T[i] + np.dot(xc.T, CtPv)
                
        return T
