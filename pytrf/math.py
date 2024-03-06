"""
pytrf math utilities

This subpackage contains several useful math routines.

"""



# External imports
#-----------------
import sys
#import mkl
#mkl.set_num_threads(1)
import numpy as np
from scipy import linalg, signal, special, sparse
from astropy.timeseries import LombScargle
from math import pi, cos, sin, tan, atan, atan2, sqrt, log, log10, exp, ceil

# Internal imports
#-----------------
from pytrf.const import ae, ee, fe

# Cholesky factorization of a symmetric positive definite matrix
#---------------------------------------------------------------
def identity(x):
    """
    Identity function used in some reparametrization cases.

    Returns
    -------
    x : float
        Parameter.

    Parameters
    ----------
    x : float
        Parameter.
    """
    return x

def a2ar(a) :
    """
    Reparametrization function for the spectral index "a" into a reparametrized "ar"
    so that a cannot reach 3.0.

    Returns
    -------
    ar : float
        Reparametrized spectral index.

    Parameters
    ----------
    a : float
        Spectral index.
    """
    
    return -log(exp(3-a)-1)

def ar2a(ar) :
    """
    Inverse reparametrization function for the spectral index.

    Returns
    ----------
    a : float
        Spectral index.
        
    Parameters
    -------
    ar : float
        Reparametrized spectral index.
    """
    
    return 3-log(1+exp(-ar))

def dar2a(ar) :
    """
    Derivative of the inverse reparametrization function for the spectral index.

    Returns
    ----------
    dar2a : float
        Derivative of the inverse reparametrization function.
        
    Parameters
    -------
    ar : float
        Reparametrized spectral index.
    """
    
    return 1-exp(ar-3)

# Cholesky factorization of a symmetric positive definite matrix
#---------------------------------------------------------------
def cholesky(A):
  
    """
    Cholesky factorization of a symmetric positive definite matrix

    Returns
    -------
    f : (n,) array_like
        Vector of the inverse square roots of the diagonal elements of A
    L : (n, n) array_like
        Cholesky factorization of A

    Parameters
    ----------
    A : (n, n) array_like
        Symmetric positive definite matrix
    """
    
    f = 1. / np.sqrt(np.diag(A))
    (L, info) = linalg.lapack.dpotrf(f*(A*f).T)
    if (info != 0):
        raise RuntimeError('Cholesky factorization failed')

    return (f, L)

# Solve linear system A*x = b given Cholesky factorization of A
#--------------------------------------------------------------
def cholsolve(fL, b):
  
    """
    Solve linear system A*x = b given Cholesky factorization of A

    Returns
    -------
    x : (n, p) array_like
        Solution of A*x = b

    Parameters
    ----------
    fL : tuple
        Output of cholesky(A)
    b : (n, p) array_like
        Right-hand side
    """
    f = fL[0]
    L = fL[1]
    (x, info) = linalg.lapack.dpotrs(L, (f*b.T).T)
    if (info != 0):
        raise RuntimeError('Cholesky solving failed')

    return (f*x.T).T

# Inversion of a symmetric positive definite matrix
#--------------------------------------------------
def invspd(A, return_det=False):
  
    """
    Inversion of a symmetric positive definite matrix

    Returns
    -------
    L : (n, n) array_like
        The inverse of A
    ld : float, optional
        The log-determinant of A

    Parameters
    ----------
    A : (n, n) array_like
        Symmetric positive definite matrix
    return_det : bool, optional
        Whether the log-determinant of A should be returned
        
    """

    (f, L) = cholesky(A)
    
    if (return_det):
        ld = 2 * (np.sum(np.log(np.diag(L))) - np.sum(np.log(f)))
    
    (L, info) = linalg.lapack.dpotri(L, overwrite_c=True)
    if (info != 0):
        raise RuntimeError('Cholesky inversion failed')
    L = np.triu(L) + np.triu(L, 1).T
    
    if (return_det):
        return f*(L*f).T, ld
    else:
        return f*(L*f).T

# Pseudo-inversion of a symmetric positive semi-definite matrix
#--------------------------------------------------------------
def pinvspd(A):
  
    """
    Pseudo-inversion of a symmetric positive semi-definite matrix

    Returns
    -------
    B : (n, n) array_like
        The pseudo-inverse of A

    Parameters
    ----------
    A : (n, n) array_like
        Symmetric positive semi-definite matrix
    """
    
    return linalg.pinvh(A)

# Compute trace of the product of two matrices
#---------------------------------------------
def trdot(A, B):
  
    """
    Compute trace of the product of two matrices

    Returns
    -------
    t : float
        trace(A*B)

    Parameters
    ----------
    A : (...,n) array_like
    B : (n,...) array_like
    """
    
    return np.einsum('ij,ji->', A, B)

# Fit linear trend to a time series
#----------------------------------
def trend(t, x):
  
    """
    Fit linear trend to a time series

    Returns
    -------
    a : (2,) array_like
        a[0] = intercept; a[1] = slope

    Parameters
    ----------
    t : (n,) array_like
        Dates
    x : (n,) array_like
        Time series values
    """

    n = len(t)
    tm = np.mean(t)
    dt = t-tm
    
    a = np.zeros(2)
    a[1] = np.sum(dt*x) / np.sum(dt**2)
    a[0] = np.sum(x)/n - a[1]*tm
    
    return a

# Remove linear trend from a time series
#---------------------------------------
def detrend(t, x):
  
    """
    Remove linear trend from a time series

    Returns
    -------
    d : (n,) array_like
        Detrended time series

    Parameters
    ----------
    t : (n,) array_like
        Dates
    x : (n,) array_like
        Time series values
    """

    a = trend(t, x)
    d = x - a[0] - a[1]*t

    return d

# Transform cartesian coordinates into geographical coordinates
#--------------------------------------------------------------
def cart2geo(X):
  
    """
    Transform cartesian coordinates into geographical coordinates

    Returns
    -------
    phi : (...) array_like or float
        Latitude(s) [rad]
    lam : (...) array_like or float
        Longitude(s) [rad]
    h : (...) array_like or float
        Ellipsoidal height(s) [m]

    Parameters
    ----------
    X : (...,3) array_like or (3,) array_like
        Cartesian coordinates [m]
    """
    
    p = np.sqrt(X[...,0]**2 + X[...,1]**2)
    r = np.sqrt(X[...,0]**2 + X[...,1]**2 + X[...,2]**2)
    u = np.arctan2(X[...,2]/p * (1 - fe + ee**2 * ae/r), 1)

    lam = 2 * np.arctan2(X[...,1], X[...,0] + p)
    phi = np.arctan2(X[...,2] * (1-fe) + ee**2 * ae * np.sin(u)**3, (1-fe) * (p - ee**2 * ae * np.cos(u)**3))
    h = p * np.cos(phi) + X[...,2] * np.sin(phi) - ae * np.sqrt(1 - ee**2 * np.sin(phi)**2)

    return (phi, lam, h)

# Transform geographical coordinates into cartesian coordinates
#--------------------------------------------------------------
def geo2cart(phi, lam, h):
  
    """
    Transform geographical coordinates into cartesian coordinates

    Returns
    -------
    X : (...,3) array_like or (3,) array_like
        Cartesian coordinates [m]

    Parameters
    ----------
    phi : (...) array_like or float
        Latitude(s) [rad]
    lam : (...) array_like or float
        Longitude(s) [rad]
    h : (...) array_like or float
        Ellipsoidal height(s) [m]
    """

    N = ae / np.sqrt(1 - (ee*np.sin(phi))**2)

    if (isinstance(phi, float)):
        X = np.zeros(3,)
    else:
        X = np.zeros(phi.shape + (3,))

    X[...,0] = (N + h) * np.cos(phi) * np.cos(lam)
    X[...,1] = (N + h) * np.cos(phi) * np.sin(lam)
    X[...,2] = (N*(1-ee**2) + h) * np.sin(phi)

    return X

# Compute rotation matrices from geocentric to topocentric coordinates
#---------------------------------------------------------------------
def xyz2enh(X):
  
    """
    Compute rotation matrices from geocentric to topocentric coordinates

    Returns
    -------
    R : (...,3,3) array_like or (3,3) array_like
        Rotation matrice(s) from geocentric to topocentric coordinates

    Parameters
    ----------
    X : (...,3) array_like or (3,) array_like
        Cartesian coordinates [m]
    """

    (phi, lam, h) = cart2geo(X)

    R = np.zeros(phi.shape + (3, 3))
    R[..., 0, 0] = -np.sin(lam)
    R[..., 0, 1] =  np.cos(lam)
    R[..., 1, 0] = -np.sin(phi) * np.cos(lam)
    R[..., 1, 1] = -np.sin(phi) * np.sin(lam)
    R[..., 1, 2] =  np.cos(phi)
    R[..., 2, 0] =  np.cos(phi) * np.cos(lam)
    R[..., 2, 1] =  np.cos(phi) * np.sin(lam)
    R[..., 2, 2] =  np.sin(phi)

    return R

# Compute rotation matrices from topocentric frames of two stations to their UVH frame
#-------------------------------------------------------------------------------------
def enh2uvh(X1, X2):
  
    """
    Compute rotation matrices from topocentric frames of two stations to their UVH frame

    Returns
    -------
    R1 : (3,3) array_like
        Rotation matrix from topocentric frame of first station to UVH frame
    R2 : (3,3) array_like
        Rotation matrix from topocentric frame of second station to UVH frame

    Parameters
    ----------
    X1 : (3,) array_like
        Cartesian coordinates of first station [m]
    X2 : (3,) array_like
        Cartesian coordinates of second station [m]
    """

    (p1, l1, h1) = cart2geo(X1)
    (p2, l2, h2) = cart2geo(X2)
      
    az = atan2(sin(l2-l1), cos(p1)*tan(p2) - sin(p1)*cos(l2-l1))
    s = sin(az)
    c = cos(az)
    
    R1 = np.zeros((3, 3))
    R1[0,0] =  s
    R1[0,1] =  c
    R1[1,0] = -c
    R1[1,1] =  s
    R1[2,2] =  1
    
    az = atan2(sin(l1-l2), cos(p2)*tan(p1) - sin(p2)*cos(l1-l2))
    s = sin(az+pi)
    c = cos(az+pi)
    
    R2 = np.zeros((3, 3))
    R2[0,0] =  s
    R2[0,1] =  c
    R2[1,0] = -c
    R2[1,1] =  s
    R2[2,2] =  1
    
    return (R1, R2)

# Low-pass Vondrak filter
#------------------------
def vondrak(t, x, fc):
  
    """
    Low-pass Vondrak filter

    Returns
    -------
    xs : (n,) array_like
        Low-pass filtered time series

    Parameters
    ----------
    t : (n,) array_like
        Dates
    x : (n,) array_like
        Time series values
    fc : float
        Cutoff frequency in units of 1/t
    """

    eps = (7.223147119819503*fc) ** 6 / (len(t) - 3)
    num = 6 * np.sqrt(t[2:-1] - t[1:-2])
    den = sqrt(t[-1] - t[0])

    a = np.hstack((0, 0, 0, num / den / ((t[0:-3] - t[1:-2]) * (t[0:-3] - t[2:-1]) * (t[0:-3] - t[3:])),   0, 0, 0))
    b = np.hstack((0, 0, 0, num / den / ((t[1:-2] - t[0:-3]) * (t[1:-2] - t[2:-1]) * (t[1:-2] - t[3:])),   0, 0, 0))
    c = np.hstack((0, 0, 0, num / den / ((t[2:-1] - t[0:-3]) * (t[2:-1] - t[1:-2]) * (t[2:-1] - t[3:])),   0, 0, 0))
    d = np.hstack((0, 0, 0, num / den / ((t[3:]   - t[0:-3]) * (t[3:]   - t[1:-2]) * (t[3:]   - t[2:-1])), 0, 0, 0))

    d0 = eps + a[3:]**2 + b[2:-1]**2 + c[1:-2]**2 + d[0:-3]**2
    d1 = a[3:-1] * b[3:-1] + b[2:-2] * c[2:-2] + c[1:-3] * d[1:-3]
    d2 = a[3:-2] * c[3:-2] + b[2:-3] * d[2:-3]
    d3 = a[3:-3] * d[3:-3]

    A = np.zeros((4, len(d0)))
    A[0,:] = d0
    A[1,:-1] = d1
    A[2,:-2] = d2
    A[3,:-3] = d3

    return linalg.solveh_banded(A, eps*x, lower=True)

# Lomb-Scargle periodogram
#-------------------------
def lombscargle(t, x, sf=4, f=None, dtrd=0, normalize=False):
    
    """
    Lomb-Scargle periodogram

    Returns
    -------
    f : array_like
        Frequencies of output periodogram in units of 1/t
    p : array_like
        Lomb-Scargle periodogram

    Parameters
    ----------
    t : (n,) array_like
        Dates
    x : (n,) array_like
        Time series values
    sf : int, optional
        Oversampling factor. Default is 4.
    f : array_like, optional
        Frequencies of output periodogram in units of 1/t. Automatically set by default.
    dtrd : int, optional
        Degree of detrending polynomial.
        - If 0, then an average is removed from the time series before computing the periodogram.
        - If 1, then a linear trend is removed from the time series before computing the periodogram.
        - If anything else, the time series is not detrended.
        Default is 0.
    normalize : bool, optional
        True for a normalized periodogram. Default is False.
    """

    # Time span and sampling of the series
    T = t[-1] - t[0]
    dt = np.min(t[1:]-t[:-1])
    
    # Detrend time series
    if (dtrd == 0):
        x = x - np.mean(x)
    elif (dtrd == 1):
        x = detrend(t, x)
    
    # Normalize time series by its standard deviation if needed
    if (normalize):
        x = x / np.std(x)

    # Build list of frequencies if needed
    if (f is None):
        f0 = 1/T
        fc = 1/(2*dt)
        df = f0/sf
        f = np.arange(f0, fc+df, df)

    # Compute periodogram
    p = LombScargle(t, x, normalization='psd', fit_mean=False, center_data=False).power(f)
    
    return (f, p)

# Morlet wavelet scalogram
#-------------------------
def scalogram(t, y, w=5, f=None):
    
    """
    Morlet wavelet scalogram
    
    Warning: The time series may have gaps. On the other hand,
    it is supposed to have a constant integration interval.

    Returns
    -------
    tf : array_like
        Dates of output scalogram
    f : array_like
        Frequencies of output scalogram
    p : array_like
        Scalogram

    Parameters
    ----------
    t : (n,) array_like
        Dates
    y : (n,) array_like
        Time series values
    w : float, optional
        Morlet wavelet omega0. Default is 5.
        Larger w -> better frequency resolution
        Lower  w -> better time resolution
    f : array_like, optional
        Frequencies of output periodogram in units of 1/t. Automatically set by default.
    """

    # Time series integration interval
    T = np.min(t[1:] - t[:-1])
    
    # Normalize time and frequency in units of integration interval
    t = t / T
    if (f is not None):
        f = f * T
        
    # Full array of dates
    tf = np.arange(t[0], t[-1]+1)

    # Indices of observed dates
    ind = []
    j = 0
    for i in range(len(t)):
        while (tf[j] < t[i]):
            j = j+1
        ind.append(j)

    # Full time series
    yf = np.zeros(len(tf))
    yf[ind] = y

    # Observation mask
    mask = np.zeros(len(tf))
    mask[ind] = 1
    
    # If necessary, set array of frequencies
    n = t[-1] - t[0]
    if (f is None):
        f = np.arange(1/n, 1/2+1/n, 1/n)
        
    # Initialize scalogram
    p = np.zeros((len(tf), len(f)))
    
    # Loop over frequencies
    for i in range(len(f)):
        
        # Wavelet of current frequency
        n = ceil(5*w/(pi*f[i]))
        s = n*f[i]/(2*w)
        m = signal.morlet(n, s=s, w=w)
        mc = np.real(m)
        ms = np.imag(m)
        
        # Convolve full time series with wavelet
        pc = signal.convolve(yf, mc, mode='same')
        ps = signal.convolve(yf, ms, mode='same')
        
        # Squared wavelet
        mcc = mc**2
        mcs = mc*ms
        mss = ms**2

        # Convolve squared wavelet with observation mask
        pcc = signal.convolve(mask, mcc, mode='same')
        pcs = signal.convolve(mask, mcs, mode='same')
        pss = signal.convolve(mask, mss, mode='same')

        # Time series / wavelet regression coefficients
        d = pcc*pss - pcs**2
        ind = np.nonzero(d / (np.sum(mcc)*np.sum(mss)) < 0.25)[0]
        d[ind] = np.nan
        xc = (pss*pc - pcs*ps) / d
        xs = (pcc*ps - pcs*pc) / d

        # Scalogram
        p[:,i] = pc*xc + ps*xs
        
    # Un-normalize time & frequency
    tf = tf * T
    f = f / T

    return (tf, f, p)

# Compute correlation matrix from covariance matrix
#--------------------------------------------------
def cov2corr(Q):
  
    """
    Compute correlation matrix from covariance matrix

    Returns
    -------
    C : (n, n) array_like
        The corresponding correlation matrix

    Parameters
    ----------
    Q : (n, n) array_like
        A covariance matrix
    """

    f = 1. / np.sqrt(np.diag(Q))

    return f*(Q*f).T

# Compute co-seismic displacements at given point
#------------------------------------------------
def compute_csd(eq, lon, lat, return_max=False):
  
    """
    Compute co-seismic displacements at given point
    
    Returns
    -------
    denh1 : (3,) array_like
        Co-seismic displacement for 1st nodal plane [mm]
    denh2 : (3,) array_like
        Co-seismic displacement for 2nd nodal plane [mm]
        
    or if return_max==True,
    
    denh : (3,) array_like
        Max of both co-seismic displacements along each axis [mm]

    Parameters
    ----------
    eq : earthquake record
        Earthquake record (from CMT catalog)
    lon : float
        Longitude (deg)
    lat : float
        Latitude (deg)
        
    """
    
    # Point coordinates wrt earthquake location [km]
    e = cos(pi/180*eq.lat) * pi/180 * (lon-eq.lon) * ae/1000
    n = pi/180 * (lat-eq.lat) * ae/1000
    
    # Compute co-seismic displacement for each nodal plane
    denh1 = 1000 * okada(e, n, eq.depth, eq.strike1, eq.dip1, eq.length/1000, eq.width/1000, eq.rake1, eq.slip)
    denh2 = 1000 * okada(e, n, eq.depth, eq.strike2, eq.dip2, eq.length/1000, eq.width/1000, eq.rake2, eq.slip)
    
    # If return_max==True, return max co-seismic displacement along each axis
    if (return_max):
        if (abs(denh2[0]) > abs(denh1[0])):
            denh1[0] = denh2[0]
        if (abs(denh2[1]) > abs(denh1[1])):
            denh1[1] = denh2[1]
        if (abs(denh2[2]) > abs(denh1[2])):
            denh1[2] = denh2[2]
            
        return denh1
    
    # Else, return both co-seismic displacements
    else:
        return (denh1, denh2)
    
# Adaptation of Beauducel's okada85.m routine
#--------------------------------------------
def okada(e, n, depth, strike, dip, length, width, rake, slip, opening=0, nu=0.25):

    strike = pi/180*strike
    dip = pi/180*dip
    rake = pi/180*rake
    
    L = length
    W = width

    U1 = cos(rake) * slip
    U2 = sin(rake) * slip
    U3 = opening
    
    d = depth + sin(dip) * W / 2
    ec = e + cos(strike) * cos(dip) * W / 2
    nc = n - sin(strike) * cos(dip) * W / 2
    x = cos(strike) * nc + sin(strike) * ec + L / 2
    y = sin(strike) * nc - cos(strike) * ec + cos(dip) * W
    p = y * cos(dip) + d * sin(dip)
    q = y * sin(dip) - d * cos(dip)

    ux = - U1 / (2 * pi) * chinnery(ux_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * pi) * chinnery(ux_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * pi) * chinnery(ux_tf, x, p, L, W, q, dip, nu)
    uy = - U1 / (2 * pi) * chinnery(uy_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * pi) * chinnery(uy_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * pi) * chinnery(uy_tf, x, p, L, W, q, dip, nu)
    uz = - U1 / (2 * pi) * chinnery(uz_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * pi) * chinnery(uz_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * pi) * chinnery(uz_tf, x, p, L, W, q, dip, nu)
    ue = sin(strike) * ux - cos(strike) * uy
    un = cos(strike) * ux + sin(strike) * uy

    return np.array([ue, un, uz])
  
# Adaptation of Beauducel's okada85.m subroutines
#------------------------------------------------
def chinnery(f, x, p, L, W, q, dip, nu):
    return f(x, p, q, dip, nu) - f(x, p - W, q, dip, nu) - f(x - L, p, q, dip, nu) + f(x - L, p - W, q, dip, nu) 

def ux_ss(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = xi * q / (R * (R + eta)) + I1(xi, eta, q, dip, nu, R) * sin(dip)
    if (q != 0):
        u = u + atan( (xi * eta) / (q * R) )
    return u

def uy_ss(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = (eta * cos(dip) + q * sin(dip)) * q / (R * (R + eta)) + q * cos(dip) / (R + eta) + I2(eta, q, dip, nu, R) * sin(dip)
    return u

def uz_ss(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    db = eta * sin(dip) - q * cos(dip)
    u = (eta * sin(dip) - q * cos(dip)) * q / (R * (R + eta)) + q * sin(dip) / (R + eta) + I4(db, eta, q, dip, nu, R) * sin(dip)
    return u

def ux_ds(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = q / R - I3(eta, q, dip, nu, R) * sin(dip) * cos(dip)
    return u

def uy_ds(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = (eta * cos(dip) + q * sin(dip)) * q / (R * (R + xi)) - I1(xi, eta, q, dip, nu, R) * sin(dip) * cos(dip)
    if (q != 0):
        u = u + cos(dip) * atan( (xi * eta) / (q * R))
    return u

def uz_ds(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    db = eta * sin(dip) - q * cos(dip)
    u = db * q / (R * (R + xi)) - I5(xi, eta, q, dip, nu, R, db) * sin(dip) * cos(dip)
    if (q != 0):
        u = u + sin(dip) * atan( (xi * eta) / (q * R))
    return u

def ux_tf(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = q**2 / (R * (R + eta)) - I3(eta, q, dip, nu, R) * (sin(dip)**2)
    return u

def uy_tf(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    u = - (eta * sin(dip) - q * cos(dip)) * q / (R * (R + xi)) - sin(dip) * xi * q / (R * (R + eta)) - I1(xi, eta, q, dip, nu, R) * (sin(dip)**2)
    if (q != 0):
        u = u + sin(dip) * atan( (xi * eta) / (q * R) )
    return u

def uz_tf(xi, eta, q, dip, nu):
    R = sqrt(xi**2 + eta**2 + q**2)
    db = eta * sin(dip) - q * cos(dip)
    u = (eta * cos(dip) + q * sin(dip)) * q / (R * (R + xi)) + cos(dip) * xi * q / (R * (R + eta)) - I5(xi, eta, q, dip, nu, R, db) * sin(dip)**2
    if (q != 0):
        u = u - cos(dip) * atan( (xi * eta) / (q * R) )
    return u

def I1(xi, eta, q, dip, nu, R):
    db = eta * sin(dip) - q * cos(dip)
    if cos(dip) > 1e-14:
        I = (1 - 2 * nu) * (- xi / (cos(dip) * (R + db))) - sin(dip) / cos(dip) * I5(xi, eta, q, dip, nu, R, db)
    else:
        I = -(1 - 2 * nu) / 2 * xi * q / (R + db)**2
    return I

def I2(eta, q, dip, nu, R):
    I = (1 - 2 * nu) * (-log(R + eta)) - I3(eta, q, dip, nu, R)
    return I

def I3(eta, q, dip, nu, R):
    yb = eta * cos(dip) + q * sin(dip)
    db = eta * sin(dip) - q * cos(dip)
    if cos(dip) > 1e-14:
        I = (1 - 2 * nu) * (yb / (cos(dip) * (R + db)) - log(R + eta)) + sin(dip) / cos(dip) * I4(db, eta, q, dip, nu, R)
    else:
        I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db)**2 - log(R + eta))
    return I

def I4(db, eta, q, dip, nu, R):
    if cos(dip) > 1e-14:
        I = (1 - 2 * nu) * 1.0 / cos(dip) * (log(R + db) - sin(dip) * log(R + eta))
    else:
        I = - (1 - 2 * nu) * q / (R + db)
    return I

def I5(xi, eta, q, dip, nu, R, db):
    X = sqrt(xi**2 + q**2)
    if (xi == 0):
        I = 0
    elif cos(dip) > 1e-14:
        I = (1 - 2 * nu) * 2 / cos(dip) * atan((eta * (X + q*cos(dip)) + X*(R + X) * sin(dip)) / (xi*(R + X) * cos(dip))) 
    else:
        I = -(1 - 2 * nu) * xi * sin(dip) / (R + db)
    return I
