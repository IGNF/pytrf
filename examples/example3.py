# Imports
from pytrf import date
from pytrf.ts import ts, model, function, param, scale_param
import numpy as np
import matplotlib.pyplot as pp

#-------------------------------------------------------------------------------

# Custom "arctan" function class: f(t) = amp * atan((t-t0)/tau)
class arctan(function):

    # "arctan" function initialization with 3 parameters: t0, amp & tau
    def __init__(f, t0, amp=None, tau=None, fix_t0=False, fix_amp=False, fix_tau=False, tunit='d', yunit='m'):
        super().__init__()
        f.par.append(param(type='arctan ref time', t=-np.inf, x=t0, fixed=fix_t0, unit=tunit))        
        f.par.append(param(type='arctan amplitude', t=-np.inf, x=amp, fixed=fix_amp, unit=yunit))
        f.par.append(scale_param(type='arctan time constant', t=-np.inf, x=tau, fixed=fix_tau, unit=tunit))
            
    # Set default a priori values for unknown parameters
    def set_x0(f, m):
        if (f.par[1].x is None):
            f.par[1].x = 1
        if (f.par[2].x is None):
            f.par[2].x = 10

    # Compute predicted observations and design matrix
    def set_oeq(f, m):
        
        # Get parameters
        t0 = f.par[0].x
        amp = f.par[1].x
        tau = f.par[2].x
        
        # Useful stuff
        dt = m.r.t - t0
        da = np.arctan(dt/tau)
        d = tau**2+dt**2
        
        # Predicted observations
        f.yc = amp * da
        
        # Design matrix
        f.A = []
        if not(f.par[0].fixed):
            f.A.append(-amp*tau / d)    # Partial derivatives wrt t0
        if not(f.par[1].fixed):
            f.A.append(da)              # Partial derivatives wrt amp
        if not(f.par[2].fixed):
            f.A.append(-amp*dt / d)     # Partial derivatives wrt tau

#-------------------------------------------------------------------------------

# Read (and detrend) NGL time series of station IGUA
r = ts.read('IGUA.tenv3', skiprows=1, format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'), usecols=(3, 8, 10, 12, 14, 15, 16), dims=['East', 'North', 'Up'], dtrd=1)

# Plot time series
r.plot(tunit='y')

#-------------------------------------------------------------------------------

# Define deterministic model
m = model(r, deg=[0, 1])                  # Linear trend
m[0].add_function(arctan(t0=52300))       # Slow-slip arctan in East
m[1].add_function(arctan(t0=52300))       # Slow-slip arctan in North

# Add only variable white noise for now
m.add_vw()

# Fit model and show results
m.fit()
print(m)
m.plot_all(tunit='y')

#-------------------------------------------------------------------------------

# Add power-law noise
m.add_pl()

# Fit model and show results
m.fit(estimator='ml', method='BFGS')
print(m)
m.plot_all(tunit='y')
