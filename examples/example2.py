# Imports
from pytrf import date
from pytrf.ts import ts, model
import numpy as np
import matplotlib.pyplot as pp

#-------------------------------------------------------------------------------

# Read JPL time series of station RIOP
r = ts.read('RIOP.series', format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz'), usecols=(10, 1, 2, 3, 4, 5, 6), dims=['East', 'North', 'Up'])

# Convert original dates (seconds since J2000) into MJDs (at noon)
r.t = date.from_ymdhms(2000, 1, 1, 12).mjd + r.t/86400
r.t = np.ceil(r.t) - 0.5

# Set time series integration interval to 1 day
r.T = 1

# Remove occasional points at the same dates as the previous ones
ind = []
for i in range(r.n-1):
    if (r.t[i] == r.t[i+1]):
        ind.append(i)
r.del_points(ind)

# Remove points with unusually large formal errors
r.clean_sigmas()

# Restrict time series to after 2010.0
r.trim(date.from_ymdhms(2010, 1, 1).mjd, np.inf)

# Detrend time series
r.detrend()

# Plot time series
r.plot(tunit='y')

#-------------------------------------------------------------------------------

# Define deterministic model
m = model(r, deg=[0, 1], per=[365.25, 182.625])     # Linear trend + annual & semi-annual sine waves
m.add_jumps([57495], deg=[0, 1])                    # Position + velocity jump (M7.8 earthquake - 27km SSE of Muisne, Ecuador)
m[0].add_exp(t0=57495)                              # Post-seismic exponential in East
m[1].add_exp(t0=57495)                              # Post-seismic exponential in North

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
m.fit(method='BFGS')
print(m)
m.plot_all(tunit='y')
