#--------------------------------------------------------------------------------------------------------------------------------
# This script illustrates two features of pytrf's "ts" module:
#  - the simulation of time series
#  - the estimation of the different noise components contained in a time series by Wiener-filtering.
#
# In step 1, a time series is simulated, which is composed of a linear trend, a periodic signal, white noise and flicker noise.
# 
# In step 2, a model is adjusted to the time series. The model is also composed of a linear trend, a periodic signal, white noise
# and flicker noise, but all the parameters are unknown and estimated from the time series. At the end of the adjustment, the
# "model" object contains least-squares estimates of the linear trend and of the periodic signal contained in the time series,
# but also estimates of the white noise and flicker noise components contained in the time series. Those estimates are obtained
# by Wiener-filtering the deterministic model residuals. They are the best linear unbiased estimates (BLUE) of each noise
# component.
# 
# A figure is finally produced which compares each component of the simulated time series with its estimates.
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import matplotlib.pyplot as pp
from pytrf.ts import ts, model



# Step 1: Simulate time series
#-----------------------------

# Initialize 1000-point long time series with 0 values
r = ts(1000)

# Define model composed of:
# - a linear trend        = -1.5 + 0.003*(t-t0)
# - a periodic signal     = 2.5*cos(2*pi*(t-t0)/100) - 2.5*sin(2*pi*(t-t0)/100)  
# - white noise           (with variance factor = 2)
# - flicker noise         (with variance factor = 1)
m0 = model(r)
m0.add_polynom(deg=0, x=-1.5)
m0.add_polynom(deg=1, x=0.003)
m0.add_sine(per=100, x=[2.5, -2.5])
m0.add_wn(s2=2)
m0.add_fn(s2=1)

# As long as all the deterministic and noise parameters of a model object m0 have
# values assigned, it's possible to call m0.simulate(). This will overwrite the values
# of the time series m0.r with values simulated from the model parameters.
m0.simulate()

# Plot every component of the simulated time series, and the total simulated time series.
fig = pp.figure(figsize=(12, 10), tight_layout=True)

ax = fig.add_subplot(411)                   # Deterministic component
pp.plot(r.t, m0.yc, 'k')
pp.axis([0, 1000, -6, 6])
pp.grid()
ax.set_xticklabels([])
pp.ylabel('deterministic component')

ax = fig.add_subplot(412)                   # White noise
pp.plot(r.t, m0.n[0].xi, 'k')
pp.axis([0, 1000, -6, 6])
pp.grid()
ax.set_xticklabels([])
pp.ylabel('white noise')

ax = fig.add_subplot(413)                   # Flicker noise
pp.plot(r.t, m0.n[1].xi, 'k')
pp.axis([0, 1000, -6, 6])
pp.grid()
ax.set_xticklabels([])
pp.ylabel('flicker noise')

ax = fig.add_subplot(414)                   # Total time series
pp.plot(r.t, r.y, 'k')
pp.axis([0, 1000, -10, 10])
pp.grid()
pp.xlabel('time')
pp.ylabel('total time series')

pp.show()



# Step 2: Adjust model
#---------------------

# Create a second model object with the same components, but no values assigned to parameters!
m = model(r, deg=[0, 1], per=[100], noise=['wn', 'fn'])

# Fit model to the simulated time series. Print results.
m.fit()
print(m)

# Plot each estimated component and its +/- 1 sigma envelope on top of the "true" (simulated) component
fig = pp.figure(figsize=(12, 10), tight_layout=True)

ax = fig.add_subplot(411)                   # Deterministic component
pp.plot(r.t, m0.yc, 'k', label='true')
pp.plot(r.t, m.yc, 'r', lw=2, label='estimated')
ax.fill_between(r.t, m.yc-m.sc, m.yc+m.sc, color='r', alpha=0.4)
pp.axis([0, 1000, -6, 6])
pp.grid()
pp.legend()
ax.set_xticklabels([])
pp.ylabel('deterministic component')

ax = fig.add_subplot(412)                   # White noise
pp.plot(r.t, m0.n[0].xi, 'k', label='true')
pp.plot(r.t, m.n[0].xi, 'turquoise', lw=2, label='estimated')
ax.fill_between(r.t, m.n[0].xi-m.n[0].sxi, m.n[0].xi+m.n[0].sxi, color='turquoise', alpha=0.4)
pp.axis([0, 1000, -6, 6])
pp.grid()
pp.legend()
ax.set_xticklabels([])
pp.ylabel('white noise')

ax = fig.add_subplot(413)                   # Flicker noise
pp.plot(r.t, m0.n[1].xi, 'k', label='true')
pp.plot(r.t, m.n[1].xi, 'hotpink', lw=2, label='estimated')
ax.fill_between(r.t, m.n[1].xi-m.n[1].sxi, m.n[1].xi+m.n[1].sxi, color='hotpink', alpha=0.4)
pp.axis([0, 1000, -6, 6])
pp.grid()
pp.legend()
ax.set_xticklabels([])
pp.ylabel('flicker noise')

ax = fig.add_subplot(414)                   # Total time series
pp.plot(r.t, r.y, 'k')
pp.axis([0, 1000, -10, 10])
pp.grid()
pp.xlabel('time')
pp.ylabel('total time series')

pp.show()
