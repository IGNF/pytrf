# Imports
import matplotlib.pyplot as pp
from pytrf.ts import ts, model

#-------------------------------------------------------------------------------

# Initialize 3000-point long time series with 0 values
r = ts(3000)

# Define model including:
# - linear trend        = 1 + 0.003*(t-t0)
# - sine wave           = 2.5*cos(2*pi*(t-t0)/200) - 2.5*sin(2*pi*(t-t0)/200)  
# - white noise         (with variance factor = 2)
# - flicker noise       (with variance factor = 1)
# - AR(1)*sine process  (to simulate a non-stationary behaviour of the above sine wave)
m0 = model(r)
m0.add_polynom(deg=0, x=1)
m0.add_polynom(deg=1, x=0.003)
m0.add_sine(per=200, x=[2.5, -2.5])
m0.add_wn(s2=2)
m0.add_fn(s2=1)
m0.add_ar1(per=200, s2=0.005, tau=600)

# Simulate time series
m0.simulate()

# Plot simulated time series and its different elements
fig = pp.figure(figsize=(10, 10), tight_layout=True)

ax = fig.add_subplot(4, 1, 1)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Deterministic model')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.yc, 'r', linewidth=2, zorder=4)

ax = fig.add_subplot(4, 1, 2)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('White noise')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[0].xi, 'r', linewidth=2, zorder=4)

ax = fig.add_subplot(4, 1, 3)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Flicker noise')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[1].xi, 'r', linewidth=2, zorder=4)

ax = fig.add_subplot(4, 1, 4)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('AR(1)*sine')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[2].xi, 'r', linewidth=2, zorder=4)
pp.show()

#-------------------------------------------------------------------------------

# Define 2nd model instance with the same components
# (but no values assigned to parameters)
m = model(r, deg=[0, 1], per=[200], noise=['wn', 'fn'])
m.add_ar1(per=200)

# Fit model and print results
m.fit(estimator='reml', method='Newton')
print(m)

# Plot estimated elements on top of true elements
fig = pp.figure(figsize=(10, 10), tight_layout=True)

ax = fig.add_subplot(4, 1, 1)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Deterministic model')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.yc, 'r', linewidth=2, zorder=4)
ax.plot(r.t, m.yc, 'c', linewidth=2, zorder=4)
ax.fill_between(r.t, m.yc-m.sc, m.yc+m.sc, alpha=0.4)

ax = fig.add_subplot(4, 1, 2)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('White noise')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[0].xi, 'r', linewidth=2, zorder=4)
ax.plot(r.t, m.n[0].xi, 'c', linewidth=2, zorder=4)
ax.fill_between(r.t, m.n[0].xi-m.n[0].sxi, m.n[0].xi+m.n[0].sxi, alpha=0.4)

ax = fig.add_subplot(4, 1, 3)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Flicker noise')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[1].xi, 'r', linewidth=2, zorder=4)
ax.plot(r.t, m.n[1].xi, 'c', linewidth=2, zorder=4)
ax.fill_between(r.t, m.n[1].xi-m.n[1].sxi, m.n[1].xi+m.n[1].sxi, alpha=0.4)

ax = fig.add_subplot(4, 1, 4)
ax.margins(0.01, 0.01)
ax.grid(zorder=0)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('AR(1)*sine')
ax.plot(r.t, r.y, '.k', zorder=3)
ax.plot(r.t, m0.n[2].xi, 'r', linewidth=2, zorder=4)
ax.plot(r.t, m.n[2].xi, 'c', linewidth=2, zorder=4)
ax.fill_between(r.t, m.n[2].xi-m.n[2].sxi, m.n[2].xi+m.n[2].sxi, alpha=0.4)
pp.show()
