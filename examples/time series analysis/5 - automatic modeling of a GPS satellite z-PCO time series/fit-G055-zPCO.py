#--------------------------------------------------------------------------------------------------------------------------------
# The purpose of this script is to automatically build a model for a time series a daily z-PCO estimaties of GPS satellite G055.
# 
# The script starts with an initial model composed a linear trend, a known offset, variable white noise and flicker noise. Then,
# the model is iteratively refined with the addition of outliers, offsets, trend changes and periodic signals. At each iteration:
#  - The current model is adjusted to the time series.
#  - Likelihood ratio tests are performed to evaluate how alternative models with an additional outlier, offset or trend change
#    at each possible epoch, or an additional periodic signal at each possible frequency, would be more likely than the current
#    model.
#  - The likelihood test statistics are converted into "inverse probabilities" to make them comparable with each other.
#  - If the maximum "inverse probability" exceeds 1e6, then the most likely alternate model is chosen as the new current model
#    (i.e., either a new outlier, offset, trend change or periodic signal is introduced), and iterations continue.
#  - Else, iterations stop.
#
# The script produces one figure at each iteration in which the likelihood ratio test statistics are shown below the residuals
# of the current model. These figures are not shown, but saved as iter0.png, iter1.png, ..., iter8.png.
#
# In the end:
#  - one offset is detected on 2007-11-06, near the beginning of the series.
#  - 9 periodic signals are detected, at periods ranging from ~2 to ~350 days.
#
# Note: The likelihood ratio tests implemented in pytrf provide sensible results only if the stochastic variations in the time
#       series are well represented by the specified noise model. In this example, a variable white + flicker noise model is
#       specified, as its power spectrum appears to match the shape of the periodogram of the model residuals. (See figures.)
#
# Warning: Expect this script to be quite CPU-intensive and last several minutes.
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import numpy as np
import matplotlib.pyplot as pp
from scipy import signal
from scipy.stats import chi2
from pytrf import date
from pytrf.ts import ts, model, sine
from pytrf.math import lombscargle



# Function that draws the figure at each iteration
#-------------------------------------------------
def plot_iteration():
    
    # Gaussian window used to smooth periodograms in the figure
    w = signal.windows.gaussian(7, 1)
    w = w / np.sum(w)
    
    # Initialize figure
    pp.figure(figsize=(20, 10), tight_layout=True)

    # Plot time series and deterministic model
    pp.subplot(321)
    for tc in m.f[0].t + m.f[1].t:
        yc = date.from_mjd(tc).ydec()
        pp.plot([yc, yc], [0.3, 0.6], '--r', linewidth=2, zorder=2)
    pp.errorbar(t, m.r.y, yerr=np.sqrt(m.r.Q), fmt='.k', ecolor='gray', zorder=3, label='time series')
    pp.plot(t, m.yc, 'r', linewidth=2, zorder=4, label='deterministic model')
    pp.fill_between(t, m.yc-m.sc, m.yc+m.sc, color='r', alpha=0.6, zorder=4)
    pp.axis([t[0], t[-1], 0.3, 0.6])
    pp.grid()
    pp.ylabel('z-PCO [m]')
    pp.legend(loc='upper left')

    # Plot residuals
    pp.subplot(323)
    pp.errorbar(t, m.v, yerr=m.sv, fmt='.k', ecolor='gray', zorder=3, label='residuals')
    pp.axis([t[0], t[-1], -0.15, 0.15])
    pp.grid()
    pp.ylabel('z-PCO residuals [m]')
    pp.legend(loc='upper right')

    # Plot results of likelihood ratio tests for outliers, offsets and trend changes
    pp.subplot(325)
    pp.semilogy(t, To, 'k', label='outlier test')
    pp.semilogy(t, Tm, color='cornflowerblue', label='offset test')
    pp.semilogy(t, Tt, color='green', label='trend change test')
    pp.axis([t[0], t[-1], 1, 1e7])
    pp.grid()
    pp.xlabel('time [yr]')
    pp.ylabel('1 / (1 - CDF)')
    pp.legend(loc='upper right')

    # Plot periodogram of full time series + frequencies of periodic signals included in current model
    pp.subplot(322)
    for f in m.f:
        if isinstance(f, sine):
            pp.plot([365.25/f.per, 365.25/f.per], [1e-5, 1], '--r', linewidth=2, zorder=2)
    (f, p) = lombscargle(m.r.t, m.r.y)
    pp.loglog(365.25*m.fr, signal.convolve(p, w, mode='same'), 'k', zorder=3, label='periodogram of full time series')
    pp.axis([0.1, 182.625, 1e-5, 1])
    pp.grid()
    pp.ylabel('power spectral density [m²/cpy]')
    pp.legend(loc='upper right')
    
    # Plot periodogram of residuals and power spectrum of noise model
    pp.subplot(324)
    pp.loglog(365.25*m.fr, signal.convolve(m.pv, w, mode='same'), 'k', zorder=3, label='periodogram of residuals')
    pp.loglog(365.25*m.fr, m.pn, 'r', linewidth=2, zorder=4, label='PSD of noise model')
    pp.fill_between(365.25*m.fr, m.pn-m.spn, m.pn+m.spn, color='r', alpha=0.6, zorder=4)
    pp.axis([0.1, 182.625, 1e-5, 1])
    pp.grid()
    pp.ylabel('power spectral density [m²/cpy]')
    pp.legend(loc='upper right')

    # Plot results of likelihood ratio tests for periodic signals
    pp.subplot(326)
    pp.loglog(365.25*m.fr, Ts, color='purple', label='periodic signal test')
    pp.axis([0.1, 182.625, 1, 1e7])
    pp.grid()
    pp.xlabel('frequency [cpy]')
    pp.ylabel('1 / (1 - CDF)')
    pp.legend(loc='upper left')

    # Save figure
    pp.savefig('iter'+str(iter))



# Main code
#----------

# Read time series (daily z-PCO estimates for GPS satellite G055
# derived from the contribution of ESA to the IGS repro3 campaign)
r = ts.read('G055.zpco', format=('t', 'x', 'qx'), dims=('z-PCO'))
t = [date.from_mjd(d).ydec() for d in r.t]      # Decimal years

# Specify initial model
m = model(r, deg=[0, 1], noise=['vw', 'fn'])    # Linear trend, variable white noise & flicker noise
m.add_jumps(deg=[0, 1], t=[57892])              # Known offset + trend change



# Iteratively refine model
iter = -1
end = False
while not(end):
    iter += 1
    
    # Fit current model
    m.fit()
    
    # Likelihood ratio tests for outliers, offsets, trend changes, mean+trend changes, and periodic signals.
    # The test results are transformed into "1 / odds" for illustration purposes.
    print('')
    print('Likelihood ratio tests')
    print('----------------------')
    print('')
    To = m.glr_outlier()
    Tm = m.glr_mean()
    Tt = m.glr_trend()
    Ts = m.glr_sine()

    # Under the null hypothesis that the current model is appropriate, the likelihood ratio test statistics
    # are distributed as chi2(1), chi2(1), chi2(1) and chi2(2), respectively. To make them comparable with
    # each other, let's transform them as 1 / the probability that an outlier / offset / trend change /
    # periodic signal as big as the one observed in the current model residuals could have occurred by chance
    # under the null hypothesis.
    To = 1 / chi2.sf(To, 1)
    Tm = 1 / chi2.sf(Tm, 1)
    Tt = 1 / chi2.sf(Tt, 1)
    Ts = 1 / chi2.sf(Ts, 2)
    
    # Draw figure
    plot_iteration()

    # Maximum inverse probability
    Tmax = np.max([np.max(To), np.max(Tm), np.max(Tt), np.max(Ts)])
    print('    Maximum inverse probability = {0:5.3e}'.format(Tmax))
    
    # If maximum inverse probability exceeds 1e6,
    if (Tmax > 1e6):
    
        # If maximum inverse probability corresponds to an outlier, remove it.
        if (Tmax == np.max(To)):
            i = np.argmax(To)
            m.del_points([i])
            print('    -> Add outlier at t =', r.t[i])

        # If maximum inverse probability corresponds to an offset, add it to the model.
        elif (Tmax == np.max(Tm)):
            i = np.argmax(Tm)
            m.add_jumps(deg=[0], t=[r.t[i]])
            print('    -> Add offset at t =', r.t[i])
            
        # If maximum inverse probability corresponds to a trend change, add it to the model.
        elif (Tmax == np.max(Tt)):
            i = np.argmax(Tt)
            m.add_jumps(deg=[1], t=[r.t[i]])
            print('    -> Add trend change at t =', r.t[i])
        
        # If maximum inverse probability corresponds to a periodic signal, add it to the model.
        else:
            i = np.argmax(Ts)
            m.add_sine(per=1/m.fr[i])
            print('    -> Add periodic signal at period T =', 1/m.fr[i], 'd')
            
    # Else, stop iterations.
    else:
        end = True
        print('    -> Stop iterations')



# Print final model
print('')
print(m)
