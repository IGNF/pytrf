#--------------------------------------------------------------------------------------------------------------------------------
# This script demonstrates how to automatically detect the position discontinuities (offsets), velocity discontinuities, and
# outliers present in a GNSS station position time series. The IGS position time series of station TXFS (Fort Stockton, TX, USA)
# is used as an example. The employed automatic detection procedure is the detection-identification-adaptation (DIA) procedure
# described in Gobron et al. (2021; https://doi.org/10.1029/2021JB022370).
# 
# The script starts by some quick automatic outlier cleaning of the time series with the function ts.clean_outliers(). An initial
# model is then specified, which consists of a linear trend, several periodic signals, variable white noise and flicker noise.
# Then, the model is iteratively refined with the addition of position discontinuities, velocity discontinuities and possible
# outliers left undetected by ts.clean_outliers(). At each iteration:
#  - The current model is adjusted to the time series.
#  - Likelihood ratio tests are performed to evaluate how alternative models with an additional outlier, position discontinuity
#    or velocity discontinuity introduced at each possible date would be more likely than the current model.
#  - If the maximum likelihood ratio test statistic exceeds a certain threshold (about 30.665), then the most likely alternate
#    model is chosen as the new current model (i.e., either a new outlier, position discontinuity or velocity discontinuity is
#    introduced), and iterations continue.
#  - Else, iterations stop.
#
# The script produces one figure at each iteration in which the likelihood ratio test statistics are shown below the residuals
# of the current model. These figures are not shown, but saved as iter0.png, iter1.png, ..., iter8.png.
#
# In the end:
#  - a velocity change is detected on 2018-11-07 (which is evident in the series, but for which there is no known explanation),
#  - a position change is detected on 2013-11-14 (due to an actual change of the station's antenna on that date),
#  - 18 outliers are detected (12 by ts.clean_outliers() + 6 during the following iterations).
#
# Note: The likelihood ratio tests implemented in pytrf provide sensible results only if the stochastic variations in the time
#       series are well represented by the specified noise model. For GNSS station position time series (corrected for non-tidal
#       loading deformation - see note below), automatic offset detection based on likelihood ratio tests thus requires a full
#       [variable-]white + flicker or [variable-]white + power-law noise model.
#
# Note: In this example, a white + flicker noise model is adjusted to a GNSS time series NOT corrected for non-tidal loading
#       deformation. The white + flicker noise model appears to be more or less appropriate in this particular case. However,
#       in the general case, non-tidal loading deformation introduces variations in GNSS time series that cannot be represented
#       by a white + flicker noise model. Adjusting a white + flicker noise model to GNSS time series NOT corrected for non-tidal
#       loading deformation therefore yields, in general, biased noise parameter estimates. Correcting GNSS time series for
#       non-tidal loading deformation before modeling their noise content is thus strongly recommended. See an extensive
#       discussion of this issue in Gobron et al. (2021; https://doi.org/10.1029/2021JB022370).
#
# Note: pytrf does not (yet) support automatic detection and modeling of the post-seismic deformation signals present in certain
#       GNSS station position time series. Such series should not be automatically segmented, but rather dealt with manually.
#       See an example in pytrf/examples/time series/modeling of a GNSS time series with post-seismic deformation.
#
# Warning: Expect this script to be quite CPU-intensive and last several minutes.
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
import numpy as np
import matplotlib.pyplot as pp
from pytrf import date
from pytrf.ts import ts, model
from pytrf.io import read_solns
from scipy.stats import chi2



# Download time series
os.system('wget -c ftp://igs-rf.ign.fr/pub/crd/TXFS_igs.xyz')

# Read time series
r = ts.read('TXFS_igs.xyz',
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),                              # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),     # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas; pairwise correlations
            dtrd=1,                                                                 # Remove (and store) mean linear trend
            rotate=True)

# Plot time series
r.plot(tunit='y')

# Remove points with unusually large formal errors
r.clean_sigmas()

# Automatically identify and flag outliers
r.clean_outliers()
# (Beware that the function ts.clean_outliers() may be used for any type of time series
# but has many parameters that are tailored by default to GNSS station position time series
# with time in days and values in meters. Those parameters would need to be adapted for
# other types of time series.)

# Plot clean time series
r.plot(tunit='y')



# Initialize model with a linear trend, variable white noise and flicker noise
m = model(r, deg=[0, 1], noise=['vw', 'fn'])

# Add periodic terms in the trajectory model, at the annual and semi-annual periods,
# at the first 8 harmonics of the GPS draconitic year, and at the three main fortnightly
# periods reported in GNSS station position time series.
for T in [365.25, 182.625] + [351.5/k for k in range(1, 8)] + [14.76, 14.19, 13.62]:
    m.add_sine(T)
    
# Iteratively refine model with position discontinuities, velocity discontinuities
# and possible remaining outliers undetected by clean_outliers().
iter = -1
end = False
while not(end):
    iter += 1
    
    # Fit current model
    m.fit()
    
    # Compute likelihood ratio test statistics for outliers, position discontinuities,
    # and velocity discontinuities
    print('')
    print('Likelihood ratio tests')
    print('----------------------')
    print('')
    To = m.glr_outlier()
    Tp = m.glr_mean()
    Tv = m.glr_trend()

    # Plot residuals of current model and test statistics
    y = [date.from_mjd(d).ydec() for d in r.t]
    fig = pp.figure(figsize=(8, 12), tight_layout=True)
    for d in range(3):
        ax = fig.add_subplot(4, 1, d+1)
        pp.errorbar(y, m[d].v, m[d].sv, fmt='.k', ecolor='gray')
        ax.margins(0.01, 0.01)
        pp.grid()
        pp.ylabel(r.dims[d]+' residuals ['+r.yunit+']')
        ax.set_xticklabels([])
    ax = fig.add_subplot(4, 1, 4)
    pp.plot(y, To, label='outliers')
    pp.plot(y, Tp, label='pos. disc.')
    pp.plot(y, Tv, label='vel. disc.')
    ax.margins(0.01, 0.01)
    pp.grid()
    pp.xlabel('Time [yr]')
    pp.ylabel('Likelihood ratio test statistics')
    pp.legend()
    pp.savefig('iter'+str(iter)+'.png')
    pp.close()
    
    # Overall maximum test statistic
    Tmax = np.max([np.max(To), np.max(Tp), np.max(Tv)])
    print('    Maximum test statistic =', Tmax)
    
    # Under the null hypothesis that the current model is appropriate, the likelihood ratio
    # test statistics are distributed as chi2(3). We can therefore compute the probability
    # that the maximum statistic Tmax could have occurred by chance under the null hypothesis
    # as chi2.sf(Tmax, 3). If that probability is "small", this means that either an outlier,
    # a position discontinuity, or a velocity discontinuity is likely still missing in the model.
    # We chose here < 1e-6 as a limit for "small". This corresponds to Tmax > ~30.665.
    if (chi2.sf(Tmax, 3) < 1e-6):
        
        # If the overall maximum statistic corresponds to an outlier,
        # remove corresponding point from the time series.
        if (Tmax == np.max(To)):
            i = np.argmax(To)
            m.del_points([i])
            print('    -> Add outlier at t =', r.t[i])
            
        # Else, if the overall maximum statistic corresponds to a position discontinuity
        # add it into the trajectory model.
        elif (Tmax == np.max(Tp)):
            i = np.argmax(Tp)
            m.add_jumps(deg=[0], t=[r.t[i]])
            print('    -> Add position discontinuity at t =', r.t[i])
            
        # Else, the overall maximum statistic corresponds to a velocity discontinuity.
        # Add it into the trajectory model.
        else:
            i = np.argmax(Tv)
            m.add_jumps(deg=[1], t=[r.t[i]])
            print('    -> Add velocity discontinuity at t =', r.t[i])
            
    # Else, stop iterations
    else:
        end = True
        print('    -> Stop iterations')
        
# Print and plot final model
print('')
print(m)
m.plot_all(tunit='y')

# You could then obtain the lists of the dates of the identified position and velocity discontinuities as
# m[0].f[0].t and m[0].f[1].t, respectively.
