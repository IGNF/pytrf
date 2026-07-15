#--------------------------------------------------------------------------------------------------------------------------------
# This script demonstrates how to adjust a complete trajectory + noise model to a GNSS station position time series with
# post-seismic deformation (PSD). The IGS position time series of station P101 (Oshoro, Hokkaidō, Japan) is used as an example.
# Its modeling is done in three steps:
#  - In the first step, the trajectory model includes a piecewise linear component, annual and semi-annual sine waves, as well as
#    exponential and logarithmic functions to account for PSD. The noise model includes only variable white noise. The purpose of
#    this first step is only to quickly detect and remove outliers.
#  - In the second step, some more periodic terms are added into the trajectory model, and flicker noise is added into the noise
#    model. The relaxation times of the PSD functions are fixed to the values estimated in the first step. It is indeed
#    recommended not to solve simultaneously for PSD relaxation times and a complete noise model, unless starting from
#    near-optimal a priori values for both the PSD relaxation times and the noise parameters.
#  - In the third step, the relaxation times of the PSD functions are unfixed, and a final adjustment of the complete trajectory
#    + noise model is performed. Starting from near-optimal a priori values for both the PSD relaxation times (from step 1) and
#    the noise parameters (from step 2), and using the 'Newton' optimization method guarantees convergence almost surely.
#
# Note: In this example, it is assumed that the dates of offsets (position discontinuities) and velocity changes are known
#       a priori. For an example of automatic offset detection, see:
#       pytrf/examples/time series/automatic offset detection in a GNSS time series.
#
# Note: In this example, it is assumed that the dates and types (exponential or logarithm) of PSD functions are known a priori.
#       There is no automatic method (yet) in pytrf for the specification of PSD models. Meanwhile, users who need to specify PSD
#       models for their own station position time series are advised to follow the "semi-automatable" approach outlined below:
#       for every E, N, U component:
#           while post-seismic deformation remains visible in the trajectory model residuals:
#               - try to add an exp in the trajectory model at the time when the largest post-seismic signal starts
#               - try to add a log into the trajectory model at the time when the largest post-seismic signal starts
#               - select the function type (exp or log) that yields the lowest residual WRMS and add it permanently to the
#                 trajectory model
#       (A [variable-]white-noise-only model is sufficient for that purpose.)
#
# Note: In this example, a white + flicker noise model is adjusted to a GNSS time series NOT corrected for non-tidal loading
#       deformation. The white + flicker noise model appears to be appropriate in this particular case. However, in the general
#       case, non-tidal loading deformation introduces variations in GNSS time series that cannot be represented by a white +
#       flicker noise model. Adjusting a white + flicker noise model to GNSS time series NOT corrected for non-tidal loading
#       deformation therefore yields, in general, biased noise parameter estimates. Correcting GNSS time series for non-tidal
#       loading deformation before modeling their noise content is thus strongly recommended. See an extensive discussion of
#       this issue in Gobron et al. (2021; https://doi.org/10.1029/2021JB022370).
#
# Warning: Expect steps 2 and 3 (in which a complete noise model is adjusted) to be quite CPU-intensive and last a few minutes.
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
from pytrf import sinex
from pytrf.ts import ts, model, fexp, flog
from pytrf.io import read_solns



# Download time series
os.system('wget -c ftp://igs-rf.ign.fr/pub/crd/P101_igs.xyz')

# Read time series
r = ts.read('P101_igs.xyz',
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),                              # Columns to read?
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),     # What's in the columns? Namely: time; 1st, 2nd & 3rd dimensions; their respective sigmas; pairwise correlations
            dtrd=1,                                                                 # Remove (and store) mean linear trend
            rotate=True)

# Remove points with unusually large formal errors
r.clean_sigmas()

# Plot time series
r.plot(tunit='y')



# Step 1: Adjust trajectory model with variable white noise only in order to detect and remove outliers
#------------------------------------------------------------------------------------------------------

# Initialize model with a linear trend, annual and semi-annual sine waves, and variable white noise
m = model(r, deg=[0, 1], per=[365.25, 182.625], noise=['vw'])

# Add position discontinuities
m.add_jumps(t=[52907.82646,     # EQ M8.2 - 134 km SSW of Kushiro, Japan
               54054.46826,     # EQ M8.3 - Kuril Islands
               55631.24054],    # EQ M9.1 - 2011 Great Tohoku Earthquake, Japan
            deg=[0])

# Add PSD functions
m[0].add_log(t0=52907.82646)    # Logarithm in East following 2003 Kushiro earthquake
m[0].add_exp(t0=55631.24054)    # Exponential in East following 2011 Tohoku earthquake
m[1].add_log(t0=52907.82646)    # Logarithm in North following 2003 Kushiro earthquake
m[1].add_exp(t0=55631.24054)    # Exponential in North following 2011 Tohoku earthquake
m[1].add_log(t0=55631.24054)    # Logarithm in North following 2011 Tohoku earthquake

## For the same result as the three paragraphs above, the model could rather be initialized
## from the information available in the IGS discontinuity list and in the IGS PSD SINEX file:
#os.system('wget -c ftp://igs-rf.ign.fr/pub/discontinuities/soln.snx')
#os.system('wget -c ftp://igs-rf.ign.fr/pub/psd/psd_IGS.snx')
#solns = read_solns('soln.snx')
#psd = sinex.read('psd_IGS.snx')
#m = model.from_solns(r, solns, code='P101', per=[365.25, 182.625], noise=['vw'], psd=psd)

# Fit model
m.fit()

# Plot results (to verify that the trajectory model is appropriate)
m.plot_all(tunit='y')

# Iteratively fit model and remove outliers
m.fit_iter(thr_norm=5, thr_raw=5)

# Plot results again (to verify that outliers are gone)
m.plot_all(tunit='y')

## If you're just interested in the values of the trajectory model parameters, but don't need realistic uncertainties,
## then you may stop here and not go through steps 2 and 3. At this point, you may print the model parameters with:
#print(m)
## You may also export the adjusted post-seismic deformation model into a SINEX file, e.g., for future use in a
## long-term stacking of SINEX solutions, with:
#m.write_psdsnx('myPSDmodels.snx', code='P101', pt=' A')

# However, if you need realistic uncertainties for the trajectory model parameters, then you need to adjust
# a more realistic noise model. It is indeed clear, from the power spectrum plot produced above by m.plot_all(),
# that the periodograms of the "noise" in the time series do not match the flat power spectrum of the
# variable-white-noise-only model used so far. The periodograms rather show a slope at low frequencies,
# characteristic of the flicker noise usually observed in GNSS station position time series.



# Step 2: Adjust complete trajectory + noise model with PSD relaxation times fixed
#---------------------------------------------------------------------------------

# Add some more periodic terms into trajectory model, namely at the first 8 harmonics of the GPS draconitic year,
# and at the three main fortnightly periods reported in GNSS station position time series.
# This addition is strongly advised, as failing to account for all periodic variations in the trajectory model
# can result in biased noise parameter estimates.
for T in [351.5/k for k in range(1, 8)] + [14.76, 14.19, 13.62]:
    m.add_sine(T)
    
# Add flicker noise into noise model
m.add_fn()

# Fix the relaxation times of all PSD functions
# It is indeed recommended not to solve simultaneously for PSD relaxation times and a complete noise model,
# unless starting from near-optimal a priori values for both the PSD relaxation times and the noise parameters.
for d in range(3):
    for f in m[d].f:
        if isinstance(f, fexp) or isinstance(f, flog):
            f.par[1].fixed = True

# Fit model
m.fit()

# Plot results again (to verify that the power spectrum of the variable white + flicker noise model now matches
# the shapes of the noise periodograms)
m.plot_all(tunit='y')



# Step 3: Final adjustment of the complete trajectory + noise model with PSD relaxation times UNfixed
#----------------------------------------------------------------------------------------------------

# UNfix the relaxation times of all PSD functions
for d in range(3):
    for f in m[d].f:
        if isinstance(f, fexp) or isinstance(f, flog):
            f.par[1].fixed = False
            
# Fit model
# Here, the 'Newton' optimization method is used rather than the default 'BFGS' method.
# Although it is slower, it has the advantage of not wandering unnecessarily far from the near-optimal noise
# parameter values from step 2, which guarantees convergence of both PSD relaxation times and noise parameters.
m.fit(method='Newton')

# Print final model
print(m)

# Plot final results 
m.plot_all(tunit='y')
