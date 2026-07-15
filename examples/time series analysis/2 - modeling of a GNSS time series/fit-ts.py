#--------------------------------------------------------------------------------------------------------------------------------
# This script demonstrates how to adjust a complete trajectory + noise model to a GNSS station position time series without
# post-seismic deformation (PSD). The IGS position time series of station UFPR (Curitiba, Brazil) is used as an example.
# Its modeling is done in two steps:
#  - In the first step, the trajectory model includes a piecewise linear component, annual and semi-annual sine waves. The noise
#    model includes only variable white noise. The purpose of this first step is only to quickly detect and remove outliers.
#  - In the second step, some more periodic terms are added into the trajectory model, and flicker noise is added into the noise
#    model.
#
# Note: In this example, it is assumed that the dates of offsets (position discontinuities) and velocity changes are known
#       a priori. For an example of automatic offset detection, see:
#       pytrf/examples/time series/automatic offset detection in a GNSS time series.
#
# Note: The time series used in this example does not feature post-seismic deformation (PSD). For an example of how to deal with
#       PSD, see pytrf/examples/time series/modeling of a GNSS time series with post-seismic deformation.
#
# Note: In this example, a white + flicker noise model is adjusted to a GNSS time series NOT corrected for non-tidal loading
#       deformation. The white + flicker noise model appears to be appropriate in this particular case. However, in the general
#       case, non-tidal loading deformation introduces variations in GNSS time series that cannot be represented by a white +
#       flicker noise model. Adjusting a white + flicker noise model to GNSS time series NOT corrected for non-tidal loading
#       deformation therefore yields, in general, biased noise parameter estimates. Correcting GNSS time series for non-tidal
#       loading deformation before modeling their noise content is thus strongly recommended. See an extensive discussion of
#       this issue in Gobron et al. (2021; https://doi.org/10.1029/2021JB022370).
#
# Warning: Expect step 2 (in which a complete noise model is adjusted) to be quite CPU-intensive and last a few minutes.
#
# Requirement: The script uses wget to download some files. If you need to install it:
#  - on Debian-based Linux distributions:   sudo apt-get install wget
#  - on RPM-based Linux distributions:      sudo dnf install wget
#  - on MacOS:                              brew install wget
#  - on Windows:                            https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe
#--------------------------------------------------------------------------------------------------------------------------------



# Imports
import os
from pytrf.ts import ts, model
from pytrf.io import read_solns



# Download time series
os.system('wget -c ftp://igs-rf.ign.fr/pub/crd/UFPR_igs.xyz')

# Read time series
r = ts.read('UFPR_igs.xyz',
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

# Add position discontinuity due to 2010 Maule, Chile Earthquake
m.add_jumps(t=[55254.27373], deg=[0])

# Add position+velocity discontinuity due to 2018 antenna change
m.add_jumps(t=[58212.64583], deg=[0, 1])

## For the same result as the three paragraphs above, the model could rather be initialized
## from the information available in the IGS discontinuity list.
#os.system('wget -c ftp://igs-rf.ign.fr/pub/discontinuities/soln.snx')
#solns = read_solns('soln.snx')
#m = model.from_solns(r, solns, code='UFPR', per=[365.25, 182.625], noise=['vw'])

# Fit model
m.fit()

# Plot results (to verify that the trajectory model is appropriate)
m.plot_all(tunit='y')

# Iteratively fit model and remove outliers
m.fit_iter(thr_norm=5, thr_raw=5)

# Plot results again (to verify that outliers are gone)
m.plot_all(tunit='y')

## If you're just interested in the values of the trajectory model parameters, but don't need realistic uncertainties,
## then you may stop here and not go through step 2. At this point, you may print the model parameters with:
#print(m)

# However, if you need realistic uncertainties for the trajectory model parameters, then you need to adjust
# a more realistic noise model. It is indeed clear, from the power spectrum plot produced above by m.plot_all(),
# that the periodograms of the "noise" in the time series do not match the flat power spectrum of the
# variable-white-noise-only model used so far. The periodograms rather show a slope at low frequencies,
# characteristic of the flicker noise usually observed in GNSS station position time series.



# Step 2: Adjust complete trajectory + noise model
#-------------------------------------------------

# Add some more periodic terms into trajectory model, namely at the first 8 harmonics of the GPS draconitic year,
# and at the three main fortnightly periods reported in GNSS station position time series.
# This addition is strongly advised, as failing to account for all periodic variations in the trajectory model
# can result in biased noise parameter estimates.
for T in [351.5/k for k in range(1, 8)] + [14.76, 14.19, 13.62]:
    m.add_sine(T)
    
# Add flicker noise into noise model
m.add_fn()

# Fit model
m.fit()

# Print final model
print(m)

# Plot final results again (to verify that the power spectrum of the variable white + flicker noise model now matches
# the shapes of the noise periodograms)
m.plot_all(tunit='y')
