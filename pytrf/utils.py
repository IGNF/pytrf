"""
pytrf miscellaneous utilities

This subpackage contains miscalleanous low-level routines.

"""



# External imports
#-----------------
import os
import re
import uuid
import numpy as np
import matplotlib.pyplot as pp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import platform

# Internal imports
#-----------------
from pytrf import date
from pytrf.const import mjd_leap, gps_utc



# Generic record class
#---------------------
class record():
  
    """
    Generic record class
    """
    
    pass

# Test if a string can be converted into float
#---------------------------------------------
def isfloat(s):

    """
    Test if a string can be converted into float

    Returns
    -------
    b : bool
        Whether s can be converted into float

    Parameters
    ----------
    s : str
        Input string
    """
  
    try:
        float(s)
        return True
    
    except ValueError:
        return False

# Compare two dates in SINEX date format
#---------------------------------------
def earlier(t1, t2):

    """
    Compare two dates in SINEX date format

    Returns
    -------
    b : bool
        Whether t1 is earlier than t2

    Parameters
    ----------
    t1 : str
        Date in SINEX date format ('yy:ddd:sssss')
    t2 : str
        Date in SINEX date format ('yy:ddd:sssss')
    """

    if (int(t1[0:2]) >= 50):
        t1 = '19' + t1
    else:
        t1 = '20' + t1

    if (int(t2[0:2]) >= 50):
        t2 = '19' + t2
    else:
        t2 = '20' + t2

    return (t1 < t2)

# Get number of leap seconds at requested dates
#----------------------------------------------
def leapsec(d, timescale='UTC'):
  
    """
    Get number of leap seconds at requested dates

    Returns
    -------
    l : (n,) array_like
        GPS-UTC leap seconds at requested dates

    Parameters
    ----------
    d : (n,) array_like
        Requested dates [MJD]
    timescale : str
        'GPS' or 'UTC' depending on time scale of d. Default is 'UTC'.
    """

    # Initialization
    l = np.zeros(len(d))

    # 1st case : MJDs given in GPS time
    if (timescale == 'GPS'):
        for i in range(len(d)):
            ind = np.nonzero(mjd_leap <= d[i])[0][-1]
            l[i] = gps_utc[ind]

    # 2nd case : MJDs given in UTC
    elif (timescale == 'UTC'):
        for i in range(len(d)):
            ind = np.nonzero(mjd_leap - gps_utc/86400. <= d[i])[0][-1]
            l[i] = gps_utc[ind]

    return l

# Generate a random (UUID4) file name
#------------------------------------
def temp_file():
  
    """
    Generate a random (UUID4) file name

    Returns
    -------
    s : str
        Random file name
    """
  
    return str(uuid.uuid4())

# Convert .ps image to .png image
#--------------------------------
def ps2png(ps, png, rotate=0, margin=20):
  
    """
    Convert .ps image to .png image

    Parameters
    ----------
    ps : str
        Input .ps file
    png : str
        Output .ps file
    rotate : float
        Rotation angle [deg]. Default is 0.
    margin : int
        Margin [pixels]. Default is 20.
    """

    # Temporary file
    tmp = temp_file()+'.png'

    # Let's go!
    os.system('gs -dQUIET -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r250 -dGraphicsAlphaBits=4 -sOutputFile={0} {1}'.format(tmp, ps))
    os.system('convert {0} -rotate {1} -quality 100 -trim -mattecolor white -frame {2}x{2} {3}'.format(tmp, rotate, margin, png))
    os.system('rm {0}'.format(tmp))
  
# Substitute keywords by their values in a string
#------------------------------------------------
def sed_keywords(s, t):
  
    """
    Substitute keywords by their values in a string

    Returns
    -------
    s : str
        Output string

    Parameters
    ----------
    s : str
        Input string
    t : date instance
        Date to be used for keywords substitutions
    """
  
    # Date elements
    s = re.subn(r'\$yyyy', t.yyyy, s)[0]
    s = re.subn(r'\$doy' , t.doy,  s)[0]
    s = re.subn(r'\$yy',   t.yy,   s)[0]
    s = re.subn(r'\$mm',   t.mm,   s)[0]
    s = re.subn(r'\$dd',   t.dd,   s)[0]
    s = re.subn(r'\$hour', t.hour, s)[0]
    s = re.subn(r'\$min' , t.min,  s)[0]
    s = re.subn(r'\$sec' , t.sec,  s)[0]
    s = re.subn(r'\$week', t.week, s)[0]
    s = re.subn(r'\$dow' , t.dow,  s)[0]
    s = re.subn(r'\$wk',   t.wk,   s)[0]
    
    # Operating system
    s = re.subn(r'\$os', platform.uname().system+', '+platform.uname().machine, s)[0]
    
    return s

# Convert dictionary into record
#-------------------------------
def dict2rec(y, sed=False, t=None):
  
    """
    Convert dictionary into record

    Returns
    -------
    r : record
        Output record

    Parameters
    ----------
    y : dict
        Input dictionary
    sed : bool, optional
        If True, substitute keywords by their values in each key. Default is False.
    t : date instance, optional
        Date to be used for keywords substitutions. Default is None.
    """
  
    # Initialization
    r = record()
    
    # Loop over dictionary keys
    for key in y:
      
        # If y[key] is a dictionary,
        if (isinstance(y[key], dict)):
            setattr(r, key, dict2rec(y[key], sed, t))
          
        # If y[key] is an empty list,
        elif (isinstance(y[key], list)) and (len(y[key]) == 0):
            setattr(r, key, y[key])
          
        # If y[key] is a list of dictionaries,
        elif (isinstance(y[key], list)) and (isinstance(y[key][0], dict)):
            setattr(r, key, [dict2rec(d, sed, t) for d in y[key]])
          
        # If y[key] is a list of strings and sed is needed,
        elif (isinstance(y[key], list)) and (isinstance(y[key][0], str)) and (sed):
            setattr(r, key, [sed_keywords(s, t) for s in y[key]])
          
        # If y[key] is a string and sed is needed,
        elif (isinstance(y[key], str)) and (sed):
            setattr(r, key, sed_keywords(y[key], t))
        
        # Other cases
        else:
            setattr(r, key, y[key])
      
    return r

# Convert record into dictionary
#-------------------------------
def rec2dict(r):
  
    """
    Convert record into dictionary

    Returns
    -------
    d : dict
        Output dictionary

    Parameters
    ----------
    r : record
        Input record
    """
  
    # Initialization
    d = vars(r)
    
    # Loop over dictionary keys
    for key in d:
      
        # If d[key] is a record,
        if (isinstance(d[key], record)):
            d[key] = rec2dict(d[key])
          
        # If d[key] is a non-empty list of records,
        elif (isinstance(d[key], list)) and (len(d[key]) > 0):
            if (isinstance(d[key][0], record)):
                d[key] = [rec2dict(r) for r in d[key]]
          
    return d

# Execute multiple commands in parallel
#--------------------------------------
def parallel_sh(file, nproc, quiet=False, ignore_errors=False):
  
    """
    Execute multiple commands in parallel

    Parameters
    ----------
    file : str
        File containing list of commands to execute
    nproc : int
        Number of CPUs to use
    quiet : bool
        Whether not to print executed commands
    ignore_errors : bool
        Whether to run further jobs after one job failed
    """
  
    # Read list of commands
    commands = open(file).readlines()
    
    # Write make file
    makefile = temp_file()+'.make'
    with open(makefile, 'w') as f:
        f.write('all :')
        for i in range(len(commands)):
            f.write(' job{0}'.format(i))
        f.write('\n')
        for i in range(len(commands)):
            f.write('job{0} :\n\t{1}'.format(i, commands[i]))
    
    # Execute make file
    command = 'make -j{0} -f {1}'.format(nproc, makefile)
    if (quiet):
        command += ' -s'
    if (ignore_errors):
        command += ' -i'
    os.system(command)
    
    # Remove make file
    os.system('rm {0}'.format(makefile))

# Draw station map
#-----------------
def station_map(lon, lat, code, write_codes=True, title=None, output=None):

    """
    Draw station map

    Parameters
    ----------
    lon : (...) array_like
        Longitudes [deg]
    lat : (...) array_like
        Latitudes [deg]
    code : list
        4-char station IDs
    write_codes : bool, optional
        Whether to print 4-char station codes on map. Default is True.
    title : str, optional
        Map title. Default is None.
    output : str, optional
        Output file. Default is None (i.e. map shown on screen).

    """

    # Draw basemap
    pp.figure()
    ax = pp.axes(projection=ccrs.Robinson())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    # Add title if necessary
    if (title):
        pp.title(title)

    # Plot points
    ax.plot(lon, lat, '.k', markersize=6, transform=ccrs.Geodetic())

    # Write 4-char codes if requested
    if (write_codes):
        for i in range(len(code)):
            ax.text(lon[i], lat[i], code[i], fontsize=10, transform=ccrs.Geodetic())

    # Save figure into output file...
    if (output):
        pp.savefig(output, bbox_inches='tight')

    # ...or show it
    else:
        pp.show()
