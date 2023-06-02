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
from pytrf.const import mjd_leap, gps_utc, PERIODS



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
    s = re.subn('\$yyyy', t.yyyy, s)[0]
    s = re.subn('\$doy' , t.doy,  s)[0]
    s = re.subn('\$yy',   t.yy,   s)[0]
    s = re.subn('\$mm',   t.mm,   s)[0]
    s = re.subn('\$dd',   t.dd,   s)[0]
    s = re.subn('\$hour', t.hour, s)[0]
    s = re.subn('\$min' , t.min,  s)[0]
    s = re.subn('\$sec' , t.sec,  s)[0]
    s = re.subn('\$week', t.week, s)[0]
    s = re.subn('\$dow' , t.dow,  s)[0]
    s = re.subn('\$wk',   t.wk,   s)[0]
    
    # Operating system
    s = re.subn('\$os', platform.uname().system+', '+platform.uname().machine, s)[0]
    
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
def parallel_sh(file, nproc):
  
    """
    Execute multiple commands in parallel

    Parameters
    ----------
    file : str
        File containing list of commands to execute
    nproc : int
        Number of CPUs to use
    """
  
    # Read list of commands
    commands = open(file).readlines()
    
    # Write make file
    makefile = temp_file()+'.make'
    f = open(makefile, 'w')
    f.write('all :')
    for i in range(len(commands)):
        f.write(' job{0}'.format(i))
    f.write('\n')
    for i in range(len(commands)):
        f.write('job{0} :\n\t{1}'.format(i, commands[i]))
    f.close()
    
    # Execute make file
    os.system('make -j{0} -f {1}'.format(nproc, makefile))
    
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
        
        
# Period object
#-----------------
class Period():
    """
    This class builds a Period object. Useful to ensures the correct format and attribute values given by user
            
    Constructors:
        - Period()                  : blank object with default attibuts, useful to create file option.
        - Period.from_record()      : from record object, for example after reading YAML file
       
        
    Code format : 2 characters (type + harmonic number)
    Possible types :
        A : annual (A1, A2, etc)
        D : draconitic (D1, D2, etc)
        P : other period, in this case 2nd characters is not a harmonic but a simple id
            
    """
    #####----------------------------------------------------------------------------------------
    #####                               Constructors
    #####----------------------------------------------------------------------------------------
    def __init__(self, code="P1", **kwargs):
        """
        Default constructor. 2 characters code is necessary
        If 'code' in PERIODS, setup 'value', else 'value'=0.
        Useful to build blank file options
        
        Other attributes can bee specify manually witn kwargs : mc_per, ic_per, etc

        Parameters
        ----------
        code : str, 2 characters optional
            Period code value. The default is "P1", user can setup a custom value

        Returns
        -------
        None.

        """
        ## init Period attributes
        self.code = code
        
        # necessary attributes and default values
        self.value = 0
        self.mc_per = ""
        self.ic_per = ""
        
        #if necessary, update attributes from kwargs
        self.__dict__.update(kwargs)
        
        if code in PERIODS.keys():
            self.value = PERIODS[code]
        
    @classmethod   
    def from_record(self, record_obj):
        
        """
        Period builds from record() object
        
        Check:
            * if record_obj values are consistent
            * if record_obj.name is key in PERIODS dict, or create a default code "P1", "P2" etc values
            * In case where value is not consistent with code, value is setup from PERIODS[code]
        
        Parameters
        ----------
        record_obj : record object
            record object, at least with "code" and "value" attribute

        """
        ## update with record_obj value or correct if necessary
        if record_obj.code in PERIODS.keys():
            code = record_obj.code
            value = PERIODS[code] # anyway take PERIODS value
        elif (record_obj.code[0] =="P") and (len(record_obj.code)==2): #P1...P9
            code = record_obj.code
            value = record_obj.value
        else:
            raise ValueError(""""Code must be a value in {}. If you want to specify another period value,
                             code format must be a 2 characters string as 'Pk', where k is an int between 0 and 9.""".format(list(PERIODS.keys())))
        
        ## check constraints format
        if self.check_RST(record_obj.mc_per):
            mc_per = record_obj.mc_per
        else:
            raise ValueError(""""{}: mc_per = '{}'. Must be a combination of 'R', 'S', 'T' or ''. """.format(record_obj.code, record_obj.mc_per))
            
        if self.check_RST(record_obj.ic_per):
            ic_per = record_obj.ic_per
        else:
            raise ValueError(""""{}: ic_per = '{}'. Must be a combination of  'R', 'S', 'T' or ''. """.format(record_obj.code, record_obj.ic_per))
        
        return Period(code=code, value=value, mc_per=mc_per, ic_per=ic_per)
    
    #####----------------------------------------------------------------------------------------
    #####                                   Help functions
    #####----------------------------------------------------------------------------------------
    def check_RST(s):
        """
        Verify constraints format "RST"

        Parameters
        ----------
        s : string
            check if s contains maximum once R,S,T

        Returns
        -------
        bool

        """
        pattern = r"^(?!.*([RST]).*\1)[RST]{0,3}$"
        match = re.match(pattern, s)
        if match:
            return True
        else:
            return False
