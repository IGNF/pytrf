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
import pandas as pd
import matplotlib.pyplot as pp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import platform
import networkx as nx
import logging

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
        
        
# Period object
#-----------------
class Period():
    """
    This class builds a Period object. Useful to ensures the correct format and attribute values given by users.
            
    Constructors:
        - Period()                  : blank object with default attibuts, useful to create file option.
        - Period.from_record()      : from record object, for example after reading YAML file
        - Period.from_snx_param()   : from snx.param.type 6 characters str (old compatible syntax: A1COSX; new: A001COSX)
        
    Attibutes:
        - code                   : str 4 characters (ex: P001)
        - cs                     : str COS' or 'SIN'
        - dim                    : str 1 character:'X', 'Y', 'Z'
        - harmonic               : int 1 to 999
        - ic_period              : str 3 characters max ('RST') internal constraints
        - mc_period              : str 3 characters max ('RST') minimal constraints
        - param_type             : str 6 characters, from snx.param.type : type + %03d harmonic + cs + dim
        - param_type_old         : str 6 characters, from snx.param.type. Old version: type + %01d harmonic + COS/SIN + dim
        - type                   : str 1 character 'A' Annual; 'D' Draconitic; 'P' other period
        - unit                   : str value unit (day)
        - value                  : float harmonic value,, compute from A1 and D1 for 'A' and 'P' or given by user with 'P'
        - verbose                : str explains Period object (ex: 'X 1st annual cosine amplitude')
       
        
    Code format : 4 characters (type + harmonic number)
    Possible types :
        A : annual (A001, A002, etc)
        D : draconitic (D001, D002, etc)
        P : other period, in this case 2nd characters is not a harmonic but a simple id
            
    """
    #####----------------------------------------------------------------------------------------
    #####                               Constructors
    #####----------------------------------------------------------------------------------------
    def __init__(self, code="P001", **kwargs):
        """
        Default constructor. Code with 4 characters is necessary
        If 'code' in PERIODS (dict var), setup 'value', else 'value'=0.
        Useful to build blank file options
        
        Other attributes can bee specified manually with kwargs : mc_period, ic_period, etc
        
        Recommended construction attributes
            'code'(str) : ex 'P001'
            'type'(str)+'harmonic'(int): ex 'P' + '1'

        Parameters
        ----------
        code : str, 4 characters optional
            Period code value. The default is "P001", user can setup a custom value

        """
        ##1 ******* init Period attributes
        self.type = code[0] #A, D, P
        self.harmonic = int(code[1:]) # 1 > 999, permissive format :  P1=P01=P001
        
        # case of construction with param_type A1 (sinex.param.type > 6 caracters old vs new format)
        self.cs = None #'C' for COS or 'S' for SIN
        self.dim = None # X Y Z
        self.param_type = None
        self.param_type_old = None
        
        # necessary attributes and default values
        self.value = 0
        self.ic_period = ""
        self.mc_period = ""
        self.unit = "day"    
        
        ##2 ******* if necessary, update attributes from kwargs (cs, di)
        self.__dict__.update(kwargs)
        self.harmonic = int(self.harmonic) #be sure of int type
        self.code = "{}{}".format(self.type, "%03d" % self.harmonic)
        
        ## check constraints format
        if not self.check_RST(self.mc_period):
            raise ValueError(""""{}: mc_period = '{}'. Must be a combination of 'R', 'S', 'T' or ''. """.format(self.code, self.mc_period))
            
        if not self.check_RST(self.ic_period):
            raise ValueError(""""{}: ic_period = '{}'. Must be a combination of  'R', 'S', 'T' or ''. """.format(self.code, self.ic_period))
         
        ##3 ******* from_snx_param() construction, > build param_type attributes
        if type(self.cs)!=None and (self.dim!=None):
            self.param_type = "{}{}{}".format(self.code, self.cs[0], self.dim) # ex: A001CX
            self.param_type_old = "{}{}{}{}".format(self.type, self.harmonic, self.cs, self.dim) # ex: A1COSX
            
        self.verbose = self.build_verbose()
        
        #if 'type' is 'A' or 'D' > value from PERIODS dict, set in any case value from this reference dictionary
        if self.type in [k[0] for k in PERIODS.keys()]: #1st letter: A, D
            h1 = "{}001".format(self.type) #A001, D001
            
            if self.value != 0 :#another value provides by user with code 'A' annual or 'D'... warning
                logging.warning(f"With code '{self.code}':{self.verbose}. You cannot set 'value' attribute. Use code 'P' in this case.")
            
            self.value = PERIODS[h1]/self.harmonic
            
        
    @classmethod   
    def from_record(self, record_obj):
        """
        Period builds from record() object
        
        Check:
            * if record_obj values are consistent
            * if record_obj.name is key in PERIODS dict, or create a default code "P001", "P002" etc values
            * In case where value is not consistent with code, value is setup from PERIODS[code]
        
        Parameters
        ----------
        record_obj : record object
            record object, at least with "code" and "value" attribute

        """
        type_p = record_obj.code[0]
        ## update with record_obj value or correct if necessary
        if type_p in [k[0] for k in PERIODS.keys()] and (len(record_obj.code)<=4): # permissive format A1=A001
            code = record_obj.code
            value = None #will set up in any case by Period() constructor
        elif (record_obj.code[0] =="P") and (len(record_obj.code)<=4): #P1...P999
            code = record_obj.code
            # with P, value set up
            value = record_obj.value
        else:
            raise ValueError(""""Code must be a value in {}. If you want to specify another period value,
                             code format must be a 4 characters string as 'Pk', where k is 3 str between '001' and '999'. Your record object : {}""".format(list(PERIODS.keys()), record_obj.__dict__))
        
        ## check constraints format
        if hasattr(record_obj, 'mc_period'):
            mc_period = record_obj.mc_period
        else:
            mc_period = ''
            
        if hasattr(record_obj, 'ic_period'):
            ic_period = record_obj.ic_period
        else:
            ic_period = ''
       
        return Period(code=code, value=value, mc_period=mc_period, ic_period=ic_period)
    
    
    @classmethod
    def from_snx_param(self, param_type):
        """
        Period build from snx.param.type 6 characters str
        Can be either new format (ex:A001CX) or old format (A1COSX)

        Parameters
        ----------
        param_type : 6 characters string
            old version 'A1COSX', new version 'A001CX'

        Returns
        -------
        Period object

        """
        if not self._is_new_format(param_type): #conversion 
            #old syntax "A1COSX"
            param_type = self._convert_to_new_format(param_type) #convert to A001CX
            
        code = param_type[0:4] #A001
        # COS or SIN
        if param_type[4]=='C':
            cs='COS'
        elif param_type[4]=='S':
            cs='SIN'
        else:
            raise ValueError("param_type[4] must be 'C' or 'S', here: '{}'".format(param_type[4]))
        
        dim = param_type[5] # X Y Z
            
        ## init Period attributes
        return Period(code=code, cs=cs, dim=dim)
        
   
    #####----------------------------------------------------------------------------------------
    #####                                   Check functions
    #####----------------------------------------------------------------------------------------
    def check_RST(per, s):
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
        
    
    def build_verbose(per):
        """
        Build Period verbose explanation
        format : "dim + hormonic + type + cs + 'amplitude'"
        example : 'X 1st annual cosine amplitude'

        Parameters
        ----------
        per : period obj
        
        Returns
        -------
        verbose str

        """
        dict_type = {"A": "annual ", "D": "draconitic ", "P": "other period "}
        dict_cs = {"COS": "cosine ", "SIN": "sine "}
        
        def ordinal_conv(num):
            """Convert harmonic 1, 2, 3, 4, etc to 1st, 2nd, 3rd, 4th, etc"""
            if 10 <= num % 100 < 20:
                return str(num) + "th"
            else:
                ordinals = {1: "st", 2: "nd", 3: "rd"}
                return str(num) + ordinals.get(num % 10, "th")
        
        ### init default verbose value
        dim = ""
        type_v = ""
        cs_v = ""
        
        ### try to find verbose value in previous dicts
        if per.dim != None:
            dim = per.dim
        if per.type in dict_type.keys():
            type_v = dict_type[per.type]
     
        if per.cs in dict_cs.keys():
            cs_v = dict_cs[per.cs]
             
        verbose = "{} {} {}{}amplitude".format(dim, ordinal_conv(per.harmonic),type_v, cs_v)
        return verbose
    
    #####----------------------------------------------------------------------------------------
    #####                                   Conversion old and new format period param_type 
    #####----------------------------------------------------------------------------------------

    def _is_new_format(param_type):
        """
        Check if code given by user is old format A1COSX or new A001CX
        True: new
        False: old
        """
        return param_type[1:4].isdigit() #simple way to make a difference between 'A001CX' (new) and 'A1COSX' (old)
        
    
    def _convert_to_new_format(param_type):
        """
        Provides new param_type format from old syntax : A1COSX > A001CX 

        Returns
        -------
        str param_type new format
        """
        # Extract the harmonic value from the old format code
        harmonic = int(param_type[1])

        # Determine the type (COS or SIN) from the old format code
        if 'COS' in param_type:
            type_str = 'C'
        elif 'SIN' in param_type:
            type_str = 'S'
        else:
            type_str = ''

        # Determine the dim (X, Y, or Z) from the old format code
        dim = param_type[-1]

        # Construct the new format code
        return f'{param_type[0]}{str(harmonic).zfill(3)}{type_str}{dim}'
    


##### WARNING -> Graph_vfconst.py not OLD version ...
class Graph_vfconst():
    """
    This class allows to build a graph from a sinex object (pytrf.sinex) and to resonate in terms of "nodes" (stations) and "edges" (constraints).
    Useful to understand links between stations and apply/build vfconst file.
    Based on python networkx library.
    """
        
    def __init__(self, snx):
        """
        Constructor, from pytrf.sinex and vfconst.yml file

        Parameters
        ----------
        snx : pytrf.sinex object
            sinex with stations of interest, for example generated for combination
        file_vfconst : list of pytrf.record() objects
            vfconst YAML file, containing stations links. Generated with pytrf.io.read_yaml('vfconst.yml')
        type_graph : str, optional
            Graph type (default "VEL"): 
                >"VEL", load only VELOCITIES type from vfconst file
                >"PERIOD", load only PERIODS type from vfconst YAML file
        Returns
        -------
        None.

        """
        self.snx = snx
        #build id dataframes : station & sites (only base on DOMES 5 first chr)
        self.df_staId, self.df_sites = self.generate_staId()
        
              
    def generate_staId(self):
        """ DataFrame summarizing stations in self.snx file
            Columns : [domes, code, pt, soln, staId]
        """
        df_domes_snx = pd.DataFrame([(sta.domes, sta.code, sta.pt, soln.soln, soln.datastart, soln.dataend) for sta in self.snx.sta for soln in sta.soln], columns=["domes", "code", "pt", "soln", "datastart", "dataend"])
        df_domes_snx["_staId_pytrf"] = df_domes_snx["code"]+ df_domes_snx["pt"]+ df_domes_snx["soln"]
        df_domes_snx["staId"] = df_domes_snx.apply(lambda row: (row["code"]+ row["pt"]+ row["soln"]).replace(" ",""), axis=1) #no space
        df_domes_snx = df_domes_snx.set_index("staId")
        
        #coordinates
        coords = self.snx.get_xyz(df_domes_snx["code"].values.tolist(), pt=df_domes_snx["pt"].values.tolist(),  soln=df_domes_snx["soln"].values.tolist())
        df_domes_snx["X"], df_domes_snx["Y"], df_domes_snx["Z"] = coords[:,0], coords[:,1], coords[:,2]
        
        ### Sites according 5 domes character
        # Select the first 5 characters of the "domes" column
        df_domes_snx['group_key'] = df_domes_snx['domes'].str[:5]
        
        # Group by the first 5 characters and aggregate into a list with respective index values
        df_sites = df_domes_snx.groupby('group_key').agg(staId=('domes', lambda x: x.index.tolist())).reset_index()
        
        # drop intermediar "group_key" column
        df_domes_snx = df_domes_snx.drop(columns='group_key')
        
        return df_domes_snx, df_sites
      
        
    #####----------------------------------------------------------------------------------------
    #####                 Build Graphes
    #####----------------------------------------------------------------------------------------  
    
    # get linked stations according vfconst.yml file
    def build_vfgraph_from_vfconst(self, vfconst, type_graph, del_sta_not_in_snx=False):
        """ Build constraints graph from vfconst: stations (node) and edges (constraints)"""
        if type_graph!='VEL' and (type_graph not in self.snx.iper_dict.keys()):
            raise ValueError(f"Unknown type graph for current sinex. Possible type_graph: 'VEL' or periods '{list(self.snx.iper_dict.keys())}'")
        
        graph = nx.Graph()
        # Loop over constrains
        for const in vfconst:
            # convert record to dict
            const = const.__dict__
            
            #VELOCITIES & PERIODS
            if (const["type"]== type_graph) and ("sta2" in const.keys()): #at least 2 stations on this site
                if (const["type"] == 'VEL') or (const["type"] in self.snx.iper_dict.keys()): #velocity or period code: "A001", etc.
                    #station of vfconst in self.snx ?
                    add_to_graph=True
                    if (const['sta1'].replace(" ","") not in self.df_staId.index) :
                        #logging.warning(f'vfconst {const["sta1"]}: station not in sinex')
                        add_to_graph=False
                    if (const['sta2'].replace(" ","") not in self.df_staId.index):
                        #logging.warning(f'vfconst {const["sta2"]}: station not in sinex')
                        add_to_graph=False
                    #del station if not in self.snx?
                    if del_sta_not_in_snx:
                        if add_to_graph:
                            #logging.warning(f'--> add vfconst: {const}')
                            graph.add_edge(const['sta1'].replace(" ",""), const['sta2'].replace(" ",""), weight=const["sigma"]) #delete possible space " " -> be more flexible
                        else:
                            logging.warning(f'vfconst {const}: not added to vfconst graph (no station in sinex)')
                            pass
                    else: #add anyway
                        graph.add_edge(const['sta1'].replace(" ",""), const['sta2'].replace(" ",""), weight=const["sigma"]) #delete possible space " " -> be more flexible
            
        return graph
    
    
    def build_snx_graph(self, limit_dist=10000, link_only_soln=False):
        # Create a graph using networkx
        G = nx.Graph()
        
        # Add edges to the graph based on the distance and threshold
        for site in self.df_sites['staId']:
            #site: a list of all stations on the site. Multiple SOLN & no dist limit yet
            complete_graph = nx.complete_graph(site) #get all pairs on site
            for staId1, staId2 in complete_graph.edges():
                #compute euclidian distance
                distance = np.sqrt(np.sum((self.df_staId.loc[staId1, ['X', 'Y', 'Z']] - self.df_staId.loc[staId2, ['X', 'Y', 'Z']])**2))
                if link_only_soln and (staId1[:5]==staId2[:5]): #add edge only for same station, between SOLN
                    G.add_edge(staId1, staId2, length=distance)
                
                elif not link_only_soln and distance <= limit_dist: #check distance only if not "link_only_soln" 
                    G.add_edge(staId1, staId2, length=distance)        
        return G
    
    
    def build_graph_same_init_x0(self, vfconst=None, type_graph='VEL'):
        """
            Build graph snx SOLN + vfconst
            Thanks to 
        """
        #whatever soln discontinuities, we will init same VEL for all SOLN
        # sinex SOLN graph:
        gr_snx_soln = self.build_snx_graph(link_only_soln=True)
        
        if type(vfconst) != type(None):
            #graĥ vfconst:
            gr_vfconst = self.build_vfgraph_from_vfconst(vfconst, type_graph)
            #add edges from vfconst
            gr_snx_soln.add_edges_from(gr_vfconst.edges)
                
        return gr_snx_soln
    
    def plot_graph(self, G):
        pp.figure()
        nx.draw(G, with_labels=True)
        
        #edge label
        # edge_labels = dict([((n1, n2), f'{round(1000 * d["length"],3)} km' ) for n1, n2, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G),)
                                     #edge_labels=edge_labels)
        #nx.draw_networkx_labels(G, pos=nx.spring_layout(G))

    #####----------------------------------------------------------------------------------------
    #####                 Get nodes & edges informations
    #####----------------------------------------------------------------------------------------  
    def get_sites(self, G):
        """
        A site is defined if at least 2 stations (node) are linked by constraints (edge)
        """
        #use networkX method to find linked point
        connected_components = list(nx.connected_components(G))
        
        #list of list
        connected_components = [list(component) for component in connected_components]
        return connected_components
    
       
    def get_connected_stations(self,  staId, G):
        staId = staId.replace(" ","")
        
        if staId in G.nodes: #this station exist
            staId_connected = list(nx.node_connected_component(G, staId))
            #filter stations if not in original self.snx but in graph G
            staId_connected_insnx = []
            for sta in staId_connected:
                if sta in self.df_staId.index:
                    staId_connected_insnx.append(sta)
                else:
                    logging.warning(f"'{sta}' in Graph but not in self.snx")
                
            return list(self.df_staId.loc[staId_connected_insnx,"code"]), list(self.df_staId.loc[staId_connected_insnx,"pt"]), list(self.df_staId.loc[staId_connected_insnx,"soln"])
        
        else:
            logging.warning(f"Station Id '{staId}' not in sinex.")
            return [],[],[]
        
        
    
    def get_connection_datum(self, staId, G, datum):
        """For staId, provide the name of the linked station referred in datum. Relations & links described by graph G."""
        
        staId = staId.replace(" ","")
        
        # connected stations to staId
        if staId in G.nodes:
            staId_connected = list(nx.node_connected_component(G, staId))
        else:
            logging.warning(f"Station Id '{staId}' not in sinex.")
            return None
        
        #station in datum
        sta_datum = [(sta.code + sta.pt + soln.soln).replace(" ","") for sta in datum.sta for soln in sta.soln]
             
        # find ref sta
        ref_sta = [sta for sta in staId_connected if sta in sta_datum]
        
        if len(ref_sta)>=1:
            ref_sta = ref_sta[0]
        else: # no ref sta find in connected stations
            ref_sta=None
            
        return ref_sta
    
        
    def map_sites_indatum(self, G, datum):
        """Sites of G link in datum"""
        sta_datum = [(sta.code + sta.pt + soln.soln).replace(" ","") for sta in datum.sta for soln in sta.soln] #staId no space " "
        list_sites = self.get_sites(G) #list sites provides by vfconst file, #staId no space " "
        
        in_datum = [] #list of True and False for each site
        for site in list_sites:
            in_datum.append([station in sta_datum for station in site])
            
        return list_sites, in_datum
            
    
    def valid_datum(self, datum,  vfconst=None, type_graph='VEL'):
        """ 
        Check if datum is valid:
        vfconst file (read_yaml: list of records) is compatible with datum (pytrf.sinex obj) if maximum 1 station by site (i.e. linked station) in datum
        If vfconst=None, build snx graph only with soln
        """
        if type(vfconst) != type(None):
            vfgraph = self.build_vfgraph_from_vfconst(vfconst, type_graph) #build vfcont graphe
        else: #no vfconst, graph edges only with soln
            vfgraph = self.build_snx_graph(link_only_soln=True)
        
        list_sites, in_datum = self.map_sites_indatum(vfgraph, datum)
            
        # 2 stations of a same site in datum ??
        sites_2sta_datum = [sum(site)>=2 for site in in_datum] #True if "2 True" in a site, else False
        
        valid_vf_datum = not any(sites_2sta_datum) #at least a True = at least 2 stations --> no valid
        
        list_pbm = [] #station pbm: del in datum or in vfconst
        if not valid_vf_datum: #no valid, wich site/ station ???
            for num, site in enumerate(list_sites):
                #print(num, site, in_datum[num])
                #list of stations on this site
                list_sta = np.array(site) #site= [sta1, sta2...]
                #for this site, stations concerned
                select = np.array(in_datum[num]) #list of bool for "index" site: [True, False,..], same lenght that list_sta
                #print(list_sta[select])
                if len(list(list_sta[select]))>1:
                    list_pbm += list(list_sta[select])
                    logging.warning(f'Datum vs vfconst, error for vfconst site "{num}": multiple stations in datum: {list_sta[select]}')
                
        return valid_vf_datum, list_pbm      
