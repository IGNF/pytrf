"""
pytrf I/O utilities

This subpackage contains read/write routines for various useful file formats.

"""



# External imports
#-----------------
import os
import re
import yaml
import copy
import unicodedata
import numpy as np
from math import sqrt, log10

# Internal imports
#-----------------
from pytrf import date, sinex
from pytrf.utils import record, isfloat, earlier, dict2rec, rec2dict, sed_keywords
from pytrf.const import agency



# Read YAML configuration file
#-----------------------------
def read_yaml(inp, sed=False, t=None):
  
    """
    Read YAML configuration file

    Returns
    -------
    r : record or list
        Content of YAML file

    Parameters
    ----------
    inp : str
        Input YAML file
    sed : bool, optional
        If True, substitute keywords by their values in each key. Default is False.
    t : date instance, optional
        Date to be used for keywords substitutions. Default is None.
    """
  
    y = yaml.load(open(inp), Loader=yaml.FullLoader)
    
    # If y is a dictionary,
    if (isinstance(y, dict)):
        return dict2rec(y, sed, t)
    
    # If y is an empty list,
    elif (isinstance(y, list)) and (len(y) == 0):
        return y
    
    # If y is a list of dictionaries,
    elif (isinstance(y, list)) and (isinstance(y[0], dict)):
        return [dict2rec(d, sed, t) for d in y]
        
    # If y is a list of strings and sed is needed,
    elif (isinstance(y, list)) and (isinstance(y[0], str)) and (sed):
        return [sed_keywords(s, t) for s in y]

    # Other cases
    else:
        return y

# Write YAML configuration file
#------------------------------
def write_yaml(r, out):
  
    """
    Write YAML configuration file

    Parameters
    ----------
    r : record or list
        Content to be written
    out : str
        Output YAML file
    """
  
    # If r is a record,
    if (isinstance(r, record)):
        l = rec2dict(r)
      
    # If r is an empty list,
    elif (isinstance(r, list)) and (len(r) == 0):
        l = r
    
    # If r is a list of records,
    elif (isinstance(r, list)) and (isinstance(r[0], record)):
        l = [rec2dict(rec) for rec in r]
        
    # Other cases
    else:
        l = r
        
    yaml.dump(l, open(out, 'w'))
    
# Read DOMES number catalogue (codomes.snx)
#------------------------------------------
def read_domes(file, coord=True):
  
    """
    Read DOMES number catalogue (codomes.snx)
    
    Returns
    -------
    domes : list
        List of records

    Parameters
    ----------
    file : str
        File to read
    coord : bool
        Whether to read station coordinates. Default is True.
        
    """

    # Initialization
    domes = []

    # Open input file
    f = open(file, 'r')

    # Read file
    line = f.readline()
    while (line):
        r = record()
        r.code = line[0:4]
        r.pt = line[5:7]
        r.domes = line[8:17]
        r.description = line[18:40].strip()
        r.description = re.subn('–', '-', r.description)[0]
        r.description = unicodedata.normalize('NFD', r.description).encode('ascii', 'ignore').decode()
        while (len(r.description) < 22):
            r.description = r.description + ' '
      
        if (coord):
            tab = line.strip().split()
            r.lon = float(tab[-2])
            r.lat = float(tab[-1])
            #r.lon = int(r.lon) + (r.lon-int(r.lon))/0.6
            #r.lat = int(r.lat) + (r.lat-int(r.lat))/0.6
        
        domes.append(r)
        line = f.readline()

    # Some manual changes to handle duplicate DOMES numbers
    for i in range(len(domes)):
        if (domes[i].code == 'GOLD'):
            domes[i].domes = '40405S031'
        elif (domes[i].code == 'IISC'):
            domes[i].domes = '22306M002'
        elif (domes[i].code == 'KELY'):
            domes[i].domes = '43005M002'
        elif (domes[i].code == 'MDVO'):
            domes[i].domes = '12309M002'
        elif (domes[i].code == 'MTKA'):
            domes[i].domes = '21741S002'
        elif (domes[i].code == 'UPAD'):
            domes[i].domes = '12750M002'
        elif (domes[i].code == 'WEL2'):
            domes[i].domes = '50208S003'

    # Close file
    f.close()

    return domes

# Read discontinuity list in (pseudo-)SINEX format
#-------------------------------------------------
def read_solns(file):
  
    """
    Read discontinuity list in (pseudo-)SINEX format
    
    Returns
    -------
    solns : list
        Discontinuity list

    Parameters
    ----------
    file : str
        File to read
    """

    # Initializations
    solns = []
    code = ''
    pt  = ''
    ista  = -1    

    # Open input file
    f = open(file, 'r')

    # Try to reach beginning of SOLUTION/DISCONTINUITY block
    end = False
    while not(end):
        line = f.readline()
        if not(line):
            return
        elif (line[0:23] == '+SOLUTION/DISCONTINUITY'):
            end = True
        
    # Read SOLUTION/DISCONTINUITY block
    line = f.readline()
    end = False
    while not(end):
        if not(line):
            end = True
        elif (line[0:1] == '-'):
            end = True
        elif (line[0:1] != '*'):

            # New station?
            if ((line[1:5] != code) or (line[6:8] != pt)):
                ista = ista+1
                code = line[1:5]
                pt = line[6:8]
                r = record()
                r.code = code
                r.pt = pt
                r.P = []
                r.V = []
                solns.append(r)

            # Position soln?
            if (line[42:43] == 'P'):
                r = record()
                r.soln = line[9:13]
                r.start = line[16:28]
                r.end = line[29:41]
                r.cause = line[46:].strip()
                solns[ista].P.append(r)

            # Velocity soln?
            elif (line[42:43] == 'V'):
                r = record()
                r.soln = line[9:13]
                r.start = line[16:28]
                r.end = line[29:41]
                r.cause = line[46:].strip()
                solns[ista].V.append(r)

        line = f.readline()

    # Close file
    f.close()

    return solns

# Get ground antenna list from ANTEX file
#----------------------------------------
def get_ant_list(file):

    """
    Get ground antenna list from ANTEX file
    
    Returns
    -------
    ant : list
        List of ground antenna types

    Parameters
    ----------
    file : str
        ANTEX file
        
    """

    # Initialization
    ant = []

    # Open input ANTEX file
    f = open(file, 'r')
    line = f.readline()

    # While there remains something to read,
    while (line):

        # New ground antenna
        if ((line[60:76] == 'TYPE / SERIAL NO') and (line[0:5] != 'BLOCK') and (line[0:7] != 'GLONASS') and (line[0:7] != 'GALILEO') and (line[0:6] != 'BEIDOU') and (line[0:4] != 'QZSS') and (line[0:5] != 'IRNSS')):
            ant.append(line[0:20])
            
        line = f.readline()
        
    # Close input file
    f.close()

    return ant

# Create sinex instance with satellite PCOs from ANTEX file
#----------------------------------------------------------
def atx2snx(file, t):

    """
    Create sinex instance with satellite PCOs from ANTEX file
    
    Returns
    -------
    snx : sinex instance
        sinex instance containing as parameters the GPS, GLONASS and Galileo
        satellite PCOs valid at date t

    Parameters
    ----------
    file : str
        ANTEX file
    t : str
        Date in SINEX format
    """

    # Initialize sinex instance
    snx = sinex.sinex()
    snx.agency = 'ATX'
    snx.t = date().tsnx()
    snx.start = t
    snx.end = t
    snx.tech = 'P'
    snx.const = '0'
    snx.content = 'A'
    snx.param = []
    snx.x = []
    snx.sig = []

    # Open input ANTEX file
    f = open(file, 'r')
    line = f.readline()

    # While there remains something to read,
    while (line):
        
        # New GPS, GLONASS or Galileo satellite
        if (line[60:76] == 'TYPE / SERIAL NO') and ((line[0:5] == 'BLOCK') or (line[0:7] in ['GLONASS', 'GALILEO'])):
            svn = line[40:44]
            pco = []
            frq = []

            # Start of validity
            while (line[60:70] != 'VALID FROM'):
                line = f.readline()
            start = (date.from_ymdhms(int(line[2:6]), int(line[10:12]), int(line[16:18]), int(line[22:24]), int(line[28:30]), int(line[33:35]))).tsnx()

            # End of validity
            line = f.readline()
            end = '00:000:00000'
            if (line[60:71] == 'VALID UNTIL'):
                end = (date.from_ymdhms(int(line[2:6]), int(line[10:12]), int(line[16:18]), int(line[22:24]), int(line[28:30]), int(line[33:35]))).tsnx()
            
            # Get frequency-specific PCOs
            while (line[60:74] != 'END OF ANTENNA'):
                if (line[60:78] == 'START OF FREQUENCY'):
                    frq.append('L'+line[5])
                    line = f.readline()
                    pco.append(np.array([float(line[1:10]), float(line[11:20]), float(line[21:30])]) / 1000)
                line = f.readline()
                    
            # Compute iono-free PCO
            if (svn[0] == 'G'):
                f1 = 1575.42
                f2 = 1227.60
                i1 = frq.index('L1')
                i2 = frq.index('L2')
            elif (svn[0] == 'R'):
                f1 = 1602
                f2 = 1246
                i1 = frq.index('L1')
                i2 = frq.index('L2')
            elif (svn[0] == 'E'):
                f1 = 1575.42
                f2 = 1176.45
                i1 = frq.index('L1')
                i2 = frq.index('L5')
            frq.append('LC')
            pco.append((f1**2*pco[i1] - f2**2*pco[i2]) / (f1**2 - f2**2))

            # If current PCO is valid at requested date,
            if (earlier(start, t)) and ((end == '00:000:00000') or earlier(t, end)):
                
                # Loop over frequencies
                for i in range(len(frq)):
                    
                    # Add SATA_X parameter
                    r = record()
                    r.type = 'SATA_X'
                    r.code = svn
                    r.pt = frq[i]
                    r.soln = '----'
                    r.tref = t
                    r.unit = 'm   '
                    r.const = '0'
                    snx.param.append(r)
                    snx.x.append(pco[i][0])
                    snx.sig.append(0)
                    
                    # Add SATA_Y parameter
                    snx.param.append(copy.deepcopy(r))
                    snx.param[-1].type = 'SATA_Y'
                    snx.x.append(pco[i][1])
                    snx.sig.append(0)
                    
                    # Add SATA_Z parameter
                    snx.param.append(copy.deepcopy(r))
                    snx.param[-1].type = 'SATA_Z'
                    snx.x.append(pco[i][2])
                    snx.sig.append(0)
        
        line = f.readline()
        
    # Close input file
    f.close()

    # Finalize sinex instance
    snx.npar = len(snx.x)
    snx.x = np.array(snx.x)
    snx.sig = np.array(snx.sig)
    snx.sort_params()
    snx.set_par_ind()
    
    return snx

# Get site log of specified station
#----------------------------------
def get_sitelog(sta, logsource, X=None, dmax=100):

    """
    Get site log of specified station
    
    Returns
    -------
    log : str or None
        Path to site log
    source : str or None
        Site log source name

    Parameters
    ----------
    sta : str
        4-char station ID
    logsource : list
        Site log source list
    X : (3,) array_like
        Station cartesian coordinates [m]. Default is None (no coordinate check).
    dmax : float
        Maximum tolerated distance [km] between X and coordinates from site log. Default is 100.
    """

    # Initializations
    log = None
    source = None
    i = 0
    f = False

    # Loop over site log sources
    while ((not(f)) and (i < len(logsource))):
        files = os.listdir(logsource[i].localdir)

        # Is there a site log for current station in directory i?
        file = None
        if (sta.lower() in [ff[0:4] for ff in files]):
            ind = [ff[0:4] for ff in files].index(sta.lower())
            file = files[ind]
        elif (sta in [ff[0:4] for ff in files]):
            ind = [ff[0:4] for ff in files].index(sta)
            file = files[ind]
          
        # If yes,
        if (file is not None):

            # Get station coordinates from site log
            Xlog = sitelog_coord(logsource[i].localdir+'/'+file)
            
            # Distance wrt input station coordinates (km)
            d = 0
            if (X is not None) and (Xlog[0] != 0):
                d = np.sqrt(np.sum((X-Xlog)**2)) / 1000

            # If site log coordinates match station coordinates within tolerance
            # or if no coordinates are available in sitelog,
            if (d < dmax):
                f = True

            # Else, try next directory.
            else:
                i = i+1

        # Else, try next directory.
        else:
            i = i+1

    # If a site log was found for current station,
    if (f):
        log = logsource[i].localdir+'/'+file
        source = logsource[i].name
        
    return (log, source)

# Extract station coordinates from site log
#------------------------------------------
def sitelog_coord(file):

    """
    Get site log of specified station
    
    Returns
    -------
    X : (3,) array_like
        Station cartesian coordinates [m]. [0, 0, 0] if no coordinates available in site log.

    Parameters
    ----------
    file : str
        Path to site log
    """


    # Initialization
    X = np.zeros(3)

    # Open input file
    f = open(file, encoding='ISO-8859-1')
    line = f.readline()

    # Look for X,Y,Z coordinates
    while (line):

        if ('X coordinate' in line):
            try:
                tab = line.strip().split(':')
                tab = tab[1].strip().split()
                X[0] = float(tab[0])
            except:
                pass

        elif ('Y coordinate' in line):
            try:
                tab = line.strip().split(':')
                tab = tab[1].strip().split()
                X[1] = float(tab[0])
            except:
                pass

        elif ('Z coordinate' in line):
            try:
                tab = line.strip().split(':')
                tab = tab[1].strip().split()
                X[2] = float(tab[0])
            except:
                pass

        line = f.readline()

    # Close input file
    f.close()
    
    # Nullify all coordinates if one of them could not be read
    if not(np.all(X)):
        X = np.zeros(3)

    return X

# Read site log
#--------------
def read_sitelog(file, sinex_formatted=False, start=None, end=None):

    """
    Read site log
    
    Returns
    -------
    rec : list
        Receiver list
    ant : list
        Antenna list
    ecc : list
        Eccentricity list

    Parameters
    ----------
    file : str
        Path to site log
    sinex_format : bool, optional
        True if outputs should be formatted for SINEX metadata blocks. Default is False.
    start : str, optional
        Start date (in SINEX date format). Default is None.
    end : str, optional
        End date (in SINEX date format). Default is None.
        
    """
  
    # Initializations
    rec = []
    ant = []
    ecc = []
    receiver = False
    antenna = False
    
    # Open input file
    f = open(file, encoding='ISO-8859-1')
    
    # Read input file
    line = f.readline()
    while (line):
      
        # New receiver
        if (re.match('3\.\d+\s+Receiver Type', line, re.I)):
            receiver = True
            antenna = False
            r = record()
            i = line.index(':')
            r.type = line[i+2:].strip()[0:20].upper()
            while (len(r.type) < 20):
                r.type = r.type + ' '
            r.system = 'GPS'
            r.serie = '-----'
            r.firmware = '-----------'
            r.cutoff = 'n/a'
            r.start = '00:000:00000'
            r.end = '00:000:00000'
            rec.append(r)
          
        # New antenna
        elif (re.match('4\.\d+\s+Antenna Type', line, re.I)):
            receiver = False
            antenna = True
            r = record()
            i = line.index(':')
            r.type = line[i+2:].strip().split()[0][0:16].upper()
            while (len(r.type) < 16):
              r.type = r.type + ' '
            r.type = r.type + 'NONE'
            r.serie = '-----'
            r.start = '00:000:00000'
            r.end = '00:000:00000'
            r.system = 'UNE'
            r.dx = ['  0.0000']*3
            r.daz = '   0'
            ant.append(r)
        
        # Satellite systems
        elif (re.match('\s*Satellite System.*:', line, re.I) and receiver):
            i = line.index(':')
            if (line[i+2:].strip()):
                rec[-1].system = line[i+2:].strip()
        
        # Serial number
        elif (re.match('\s*Serial Number.*:', line, re.I) and (receiver or antenna)):
            i = line.index(':')
            if (line[i+2:].strip()):
                if not(line[i+2:].strip().split()[0] in ['(A20,', '(A20)']):
                    if (receiver):
                        rec[-1].serie = line[i+2:].strip()
                    elif (antenna):
                        ant[-1].serie = line[i+2:].strip()
          
        # Firmware version
        elif (re.match('\s*Firmware Version.*:', line, re.I) and receiver):
            i = line.index(':')
            if (line[i+2:].strip()):
                if not(line[i+2:].strip().split()[0] in ['(A11,', '(A11)']):
                    rec[-1].firmware = line[i+2:].strip()
              
        # Cutoff angle
        elif (re.match('\s*Elevation Cutoff Setting.*:', line, re.I) and receiver):
            i = line.index(':')
            if (line[i+2:].strip()):
                tab = line[i+2:].strip().split()
                if (isfloat(tab[0])):
                    rec[-1].cutoff = '{0:>3d}'.format(int(float(tab[0])))

        # Date installed
        elif (re.match('\s*Date Installed.*:', line, re.I) and (receiver or antenna)):
            i = line.index(':')
            if (receiver):
                rec[-1].start = sitelog_date(line[i+2:].strip())
            elif (antenna):
                ant[-1].start = sitelog_date(line[i+2:].strip())
          
        # Date removed
        elif (re.match('\s*Date Removed.*:', line, re.I) and (receiver or antenna)):
            i = line.index(':')
            if (receiver):
                rec[-1].end = sitelog_date(line[i+2:].strip())
            elif (antenna):
                ant[-1].end = sitelog_date(line[i+2:].strip())
            
        # Up eccentricity
        elif ((re.match('\s*Antenna Height.*:', line, re.I) or re.match('\s*Marker->ARP Up Ecc.*:', line, re.I)) and antenna):
            i = line.index(':')
            tab = line[i+2:].strip().split()
            if (len(tab) > 0):
                tab[0] = re.subn('m', '', tab[0])[0]
                if (isfloat(tab[0])):
                    ant[-1].dx[0] = '{0:8.4f}'.format(float(tab[0]))
          
        # North eccentricity
        elif (re.match('\s*Marker->ARP North Ecc.*:', line, re.I) and antenna):
            i = line.index(':')
            tab = line[i+2:].strip().split()
            if (len(tab) > 0):
                tab[0] = re.subn('m', '', tab[0])[0]
                if (isfloat(tab[0])):
                    ant[-1].dx[1] = '{0:8.4f}'.format(float(tab[0]))
          
        # East eccentricity
        elif (re.match('\s*Marker->ARP East Ecc.*:', line, re.I) and antenna):
            i = line.index(':')
            tab = line[i+2:].strip().split()
            if (len(tab) > 0):
                tab[0] = re.subn('m', '', tab[0])[0]
                if (isfloat(tab[0])):
                    ant[-1].dx[2] = '{0:8.4f}'.format(float(tab[0]))
              
        # Radome type
        elif (re.match('\s*Antenna Radome Type.*:', line, re.I) and antenna):
            i = line.index(':')
            rad = line[i+2:].strip()[0:4].strip().upper()
            if (len(rad) == 4):
                ant[-1].type = ant[-1].type[0:16] + rad
        
        # Alignment from true North
        elif (re.match('\s*Alignment from True N.*:', line, re.I) or re.match('\s*Degree Offset from North.*:', line, re.I)) and (antenna):
            i = line.index(':')
            tab = line[i+2:].strip().split()
            if (len(tab) > 0):
                tab[0] = re.subn('deg', '', tab[0])[0]
                if (isfloat(tab[0])):
                    ant[-1].daz = '{0:4d}'.format(round(float(tab[0])))
        
        # Default receiver or default antenna or new section
        elif ((line[0:3] == '3.x') or (line[0:3] == '4.x') or (line[0:2] == '5.')):
            receiver = False
            antenna = False
        
        line = f.readline()
    
    # Close input file
    f.close()
    
    # Remove receivers with undefined dates
    i = 0
    while (i < len(rec)):
        if ((rec[i].start == '00:000:00000') and (rec[i].end == '00:000:00000')):
            rec.pop(i)
        else:
            i = i+1

    # Remove antennas with undefined dates
    i = 0
    while (i < len(ant)):
        if ((ant[i].start == '00:000:00000') and (ant[i].end == '00:000:00000')):
            ant.pop(i)
        else:
            i = i+1
    
    # Change receiver removal dates if needed
    for i in range(len(rec)-1):
        if (rec[i].end == '00:000:00000'):
            rec[i].end = rec[i+1].start
    for i in range(1, len(rec)):
        if (rec[i].start == '00:000:00000'):
            rec[i].start = rec[i-1].end
    for i in range(len(rec)-1):
        if (earlier(rec[i+1].start, rec[i].end)):
            rec[i].end = rec[i+1].start      

    # Change antenna removal dates if needed
    for i in range(len(ant)-1):
        if (ant[i].end == '00:000:00000'):
            ant[i].end = ant[i+1].start
    for i in range(1, len(ant)):
        if (ant[i].start == '00:000:00000'):
            ant[i].start = ant[i-1].end
    for i in range(len(ant)-1):
        if (earlier(ant[i+1].start, ant[i].end)):
            ant[i].end = ant[i+1].start      
        
    # Some more work to do if outputs should be SINEX formatted
    if (sinex_formatted):
        
        # Copy antenna list into eccentricty list
        ecc = copy.deepcopy(ant)
        
        # Merge successive receivers if needed
        i = 0
        while (i < len(rec)-1):
            if ((rec[i].type == rec[i+1].type) and (rec[i].serie == rec[i+1].serie) and (rec[i].firmware == rec[i+1].firmware)):
                rec[i].end = rec[i+1].end
                rec.pop(i+1)
            else:
                i = i+1
            
        # Merge successive antennas if needed
        i = 0
        while (i < len(ant)-1):
            if ((ant[i].type == ant[i+1].type) and (ant[i].serie == ant[i+1].serie) and (ant[i].daz == ant[i+1].daz)):
                ant[i].end = ant[i+1].end
                ant.pop(i+1)
            else:
              i = i+1
            
        # Merge successive eccentricities if needed
        i = 0
        while (i < len(ecc)-1):
            if ((ecc[i].dx[0] == ecc[i+1].dx[0]) and (ecc[i].dx[1] == ecc[i+1].dx[1]) and (ecc[i].dx[2] == ecc[i+1].dx[2])):
                ecc[i].end = ecc[i+1].end
                ecc.pop(i+1)
            else:
                i = i+1
            
        # Format receiver serial numbers
        for r in rec:
            r.serie = r.serie[0:5]
            while (len(r.serie) < 5):
                r.serie = r.serie + ' '
            
        # Format receiver firmware versions
        for r in rec:
            r.firmware = r.firmware[0:11]
            while (len(r.firmware) < 11):
                r.firmware = r.firmware + ' '
            
        # Format antenna serial numbers
        for r in ant:
            r.serie = r.serie[0:5]
            while (len(r.serie) < 5):
                r.serie = r.serie + ' '
            
        # Change some antenna types
        for r in ant:
            if (r.type[0:16] == 'DORNE           '):
                r.type = 'AOAD/M_T        '+r.type[16:20]
            elif ((r.type[0:16] == 'ASHTECH         ') or (r.type[0:16] == 'ASH             ')):
                r.type = 'ASH700936A_M    '+r.type[16:20]
            elif (r.type[0:16] == 'TR              '):
                r.type = 'TRM22020.00+GP  '+r.type[16:20]
            elif (r.type[0:16] == '4000ST          '):
                r.type = 'TRM14532.00     '+r.type[16:20]
            elif (r.type[0:16] == 'TRIMBLE         '):
                r.type = 'TRM29659.00     '+r.type[16:20]
    
    # Finally remove records that do not fall within period of interest
    if (start):
        i = 0
        while (i < len(rec)):
            r = rec[i]
            if earlier(start, r.end) or (r.end == '00:000:00000'):
                i = i+1
            else:
                rec.pop(i)

        i = 0
        while (i < len(ant)):
            r = ant[i]
            if earlier(start, r.end) or (r.end == '00:000:00000'):
                i = i+1
            else:
                ant.pop(i)

        i = 0
        while (i < len(ecc)):
            r = ecc[i]
            if earlier(start, r.end) or (r.end == '00:000:00000'):
                i = i+1
            else:
                ecc.pop(i)

    if (end):
        i = 0
        while (i < len(rec)):
            r = rec[i]
            if earlier(r.start, end) or (r.start == '00:000:00000'):
                i = i+1
            else:
                rec.pop(i)

        i = 0
        while (i < len(ant)):
            r = ant[i]
            if earlier(r.start, end) or (r.start == '00:000:00000'):
                i = i+1
            else:
                ant.pop(i)

        i = 0
        while (i < len(ecc)):
            r = ecc[i]
            if earlier(r.start, end) or (r.start == '00:000:00000'):
                i = i+1
            else:
                ecc.pop(i)

    return (rec, ant, ecc)

# Convert date from site log into SINEX date format
#--------------------------------------------------
def sitelog_date(t):

    """
    Convert date from site log into SINEX date format
    
    Returns
    -------
    s : str
        Date in SINEX format ('yy:ddd:sssss')
        
    Parameters
    ----------
    t : str
        Date from site log
    """
  
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # yyyy-mm-ddThh:mm
    if (re.match('\d{4}.\d{2}.\d{2}.\d{2}.\d{2}', t)):
        try:
            return date.from_ymdhms(int(t[0:4]), int(t[5:7]), int(t[8:10]), int(t[11:13]), int(t[14:16])).tsnx()
        except ValueError:
            return '00:000:00000'
    
    # dd-mon-yyyy hh:mm
    elif (re.match('\d{2}.[a-z]{3}.\d{4}.\d{2}.\d{2}', t, re.I)):
        try:
            i = months.index(t[3:6].lower())
            return date.from_ymdhms(int(t[7:11]), i+1, int(t[0:2]), int(t[12:14]), int(t[15:17])).tsnx()
        except ValueError:
            return '00:000:00000'
      
    # yyyy-mm-dd
    elif (re.match('\d{4}.\d{2}.\d{2}', t)):
        try:
            return date.from_ymdhms(int(t[0:4]), int(t[5:7]), int(t[8:10])).tsnx()
        except ValueError:
            return '00:000:00000'
    
    # dd-mon-yyyy
    elif (re.match('\d{2}.[a-z]{3}.\d{4}', t, re.I)):
        try:
            i = months.index(t[3:6].lower())
            return date.from_ymdhms(int(t[7:11]), i+1, int(t[0:2])).tsnx()
        except ValueError:
            return '00:000:00000'
    
    # dd-mm-yyyy
    elif (re.match('\d{2}.\d{2}.\d{4}', t)):
        try:
            return date.from_ymdhms(int(t[6:10]), int(t[3:5]), int(t[0:2])).tsnx()
        except ValueError:
            return '00:000:00000'
    
    # yyyy-mm-d
    elif (re.match('\d{4}.\d{2}.\d{1}', t)):
        try:
            return date.from_ymdhms(int(t[0:4]), int(t[5:7]), int(t[8])).tsnx()
        except ValueError:
            return '00:000:00000'
    
    # Unsupported date format
    else:
        return '00:000:00000'

# Create sinex instance from site logs
#-------------------------------------
def sitelogs2snx(logsource):

    """
    Create sinex instance from site logs
    
    Returns
    ------
    snx : sinex instance
        sinex instance containing station metadata
    
    Parameters
    ----------
    logsource : list
        Site log source list
    """
    
    # Initializations
    snx = sinex.sinex()
    snx.agency = agency
    snx.start = '00:000:00000'
    snx.end = '00:000:00000'
    snx.tech = 'P'
    snx.npar = 0
    snx.const = '0'
    snx.content = ''
    snx.sta = []
    
    # Loop over sitelog sources
    for source in logsource:
        
        # List of site logs in local directory of current source
        files = np.sort(os.listdir(source.localdir))
        
        # Loop over site logs from current source
        for f in files:
            
            # 4-char ID and coordinates of current station
            code = f[0:4].upper()
            X = sitelog_coord(source.localdir+'/'+f)
            
            # Is there already a station with the same 4-char ID less than 100 km away in snx.sta?
            b = False
            keys = [s.code for s in snx.sta]
            if (code in keys):
                ind = np.nonzero(np.array(keys) == code)[0]
                d = [sqrt(np.sum((s.X-X)**2)) / 1000 for s in [snx.sta[i] for i in ind]]
                if (np.min(d) < 100):
                    b = True
                    
            # If not, then add a new station into snx.sta
            if not(b):
                r = record()
                r.code = code
                r.pt = '--'
                r.domes = 9*' '
                r.tech = 'P'
                r.description = 'Sitelog source: {0:<6s}'.format(source.name)
                r.lon = 11*' '
                r.lat = 11*' '
                r.h = 7*' '
                r.X = X
                (r.rec, r.ant, r.ecc) = read_sitelog(source.localdir+'/'+f, sinex_formatted=True)
                r.source = source.name
                snx.sta.append(r)
                
    # Sort snx.sta
    ind = np.argsort([s.code for s in snx.sta])
    snx.sta = [snx.sta[i] for i in ind]
    
    return snx

# Read CMT catalog (.ndk files)
#------------------------------
def read_ndk(files):
  
    """
    Read CMT catalog (.ndk files)
    
    Returns
    -------
    eqs : list
        Earthquake list

    Parameters
    ----------
    file : list
        List of files to read
    """
    
    # Initialization
    eqs = []
    
    # Loop over input files
    for file in files:
  
        # Open input file and read 1st line
        f = open(file)
        line = f.readline()
        
        # While there remains lines to read,
        while (line):
            
            # New earthquake record
            r = record()
            
            # Date
            sec = int(round(float(line[22:26])))
            if (sec < 60):
                r.t = date.from_ymdhms(int(line[5:9]), int(line[10:12]), int(line[13:15]), int(line[16:18]), int(line[19:21]), sec)
            else:
                r.t = date.from_ymdhms(int(line[5:9]), int(line[10:12]), int(line[13:15]), int(line[16:18]), int(line[19:21]), sec-1)
                r.t.add_s(1)
            
            # Location
            r.lat = float(line[27:33])
            r.lon = float(line[34:41])
            r.depth = float(line[42:47])
            r.location = line[56:].strip().title()
            
            # Read next 3 lines
            line = f.readline()
            line = f.readline()
            line = f.readline()
            
            # Moment exponent converted into N.m
            momentexp = int(line[0:2]) - 7
            
            # Read next line
            line = f.readline()
            
            # Log-moment
            r.logmoment = log10(float(line[49:56])) + momentexp
            
            # Magnitude
            r.Mw = round(2./3*(r.logmoment-9.1)*10)/10
            
            # Moment tensor
            r.Tval = float(line[4:11]) * 10**momentexp
            r.Tpl = float(line[12:14])
            r.Taz = float(line[15:18])
            r.Nval = float(line[19:26]) * 10**momentexp
            r.Npl = float(line[27:29])
            r.Naz = float(line[30:33])
            r.Pval = float(line[34:41]) * 10**momentexp
            r.Ppl = float(line[42:44])
            r.Paz = float(line[45:48])
            
            # Nodal planes
            r.strike1 = float(line[57:60])
            r.dip1 = float(line[61:63])
            r.rake1 = float(line[64:68])
            r.strike2 = float(line[69:72])
            r.dip2 = float(line[73:75])
            r.rake2 = float(line[76:80])
            
            # Compute rupture parameters from either Mai & Beroza (2000)'s scaling law for large earthquakes
            # or Yen & Ma (2012)'s scaling law for smaller earthquakes
            if (r.Mw > 7.6):
                r.width = 10**(-1.28 + 0.29*r.logmoment)
                r.length = 10 **(-2.20 + 0.35*r.logmoment)
                r.slip = 10**(-6.98 + 0.35*r.logmoment)
            elif (r.logmoment >= 20):
                r.width = 10**(-1.84 + r.logmoment/3)
                r.length = 10**(-2.27 + r.logmoment/3)
                r.slip = 10**(-6.37 + r.logmoment/3)
            else:
                r.width = 10**(-5.08 + r.logmoment/2)
                r.length = r.width
                r.slip = 10**(-0.32)
            
            # Store earthquake
            eqs.append(r)
            
            # Read next line
            line = f.readline()
    
        # Close input file
        f.close()
    
    return eqs
