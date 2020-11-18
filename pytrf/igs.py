"""
pytrf IGS utilities

This subpackage contains various useful routines for the IGS SINEX combination.

"""

# External imports
#-----------------
import os
import numpy as np

# Internal imports
#-----------------
from pytrf.io import read_yaml



# Read list of stations to be rejected from AC solutions
#-------------------------------------------------------
def read_rej(f, d, ac):
    
    """
    Read list of stations to be rejected from AC solutions

    Parameters
    ----------
    f : str
        Input YAML file
    d : int
        Day in week
    ac : list
        List of ACs
    """

    # Initialize ac[*].rej fields
    for a in ac:
        a.rej = []

    # Read station rejection file
    rej = read_yaml(f)
    
    # Loop over rejection records
    if (rej):
        for r in rej:
        
            # Translate r.days into a list of days
            daylist = []
            if (isinstance(r.days, int)):
                daylist.append(r.days)
            else:
                for s in r.days.split(';'):
                    if (len(s) == 1):
                        daylist.append(int(s))
                    else:
                        daylist.extend(range(int(s[0]), int(s[2])+1))
                
            # If current day is in this list of days
            if (d in daylist):
                
                # Update list of stations to reject from concerned ACs
                for a in r.acs.split(';'):
                    if (a in [c.name for c in ac]):
                        i = [c.name for c in ac].index(a)
                        ac[i].rej.extend(r.stas.split(';'))
            
    # Sort and remove doublons from each station list
    for i in range(len(ac)):
        ac[i].rej = np.sort(np.unique(ac[i].rej))

# Write residuals from daily IGS combination
#-------------------------------------------
def write_res(snx, acs, w, d, f):
    
    """
    Write residuals from daily IGS combination

    Parameters
    ----------
    snx : sinex instance
        Daily combined solution
    acs : list
        List of combination inputs
    w : str
        GPS week
    d : str
        Day of week
    f : str
        Output file
    """

    # Open output file
    f = open(f, 'w')
    
    # Write header
    print('--------------------------------------------------------------------------------', file=f)
    print('Residuals from IGS combination of daily AC repro3 solutions for week {0}, day {1}'.format(w, d), file=f)
    print('--------------------------------------------------------------------------------', file=f)
    print('', file=f)
    print(' Daily AC solutions:', file=f)
    for ac in acs:
        print('  - {0} = {1}'.format(ac.name, os.path.basename(ac.file)), file=f)
    print('', file=f)
    print(' Daily combined solution :', file=f)
    print('  - igs = {0}'.format(snx.file), file=f)
    print('', file=f)
    print(' residual = AC estimate - IGS combined estimate - 7-parameter transfo', file=f)
    print(' sigma    = formal error of AC estimate', file=f)
    #print(' WAVG      = sum(res/sig**2)/sum(1/sig**2)', file=f)
    #print(' WRMS      = sqrt(sum(res**2/sig**2)/sum(1/sig**2))', file=f)
    print('', file=f)
    #s = ''
    #for ac in acs:
        #if not(ac.comb):
            #s = s+' '+ac.name
    #if (len(s) > 0):
        #print(' Solutions excluded from parameter statistics:'+s, file=f)
        #print('', file=f)
        
    # Station position residuals header
    print('', file=f)
    print('', file=f)
    print(' 1) Station position residuals:', file=f)
    print(' ------------------------------', file=f)
    print('', file=f)
    print('            ___________residual__________ _____________sigma___________', file=f)
    print(' sta    AC     E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm] ', file=f)
    print(' ----------------------------------------------------------------------', file=f)
    
    # Station position residuals
    for ac in acs:
        ac.res = []
        ac.sig = []
        
    for i in snx.ix:
        code = snx.param[i].code
        #res = []
        #sig = []
        for ac in acs:
            if (code in [p.code for p in [ac.snx.param[k] for k in ac.snx.ix]]):
                ix = ac.snx.ix[[p.code for p in [ac.snx.param[k] for k in ac.snx.ix]].index(code)]
                print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3]), file=f)
                ac.res.append(ac.v[ix:ix+3])
                ac.sig.append(ac.sv[ix:ix+3])
                #if (ac.comb):
                    #res.append(ac.v[ix:ix+3])
                    #sig.append(ac.sv[ix:ix+3])
        #res = np.array(res)
        #sig = np.array(sig)
        #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
        #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))
        #print(' {0}      WAVG/WRMS {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(code, wavg, wrms), file=f)
        print(' ----------------------------------------------------------------------', file=f)
    
    #res = []
    #sig = []
    #for ac in acs:
        #if (ac.comb):
            #res.extend(ac.res)
            #sig.extend(ac.sig)
        #ac.res = np.array(ac.res)
        #ac.sig = np.array(ac.sig)
        #ac.nsta = len(ac.res)
        #ac.wavg = np.sum(ac.res/ac.sig**2, axis=0) / np.sum(1/ac.sig**2, axis=0)
        #ac.wrms = np.sqrt(np.sum((ac.res/ac.sig)**2, axis=0) / np.sum(1/ac.sig**2, axis=0))
        #print(' WAVG/WRMS {0}       {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(ac.name, ac.wavg, ac.wrms), file=f)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))
    #print(' WAVG/WRMS WAVG/WRMS {0[0]:9.3f} {0[1]:9.3f} {0[2]:9.3f} {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f}'.format(wavg, wrms), file=f)
    #print(' -------------------------------------------------------------------------------', file=f)

    # Other parameter residuals header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 2) ERP, geocenter and scale residuals:', file=f)
    print(' --------------------------------------', file=f)
    print('', file=f)
    print(' param unit  AC   residual    sigma ', file=f)
    print(' -----------------------------------', file=f)

    # XPO residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.xpo):
            ix = ac.snx.ixpo[0]
            ac.vxpo = 1000*ac.v[ix]
            ac.svxpo = 1000*ac.sv[ix]
            print(' XPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpo, ac.svxpo), file=f)
            #if (ac.comb):
                #res.append(ac.vxpo)
                #sig.append(ac.svxpo)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' XPO   uas   WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # YPO residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.ypo):
            ix = ac.snx.iypo[0]
            ac.vypo = 1000*ac.v[ix]
            ac.svypo = 1000*ac.sv[ix]
            print(' YPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypo, ac.svypo), file=f)
            #if (ac.comb):
                #res.append(ac.vypo)
                #sig.append(ac.svypo)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' YPO   uas   WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # XPOR residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.xpor):
            ix = ac.snx.ixpor[0]
            ac.vxpor = 1000*ac.v[ix]
            ac.svxpor = 1000*ac.sv[ix]
            print(' XPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpor, ac.svxpor), file=f)
            #if (ac.comb):
                #res.append(ac.vxpor)
                #sig.append(ac.svxpor)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' XPOR  uas/d WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # YPOR residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.ypor):
            ix = ac.snx.iypor[0]
            ac.vypor = 1000*ac.v[ix]
            ac.svypor = 1000*ac.sv[ix]
            print(' YPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypor, ac.svypor), file=f)
            #if (ac.comb):
                #res.append(ac.vypor)
                #sig.append(ac.svypor)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' YPOR  uas/d WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # LOD residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.lod):
            ix = ac.snx.ilod[0]
            ac.vlod = 1000*ac.v[ix]
            ac.svlod = 1000*ac.sv[ix]
            print(' LOD   us    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vlod, ac.svlod), file=f)
            #if (ac.comb):
                #res.append(ac.vlod)
                #sig.append(ac.svlod)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' LOD   us    WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # XGC residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]
            ac.vxgc = ac.v[ix]
            ac.svxgc = ac.sv[ix]
            print(' XGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxgc, ac.svxgc), file=f)
            #if (ac.comb):
                #res.append(ac.vxgc)
                #sig.append(ac.svxgc)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' XGC   mm    WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # YGC residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]+1
            ac.vygc = ac.v[ix]
            ac.svygc = ac.sv[ix]
            print(' YGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vygc, ac.svygc), file=f)
            #if (ac.comb):
                #res.append(ac.vygc)
                #sig.append(ac.svygc)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' YGC   mm    WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # ZGC residuals
    #res = []
    #sig = []
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]+2
            ac.vzgc = ac.v[ix]
            ac.svzgc = ac.sv[ix]
            print(' ZGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vzgc, ac.svzgc), file=f)
            #if (ac.comb):
                #res.append(ac.vzgc)
                #sig.append(ac.svzgc)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' ZGC   mm    WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # SC residuals
    #res = []
    #sig = []
    for ac in acs:
        ix = ac.snx.isc[0]
        ac.vsc = ac.v[ix]
        ac.svsc = ac.sv[ix]
        print(' SC    ppb   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vsc, ac.svsc), file=f)
        #if (ac.comb):
            #res.append(ac.vsc)
            #sig.append(ac.svsc)
    #res = np.array(res)
    #sig = np.array(sig)
    #wavg = np.sum(res/sig**2, axis=0) / np.sum(1/sig**2, axis=0)
    #wrms = np.sqrt(np.sum((res/sig)**2, axis=0) / np.sum(1/sig**2, axis=0))                
    #print(' SC    ppb   WAVG/WRMS {0:9.3f} {1:9.3f}'.format(wavg, wrms), file=f)
    print(' -----------------------------------', file=f)

    # Close output file
    f.close()
    
# Write summary of weekly IGS combination
#----------------------------------------
def write_sum(dacs, w, opt, nsta, ndat, Tdat, wdat, nolog, f):
    
    """
    Write summary of weekly IGS combination

    Parameters
    ----------
    acs : list
        List of list of daily combination inputs
    w : str
        GPS week
    opt : record
        Combination options
    nsta : list
        Numbers of stations in daily combined solutions
    ndat : list
        Numbers of RF stations in daily combined solutions
    Tdat : list
        RF -> daily combined solutions transformation parameters
    wdat : list
        RMS of RF -> daily combined solutions transformations
    nolog : list
        List of stations without sitelogs
    f : str
        Output file
    """
    
    # Build full list of ACs
    acs = []
    files = []
    for d in range(7):
        for ac in dacs[d]:
            if not(ac.name in acs):
                acs.append(ac.name)
                file = os.path.basename(ac.file)
                files.append(file[:11] + '${yyyy}${doy}' + file[18:])
    ind = np.argsort(acs)
    acs = [acs[i] for i in ind]
    files = [files[i] for i in ind]
    
    # Read list of sitelog sources
    logsource = read_yaml(opt.logsource)
    
    # Open output file
    f = open(f, 'w')
    
    # Write header
    print('----------------------------------------------------------', file=f)
    print('IGS combination of daily AC repro3 solutions for week {0}'.format(w), file=f)
    print('----------------------------------------------------------', file=f)
    print('', file=f)
    print(' Author:  Paul Rebischung', file=f)
    print(' Contact: igs-rf@ign.fr', file=f)
    print('', file=f)
    print(' Daily AC solutions:', file=f)
    for i in range(len(acs)):
        print('  - {0} = {1}'.format(acs[i], files[i]), file=f)
    print('', file=f)
    print(' Daily combined solutions:', file=f)
    file = opt.dailysnx[:11] + '${yyyy}${doy}' + opt.dailysnx[18:]
    print('  - igs = {0}'.format(file), file=f)

    # Main statistics header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 1) Main combination statistics:', file=f)
    print(' -------------------------------', file=f)
    print('  - #sta   = number of stations (after rejection of outliers)', file=f)
    print('  - #RF    = number of stations used for alignment to {0}'.format(opt.datumname), file=f)
    print('  - VF^0.5 = square root of estimated variance factor', file=f)
    print('  - WRMS   = WRMS of "AC - igs" station position residuals', file=f)
    print('', file=f)
    print('                                _____________WRMS____________', file=f)
    print(' AC  day  #sta   #RF    VF^0.5     E[mm]     N[mm]     H[mm] ', file=f)
    print(' ------------------------------------------------------------', file=f)

    # Main statistics
    for i in range(len(acs)):
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                print(' {0.name}   {1} {0.nsta:5d} {0.ndat:5d} {0.sf:10.6f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(ac, d, ac.wrms), file=f)
        print(' ------------------------------------------------------------', file=f)
    for d in range(7):
        print(' igs   {0} {1:5d} {2:5d}'.format(d, nsta[d], ndat[d]), file=f)
    print(' ------------------------------------------------------------', file=f)
        
    ## Station position residuals header
    #print('', file=f)
    #print('', file=f)
    #print('', file=f)
    #print(' 2) Station position residuals:', file=f)
    #print(' ------------------------------', file=f)
    #print('  - residuals = AC estimates - IGS combined estimates - 7-parameter transfo', file=f)
    #print('  - sigmas    = formal errors of AC estimates', file=f)
    #print('  - WAVG      = sum(res/sig**2)/sum(1/sig**2)', file=f)
    #print('  - WRMS      = sqrt(sum(res**2/sig**2)/sum(1/sig**2))', file=f)
    #print('', file=f)
    #print('         _____________WAVG____________ _____________WRMS____________', file=f)
    #print(' AC  day    E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm] ', file=f)
    #print(' -------------------------------------------------------------------', file=f)

    ## Station position residuals
    #for i in range(len(acs)):
        #for d in range(7):
            #if (acs[i] in [ac.name for ac in dacs[d]]):
                #ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                #print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac.name, d, ac.wavg, ac.wrms), file=f)
        #print(' -------------------------------------------------------------------', file=f)

    # ERP, geocenter and scale residuals header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 2) "AC - igs" ERP, geocenter and scale residuals:', file=f)
    print(' -------------------------------------------------', file=f)
    #print('  - residuals = AC estimates - IGS combined estimates - 7-parameter transfo', file=f)
    print('', file=f)
    print('             XPO       YPO       XPOR      YPOR      LOD       XGC       YGC       ZGC       SC', file=f)
    print(' AC  day    [uas]     [uas]    [uas/d]   [uas/d]     [us]      [mm]      [mm]      [mm]     [ppb]', file=f)
    print('--------------------------------------------------------------------------------------------------', file=f)

    # ERP, geocenter and scale residuals
    for i in range(len(acs)):
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                s = ' '+ac.name+'   '+str(d)
                if hasattr(ac, 'vxpo'):
                    s = s + ' {0:9.3f}'.format(ac.vxpo)
                else:
                    s = s + '          '
                if hasattr(ac, 'vypo'):
                    s = s + ' {0:9.3f}'.format(ac.vypo)
                else:
                    s = s + '          '
                if hasattr(ac, 'vxpor'):
                    s = s + ' {0:9.3f}'.format(ac.vxpor)
                else:
                    s = s + '          '
                if hasattr(ac, 'vypor'):
                    s = s + ' {0:9.3f}'.format(ac.vypor)
                else:
                    s = s + '          '
                if hasattr(ac, 'vlod'):
                    s = s + ' {0:9.3f}'.format(ac.vlod)
                else:
                    s = s + '          '
                if hasattr(ac, 'vxgc'):
                    s = s + ' {0:9.3f} {1:9.3f} {2:9.3f}'.format(ac.vxgc, ac.vygc, ac.vzgc)
                else:
                    s = s + 3*'          '
                if hasattr(ac, 'vsc'):
                    s = s + ' {0:9.3f}'.format(ac.vsc)
                else:
                    s = s + '          '
                print(s, file=f)
        print('--------------------------------------------------------------------------------------------------', file=f)
                
    # Transformation parameters header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 3) "AC -> {0}" 7-parameter transformations:'.format(opt.datumname), file=f)
    print(' ---------------------------------------------', file=f)
    print('  - RMS = RMS of residuals from "AC -> {0}" 7-parameter transformations'.format(opt.datumname), file=f)
    print('', file=f)
    print('             TX        TY        TZ        SC        RX        RY        RZ    _____________RMS_____________', file=f)
    print(' AC  day    [mm]      [mm]      [mm]      [ppb]     [mas]     [mas]     [mas]     E[mm]     N[mm]     H[mm] ', file=f)
    print(' -----------------------------------------------------------------------------------------------------------', file=f)

    # Transformation parameters
    for i in range(len(acs)):
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {2[3]:9.3f} {2[4]:9.3f} {2[5]:9.3f} {2[6]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac.name, d, -ac.Tdat, ac.wdat), file=f)
        print(' -----------------------------------------------------------------------------------------------------------', file=f)
    for d in range(7):
        print(' igs   {0} {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f} {1[3]:9.3f} {1[4]:9.3f} {1[5]:9.3f} {1[6]:9.3f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(d, -Tdat[d], wdat[d]), file=f)
    print(' -----------------------------------------------------------------------------------------------------------', file=f)

    # Manually rejected stations header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 4) Manually rejected stations:', file=f)
    print(' ------------------------------', file=f)
    print('', file=f)
    print(' AC  day   stations', file=f)
    print(' -------------------------------------------------------------------------------', file=f)

    # Manually rejected stations
    for i in range(len(acs)):
        b = False
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                if (len(ac.rej) > 0):
                    b = True
                    s = ' {0}   {1}   '.format(ac.name, d)
                    for sta in ac.rej:
                        s = s + sta + ' '
                    print(s[:-1], file=f)
        if (b):
            print(' -------------------------------------------------------------------------------', file=f)
        
    # Outliers header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 5) Outliers:', file=f)
    print(' ------------', file=f)
    print('', file=f)
    print('                __________residuals__________', file=f)
    print(' AC  day   sta     E[mm]     N[mm]     H[mm] ', file=f)
    print(' --------------------------------------------', file=f)

    # Outliers
    for i in range(len(acs)):
        b = False
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                if hasattr(ac, 'staout'):
                    ind = np.argsort(ac.staout)
                    for j in ind:
                        b = True
                        print(' {0}   {1}   {2} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac.name, d, ac.staout[j], ac.resout[j]), file=f)
        if (b):
            print(' --------------------------------------------', file=f)

    # Station metadata inconsistencies header
    print('', file=f)
    print('', file=f)
    print('', file=f)
    print(' 6) Station metadata inconsistencies:', file=f)
    print(' ------------------------------------', file=f)
    for source in logsource:
        print('  - {0.name:<6s} = {0.server}{0.remotedir}'.format(source), file=f)
    print('', file=f)
    if (len(nolog) > 0):
        s = '  - No sitelog found for station(s): '
        for sta in nolog:
            s = s + sta + ', '
        print(s[:-2], file=f)
        print('', file=f)
    print(' AC    sta    ___metadata_type___   __info_from_sinex___   _info_from_sitelog__   source', file=f)
    print(' ---------------------------------------------------------------------------------------', file=f)

    # Station metadata inconsistencies
    for i in range(len(acs)):
        metaerr = []
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                metaerr.extend(ac.metaerr)
        if (len(metaerr) > 0):
            metaerr = np.sort(np.unique(metaerr))
            for err in metaerr:
                print(' {0}   {1}'.format(ac.name, err), file=f)
            print(' ---------------------------------------------------------------------------------------', file=f)

    # Close output file
    f.close()
