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
def write_res(snx, acs, w, d, fres, fyml):
    
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
    fres : str
        Output residual file
    fyml : str
        Output YAML file
    """

    # Open output files
    fres = open(fres, 'w')
    fyml = open(fyml, 'w')
    
    # Write header
    print('--------------------------------------------------------------------------------', file=fres)
    print('Residuals from IGS combination of daily AC repro3 solutions for week {0}, day {1}'.format(w, d), file=fres)
    print('--------------------------------------------------------------------------------', file=fres)
    print('', file=fres)
    print(' Daily AC solutions:', file=fres)
    for ac in acs:
        print('  - {0} = {1}'.format(ac.name, os.path.basename(ac.file)), file=fres)
    print('', file=fres)
    print(' Daily combined solution :', file=fres)
    print('  - igs = {0}'.format(snx.file), file=fres)
    print('', file=fres)
    print(' residual = AC estimate - IGS combined estimate - 7-parameter transfo', file=fres)
    print(' sigma    = formal error of AC estimate', file=fres)
    print('', file=fres)
    
    # Write YML header
    print('#---------------------------------------------------------------------------------', file=fyml)
    print('# Residuals from IGS combination of daily AC repro3 solutions for week {0}, day {1}'.format(w, d), file=fyml)
    print('#---------------------------------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('# Daily AC solutions:', file=fyml)
    print('ac:', file=fyml)
    for ac in acs:
        print('  - {{name: {0}, file: {1}}}'.format(ac.name, os.path.basename(ac.file)), file=fyml)
    print('', file=fyml)
    print('# IGS combined solution:', file=fyml)
    print('igs: {{file: {0}}}'.format(snx.file), file=fyml)
    print('', file=fyml)
    print('# res = AC estimate - IGS combined estimate - 7-parameter transfo', file=fyml)
    print('# sig = formal error of AC estimate', file=fyml)
    print('', file=fyml)
    
    # Station position residuals header
    print('', file=fres)
    print('', file=fres)
    print(' 1) Station position residuals:', file=fres)
    print(' ------------------------------', file=fres)
    print('', file=fres)
    print('            ___________residual__________ _____________sigma___________', file=fres)
    print(' sta    AC     E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm] ', file=fres)
    print(' ----------------------------------------------------------------------', file=fres)

    # Station position residuals YML header
    print('', file=fyml)
    print('', file=fyml)
    print('# 1) Station position residuals (unit: mm, frame: ENH):', file=fyml)
    print('#------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('stares:', file=fyml)
    
    # Station position residuals
    for i in snx.ix:
        code = snx.param[i].code
        for ac in acs:
            if (code in [p.code for p in [ac.snx.param[k] for k in ac.snx.ix]]):
                ix = ac.snx.ix[[p.code for p in [ac.snx.param[k] for k in ac.snx.ix]].index(code)]
                print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3]), file=fres)
                print('  - {{sta: {0}, ac: {1}, res:[{2[0]:9.3f},{2[1]:9.3f},{2[2]:9.3f}], sig:[{3[0]:9.3f},{3[1]:9.3f},{3[2]:9.3f}]}}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3]), file=fyml)
        print(' ----------------------------------------------------------------------', file=fres)
        print('', file=fyml)
    
    # Other parameter residuals header
    print('', file=fres)
    print('', file=fres)
    print('', file=fres)
    print(' 2) ERP, geocenter and scale residuals:', file=fres)
    print(' --------------------------------------', file=fres)
    print('', file=fres)
    print(' param unit  AC   residual    sigma ', file=fres)
    print(' -----------------------------------', file=fres)

    # Other parameter residuals YML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 2) ERP, geocenter and scale residuals:', file=fyml)
    print('#---------------------------------------', file=fyml)
    print('', file=fyml)
    print('globres:', file=fyml)
    
    # XPO residuals
    for ac in acs:
        if (ac.xpo):
            ix = ac.snx.ixpo[0]
            ac.vxpo = 1000*ac.v[ix]
            ac.svxpo = 1000*ac.sv[ix]
            print(' XPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpo, ac.svxpo), file=fres)
            print('  - {{param: XPO , unit: uas  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxpo, ac.svxpo), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YPO residuals
    for ac in acs:
        if (ac.ypo):
            ix = ac.snx.iypo[0]
            ac.vypo = 1000*ac.v[ix]
            ac.svypo = 1000*ac.sv[ix]
            print(' YPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypo, ac.svypo), file=fres)
            print('  - {{param: YPO , unit: uas  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vypo, ac.svypo), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # XPOR residuals
    for ac in acs:
        if (ac.xpor):
            ix = ac.snx.ixpor[0]
            ac.vxpor = 1000*ac.v[ix]
            ac.svxpor = 1000*ac.sv[ix]
            print(' XPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpor, ac.svxpor), file=fres)
            print('  - {{param: XPOR, unit: uas/d, ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxpor, ac.svxpor), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YPOR residuals
    for ac in acs:
        if (ac.ypor):
            ix = ac.snx.iypor[0]
            ac.vypor = 1000*ac.v[ix]
            ac.svypor = 1000*ac.sv[ix]
            print(' YPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypor, ac.svypor), file=fres)
            print('  - {{param: YPOR, unit: uas/d, ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vypor, ac.svypor), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # LOD residuals
    for ac in acs:
        if (ac.lod):
            ix = ac.snx.ilod[0]
            ac.vlod = 1000*ac.v[ix]
            ac.svlod = 1000*ac.sv[ix]
            print(' LOD   us    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vlod, ac.svlod), file=fres)
            print('  - {{param: LOD , unit: us   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vlod, ac.svlod), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # XGC residuals
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]
            ac.vxgc = ac.v[ix]
            ac.svxgc = ac.sv[ix]
            print(' XGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxgc, ac.svxgc), file=fres)
            print('  - {{param: XGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxgc, ac.svxgc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YGC residuals
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]+1
            ac.vygc = ac.v[ix]
            ac.svygc = ac.sv[ix]
            print(' YGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vygc, ac.svygc), file=fres)
            print('  - {{param: YGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vygc, ac.svygc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # ZGC residuals
    for ac in acs:
        if (ac.gc):
            ix = ac.snx.igc[0]+2
            ac.vzgc = ac.v[ix]
            ac.svzgc = ac.sv[ix]
            print(' ZGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vzgc, ac.svzgc), file=fres)
            print('  - {{param: ZGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vzgc, ac.svzgc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # SC residuals
    for ac in acs:
        ix = ac.snx.isc[0]
        ac.vsc = ac.v[ix]
        ac.svsc = ac.sv[ix]
        print(' SC    ppb   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vsc, ac.svsc), file=fres)
        print('  - {{param: SC  , unit: ppb  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vsc, ac.svsc), file=fyml)
    print(' -----------------------------------', file=fres)

    # Close output files
    fres.close()
    fyml.close()
    
# Write summary of weekly IGS combination
#----------------------------------------
def write_sum(dacs, w, opt, nsta, ndat, Tdat, wdat, ncore, Tcore, wcore, dxpo, dypo, dxpor, dypor, dlod, nolog, fsum, fyml):
    
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
    ncore : list
        Numbers of core stations in daily combined solutions
    Tcore : list
        Core -> daily combined solutions transformation parameters
    wcore : list
        RMS of core -> daily combined solutions transformations
    dxpo : list
        "Bulletin A - IGS" XPO differences
    dypo : list
        "Bulletin A - IGS" YPO differences
    dxpor : list
        "Bulletin A - IGS" XPOR differences
    dypor : list
        "Bulletin A - IGS" YPOR differences
    dlod : list
        "Bulletin A - IGS" LOD differences
    nolog : list
        List of stations without sitelogs
    fsum : str
        Output summary file
    fyml : str
        Output YAML file
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
    
    # Open output files
    fsum = open(fsum, 'w')
    fyml = open(fyml, 'w')
    
    # Write header
    print('----------------------------------------------------------', file=fsum)
    print('IGS combination of daily AC repro3 solutions for week {0}'.format(w), file=fsum)
    print('----------------------------------------------------------', file=fsum)
    print('', file=fsum)
    print(' Author:  Paul Rebischung', file=fsum)
    print(' Contact: igs-rf@ign.fr', file=fsum)
    print('', file=fsum)
    print(' Daily AC solutions:', file=fsum)
    for i in range(len(acs)):
        print('  - {0} = {1}'.format(acs[i], files[i]), file=fsum)
    print('', file=fsum)
    print(' Daily combined solutions:', file=fsum)
    file = opt.dailysnx[:11] + '${yyyy}${doy}' + opt.dailysnx[18:]
    print('  - igs = {0}'.format(file), file=fsum)
    print('', file=fsum)
    print(' IERS Bulletin A:', file=fsum)
    print('  - BuA = finals2000A.data', file=fsum)

    # Write YML header
    print('#-----------------------------------------------------------', file=fyml)
    print('# IGS combination of daily AC repro3 solutions for week {0}'.format(w), file=fyml)
    print('#-----------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('author: Paul Rebischung', file=fyml)
    print('contact: igs-rf@ign.fr', file=fyml)
    print('', file=fyml)
    print('# Daily AC solutions:', file=fyml)
    print('ac:', file=fyml)
    for i in range(len(acs)):
        print('  - {{name: {0}, file: \'{1}\'}}'.format(acs[i], files[i]), file=fyml)
    print('', file=fyml)
    print('# Daily combined solutions:', file=fyml)
    print('igs: {{file: \'{0}\'}}'.format(file), file=fyml)
    print('', file=fyml)
    print('# IERS Bulletin A:', file=fyml)
    print('bua: {file: finals2000A.data}', file=fyml)
    
    # Main statistics header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 1) Main combination statistics:', file=fsum)
    print(' -------------------------------', file=fsum)
    print('  - #sta   = number of stations (after rejection of outliers)', file=fsum)
    print('  - #RF    = number of usable {0} stations'.format(opt.datumname), file=fsum)
    print('  - #core  = number of core stations used for alignament to {0}'.format(opt.datumname), file=fsum)
    print('  - VF^0.5 = square root of estimated variance factor', file=fsum)
    print('  - WRMS   = WRMS of "AC - igs" station position residuals', file=fsum)
    print('', file=fsum)
    print('                                      _____________WRMS____________', file=fsum)
    print(' AC  day  #sta   #RF #core    VF^0.5     E[mm]     N[mm]     H[mm] ', file=fsum)
    print(' ------------------------------------------------------------------', file=fsum)

    # Main statistics YAML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 1) Main combination statistics', file=fyml)
    print('#-------------------------------', file=fyml)
    print('# - nsta   = number of stations (after rejection of outliers)', file=fyml)
    print('# - nrf    = number of usable IGSR3 stations', file=fyml)
    print('# - ncore  = number of core stations used for alignament to IGSR3', file=fyml)
    print('# - sqrtvf = square root of estimated variance factor', file=fyml)
    print('# - wrms   = WRMS of "AC - igs" station position residuals (unit: mm, frame: ENH)', file=fyml)
    print('', file=fyml)
    print('stats:', file=fyml)

    # Main statistics
    for i in range(len(acs)):
        print('  -', file=fyml)
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                print(' {0.name}   {1} {0.nsta:5d} {0.ndat:5d} {0.ncore:5d} {0.sf:10.6f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(ac, d, ac.wrms), file=fsum)
                print('    - {{ac: {0.name}, day: {1}, nsta: {0.nsta:5d}, nrf: {0.ndat:5d}, ncore: {0.ncore:5d}, sqrtvf: {0.sf:10.6f}, wrms: [{2[0]:9.3f},{2[1]:9.3f},{2[2]:9.3f}]}}'.format(ac, d, ac.wrms), file=fyml)
        print(' ------------------------------------------------------------------', file=fsum)
    
    print('  -', file=fyml)
    for d in range(7):
        print(' igs   {0} {1:5d} {2:5d} {3:5d}'.format(d, nsta[d], ndat[d], ncore[d]), file=fsum)
        print('    - {{ac: igs, day: {0}, nsta: {1:5d}, nrf: {2:5d}, ncore: {3:5d}}}'.format(d, nsta[d], ndat[d], ncore[d]), file=fyml)
    print(' ------------------------------------------------------------------', file=fsum)
    
    # ERP, geocenter and scale residuals header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 2) "AC - igs" ERP, geocenter and scale residuals:', file=fsum)
    print(' -------------------------------------------------', file=fsum)
    print('', file=fsum)
    print('             XPO       YPO       XPOR      YPOR      LOD       XGC       YGC       ZGC       SC', file=fsum)
    print(' AC  day    [uas]     [uas]    [uas/d]   [uas/d]     [us]      [mm]      [mm]      [mm]     [ppb]', file=fsum)
    print(' -------------------------------------------------------------------------------------------------', file=fsum)

    # ERP, geocenter and scale residuals YAML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 2) "AC - igs" ERP, geocenter and scale residuals:', file=fyml)
    print('#--------------------------------------------------', file=fyml)
    print('# - dxpo  = X-pole residuals (unit: uas)', file=fyml)
    print('# - dypo  = Y-pole residuals (unit: uas)', file=fyml)
    print('# - dxpor = X-pole rate residuals (unit: uas/d)', file=fyml)
    print('# - dypor = Y-pole rate residuals (unit: uas/d)', file=fyml)
    print('# - dlod  = LOD residuals (unit: us)', file=fyml)
    print('# - dgc   = geocenter rediduals (unit: mm, frame: XYZ)', file=fyml)
    print('# - dsc   = terrestrial scale residuals (unit: ppb)', file=fyml)
    print('', file=fyml)
    print('globres:', file=fyml)

    # ERP, geocenter and scale residuals
    for i in range(len(acs)):
        print('  -', file=fyml)
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                s = ' '+ac.name+'   '+str(d)
                y = '    - {ac: '+ac.name+', day: '+str(d)

                if hasattr(ac, 'vxpo'):
                    s = s + ' {0:9.3f}'.format(ac.vxpo)
                    y = y + ', dxpo:{0:9.3f}'.format(ac.vxpo)
                else:
                    s = s + '          '
                    y = y + ', dxpo:         '

                if hasattr(ac, 'vypo'):
                    s = s + ' {0:9.3f}'.format(ac.vypo)
                    y = y + ', dypo:{0:9.3f}'.format(ac.vypo)
                else:
                    s = s + '          '
                    y = y + ', dypo:         '

                if hasattr(ac, 'vxpor'):
                    s = s + ' {0:9.3f}'.format(ac.vxpor)
                    y = y + ', dxpor:{0:9.3f}'.format(ac.vxpor)
                else:
                    s = s + '          '
                    y = y + ', dxpor:         '

                if hasattr(ac, 'vypor'):
                    s = s + ' {0:9.3f}'.format(ac.vypor)
                    y = y + ', dypor:{0:9.3f}'.format(ac.vypor)
                else:
                    s = s + '          '
                    y = y + ', dypor:         '

                if hasattr(ac, 'vlod'):
                    s = s + ' {0:9.3f}'.format(ac.vlod)
                    y = y + ', dlod:{0:9.3f}'.format(ac.vlod)
                else:
                    s = s + '          '
                    y = y + ', dlod:         '

                if hasattr(ac, 'vxgc'):
                    s = s + ' {0:9.3f} {1:9.3f} {2:9.3f}'.format(ac.vxgc, ac.vygc, ac.vzgc)
                    y = y + ', dgc:[{0:9.3f},{1:9.3f},{2:9.3f}]'.format(ac.vxgc, ac.vygc, ac.vzgc)
                else:
                    s = s + 3*'          '
                    y = y + ', dgc:                               '

                if hasattr(ac, 'vsc'):
                    s = s + ' {0:9.3f}'.format(ac.vsc)
                    y = y + ', dsc:{0:9.3f}}}'.format(ac.vsc)
                else:
                    s = s + '          '
                    y = y + ', dsc:         }'

                print(s, file=fsum)
                print(y, file=fyml)

        print(' -------------------------------------------------------------------------------------------------', file=fsum)
    
    print('  -', file=fyml)
    for d in range(7):
        print(' BuA   {0} {1:9.3f} {2:9.3f} {3:9.3f} {4:9.3f} {5:9.3f}'.format(d, 1000*dxpo[d], 1000*dypo[d], 1000*dxpor[d], 1000*dypor[d], 1000*dlod[d]), file=fsum)
        print('    - {{ac: bua, day: {0}, dxpo:{1:9.3f}, dypo:{2:9.3f}, dxpor:{3:9.3f}, dypor:{4:9.3f}, dlod:{5:9.3f}}}'.format(d, 1000*dxpo[d], 1000*dypo[d], 1000*dxpor[d], 1000*dypor[d], 1000*dlod[d]), file=fyml)
    print(' -------------------------------------------------------------------------------------------------', file=fsum)
    
    # Transformation parameters header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 3) "AC -> {0}" 7-parameter transformations:'.format(opt.datumname), file=fsum)
    print(' ---------------------------------------------', file=fsum)
    print('  - RMS = RMS of residuals from "AC -> {0}" 7-parameter transformations'.format(opt.datumname), file=fsum)
    print('', file=fsum)
    print('             TX        TY        TZ        SC        RX        RY        RZ    _____________RMS_____________', file=fsum)
    print(' AC  day    [mm]      [mm]      [mm]      [ppb]     [mas]     [mas]     [mas]     E[mm]     N[mm]     H[mm] ', file=fsum)
    print(' -----------------------------------------------------------------------------------------------------------', file=fsum)

    # Transformation parameters YAML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 3) "AC -> IGSR3" 7-parameter transformations:', file=fyml)
    print('#----------------------------------------------', file=fyml)
    print('# - T   = translations (unit: mm, frame: XYZ)', file=fyml)
    print('# - S   = scale factors (unit: ppb)', file=fyml)
    print('# - R   = rotations (unit: mas, frame: XYZ)', file=fyml)
    print('# - rms = RMS of "AC -> IGSR3" transformation residuals (unit: mm, frame: ENH)', file=fyml)
    print('', file=fyml)
    print('transfo:', file=fyml)

    # Transformation parameters
    for i in range(len(acs)):
        print('  -', file=fyml)
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                print(' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {2[3]:9.3f} {2[4]:9.3f} {2[5]:9.3f} {2[6]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac.name, d, -ac.Tcore, ac.wcore), file=fsum)
                print('    - {{ac: {0}, day: {1}, T:[{2[0]:9.3f},{2[1]:9.3f},{2[2]:9.3f}], S:{2[3]:9.3f}, R:[{2[4]:9.3f},{2[5]:9.3f},{2[6]:9.3f}], rms:[{3[0]:9.3f},{3[1]:9.3f},{3[2]:9.3f}]}}'.format(ac.name, d, -ac.Tcore, ac.wcore), file=fyml)
        print(' -----------------------------------------------------------------------------------------------------------', file=fsum)
    
    print('  -', file=fyml)
    for d in range(7):
        print(' igs   {0} {1[0]:9.3f} {1[1]:9.3f} {1[2]:9.3f} {1[3]:9.3f} {1[4]:9.3f} {1[5]:9.3f} {1[6]:9.3f} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f}'.format(d, -Tcore[d], wcore[d]), file=fsum)
        print('    - {{ac: igs, day: {0}, T:[{1[0]:9.3f},{1[1]:9.3f},{1[2]:9.3f}], S:{1[3]:9.3f}, R:[{1[4]:9.3f},{1[5]:9.3f},{1[6]:9.3f}], rms:[{2[0]:9.3f},{2[1]:9.3f},{2[2]:9.3f}]}}'.format(d, -Tcore[d], wcore[d]), file=fyml)
    print(' -----------------------------------------------------------------------------------------------------------', file=fsum)

    # Manually rejected stations header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 4) Manually rejected stations:', file=fsum)
    print(' ------------------------------', file=fsum)
    print('', file=fsum)
    print(' AC  day   stations', file=fsum)
    print(' -------------------------------------------------------------------------------', file=fsum)

    # Manually rejected stations YAML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 4) Manually rejected stations:', file=fyml)
    print('#-------------------------------', file=fyml)
    print('', file=fyml)
    print('rejections:', file=fyml)

    # Manually rejected stations
    for i in range(len(acs)):
        b = False
        for d in range(7):
            if (acs[i] in [ac.name for ac in dacs[d]]):
                ac = dacs[d][[ac.name for ac in dacs[d]].index(acs[i])]
                if (len(ac.rej) > 0):
                    b = True
                    s = ' {0}   {1}   '.format(ac.name, d)
                    y = '  - {{ac: {0}, day: {1}, sta: ['.format(ac.name, d)
                    for sta in ac.rej:
                        s = s + sta + ' '
                        y = y + sta + ', '
                    s = s[:-1]
                    y = y[:-2] + ']}'
                    print(s, file=fsum)
                    print(y, file=fyml)
        if (b):
            print(' -------------------------------------------------------------------------------', file=fsum)
        
    # Outliers header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 5) Outliers:', file=fsum)
    print(' ------------', file=fsum)
    print('', file=fsum)
    print('                __________residuals__________', file=fsum)
    print(' AC  day   sta     E[mm]     N[mm]     H[mm] ', file=fsum)
    print(' --------------------------------------------', file=fsum)

    # Outliers YAML header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 5) Outliers:', file=fyml)
    print('#-------------', file=fyml)
    print('# - res = "AC - igs" station position residuals before rejection (unit: mm, frame: ENH)', file=fyml)
    print('', file=fyml)
    print('outliers:', file=fyml)

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
                        print(' {0}   {1}   {2} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(ac.name, d, ac.staout[j], ac.resout[j]), file=fsum)
                        print('  - {{ac: {0}, day: {1}, sta: {2}, res: [{3[0]:9.3f},{3[1]:9.3f},{3[2]:9.3f}]}}'.format(ac.name, d, ac.staout[j], ac.resout[j]), file=fyml)
        if (b):
            print(' --------------------------------------------', file=fsum)

    # Station metadata inconsistencies header
    print('', file=fsum)
    print('', file=fsum)
    print('', file=fsum)
    print(' 6) Station metadata inconsistencies:', file=fsum)
    print(' ------------------------------------', file=fsum)
    for source in logsource:
        print('  - {0.name:<6s} = {0.server}{0.remotedir}'.format(source), file=fsum)
    print('', file=fsum)
    if (len(nolog) > 0):
        s = '  - No sitelog found for station(s): '
        for sta in nolog:
            s = s + sta + ', '
        print(s[:-2], file=fsum)
        print('', file=fsum)
    print(' AC    sta    ___metadata_type___   __info_from_sinex___   _info_from_sitelog__   source', file=fsum)
    print(' ---------------------------------------------------------------------------------------', file=fsum)

    # Station metadata inconsistencies header
    print('', file=fyml)
    print('', file=fyml)
    print('', file=fyml)
    print('# 6) Station metadata inconsistencies:', file=fyml)
    print('#-------------------------------------', file=fyml)
    print('# - sources = site log repositories', file=fyml)
    print('# - nologs  = list of stations for which no site log was found', file=fyml)
    print('# - errors  = inconsistencies between AC solutions and site logs', file=fyml)
    print('', file=fyml)
    print('sources:', file=fyml)
    for source in logsource:
        print('  - {{name: {0.name:<6s}, address: \'{0.server}{0.remotedir}\'}}'.format(source), file=fyml)
    print('', file=fyml)
    s = 'nologs: ['
    for sta in nolog:
        s = s + sta + ', '
    s = s[:-2] + ']'
    print(s, file=fyml)
    print('', file=fyml)
    print('errors:', file=fyml)    

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
                print(' {0}   {1}'.format(ac.name, err), file=fsum)
                print('  - {{ac: {0}, sta: {1}, type: {2}, from_ac: {3}, from_log: {4}, source: {5}}}'.format(ac.name, err[0:4], err[7:26], err[29:49], err[52:72], err[75:81]), file=fyml)
            print(' ---------------------------------------------------------------------------------------', file=fsum)

    # Close output files
    fsum.close()
    fyml.close()
    
