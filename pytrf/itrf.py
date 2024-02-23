#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytrf ITRF utilities

This subpackage contains various useful routines for the ITRF SINEX combination.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from scipy.stats import median_abs_deviation as mad

from pytrf import date, sinex
from pytrf.ts import ts, model
from pytrf.io import read_solns
from pytrf.const import ae

# Write residuals from daily IGS combination
#-------------------------------------------
def write_res(snx, acs, fres, fyml):
    
    """
    Write residuals from combination
    1 file by station, save in "res" folder (create if doesn't exist yet)

    Parameters
    ----------
    snx : sinex instance
        Daily combined solution
    acs : list
        List of combination inputs
    fres : str
        Output residual file
    fyml : str
        Output YAML file
    """
    #create 'res' folder if doesn't exist
    if not os.path.exists("res"):
        os.makedirs("res")

    # Open output files
    fres = open(os.path.join("res",fres), 'w')
    fyml = open(os.path.join("res",fyml), 'w')
    
    # Write header
    print('-------------------------------------------------------------------------------', file=fres)
    print('Residuals from combination of AC SINEX solutions', file=fres)
    print('-------------------------------------------------------------------------------', file=fres)
    print('', file=fres)
    print(' AC solutions:', file=fres)
    for ac in acs:
        print('  - {0} = {1}'.format(ac.name, os.path.basename(ac.file)), file=fres)
    print('', file=fres)
    print(' Combined solution :', file=fres)
    print('  - igs = {0}'.format(snx.file), file=fres)
    print('', file=fres)
    print(' residual = AC estimate - combined estimate - 7-parameter transfo', file=fres)
    print(' sigma    = formal error of AC estimate', file=fres)
    print('', file=fres)
    
    # Write YML header
    print('#--------------------------------------------------------------------------------', file=fyml)
    print('# Residuals from combination of AC SINEX solutions', file=fyml)
    print('#--------------------------------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('# AC solutions:', file=fyml)
    print('ac:', file=fyml)
    for ac in acs:
        print('  - {{name: {0}, file: {1}}}'.format(ac.name, os.path.basename(ac.file)), file=fyml)
    print('', file=fyml)
    print('# Combined solution:', file=fyml)
    print('igs: {{file: {0}}}'.format(snx.file), file=fyml)
    print('', file=fyml)
    print('# res = AC estimate - combined estimate - 7-parameter transfo', file=fyml)
    print('# sig = formal error of AC estimate', file=fyml)
    print('', file=fyml)
    
    # Station position residuals header
    print('', file=fres)
    print('', file=fres)
    print(' 1) Station position residuals:', file=fres)
    print(' ------------------------------', file=fres)
    print('', file=fres)
    
    header1 = '                  ___________residual__________ _____________sigma___________'
    header2 = ' sta          AC     E[mm]     N[mm]     H[mm]     E[mm]     N[mm]     H[mm]     mjd        epoch '
    header3 = ' ------------------------------------------------------------------------------------------------------'
    print(header1, file=fres)
    print(header2, file=fres)
    print(header3, file=fres)

    # Station position residuals YML header
    print('', file=fyml)
    print('', file=fyml)
    print('# 1) Station position residuals (unit: mm, frame: ENH):', file=fyml)
    print('#------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('stares:', file=fyml)
    
    # Station position residuals
    for stasnx in snx.sta:
        code = stasnx.code
        #write 1 file by station
        with open(f'res/{code}.res', 'w') as f_stares:
            f_stares.write(header1 +'\n')
            f_stares.write(header2 +'\n')
            f_stares.write(header3 +'\n')
            
            for ac in acs:
                if (code in [p.code for p in [ac.snx.param[k] for k in ac.snx.ix]]):
                    ix = ac.snx.ix[[p.code for p in [ac.snx.param[k] for k in ac.snx.ix]].index(code)]
                    sta = [sta for sta in ac.snx.sta if sta.code==code][0]
                    if len(sta.soln)!=1:
                        print(f"WARNING {ac.file}: station {code}: len(soln) = {len(sta.soln)}")
                        datemean=0
                    else:
                        datemean = sta.soln[0].datamean
                    #mjd date conversion, usefull for time format pytrf.ts    
                    mjddatemean = date.from_tsnx(datemean).mjd
                    res_sta_line = ' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}   {4}   {5}'.format(sta.code+sta.pt+sta.soln[0].soln, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3], mjddatemean, datemean)
                    # write res_sta_line 
                    f_stares.write(res_sta_line +'\n') #in station res file
                    print(res_sta_line, file=fres)
                    print('  - {{sta: {0}, ac: {1}, res:[{2[0]:9.3f},{2[1]:9.3f},{2[2]:9.3f}], sig:[{3[0]:9.3f},{3[1]:9.3f},{3[2]:9.3f}], mjd:{4:9}, epoch: {5}}}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3], mjddatemean, datemean), file=fyml)
            print('---------------------------------------------------------------------------------------', file=fres)
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
        if hasattr(ac,'xpo'):
            ix = ac.snx.ixpo[0]
            ac.vxpo = 1000*ac.v[ix]
            ac.svxpo = 1000*ac.sv[ix]
            print(' XPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpo, ac.svxpo), file=fres)
            print('  - {{param: XPO , unit: uas  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxpo, ac.svxpo), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YPO residuals
    for ac in acs:
        if hasattr(ac, 'ypo'):
            ix = ac.snx.iypo[0]
            ac.vypo = 1000*ac.v[ix]
            ac.svypo = 1000*ac.sv[ix]
            print(' YPO   uas   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypo, ac.svypo), file=fres)
            print('  - {{param: YPO , unit: uas  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vypo, ac.svypo), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # XPOR residuals
    for ac in acs:
        if hasattr(ac, 'xpor'):
            ix = ac.snx.ixpor[0]
            ac.vxpor = 1000*ac.v[ix]
            ac.svxpor = 1000*ac.sv[ix]
            print(' XPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxpor, ac.svxpor), file=fres)
            print('  - {{param: XPOR, unit: uas/d, ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxpor, ac.svxpor), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YPOR residuals
    for ac in acs:
        if hasattr(ac, 'ypor'):
            ix = ac.snx.iypor[0]
            ac.vypor = 1000*ac.v[ix]
            ac.svypor = 1000*ac.sv[ix]
            print(' YPOR  uas/d {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vypor, ac.svypor), file=fres)
            print('  - {{param: YPOR, unit: uas/d, ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vypor, ac.svypor), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # LOD residuals
    for ac in acs:
        if hasattr(ac, 'lod'):
            ix = ac.snx.ilod[0]
            ac.vlod = 1000*ac.v[ix]
            ac.svlod = 1000*ac.sv[ix]
            print(' LOD   us    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vlod, ac.svlod), file=fres)
            print('  - {{param: LOD , unit: us   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vlod, ac.svlod), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # XGC residuals
    for ac in acs:
        if hasattr(ac, 'gc'):
            ix = ac.snx.igc[0]
            ac.vxgc = ac.v[ix]
            ac.svxgc = ac.sv[ix]
            print(' XGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vxgc, ac.svxgc), file=fres)
            print('  - {{param: XGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vxgc, ac.svxgc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # YGC residuals
    for ac in acs:
        if hasattr(ac, 'gc'):
            ix = ac.snx.igc[0]+1
            ac.vygc = ac.v[ix]
            ac.svygc = ac.sv[ix]
            print(' YGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vygc, ac.svygc), file=fres)
            print('  - {{param: YGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vygc, ac.svygc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # ZGC residuals
    for ac in acs:
        if hasattr(ac, 'gc'):
            ix = ac.snx.igc[0]+2
            ac.vzgc = ac.v[ix]
            ac.svzgc = ac.sv[ix]
            print(' ZGC   mm    {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vzgc, ac.svzgc), file=fres)
            print('  - {{param: ZGC , unit: mm   , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vzgc, ac.svzgc), file=fyml)
    print(' -----------------------------------', file=fres)
    print('', file=fyml)

    # SC residuals
    for ac in acs:
        if hasattr(ac, 'sc'):
            ix = ac.snx.isc[0]
            ac.vsc = ac.v[ix]
            ac.svsc = ac.sv[ix]
            print(' SC    ppb   {0} {1:9.3f} {2:9.3f}'.format(ac.name, ac.vsc, ac.svsc), file=fres)
            print('  - {{param: SC  , unit: ppb  , ac: {0}, res: {1:9.3f}, sig: {2:9.3f}}}'.format(ac.name, ac.vsc, ac.svsc), file=fyml)
    print(' -----------------------------------', file=fres)
    
    
    # Close output files
    fres.close()
    fyml.close()
    
    
def plot_coord_res_1_sta(snx, sta, solns, psd, detrend=True, res_folder="res"):
    
    #open sta residues file
    df_sta = pd.read_csv('{}/{}.res'.format(res_folder, sta), sep="\s+", header=None, skiprows=3, names=["code","pt","soln","E","N","U","sE","sN","sU","epoch"])
    
    # Read station position time series
    r = ts.read('../data/coord/'+sta+'_igs.plh', usecols=(3, 6, 7, 8, 9, 10, 11), format=('t', 'y', 'x', 'z', 'sy', 'sx', 'sz'), dims=['East', 'North', 'Up'])
    
    # Degree -> m conversion
    r.y[:,0:2] = np.pi/180*ae * r.y[:,0:2]
    r.Q[:,0:2,0:2] = (np.pi/180*ae)**2 * r.Q[:,0:2,0:2]
    
    # Detrend time series
    if detrend:
        r.detrend()
    
    # Read discontinuity list and PSD models
    # solns = read_solns('../data/discontinuities/soln.snx')
    # psd = sinex.read('../data/psd/psd_IGS.snx')
    
    # Get dates and causes of discontinuities
    i = [s.code for s in solns].index(sta)
    yd = [date.from_tsnx(p.end).ydec() for p in solns[i].P[:-1]]
    cd = [p.cause for p in solns[i].P[:-1]]
    
    # Format causes of discontinuities
    for i in range(len(cd)):
        if (cd[i][:2] == 'EQ'):
            cd[i] = cd[i][:7]
        elif (len(cd[i]) > 20):
            cd[i] = cd[i][:20]
    
    # Initialize model
    m = model.from_solns(r, solns, sta, per=[365.25, 182.625], noise=['vw'], psd=psd, fix_tau=True, fix_amp=True)
    
    # Iteratively fit model and remove outliers
    m.fit_iter(thr_norm=5, finalize=False, quiet=True)
    
    # Plot station position time series + residuals
    pp.figure(figsize=(20, 10), tight_layout=True)
    y = [date.from_mjd(d).ydec() for d in m.r.t]
    mv = [mad(m[d].v) for d in range(3)]
    
    for d in range(3): #ENU
        pp.subplot(3, 2, 2*d+1)
        pp.grid()
        pp.errorbar(y, 1000*m.r[d].y, yerr=1000*m[d].sv, fmt='.k', ecolor='grey', zorder=10)
        pp.plot(y, 1000*m[d].yc, 'r', linewidth=2, zorder=20)
        ymin = 1000*np.min(m[d].yc)-5000*mv[d]
        ymax = 1000*np.max(m[d].yc)+5000*mv[d]
        pp.axis([y[0], y[-1], ymin, ymax])
        pp.ylabel(m[d].r.dims+' [mm]')

        for i in range(len(yd)):
            pp.plot([yd[i], yd[i]], [ymin, ymax], '--r', linewidth=2, zorder=5)
            pp.text(yd[i], (ymin+ymax)/2, cd[i], ha='right', va='center', color='r', rotation=90, fontsize=10, zorder=15, fontweight='bold')
        if (d == 0):
            pp.title('Detrended station position time series + model')
           
        if (d == 2):
            pp.xlabel('time [yr]')
        
        #residues     
        pp.subplot(3, 2, 2*d+2)
        pp.grid()
        pp.errorbar(y, 1000*m[d].v, yerr=1000*m[d].sv, fmt='.k', ecolor='gray', zorder=10)
        ymin = -5000*mv[d]
        ymax = +5000*mv[d]
        pp.axis([y[0], y[-1], ymin, ymax])
        pp.ylabel(m[d].r.dims+' residuals [mm]')
    
        for i in range(len(yd)):
            pp.plot([yd[i], yd[i]], [ymin, ymax], '--r', linewidth=2, zorder=5)
            pp.text(yd[i], (ymin+ymax)/2, cd[i], ha='right', va='center', color='r', rotation=90, fontsize=10, zorder=15, fontweight='bold')
        if (d == 0):
            pp.title('Residual time series')
            
        if (d == 2):
            pp.xlabel('time [yr]')
            
    pp.show()
    