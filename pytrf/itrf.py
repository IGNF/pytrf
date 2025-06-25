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

# Write residuals from ITRF combination
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
    
    header1 = '                    __________________________________model_______________________   ___________________residues______________________   __________obs_________'
    header2 = ' sta         AC     Em[m]     Nm[m]          Um[m]         sEm[mm] sNm[mm] sUm[mm]   Er[mm]  Nr[mm]  Ur[mm]   sEr[mm] sNr[mm] sUr[mm]    sE[mm]  sN[mm]  sU[mm]      mjd      epoch '
    header3 = ' ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
    print(header1, file=fres)
    print(header2, file=fres)
    print(header3, file=fres)

    # Station position residuals YML header
    print('', file=fyml)
    print('', file=fyml)
    print('# 1) Station position residuals (unit: mm, frame: ENU):', file=fyml)
    print('#------------------------------------------------------', file=fyml)
    print('', file=fyml)
    print('stares:', file=fyml)
    
    # Station position residuals
    for stasnx in snx.sta:
        code = stasnx.code
        pt  = stasnx.pt.replace(" ","")
        #write 1 file by station
        with open(f'res/{code}_{pt}.res', 'w') as f_stares:
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
                        mjddatemean = 0
                    else:
                        datemean = sta.soln[0].datamean
                        #mjd date conversion, usefull for time format pytrf.ts 
                        try:
                            mjddatemean = date.from_tsnx(datemean).mjd
                        except:
                            mjddatemean = 0
                            
                    #res_sta_line = ' {0}   {1} {2[0]:9.3f} {2[1]:9.3f} {2[2]:9.3f} {3[0]:9.3f} {3[1]:9.3f} {3[2]:9.3f}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3])
                    res_sta_line = ' {0}  {1}  {2[0]:8.5f} {2[1]:14.5f} {2[2]:14.5f} {3[0]:7.3f} {3[1]:7.3f} {3[2]:7.3f}   {4[0]:7.3f} {4[1]:7.3f} {4[2]:7.3f}  {5[0]:7.3f} {5[1]:7.3f} {5[2]:7.3f}   {6[0]:7.3f} {6[1]:7.3f} {6[2]:7.3f}    {7}   {8}'.format(sta.code+sta.pt+sta.soln[0].soln, ac.name, 
                                                                                                                                                                                                                                                      ac.ym[ix:ix+3], ac.sm[ix:ix+3],
                                                                                                                                                                                                                                                      ac.v[ix:ix+3],  ac.sv[ix:ix+3],
                                                                                                                                                                                                                                                      ac.sobs[ix:ix+3], #mm
                                                                                                                                                                                                                                                      mjddatemean, datemean)
                    # write res_sta_line 
                    f_stares.write(res_sta_line +'\n') #in station res file
                    print(res_sta_line, file=fres)
                    print('  - {{sta: {0}, ac: {1}, res:[{2[0]:9.4f},{2[1]:9.4f},{2[2]:9.4f}], sig:[{3[0]:9.3f},{3[1]:9.3f},{3[2]:9.3f}], mjd:{4:9}, epoch: {5}}}'.format(code, ac.name, ac.v[ix:ix+3], ac.sv[ix:ix+3], mjddatemean, datemean), file=fyml)
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
    
    
    
def plot_coord_res_1_sta(code, pt, solns, psd, detrend=True, res_folder="res"):
    """
    Plot 1 station time serie residues + model

    Parameters
    ----------
    code : str
        station code 4 chr
    pt : str
        pt code
    solns : pytrf.record()
        solns file, from read_solns
    psd : pytrf.sinex
        psd sinex open file
    detrend : Tbool, optional
        If plot must be detrended. The default is True. If False: only mean serie is remove.
    res_folder : str, optional
        residues folder. The default is "res".

    Returns
    -------
    None.

    """
    #open sta residues file
    pt = pt.replace(" ","")
    names = ["code","pt","soln", "sol", "Em","Nm","Um","sEm","sNm","sUm","Er","Nr","Ur","sEr","sNr","sUr", "sE","sN","sU", "mjd", "epoch"]
    df_sta = pd.read_csv('res/{}_{}.res'.format(code, pt), sep="\s+", header=None, index_col=False, skiprows=3, names=names)
    
    r = ts(t=df_sta["mjd"].to_numpy(), y =df_sta.loc[:,["Em","Nm","Um"]].to_numpy(), dims=['East', 'North', 'Up'])
    
    if detrend:
        r.detrend(dtrd=1)
        
    else: #del only mean
        r.detrend(dtrd=0)
    
    #detrended model [mm]
    df_sta["Emm"] = r.y[:,0] * 1e3 
    df_sta["Nmm"] = r.y[:,1] * 1e3
    df_sta["Umm"] = r.y[:,2] * 1e3
    

    #observations: model + residues [mm]
    df_sta["E"] = df_sta["Emm"] + df_sta["Er"]
    df_sta["N"] = df_sta["Nmm"] + df_sta["Nr"]
    df_sta["U"] = df_sta["Umm"] + df_sta["Ur"]
    
    
    # Add PSD
    e_psd, n_psd, u_psd = [], [], []
    for d in df_sta["epoch"]:
        e, n, u = psd.get_psd(code, d)[0] #unit: m
        e_psd.append(e*1e3) #mm conversion
        n_psd.append(n*1e3)
        u_psd.append(u*1e3)
        
    df_sta["Emmpsd"] = df_sta["Emm"] + np.array(e_psd)
    df_sta["Nmmpsd"] = df_sta["Nmm"] + np.array(n_psd)
    df_sta["Ummpsd"] = df_sta["Umm"] + np.array(u_psd)
    
    
    # Get dates and causes of discontinuities
    try: 
        i = [s.code for s in solns].index(code)
        yd = [date.from_tsnx(p.end).ydec() for p in solns[i].P[:-1]]
        cd = [p.cause for p in solns[i].P[:-1]]
        
        # Format causes of discontinuities
        for i in range(len(cd)):
            if (cd[i][:2] == 'EQ'):
                cd[i] = cd[i][:7]
            elif (len(cd[i]) > 20):
                cd[i] = cd[i][:20]
                
    except: #not in soln
        yd = []
        pass
        
    #### Plot station position time series + residuals
    pp.figure(figsize=(20, 10), tight_layout=True)
    y = [date.from_mjd(d).ydec() for d in df_sta["mjd"]]
    mv = [mad(df_sta[dim]) for dim in ['Er', 'Nr', 'Ur']] #mm
    
    print("mv:",mv)
    
    dims = ['East', 'North', 'Up']
    for d, dim in enumerate(dims): #ENU
        pp.subplot(3, 2, 2*d+1)
        pp.grid()
        pp.errorbar(y, df_sta['{}'.format(dim[0])], yerr=df_sta['s{}'.format(dim[0])], fmt='.k', ecolor='grey', zorder=10)
        #pp.plot(y, df_sta['{}mm'.format(dim[0])], 'g', linewidth=2, zorder=20) # only model
        pp.plot(y, df_sta['{}mmpsd'.format(dim[0])], 'r', linewidth=2, zorder=20) #model + psd
        ymin = np.min(df_sta['{}mm'.format(dim[0])])-5*mv[d]
        ymax = np.max(df_sta['{}mm'.format(dim[0])])+5*mv[d]
        pp.axis([y[0], y[-1], ymin, ymax])
        pp.ylabel(dim+' [mm]')

        for i in range(len(yd)):
            pp.plot([yd[i], yd[i]], [ymin, ymax], '--r', linewidth=2, zorder=5)
            pp.text(yd[i], (ymin+ymax)/2, cd[i], ha='right', va='center', color='r', rotation=90, fontsize=10, zorder=15, fontweight='bold')
        if (d == 0):
            if detrend:
                pp.title('Detrended station position time series + model')
            else:
                pp.title('Station position time series + model')
           
        if (d == 2):
            pp.xlabel('time [yr]')
        
        #residues     
        pp.subplot(3, 2, 2*d+2)
        pp.grid()
        pp.errorbar(y, df_sta['{}r'.format(dim[0])], yerr=df_sta['s{}r'.format(dim[0])], fmt='.k', ecolor='gray', zorder=10)
        ymin = -5*mv[d]
        ymax = +5*mv[d]
        pp.axis([y[0], y[-1], ymin, ymax])
        pp.ylabel(dim+' residuals [mm]')
    
        for i in range(len(yd)):
            pp.plot([yd[i], yd[i]], [ymin, ymax], '--r', linewidth=2, zorder=5)
            pp.text(yd[i], (ymin+ymax)/2, cd[i], ha='right', va='center', color='r', rotation=90, fontsize=10, zorder=15, fontweight='bold')
        if (d == 0):
            pp.title('Residual time series')
            
        if (d == 2):
            pp.xlabel('time [yr]')
            
    pp.show()
    
    
def write_tie_baseline(list_snxtie):
    """
    Write table with tie baselines, extracted from snx tie (= CATREF contie.dat)
    Star radiation strategy

    Parameters
    ----------
    list_snxtie : list of pytrf.snx
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    df = pd.DataFrame(columns=["station1","station2", "dX[m]", "dY[m]","dZ[m]","t_snx","t_mjd","site"])
    
    idx = 0
    for tiesnx in list_snxtie:
        #1st sta as reference (star configuration (radiation) -> baseline )
        sta_ref=tiesnx.sta[0]
        # get tie time
        t_snx = tiesnx.param[tiesnx.get_sta_ind(code=[sta_ref.code], pt=[sta_ref.pt], soln=[sta_ref.soln[0].soln])[0,0]].tref
        t_mjd = date.from_tsnx(t_snx).mjd
        
        
        ref_coords = tiesnx.get_xyz(code=[sta_ref.code], pt=[sta_ref.pt], soln=[sta_ref.soln[0].soln])
       
        for sta in tiesnx.sta[1:]:
            sta_coords = tiesnx.get_xyz(code=[sta.code], pt=[sta.pt], soln=[sta.soln[0].soln])
            baseline = ref_coords - sta_coords
            
            df.loc[idx,:] = [f"{sta_ref.code}_{sta_ref.pt.strip()}_{sta_ref.soln[0].soln.strip()}", f"{sta.code}_{sta.pt.strip()}_{sta.soln[0].soln.strip()}", baseline[0,0], baseline[0,1], baseline[0,2], t_snx,t_mjd, sta.domes[:5]]
            
            idx+=1
           
    df.sort_values(by=["t_mjd","station1"]).reset_index(drop=True)
            
    
    with open("contie.txt","w") as fi:
        fi.write("{:<7} {:15} {:15} {:20} {:20} {:20} {:15} {:15} {:10}\n".format("id_tie", "station1","station2", "dX[m]", "dY[m]","dZ[m]","t_snx","t_mjd", "site"))
    
        for idx in df.index:
            fi.write("{:<7} {:15} {:15} {:<20.9f} {:<20.9f} {:<20.9f} {:15} {:15} {:10}\n".format(f"t{idx}", df.loc[idx,"station1"],df.loc[idx,"station2"], df.loc[idx,"dX[m]"], df.loc[idx,"dY[m]"],df.loc[idx,"dZ[m]"],df.loc[idx,"t_snx"], df.loc[idx,"t_mjd"], df.loc[idx,"site"]))
       
            
       
def write_tie_baseline_btw2tech(list_snxtie, list_sta_tech1, list_sta_tech2, title_baseline):
    """
    Write baseline btw 2 specific technics: GNSS - VLBI, GNSS - SLR, etc.,  extracted from snx tie (= CATREF contie.dat)
    List of stations of interest must be set as a parameter.

    Parameters
    ----------
    list_snxtie : list of pytrf.snx
        DESCRIPTION.
    list_sta_tech1: list of station belonging to technic 
        List of 2D list: [[sta1.code, sta1.pt], [sta2.code, sta2.pt]] WARNING 'ALBH' + '1' #pt on 4chr
        

    Returns
    -------
    None.

    """
    
    df = pd.DataFrame(columns=["station1","station2", "dX[m]", "dY[m]","dZ[m]","t_snx","t_mjd", "site"])
    
    idx = 0
    for tiesnx in list_snxtie:
        #1st sta as reference (star configuration (radiation) -> baseline )
        
        for num1, sta in enumerate(tiesnx.sta): #stations of list tech1
            
            if [sta.code.strip(), sta.pt.strip()] in list_sta_tech1: #this sta belongs to "tech1" list of station of interest
                sta_ref=tiesnx.sta[num1]
                # get tie time
                t_snx = tiesnx.param[tiesnx.get_sta_ind(code=[sta_ref.code], pt=[sta_ref.pt], soln=[sta_ref.soln[0].soln])[0,0]].tref
                t_mjd = date.from_tsnx(t_snx).mjd
                
                sta1_coords = tiesnx.get_xyz(code=[sta_ref.code], pt=[sta_ref.pt], soln=[sta_ref.soln[0].soln])
                
                for num2, sta2 in enumerate(tiesnx.sta): #stations of list tech2                   
                    if ([sta2.code.strip(), sta2.pt.strip()] in list_sta_tech2): #num2==0: same station1 & station2
                        sta2_coords = tiesnx.get_xyz(code=[sta2.code], pt=[sta2.pt], soln=[sta2.soln[0].soln])
                        baseline = sta1_coords - sta2_coords
                        
                        df.loc[idx,:] = [f"{sta_ref.code}_{sta_ref.pt.strip()}_{sta_ref.soln[0].soln.strip()}", f"{sta2.code}_{sta2.pt.strip()}_{sta2.soln[0].soln.strip()}", baseline[0,0], baseline[0,1], baseline[0,2], t_snx, str(t_mjd), str(sta.domes[:5])]
                        
                        idx+=1
                        
                        
    df.sort_values(by=["t_mjd","station1"]).reset_index(drop=True)
    
    with open(f"contie_{title_baseline}.txt","w") as fi:
        fi.write("{:<7} {:15} {:15} {:20} {:20} {:20} {:15} {:15} {:10}\n".format("id_tie", "station1","station2", "dX[m]", "dY[m]","dZ[m]","t_snx","t_mjd","site"))
    
        for idx in df.index:
            fi.write("{:<7} {:15} {:15} {:<20.9f} {:<20.9f} {:<20.9f} {:<15} {:<15} {:<10}\n".format(f"t{idx}", df.loc[idx,"station1"],df.loc[idx,"station2"], df.loc[idx,"dX[m]"], df.loc[idx,"dY[m]"],df.loc[idx,"dZ[m]"],df.loc[idx,"t_snx"], df.loc[idx,"t_mjd"], df.loc[idx,"site"]))