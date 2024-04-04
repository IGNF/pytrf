#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:35:11 2024

@author: julienbarneoud
"""

import logging
logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import networkx as nx
from pytrf.date import date
from pytrf.sinex import sinex
from pytrf.io import read_solns
from Graph_vfconst import Graph_vfconst
import tqdm

###open files
solns = read_solns("data/igs_catref/soln.snx")
df = pd.read_csv("data/igs_catref/list-points.dat", names=['domes', 'soln', 'code', 'pt', 'nobs', 'datastart','dataend', 'X','Y','Z'], usecols=[0,1,2,3,4,5,6,8,9,10], sep=r'\s+')
#reformat datastart & dataend with tsinex format
df['datastart'] += ':00000'
df['dataend'] += ':00000'

grB = Graph_vfconst(solns=solns, df_sta=df)
df_staId = grB.df_staId
df_sites = grB.df_sites
df_v_disc = grB.df_v_disc


## 1. graph
G_dist = grB.build_snx_graph(limit_dist=10000)

_, dict_sites_Gdist = grB.minimum_linked(G_dist)
df_sites_dist = pd.DataFrame(columns=["siteId", "staId"])

#build new df_sites from sites & subsites defines by G_dist
for numsite, site in enumerate(dict_sites_Gdist.keys()):
    df_sites_dist.loc[numsite,"siteId"] = site
    df_sites_dist.loc[numsite,"staId"] = list(dict_sites_Gdist[site].nodes)

#update default df_dites attribute by=uild with domes name
grB.df_sites = df_sites_dist

## 2.time graph
G_time, G_time_seg, pbm_names = grB.build_time_soln_graph(fit_on_dataObs=True)

with open("list_pbm_solnV.txt", "w") as fpbm: #overwrite potential existing file
    fpbm.write("## code pt solnV  datastart_solnV  dataend_solnV   -----actual time range in input data ---->  datastart_solnV_obs  dataend_solnV_obs     'id solnV segment' ##\n")
    for nums, site in enumerate(pbm_names):
     
        #fpbm.write("\n--------------------------------- SITE PROBLEM -----------------------------------\n")
        fpbm.write(f"\n{34*'>'} SITE PROBLEM {34*'>'}\n")
        for num_pbm in site.keys(): 
            if num_pbm > 0:
                fpbm.write(f"{75*'.'}\n")
                
            for numl, line in enumerate(site[num_pbm]["code"][0]):
                code=line[0]
                pt=line[1]
                solnV=line[2]
                
                dates = grB.df_v_disc.loc[(grB.df_v_disc['code']==code) & (grB.df_v_disc['pt']==pt) & (grB.df_v_disc['solnV']==solnV),['datastart','dataend']].values.reshape(-1)
                dates_obs = grB.df_v_disc_fit_dataObs.loc[(grB.df_v_disc_fit_dataObs['code']==code) & (grB.df_v_disc_fit_dataObs['pt']==pt) & (grB.df_v_disc_fit_dataObs['solnV']==solnV),['datastart_obs','dataend_obs']].values.reshape(-1)
                
                id_solnV = site[num_pbm]["segment_v"][0][numl]
             
                if len(dates_obs) == 0: #no obs for this solnV time segment
                    dates_obs = [12*'*', 12*'*']
                
                fpbm.write(f"V- {code}{pt}{solnV} {dates[0]}  {dates[1]}   -->   {dates_obs[0]}  {dates_obs[1]}       {id_solnV}\n")
             
            fpbm.write("VS. \n")
            for numl, line in enumerate(site[num_pbm]["code"][1]):
                code=line[0]
                pt=line[1]
                solnV=line[2]
                
                dates = grB.df_v_disc.loc[(grB.df_v_disc['code']==code) & (grB.df_v_disc['pt']==pt) & (grB.df_v_disc['solnV']==solnV),['datastart','dataend']].values.reshape(-1)
                dates_obs = grB.df_v_disc_fit_dataObs.loc[(grB.df_v_disc_fit_dataObs['code']==code) & (grB.df_v_disc_fit_dataObs['pt']==pt) & (grB.df_v_disc_fit_dataObs['solnV']==solnV),['datastart_obs','dataend_obs']].values.reshape(-1)
                
                id_solnV = site[num_pbm]["segment_v"][1][numl]
                
                if len(dates_obs) == 0: #no obs for this solnV time segment
                    dates_obs = [12*'*', 12*'*']
                fpbm.write(f"V- {code}{pt}{solnV} {dates[0]}  {dates[1]}   -->   {dates_obs[0]}  {dates_obs[1]}       {id_solnV}\n")      
        #fpbm.write("---------------------------------------------------------------------------------\n")
        fpbm.write(f"{82*'<'}\n")        

    
##3. manual correction(s) on G_time: add edges after check 'pbm_names' list
#0 CIT1 A vs JPLM?
# G_time.add_edge('site_40400_58670.13880787037_999999.0','site_40400_53269.0_999999.0') ### CIT1 A 6 <> JPLM A 3 (last segment)
# G_time.add_edge('site_49886_58144.0_999999.0', 'site_49886_55290.94494212963_999999.0') ### PLO5 A 3 <> P475 A 2 & PLO6 A 3 <> P475 A 2

##4. sub-complete graphes (linked btw them all neighbouring nodes)
#g_dist, list_g_dist = grB.sub_complete_graph(G_dist) #  >>>> no complete graph for dist, because loop over complete all pairs on site, normaly same result..
g_time, list_g_time = grB.sub_complete_graph(G_time)


##5.  intersection dist & time graphes
intersection_graph = nx.intersection(G_dist, g_time)

##6. absolute constraints
sta_const_abs = grB.build_absolute_const(const_time=1, const_nobs=20, G_relative_const=intersection_graph)# get 'intersection_graph': check if stations keept already linked to another station in a site


##7. minimum link in each clusters of 'intersection_graph' & export
##### VCONTR.dat
graph_final_vcontr, dict_graph_final_vcontr = grB.minimum_linked(intersection_graph)        
#### write & export vcontr CATREF
with open('output/vcontr.dat', 'w') as file:
    
    # abs constraints
    for staId in sta_const_abs:
        
        file.write("SIT\n")
        domes = df_staId.loc[staId,'domes']
        code= staId[:4]
        pt = staId[4]
        soln = staId[5:]
        sigma = '0.100000'
        
        file.write("{:>9}{:>3}{:>10}{:>3}{:>37}{:>6}{:>3}\n".format(
            domes, soln, "", "", sigma, code, pt))
    
    # relative constraints
    for site in sorted(dict_graph_final_vcontr.keys()):
        file.write("SIT\n")
        gr = dict_graph_final_vcontr[site]
        
        edges = list(gr.edges) #list of tuple
        
        for (staId1, staId2) in edges:
            domes1 = df_staId.loc[staId1,'domes']
            domes2 = df_staId.loc[staId2,'domes']
            
            code1, code2 = staId1[:4], staId2[:4]
            pt1, pt2 = staId1[4], staId2[4]
            soln1, soln2 = staId1[5:], staId2[5:]
            
            sigma = '0.000001'
            
            file.write("{:>9}{:>3}{:>10}{:>3}{:>37}{:>6}{:>3}{:>5}{:>3}\n".format(
                domes1, soln1, domes2, soln2, sigma, code1, pt1, code2, pt2))
            
            
##### FCONTR.dat
graph_final_fcontr, dict_graph_final_fcontr = grB.minimum_linked(G_dist) #fconst: based only on distance, no amplitude discontinuities yet     
#### write & export vcontr CATREF
with open('output/fcontr.dat', 'w') as file:
    for site in sorted(dict_graph_final_fcontr.keys()):
        file.write("SIT\n")
        gr = dict_graph_final_fcontr[site]
        
        edges = list(gr.edges) #list of tuple
        
        for (staId1, staId2) in edges:
            domes1 = df_staId.loc[staId1,'domes']
            domes2 = df_staId.loc[staId2,'domes']
            
            code1, code2 = staId1[:4], staId2[:4]
            pt1, pt2 = staId1[4], staId2[4]
            soln1, soln2 = staId1[5:], staId2[5:]
            
            sigma = '0.000001'
            
            file.write("{:>9}{:>3}{:>10}{:>3}{:>37}{:>6}{:>3}{:>5}{:>3}\n".format(
                domes1, soln1, domes2, soln2, sigma, code1, pt1, code2, pt2))
            #annual & semi annual fcontr.dat: insert blank line
            file.write("\n") 
            
            

logging.info("END vcontr.dat & fcontr.dat generation.")