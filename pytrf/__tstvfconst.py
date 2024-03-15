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
solns = read_solns("soln-gnss-i20b+start-end.snx")
df = pd.read_csv("gnss-i20b.xyz", names=['code', 'pt', 'domes', 'X','Y','Z'], sep=r'\s+')
df['pt'] = ' ' + df['pt'] #WARNING >> essential to be consistent with soln/ sinex 'pt' format (2chr)


grB = Graph_vfconst(solns=solns, df_sta=df)
df_staId = grB.df_staId


## 1. graph
G_dist = grB.build_snx_graph(limit_dist=10000)

## 2.time graph
G_time, G_time_seg, pbm_names = grB.build_time_soln_graph()
pbmv = [p[0] for p in pbm_names]
list_pbm = []
for sta in pbmv:
    right = []
    left = []
    for line in sta[0]:
        right.append(''.join(line))
    for line in sta[1]:
        left.append(''.join(line))
    list_pbm.append([right, left])


##3. manual correction(s) on G_time: add edges after check 'pbm_names' list
#0 PEN2 A vs PENC?
G_time.add_edge('site_13407_50636.5_60364.5','site_13407_52941.0_60364.5') #1 MADR A > linked to MAD2 after 2003
G_time.add_edge('site_21602_56603.0_60364.5','site_21602_50082.5_60364.5') #2 last WUH2 with WUHN
#3 JPLM vs WLSN?
#4 JPLM vs CIT1?
#5 1st PIN1 with 1st PIN2 ; last PIN1 with last PIN2
#G_time.add_edge('PIN1A1', 'PIN2A1')    #G_time.add_edge('site_40407_51467.4074537037_53258.145833333336','site_40407_51467.4074537037_55290.94494212963')
G_time.add_edge('site_40407_58670.13880787037_60364.5', 'site_40407_55290.94494212963_60364.5') #dernier vitesse PIN1/PIN2 #G_time.add_edge('PIN1A8','PIN2A5')> pin1A3 anc PIN2A5 

G_time.add_edge('site_40407_51467.4074537037_53258.145833333336', 'site_40407_51467.4074537037_55290.94494212963' ) # contrainte 2e V PIN1/PIN2
G_time.add_edge('site_40451_53530.5_56608.0', 'site_40451_49354.5_60364.5')#6 NLR1, GODE > change nothing because more than 10 km
#7 RIO2 vs RGDG : nothing to do (no constraint)



##4. sub-complete graphes (linked btw them all neighbouring nodes)
#g_dist, list_g_dist = grB.sub_complete_graph(G_dist) #  >>>> no complete graph for dist, because loop over complete all pairs on site, normaly same result..
g_time, list_g_time = grB.sub_complete_graph(G_time)


##5.  intersection dist & time graphes
intersection_graph = nx.intersection(G_dist, g_time)


##6. minimum link in each clusters of 'intersection_graph'
def minimum_linked(graph):
    clusters = list(nx.connected_components(graph))

    # Create a new graph to store the minimum spanning trees
    graph_final_min_link = nx.Graph()
    dict_graph_final_min_link = {}

    # Create minimum spanning tree for each cluster and add edges to graph_final_min_link
    for cluster in clusters:
        if len(cluster) > 1: #at least 1 connected component  ---> if 1 sta, no link, we remove
                    
            # Create a subgraph for the current cluster
            subgraph = graph.subgraph(cluster)
            
            site = df_staId.loc[list(subgraph.nodes)[0], "domes"][:5]
            
            if site in dict_graph_final_min_link.keys(): #case of subsites
                #logging.info(f"Sub-site: {site}")
                num_subsite = len([s for s in dict_graph_final_min_link.keys() if site in s]) + 1
                site = f"{site}_{num_subsite}"
            
            # Calculate minimum spanning tree for the subgraph
            mst = nx.minimum_spanning_tree(subgraph)
            # Add edges of the minimum spanning tree to mst_graph
            graph_final_min_link.add_edges_from(mst.edges)
            
            dict_graph_final_min_link[site] = mst
            
            # if 'PIN2A1' in list(cluster):
            #     print(cluster, subgraph.edges, mst.edges,site)
        
    return graph_final_min_link, dict_graph_final_min_link
        

##### VCONTR.dat
graph_final_vcontr, dict_graph_final_vcontr = minimum_linked(intersection_graph)        
#### write & export vcontr CATREF
with open('output/vcontr.dat', 'w') as file:
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
graph_final_fcontr, dict_graph_final_fcontr = minimum_linked(G_dist) #fconst: based only on distance, no amplitude discontinuities yet     
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
        
        

