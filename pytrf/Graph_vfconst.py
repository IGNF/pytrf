#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External imports
#-----------------
import os
import re
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

import networkx as nx
import logging
import tqdm

from shapely.geometry import LineString, MultiLineString

# Internal imports
#-----------------
from pytrf import date
from pytrf.const import mjd_leap, gps_utc, PERIODS
from pytrf.io import read_solns


class Graph_vfconst():
    """
    This class allows to build a graph from a sinex object (pytrf.sinex) and to resonate in terms of "nodes" (stations) and "edges" (constraints).
    Useful to understand links between stations and apply/build vfconst file.
    Based on python networkx library.
    
    
    Vocabulary & graph modeling
    ---------------------------
    A station: a node (station ID -> staId = code(4chr) + pt(1chr) + solnP(1+chr))
    A constrain btw 2 stations: an edge btw 2 nodes
    
    A site: based on DOMES 5chr
        In a site, some constraints must/ mustn't be applied btw stations
         
    A soln time segment -> modelized as a geometric segment (built with shapely.geometry)
        Line: a time segment (i.e. 1 soln)
        MultiLine: set of Line (i.e. a station modelisation, with several solns)
    
    
    Build a vfconst file
    ---------------------
    from:
    > SOLNS discontinuities (pytrf.io.read_solns)
    > stations info: coordinates, domes, etc. (pandas.dataframe)
    
    """
        
    def __init__(self, snx=None, solns=None, df_sta=None, dates_from_obs=True):
        """
        Constructor, from pytrf.sinex and vfconst.yml file

        Parameters
        ----------
        snx : pytrf.sinex object
            sinex with stations of interest, for example generated for combination
        solns : pytrf.io.read_solns, list of records
            soln file open with pytrf.io.read_solns
        df_sta: pandas.DataFrame
            stations observation info. At least with columns 'domes', 'code', 'pt', 'X', 'Y', 'Z', 'nobs' (optional: datestart & dateend, 'soln' (default set to '   1'))
        
        dates_from_obs: bool, default True.
            keep anyway dates from obs, not fit on soln dates. 
            If no datastart or dataend in df_sta, set dby default dates_from_obs = False (-> 'dates from SOLNS')
        
        Returns
        -------
        None.

        """
        self.snx = None
        self.solns = None
        self.df_sta = None
        self.dates_from_obs = dates_from_obs
        
        self.df_staId, self.df_sites = None, None
        
        if type(snx) != type(None): #init from snx object
            self.snx = snx
            #build id dataframes : station & sites (only base on DOMES 5 first chr)
            self.df_staId = self.generate_staId_from_snx()

        if type(solns) != type(None):
            self.solns = solns
            self.df_solns, self.df_v_disc = self.generate_df_solns_vdisc()
            
        if type(df_sta) != type(None):
            self.df_sta = df_sta
            if "pt" in self.df_sta.columns:
                self.df_sta["pt"] = self.df_sta["pt"].astype(str).str.rjust(2) #reformat pt as 2 chr (' A')
            if "soln" in self.df_sta.columns:
                self.df_sta["soln"] = self.df_sta["soln"].astype(str).str.rjust(4) #reformat soln as 4 chr ('   1')
            else: #set default soln
                self.df_sta["soln"] = '   1'
                
         
        if type(self.df_sta) != type(None) and type(self.solns) != type(None): ###init df_staId & df_sites
        
            #check if all sta are in solns
            list_staId = self.df_sta.apply(lambda row: (row["code"]+ row["pt"]+ row["soln"]).replace(" ",""), axis=1).values
            list_staId = [staId for staId in list_staId if staId not in self.df_solns.index]
            
            if len(list_staId)>0:#not all sta in solns: add default soln values
                logging.info(f"{list_staId} not in soln file, add default values.")
                self.add_default_soln(list_staId)
                
            #build StaId using in priority station in 'df_sta'
            #print(("datastart" not in self.df_sta.columns) or ("dataend" not in self.df_sta.columns) or ("soln" not in self.df_sta.columns))
            if ("datastart" not in self.df_sta.columns) or ("dataend" not in self.df_sta.columns) or (not self.dates_from_obs): #BASED on SOLN take datastart & dataend from soln
                print("based on SOLN", len(self.df_solns))
                self.df_solns = pd.merge(self.df_solns.reset_index(), self.df_sta[['code', 'pt', 'domes']].drop_duplicates().reset_index(drop=True), on=['code', 'pt'], how="left").set_index('staId') #set domes to solns 
                print("1. based on SOLN", len(self.df_solns))
                self.df_staId = pd.merge(self.df_solns.reset_index(), self.df_sta[['code', 'pt', 'soln', 'X', 'Y', 'Z', 'nobs']], on=['code', 'pt', 'soln'], how='left').set_index('staId')
                print("2. based on SOLN", len(self.df_solns), len(self.df_staId))
            else: #BASED on OBS: prefer to take dates from df_sta -> join also on SOLN
                self.df_staId = pd.merge(self.df_solns.reset_index().loc[:, self.df_solns.reset_index().columns.difference(['datastart', 'dataend'])], self.df_sta, on=['code', 'pt','soln'], how='right').set_index('staId') #retrive 'datastart' & 'dataend' from soln
            
            #set node name according to segment_v + site (domes[:5])
            #self.df_staId['node_v_name'] = self.df_staId.apply(lambda row: self.get_node_sitename_linestring(row['domes'][:5], row['segment_v']), axis=1)
            #reorder according domes name
            self.df_staId = self.df_staId.sort_values(by=['domes', 'staId'])
            
            if self.df_staId.isna().any().any(): #at least 1 NaN after merge
                logging.warning("No 1 soln by station -> set default soln value.")
                #if no soln for all stations -> Nan -> replace by default value
                self.df_staId['datastart'] = self.df_staId['datastart'].fillna('00:000:00000')
                self.df_staId['dataend'] = self.df_staId['dataend'].fillna('00:000:00000')
                self.df_staId['soln'] = self.df_staId['soln'].fillna('   1')
                self.df_staId['solnV'] = self.df_staId['solnV'].fillna('   1')
                
                self.df_staId["_staId_pytrf"] = self.df_staId["code"] + self.df_staId["pt"] + self.df_staId["soln"]
                self.df_staId["staId"] = self.df_staId.apply(lambda row: (row["code"]+ row["pt"]+ row["soln"]).replace(" ",""), axis=1) #no space
                self.df_staId = self.df_staId.set_index("staId")
                
            #create sub selection of solnV based on data in staId -> datastart & dataend date based on data observation (self.df_staId)
            self.df_v_disc_fit_dataObs = self.generate_v_disc_fit_dataObs()
        
        print("len df_staId", len(self.df_staId))
        self.df_sites = self.generate_sites_domes(self.df_staId)
        print("len df_staId", len(self.df_staId))
              
    def generate_staId_from_snx(self):
        """ DataFrame summarizing stations in self.snx file
            Columns : ['domes', 'code', 'pt', 'soln', 'datastart', 'dataend', '_staId_pytrf', 'X', 'Y', 'Z']
        """
        df_domes_snx = pd.DataFrame([(sta.domes, sta.code, sta.pt, soln.soln, soln.datastart, soln.dataend, soln.nobs) for sta in self.snx.sta for soln in sta.soln], columns=["domes", "code", "pt", "soln", "datastart", "dataend", "nobs"])
        df_domes_snx["_staId_pytrf"] = df_domes_snx["code"]+ df_domes_snx["pt"]+ df_domes_snx["soln"]
        df_domes_snx["staId"] = df_domes_snx.apply(lambda row: (row["code"]+ row["pt"]+ row["soln"]).replace(" ",""), axis=1) #no space
        df_domes_snx = df_domes_snx.set_index("staId")
        
        #coordinates
        coords = self.snx.get_xyz(df_domes_snx["code"].values.tolist(), pt=df_domes_snx["pt"].values.tolist(),  soln=df_domes_snx["soln"].values.tolist())
        df_domes_snx["X"], df_domes_snx["Y"], df_domes_snx["Z"] = coords[:,0], coords[:,1], coords[:,2]
        
        return df_domes_snx
    
    
    def generate_df_solns_vdisc(self):
        """ DataFrame summarizing stations in self.soln file
            Columns : ['code', 'pt', 'soln', 'solnV 'datastart', 'dataend', '_staId_pytrf']
        """
        
        def find_segment(k, list_dicontinuities):
            """Provides k segment number in a list of discontinuities
               Ex: list_dicontinuities = [1,3,8] & k=2 -> belong to segment [1,3] (seg n.1)
            """
            for i, discontinuity in enumerate(list_V_dicontinuities):
                if i == 0 and k <= discontinuity: #1st segment
                    return i
                elif i == len(list_dicontinuities) - 1 and k >= discontinuity: #last
                    return i + 1
                elif list_dicontinuities[i] < k <= list_dicontinuities[i + 1]: #middle
                    return i + 1
                
                
        df_solns = pd.DataFrame()
        df_v_disc = pd.DataFrame()
        num=0
        numV=0
        for soln in tqdm.tqdm(self.solns, desc='Building df from solns...'):
            
            list_P_dicontinuities = [date.from_tsnx(p.end).mjd for p in soln.P if p.end !='00:000:00000']
            list_V_dicontinuities = [date.from_tsnx(v.end).mjd for v in soln.V if v.end !='00:000:00000']
            valid_soln = all(element in list_P_dicontinuities for element in list_V_dicontinuities)
            if not valid_soln:
                raise ValueError(f"Invalid soln format: {soln.code} {soln.pt} P {[p.end for p in soln.P if p.end !='00:000:00000']} vs. V {[v.end for v in soln.V if v.end !='00:000:00000']}")
            
            for solP in soln.P:
                df_solns.loc[num, 'code'] = soln.code
                df_solns.loc[num, 'pt'] = soln.pt
                #depend of soln
                df_solns.loc[num, 'datastart'] = solP.start
                df_solns.loc[num, 'dataend'] = solP.end
                df_solns.loc[num, 'soln'] =solP.soln
                
                if solP.end == '00:000:00000' or len(list_V_dicontinuities)== 0: #last P segment or no V discontinuities anyway
                   solV = soln.V[-1] #the last V segment
                else: #Position discontinuities -> potential Velocity discontinuities
                    end_mjd = date.from_tsnx(solP.end).mjd
                    solV = soln.V[find_segment(end_mjd, list_V_dicontinuities)] #find correct V segment
                    
                ### mjd date
                if solV.start == '00:000:00000':
                    mjd_vstart = -1 #-inf
                else:
                    mjd_vstart = date.from_tsnx(solV.start).mjd
                    
                if solV.end == '00:000:00000':
                    mjd_vend = 999999 #+inf
                else:
                    mjd_vend = date.from_tsnx(solV.end).mjd
        
                df_solns.loc[num, 'segment_v'] = LineString([(mjd_vstart, 0), (mjd_vend, 0)])
                df_solns.loc[num, 'solnV'] = solV.soln
                num+=1

            for solV in soln.V:
                df_v_disc.loc[numV, 'code'] = soln.code
                df_v_disc.loc[numV, 'pt'] = soln.pt
                #depend of soln
                df_v_disc.loc[numV, 'datastart'] = solV.start
                df_v_disc.loc[numV, 'dataend'] = solV.end
                df_v_disc.loc[numV, 'solnV'] = solV.soln
                
                ### mjd date
                if solV.start == '00:000:00000':
                    df_v_disc.loc[numV, 'mjd_datastart'] = -1 #-inf
                else:
                    df_v_disc.loc[numV, 'mjd_datastart'] = date.from_tsnx(solV.start).mjd
                    
                if solV.end == '00:000:00000':
                    df_v_disc.loc[numV, 'mjd_dataend'] = 999999 #+inf
                else:
                    df_v_disc.loc[numV, 'mjd_dataend'] = date.from_tsnx(solV.end).mjd
                    
                df_v_disc.loc[numV,"segment"] = LineString([(df_v_disc.loc[numV, 'mjd_datastart'], 0), (df_v_disc.loc[numV, 'mjd_dataend'], 0)])
                
                numV+=1
                
        df_solns['_staId_pytrf'] = df_solns['code'] + df_solns['pt'] + df_solns['soln']
        df_solns['staId'] = df_solns.apply(lambda row: (row["_staId_pytrf"].replace(" ","")), axis=1)
        
        df_solns = df_solns.set_index("staId")
        
        return df_solns, df_v_disc
    
    
    
    def generate_v_disc_fit_dataObs(self):
        """
        Provides dataframe of v discontinuities from soln, fit on data in self.df_staId
        Add 'datastart_obs' & 'dataend_obs': soln segmentV based on data observation (fit on obs)
        
        """
        df_obs = self.df_staId[['code', 'pt', 'soln', 'solnV', 'datastart', 'dataend']]
        df_obs = df_obs.rename(columns={'datastart':'datastart_obs', 'dataend':'dataend_obs'})
        
        df_v_disc_fit_dataObs = self.df_v_disc.merge(df_obs, on=['code', 'pt', 'solnV'], how='inner')
        
        # Define custom aggregation function
        def custom_agg(group):
            min_soln = group.loc[group['soln'].astype(int).idxmin(),'soln']
            max_soln = group.loc[group['soln'].astype(int).idxmax(),'soln']
            
            segment = group.loc[:, 'segment'].iloc[0]
            datastart = group.loc[group['soln'] == min_soln, 'datastart_obs'].iloc[0]
            dataend = group.loc[group['soln'] == max_soln, 'dataend_obs'].iloc[0]
            
            return pd.Series({'segment':segment, 'datastart_obs': datastart, 'dataend_obs': dataend})
        
        df_v_disc_fit_dataObs = df_v_disc_fit_dataObs.groupby(['code', 'pt', 'datastart', 'dataend', 'solnV', 'mjd_datastart', 'mjd_dataend']).apply(custom_agg).reset_index()
        
        #### compute segment_observation
        mjd_datastart_obs = [date.from_tsnx(da).mjd if da!= '00:000:00000' else -1 for da in df_v_disc_fit_dataObs['datastart_obs']]
        mjd_dataend_obs = [date.from_tsnx(da).mjd if da!= '00:000:00000' else 999999 for da in df_v_disc_fit_dataObs['dataend_obs']]
        
        df_v_disc_fit_dataObs['segment_obs'] = [LineString([(mjd_datastart_obs[num], 0), (mjd_dataend_obs[num], 0)]) for num in range(len(mjd_datastart_obs))]
        
        return df_v_disc_fit_dataObs
    
    
    def add_default_soln(self, list_staId):
        """Update df_solns, df_v_disc with default soln values for 'list_staId' code"""
        
        
        for staId in list_staId:
            self.df_solns.loc[staId, "code"] = staId[:4]
            self.df_solns.loc[staId, "pt"] = "{0:>2}".format(staId[4])
            self.df_solns.loc[staId, "soln"] = "{0:>4}".format(staId[5:])
            self.df_solns.loc[staId, "solnV"] = '   1'
            
            self.df_solns.loc[staId, "datastart"] = self.df_solns.loc[staId, "dataend"] = '00:000:00000'
            
            self.df_solns.loc[staId, "segment_v"] = LineString([(-1, 0), (999999, 0)])
            self.df_solns.loc[staId,'_staId_pytrf'] = self.df_solns.loc[staId,'code'] + self.df_solns.loc[staId,'pt'] + self.df_solns.loc[staId,'soln']
            
            
            #### df_v_disc
            numl = len(self.df_v_disc)
            self.df_v_disc.loc[numl , "code"] = staId[:4]
            self.df_v_disc.loc[numl , "pt"] = "{0:>2}".format(staId[4])
            self.df_v_disc.loc[numl , "solnV"] = '   1'
            
            self.df_v_disc.loc[numl , "datastart"] = self.df_v_disc.loc[numl , "dataend"] = '00:000:00000'
            
            self.df_v_disc.loc[numl , "segment"] = LineString([(-1, 0), (999999, 0)])
            self.df_v_disc.loc[numl , "mjd_datastart"], self.df_v_disc.loc[numl , "mjd_dataend"] = -1, 999999
       
    
    
    def generate_sites_domes(self, df_domes_snx):
        """From df_staId, generate df_sites using domes 5chr"""
        
        if 'domes' not in df_domes_snx.columns or type(df_domes_snx) == type(None):
            return None
        
        #copy 
        df_domes_snx = df_domes_snx.copy()
        ### Sites according 5 domes character
        # Select the first 5 characters of the "domes" column
        df_domes_snx['siteId'] = df_domes_snx['domes'].str[:5]

        # Group by the first 5 characters and aggregate into a list with respective index values
        df_sites = df_domes_snx.groupby('siteId').agg(staId=('domes', lambda x: x.index.tolist())).reset_index()

        #### remove site without domes: '-----'
        df_sites = df_sites[df_sites['siteId'] != '-----'].reset_index(drop=True)
        
        if type(self.df_staId) != type (None):
            if "siteId" in self.df_staId:
                del self.df_staId["siteId"]
                
            print("shapes", len(self.df_staId.reset_index()), len(df_sites.explode('staId').reset_index(drop=True)))
            self.df_staId = pd.merge(self.df_staId.reset_index(), df_sites.explode('staId').reset_index(drop=True), on='staId').set_index('staId')
            
             
        return df_sites
    
    
    
    def set_df_sites(self, df_sites):
        """
        Add/ upgrade sites definition (__ini__ default: only based on DOMES 5 chr)
        Example: sites based in distance criteria
        
        This method update usefull attributes:
            - self.df_staId: 'node_v_name' (initially based on domes 5chr)

        Parameters
        ----------
        df_sites : pandas.DataFrame
            2 columns: siteId (str), staId (list of str)

        Returns
        -------
        None.
        Update attributes

        """
        self.df_sites = df_sites
        
        # update "site" name
        if "siteId" in self.df_staId:
            del self.df_staId["siteId"]
        self.df_staId = pd.merge(self.df_staId.reset_index(), df_sites.explode('staId').reset_index(drop=True), on='staId', how='left').set_index('staId')
         
        self.df_staId["siteId"] = self.df_staId["siteId"].fillna('-----') #keep all stations, if not in sites default '-----'

        

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
        """ Build graph from self.df_staId: based on station distance in a site
         if link_only_soln=True: no dist consideration, only station solns linked.
        """
        # Create a graph using networkx
        G = nx.Graph()
        
        # Add edges to the graph based on the distance and threshold
        for site in tqdm.tqdm(self.df_sites['staId'], desc='Building dist graph...'):
            #site: a list of all stations on the site. Multiple SOLN & no dist limit yet
            complete_graph = nx.complete_graph(site) #get all pairs on site
            for staId1, staId2 in complete_graph.edges():
                #compute euclidian distance
                distance = np.sqrt(np.sum((self.df_staId.loc[staId1, ['X', 'Y', 'Z']] - self.df_staId.loc[staId2, ['X', 'Y', 'Z']])**2))
                #print("dist", distance, staId1, staId2)
                if link_only_soln and (staId1[:5]==staId2[:5]): #add edge only for same station, between SOLN
                    G.add_edge(staId1, staId2, length=distance)
                
                
                elif not link_only_soln and distance <= limit_dist: #check distance only if not "link_only_soln" 
                    G.add_edge(staId1, staId2, length=distance)        
        return G
    
    
    def sub_complete_graph(self, G):
        """ From graph G, build a complete graph btw all stations (nodes) linked/neighbouring in G
            Useful before a graph intersection -> related station(s) will be linked, then just check if an edge exists.
        """
        # Get connected components
        clusters = list(nx.connected_components(G))
        
        # Create subgraphs for each cluster
        subgraphs = [G.subgraph(cluster) for cluster in clusters]
        
        # Create sub complete graphs for each cluster
        sub_compl_graphs = [nx.complete_graph(subgraph.nodes) for subgraph in subgraphs] #list of graphes
        
        # Accumulate all subgraphs into a single graph
        single_graph = nx.union_all(sub_compl_graphs)
        
        return single_graph, sub_compl_graphs

            
    
    def build_time_soln_graph(self):
        """
        Builds graph based on TIME relations btw stations SOLNS.
        
        Strategy:
        ---------
        1. Builds a graph based on solnV relations (G_time_seg)
           -node: segment solnV
           -edge: constraints
        
        Firstly, add all segment solnV (nodes) in graph, without constraints (edges)
        
        Secondly, about constraints (edges), there are several possibilites on each DOMES site:
            -> same V solns = same discontinuities (do nothing, not edges btw solnV segments)
            - no same solnV segment:
                -> 2 solnV segment of a station included in 1 solnV segment of another station: CONFLICT -> get station(s) name(s) to user (pbm_names) -> (default: do nothing, not edges btw solnV segments)
                -> 1 solnV segment of a station included in 1 solnV segment of another station: same V -> ADD CONSTRAINTS (edge) btw these 2 nodes.
                -> no inclusion/ time segment solnV intersection : no possible association btw solnV nodes (do nothing, not edges btw solnV segments)
                
        2. Builds a graph with stationsId (code/pt/solnP from self.df_staId) linked to their solnV node (1 station necessary linked to 1 node solnV <> 1 node solnV may be linked to 0--n stations)


        ==> Final G_all graph: nodes (staId + solnV), edges (constraints btw staId, neighbouring thanks to solnV nodes)
        
        Parameters:
        ----------
        - self.df_sites: pandas.dataframe of sites: 2 columns: "siteId" (str) + "staId" (list of str).
                        You can update BEFORE this attr.
                        Default __init__: sites based on DOMES
                        Ex: can be generated with graph G_dist after 'build_snx_graph()' -> sites based on dist proxmity
        
        - self.dates_from_obs: bool (Default: True) 
    
                    If True: Use only solnV segments occuring/usefull in current dataset (self.df_staId).
                            For example, in case of a stations & time range selections. Not considered all SOLN
                            WARNING: can hide CONFLICTS in solnV [2 solnV segment of a station described by soln included in 1 solnV segment of another station]
                                     -> here 1 solnV can be not represented in dataset  
                                     
                    If False: Use all solnV segments in self.soln -> 'dates from soln'
                    
                        
        """
        
        def get_pbms(df_v_disc, site, sub, sta, multiline, list_multilinestrings):
            """ Return list of versus pbm --> log"""
            
            dict_pbms = find_common_values_ml(multiline)
            
            multline_pbm1 = np.array(list_multilinestrings[sta])
            
            pbms = {}
            
            if (len(dict_pbms))>1:
                logging.warning(f"Multiple problems {site['siteId']} {site['staId']}")
            
            num_site_pbm = 0
            for value_pbm, list_linestring in dict_pbms.items():
                
                ml2, l2 = value_pbm
                LineString_pbm1 = np.array(list_multilinestrings[sta].geoms)[list_linestring]
                         
                multline_pbm2 = np.array(list_multilinestrings)[ml2]
                LineString_pbm2 = np.array(multline_pbm2.geoms)[l2]
                

                list_staId_pbm1 = list(sub.loc[sub['segment_multi']==multline_pbm1].index)
                list_staId_pbm1 = [(sta[:4], f" {sta[4]}") for sta in list_staId_pbm1]
                c1, pt1 = zip(*list_staId_pbm1)
                df_v_disc1 = df_v_disc[(df_v_disc['code'].isin(c1)) & (df_v_disc['pt'].isin(pt1))]
                
                list_staId_pbm2 = list(sub.loc[sub['segment_multi']==multline_pbm2].index)
                list_staId_pbm2 = [(sta[:4], f" {sta[4]}") for sta in list_staId_pbm2]
                c2, pt2 = zip(*list_staId_pbm2)
                
                df_v_disc2 = df_v_disc[(df_v_disc['code'].isin(c2)) & (df_v_disc['pt'].isin(pt2))]
                
                # solns of "df_v_disc1" > refers to soln V (Velocity!)
                #print(LineString_pbm1, df_v_disc1, LineString_pbm2)
                v_disc1 = df_v_disc1.loc[df_v_disc1['segment'].isin(LineString_pbm1)][['code','pt','solnV']].values.tolist()
                v_disc2 = df_v_disc2.loc[df_v_disc2['segment']==LineString_pbm2][['code','pt','solnV']].values.tolist()
                
                logging.warning(f" >>>{v_disc1} VS {v_disc2}")
                
                pbm = {}
                
                pbm["code"] = [v_disc1, v_disc2]
                      #[[self.get_node_sitename_linestring(site['siteId'], lpb1) for lpb1 in  LineString_pbm1], [self.get_node_sitename_linestring(site['siteId'], lpb2) for lpb2 in  LineString_pbm2]]]
                pbm["segment_v"] = [[self.df_staId.loc[(self.df_staId["code"]==code) & (self.df_staId["pt"]==pt) & (self.df_staId["solnV"]==solnV), 'node_v_name'].values for code, pt, solnV in v_disc1], [self.df_staId.loc[(self.df_staId["code"]==code) & (self.df_staId["pt"]==pt) & (self.df_staId["solnV"]==solnV), 'node_v_name'].values for code, pt, solnV in v_disc2]]
                pbm["segment_v_obs"] = [[self.df_staId.loc[(self.df_staId["code"]==code) & (self.df_staId["pt"]==pt) & (self.df_staId["solnV"]==solnV), 'node_v_name_obs'].values for code, pt, solnV in v_disc1], [self.df_staId.loc[(self.df_staId["code"]==code) & (self.df_staId["pt"]==pt) & (self.df_staId["solnV"]==solnV), 'node_v_name_obs'].values for code, pt, solnV in v_disc2]]
              
                
                #no segment value -> reformat to ''
                pbm["segment_v"][0] = [name_id[0] if len(name_id)>0 else '' for name_id in pbm["segment_v"][0]] #left
                pbm["segment_v"][1] = [name_id[0] if len(name_id)>0 else '' for name_id in pbm["segment_v"][1]] #right
                
                pbm["segment_v_obs"][0] = [name_id[0] if len(name_id)>0 else '' for name_id in pbm["segment_v_obs"][0]] #left
                pbm["segment_v_obs"][1] = [name_id[0] if len(name_id)>0 else '' for name_id in pbm["segment_v_obs"][1]] #right
                
                pbms[num_site_pbm] = pbm
                
                num_site_pbm += 1
                
            return pbms
        
        # Function to find common values between keys in a multiline
        def find_common_values_ml(dictionary):
            common_values = {}  # Dictionary to store common values and their corresponding keys
            for key, lst in dictionary.items():
                for value in lst:
                    if value not in common_values:
                        common_values[value] = [key]
                    else:
                        common_values[value].append(key)
            # Filter out values that are common to multiple keys
            common_values = {value: keys for value, keys in common_values.items() if len(keys) > 1}
            return common_values
   
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##update 'segment_obs' to df_staId
        if 'segment_v_obs' in self.df_staId: #if already exist, del it
            del self.df_staId['segment_v_obs']
        self.df_staId = pd.merge(self.df_staId.reset_index(), self.df_v_disc_fit_dataObs[['code', 'pt', 'solnV', 'segment_obs']], on=['code', 'pt', 'solnV']).set_index("staId")
        self.df_staId =self.df_staId.rename(columns={'segment_obs': 'segment_v_obs'})
        
        #### update self.df_v_disc & self.df_v_disc_fit_dataObs with "siteId"
        
        if self.dates_from_obs: #use solnV info only from observed data
            df_v_disc = self.df_v_disc_fit_dataObs
        
            ##### for each staId, associate all solnV of this station (no solnP consideration) --> considerepd.merge(df_staId, grB.df_v_disc_fit_dataObs[['code', 'pt', 'solnV', 'segment_obs']], on=['code', 'pt', 'solnV'])d segment_obs too
            df = pd.merge(self.df_staId.reset_index(), df_v_disc[['code', 'pt','segment', 'segment_obs']], on=['code', 'pt'], how='left').set_index("staId")
            self.df_grouped = df.groupby('staId')['segment'].agg(list).to_frame()
            self.df_grouped['segment_multi'] = self.df_grouped['segment'].apply(lambda x: MultiLineString(x)) #dataframe with saId as index & list of segment
            self.df_grouped['segment_multi_obs'] = df.groupby('staId')['segment_obs'].agg(list).to_frame()['segment_obs'].apply(lambda x: MultiLineString(x))
            
        
        else: # 'fit on soln'
            df_v_disc = self.df_v_disc
            
            ##### for each staId, associate all solnV of this station (no solnP consideration) --> considered segment_obs too
            df = pd.merge(self.df_staId.reset_index(), df_v_disc[['code', 'pt','segment']], on=['code', 'pt'], how='left').set_index("staId")
            self.df_grouped = df.groupby('staId')['segment'].agg(list).to_frame()
            self.df_grouped['segment_multi'] = self.df_grouped['segment'].apply(lambda x: MultiLineString(x)) #dataframe with saId as index & list of segment
        
        #set 'node_v_segment' 
        self.df_staId['node_v_name'] = self.df_staId.apply(lambda row: f"site_{row['siteId']}_{list(row['segment_v'].coords)[0][0]}_{list(row['segment_v'].coords)[1][0]}", axis=1)
        self.df_staId['node_v_name_obs'] = self.df_staId.apply(lambda row: f"site_{row['siteId']}_{list(row['segment_v_obs'].coords)[0][0]}_{list(row['segment_v_obs'].coords)[1][0]}", axis=1)
           
        
        num_pbm, num_pbm_solved = 0, 0
        list_pbm, list_pbm_solved = [], []
        pbm_names = {}
        pbm_solved = {}
        # Create a graph using networkx
        G_time_seg = nx.Graph()
        for num_site, site in tqdm.tqdm(self.df_sites.iterrows(), desc='Building time graph...', total=len(self.df_sites)):
            
            sub = self.df_grouped.loc[site['staId']] #on site, all V discontinuities time segment
            #nodes_seg_names = pd.unique([f'site_{site["siteId"]}_{list(line.coords)[0][0]}_{list(line.coords)[1][0]}' for line in sub['segment'].explode()])
            nodes_seg_names = pd.unique([self.get_node_sitename_linestring(site['siteId'], line) for line in sub['segment'].explode()])
            
            #add all temporal solnV segments as nodes
            G_time_seg.add_nodes_from(nodes_seg_names, color='red') 
            
            ### add links btw nodes_seg_names?
            if sub["segment_multi"].nunique() == 1: #All values in V 'segment' are identical >>> same discontinuities, all right, no needs to link 'node_seg_names'
                #logging.info(f"Site {site['siteId']} OK.")
                pass
       
            else:
                #logging.warning(f"No consistent V discontinuities on site {num} >> {self.df_sites.loc[num, ['siteId', 'staId']].values}")
                list_multilinestrings = list(pd.unique(sub["segment_multi"]))
                dict_included = self.check_inclusionML(list_multilinestrings)
                
                pbm_in_dc = any([len(find_common_values_ml(multiline))>0 for sta, multiline in dict_included.items()])
                pbm_in_dc_obs = True #set default to True
                
                
                if self.dates_from_obs:
                    list_multilinestrings_obs = list(pd.unique(sub["segment_multi_obs"])) #study based on datastart & dataend of observations
                    dict_included_obs = self.check_inclusionML(list_multilinestrings_obs)
                    
                    pbm_in_dc_obs = any([len(find_common_values_ml(multiline))>0 for sta, multiline in dict_included_obs.items()])
                    
                ### 1. pbm anyway, stay focus on soln or no pbm
                if (pbm_in_dc and pbm_in_dc_obs) or (not pbm_in_dc) : 
                
                    for sta, multiline in dict_included.items():
                        #multiline: list of dict -> simple list {0:[(1,2)], 1:[2,7]}
                        #### inclusion trouble: 2 or + inclusion from station1 to station2 >>>> no more V discontinuities !!  flag but do nothing on links
                        if len(find_common_values_ml(multiline))>0:
                            
                                list_pbm.append(num_site)
                                num_pbm +=1
                                
                                #generate pbm list LOG
                                pbm = get_pbms(df_v_disc, site, sub, sta, multiline, list_multilinestrings)                  
                                if site['siteId'] not in pbm_names.keys():
                                    pbm_names[site['siteId']] = []
                                pbm_names[site['siteId']].append(pbm)
                                continue # -> go to NEXT iterr. (test next multiline)
                        
                        ### 1 segment included in another sta segment -> link -> V constraints possible
                        elif len(multiline)!=0: 
                            # print(">>> possible inclusion") #possible to have several included segment for this current multline, but on different segments.
                            # print(f"site:{num_site}", sta, multiline)
                            
                            for il1 in multiline.keys():
                                
                                for iml, il in multiline[il1]:
                                
                                    line_obj1 = np.array(list_multilinestrings[sta].geoms)[il1]
                                    node1 = self.get_node_sitename_linestring(site['siteId'], line_obj1)
                                    
                                    line_obj2 = np.array(list_multilinestrings[iml].geoms)[il]
                                    node2 = self.get_node_sitename_linestring(site['siteId'], line_obj2)
                                    
                                    if (node1 not in G_time_seg.nodes) or (node2 not in G_time_seg.nodes):
                                        logging.warning("[Add included time segment] Unknown node {node1} & {node2}")
                                    
                                    G_time_seg.add_edge(node1, node2)
                                    
                
                
                ###2. pbm solved by obs...
                if pbm_in_dc and not pbm_in_dc_obs: 
                    logging.warning(f" --- !! [Segment conflict solved using Obs solnV dates] site {site} !!--")
                    
                    for sta, multiline in dict_included.items(): #loop on basic soln data, only to get 'solved pbm'
                        
                        if len(find_common_values_ml(multiline))>0: #case of pbm...
                            list_pbm_solved.append(num_site)
                            num_pbm_solved +=1
                                
                            #generate pbm list LOG
                            pbm = get_pbms(df_v_disc, site, sub, sta, multiline, list_multilinestrings)                  
                            if site['siteId'] not in pbm_solved.keys():
                                pbm_solved[site['siteId']] = []
                                
                            pbm_solved[site['siteId']].append(pbm)
                    
                    for sta, multiline in dict_included_obs.items():
                        #multiline: list of dict -> simple list {0:[(1,2)], 1:[2,7]}
                        #### inclusion trouble: 2 or + inclusion from station1 to station2 >>>> no more V discontinuities !!  flag but do nothing on links
                        if len(find_common_values_ml(multiline))>0: #always pbm NOT POSSIBLE
                            raise ValueError(f'Always pbm obs {sta}')
                            
                        ### 1 segment included in another sta segment -> link -> V constraints possible
                        elif len(multiline)!=0: # pbm solved
                            # only plot the warning message
                            # print(">>> possible inclusion") #possible to have several included segment for this current multline, but on different segments.
                            # print(f"site:{num_site}", sta, multiline)
                            
                            for il1 in multiline.keys():
                                
                                for iml, il in multiline[il1]:
                                
                                    line_obj1 = np.array(list_multilinestrings_obs[sta].geoms)[il1]
                                    node1 = self.get_node_sitename_linestring(site['siteId'], line_obj1)
                                    
                                    line_obj2 = np.array(list_multilinestrings_obs[iml].geoms)[il]
                                    node2 = self.get_node_sitename_linestring(site['siteId'], line_obj2)
                                    
                                    if (node1 not in G_time_seg.nodes) or (node2 not in G_time_seg.nodes):
                                        logging.warning("[Add included time segment] Unknown node {node1} & {node2}")
                                    
                                    G_time_seg.add_edge(node1, node2)
                                       

        print(f"Num sites: {len(self.df_sites)}")
        print(f" ->num_pbm={num_pbm} {list_pbm}")
        print(f" ->num_pbm_solved={num_pbm_solved} {list_pbm_solved}")
        
        #### add edges btw station id & its node_v_name
        G_all = G_time_seg.copy()
        G_all.add_edges_from(list(zip(list(self.df_staId['node_v_name']), list(self.df_staId.index))))
        
        return G_all, G_time_seg, [pbm_names, pbm_solved] #WARNING: pbm names: code/pt/soln V (not P!)
    
    
    def build_absolute_const(self, const_time=1, const_nobs=20, G_relative_const=None):
        """
        Build Graph of Absolute constraints: all stations with:
            - nobs < const_nobs 
            - time length < const time

        Parameters
        ----------
        const_time : float, optional
            year minimum time length. The default is 1 (1 year)
        const_nobs : int, optional
            minimal number of observation. The default is 20.
        G_relative_const: networkx.Graph
            Graph with all relative constraints btw different stations (build from dist (build_snx_graph) & time (build_time_soln_graph))
            Allows to filter stations already linked with other on sites
        Returns
        -------
        None.

        """
        # compute years of measurments
        df_years = self.df_staId.copy()
        df_years["ydatastart"] = df_years["datastart"].apply(lambda row: date.from_tsnx(row).ydec())
        df_years["ydataend"] = df_years["dataend"].apply(lambda row: date.from_tsnx(row).ydec())
        
        
        #min & max dates
        df_cumul_y = df_years.groupby(['code', 'pt']).agg({'ydatastart': 'min', 'ydataend': 'max', 'nobs':'sum'}).reset_index()
        df_cumul_y["nyear_total"] = df_cumul_y["ydataend"] - df_cumul_y["ydatastart"]
        df_cumul_y = df_cumul_y.rename(columns={"nobs":"nobs_total"})
        #merge 
        df_years = pd.merge(df_years.reset_index()[['staId', 'code', 'pt']], df_cumul_y[['code', 'pt', 'nyear_total', 'nobs_total']], on=['code','pt'], how='left').set_index('staId')
        
        self.df_staId["nyear_total"] = df_years["nyear_total"]
        self.df_staId["nobs_total"] = df_years["nobs_total"]
        
        staId_absconst = list(df_years.loc[(df_years["nyear_total"]<const_time) & (df_years["nobs_total"]<const_nobs)].index)
        
        if type(G_relative_const) != type(None): #filter station already linked with other on site
            staId_absconst = [st for st in staId_absconst if st not in G_relative_const]
        
        return staId_absconst
        

    def check_inclusionML(self, multilinestrings):
        """
        For a list of MultiLine objects, get potential inclusion btw mutilines
        A Multiline: defined as a set of LineString
        
        Modelisation:
            Line: a time segment (i.e. 1 soln)
            MultiLine: set of Line (i.e. a station modelisation, with several solns)
        
        
        Parameters
        ----------
        multilinestrings : TYPE
            DESCRIPTION.

        Returns
        -------
        dict_included : TYPE
            DESCRIPTION.

        """
        # Iterate over each MultiLineString
        dict_included = {}

        for i, mls1 in enumerate(multilinestrings):
            # Iterate over each LineString within the MultiLineString
            dict_included[i] = {} #list of tuple MultilineString, Linstring
            #dict_equal[i] = []
            for j, ls1 in enumerate(mls1.geoms):
                # Convert the LineString to a shapely LineString object
                # Check against other MultiLineStrings
                for k, mls2 in enumerate(multilinestrings):
                    if k != i:  # Skip self-comparison
                        # Iterate over each LineString within the other MultiLineString
                        for j2, ls2 in enumerate(mls2.geoms):
                            # Convert the LineString to a shapely LineString object                                
                            # Check if ls1 is completely included in ls2
                            if ls1.within(ls2):
                                ##print(f"LineString {j} of MultiLineString {i} is completely included in LineString {j2} of MultiLineString {k}.")
                                #### LineString {j} of current MultiLineString {i} is completely included in LineString {j2} of MultiLineString {k}.
                                if j not in dict_included[i]:
                                    dict_included[i][j] =  []
                                    
                                dict_included[i][j].append((k, j2)) #### LineString {j} of current MultiLineString {i} is completely included in LineString {j2} of MultiLineString {k}.
                                    
        return dict_included

    def get_node_sitename_linestring(self, site_name, linestring, obs=False):
        """ 
         From site name (siteId) + LineString obj (composed of 2 points) --> builds the name of node 'segment_v
        Get name from self.df_staId (node_v_name) or (node_v_name_obs)
        '"""
        val= self.df_staId.loc[(self.df_staId['siteId']==site_name) & ((self.df_staId['segment_v']==linestring) | (self.df_staId['segment_v_obs']==linestring))]
        if obs: #name based on obs linestring (no -inf, +inf)
            val= val['node_v_name_obs'].values
        else:
            val= val['node_v_name'].values
            
        if len(val)==0:
            raise ValueError(f"Unknown site '{site_name}' & linestring '{linestring}'.")
        else:
            val = val[0]
            
        #return f'site_{site_name}_{list(linestring.coords)[0][0]}_{list(linestring.coords)[1][0]}'
        
        return val
    
    def build_graph_same_init_x0(self, vfconst=None, type_graph='VEL'):
        """
            Build graph snx SOLN + vfconst
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
    
    
    def minimum_linked(self, graph):
        
        def custom_sort(item):
            """Custom sort (soln>=10) """
            prefix = item[:5]
            number = int(item[5:]) #soln
            return (prefix, number)
        clusters = list(nx.connected_components(graph))

        # Create a new graph to store the minimum spanning trees
        graph_final_min_link = nx.Graph()
        dict_graph_final_min_link = {}

        # Create minimum spanning tree for each cluster and add edges to graph_final_min_link
        for cluster in clusters:
            if len(cluster) > 1: #at least 1 connected component  ---> if 1 sta, no link, we remove
                        
                # Create a subgraph for the current cluster
                subgraph = graph.subgraph(cluster)
                
                site = self.df_staId.loc[list(subgraph.nodes)[0], 'siteId']
                
                if site in dict_graph_final_min_link.keys(): #case of subsites
                    #logging.info(f"Sub-site: {site}")
                    num_subsite = len([s for s in dict_graph_final_min_link.keys() if site in s]) + 1
                    site = f"{site}_{num_subsite}"
                
                # Calculate minimum spanning tree for the subgraph
                #mst = nx.minimum_spanning_tree(subgraph) #>>> native python networkx method
                
                mst = nx.Graph()
                nodes = sorted(subgraph.nodes, key=custom_sort)
                for numn in range(len(nodes[:-1])):
                    mst.add_edge(nodes[numn], nodes[numn+1])
                
                # Add edges of the minimum spanning tree to mst_graph
                graph_final_min_link.add_edges_from(mst.edges)
                
                dict_graph_final_min_link[site] = mst
                
             
            
        return graph_final_min_link, dict_graph_final_min_link
    
    def plot_graph(self, G, with_labels=True):
        pp.figure()
        nx.draw(G, with_labels=with_labels)
        
        #edge label
        # edge_labels = dict([((n1, n2), f'{round(1000 * d["length"],3)} km' ) for n1, n2, d in G.edges(data=True)])
        
        if with_labels:
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
            for num, site in list_sites:
                #list of stations on this site
                list_sta = np.array(site) #site= [sta1, sta2...]
                #for this site, stations concerned
                select = np.array(in_datum[num]) #list of bool for "index" site: [True, False,..], same lenght that list_sta
                list_pbm += list_sta[select]
                logging.warning(f'Datum vs vfconst, error for vfconst site "{num}": multiple stations in datum: {list_sta[select]}')
                
        return valid_vf_datum, list_pbm  
    
    
    def valid_solns(self, solns):
        pass



    def write_pbm_file(self, pbm_names, out_file="list_pbm_solnV.txt", no_site=False):
        
        with open(out_file, "w") as fpbm: #overwrite potential existing file
            fpbm.write("## code pt solnV  datastart_solnV  dataend_solnV   -----actual time range in input data ---->  datastart_solnV_obs  dataend_solnV_obs     'id solnV segment'  (id_solnV_obs) ##\n")
            for siteId in pbm_names.keys():
                
                site = pbm_names[siteId]
                #fpbm.write("\n--------------------------------- SITE PROBLEM -----------------------------------\n")
                fpbm.write(f"\n{34*'>'} SITE PROBLEM {34*'>'}\n")
                
                for nump, pbm in enumerate(site):
                    fpbm.write(f">> Pbm n.{nump+1}:\n")
                    for num_pbm in pbm.keys(): 
                        if num_pbm > 0:
                            fpbm.write(f"{75*'.'}\n")
                            
                        for numl, line in enumerate(pbm[num_pbm]["code"][0]):
                            code=line[0]
                            pt=line[1]
                            solnV=line[2]
                            
                            dates = self.df_v_disc.loc[(self.df_v_disc['code']==code) & (self.df_v_disc['pt']==pt) & (self.df_v_disc['solnV']==solnV),['datastart','dataend']].values.reshape(-1)
                            dates_obs = self.df_v_disc_fit_dataObs.loc[(self.df_v_disc_fit_dataObs['code']==code) & (self.df_v_disc_fit_dataObs['pt']==pt) & (self.df_v_disc_fit_dataObs['solnV']==solnV),['datastart_obs','dataend_obs']].values.reshape(-1)
                            
                            id_solnV = pbm[num_pbm]["segment_v"][0][numl]
                            id_solnV_obs = pbm[num_pbm]["segment_v_obs"][0][numl]
                            
                            if no_site:
                                id_solnV=''
                                id_solnV_obs =''
                                
                         
                            if len(dates_obs) == 0: #no obs for this solnV time segment
                                dates_obs = [12*'*', 12*'*']
                            
                            fpbm.write(f"V- {code}{pt}{solnV} {dates[0]}  {dates[1]}   -->   {dates_obs[0]}  {dates_obs[1]}       {id_solnV:<50}  {id_solnV_obs}\n")
                         
                        fpbm.write("VS. \n")
                        for numl, line in enumerate(pbm[num_pbm]["code"][1]):
                            code=line[0]
                            pt=line[1]
                            solnV=line[2]
                            
                            dates = self.df_v_disc.loc[(self.df_v_disc['code']==code) & (self.df_v_disc['pt']==pt) & (self.df_v_disc['solnV']==solnV),['datastart','dataend']].values.reshape(-1)
                            dates_obs = self.df_v_disc_fit_dataObs.loc[(self.df_v_disc_fit_dataObs['code']==code) & (self.df_v_disc_fit_dataObs['pt']==pt) & (self.df_v_disc_fit_dataObs['solnV']==solnV),['datastart_obs','dataend_obs']].values.reshape(-1)
                            
                            id_solnV = pbm[num_pbm]["segment_v"][1][numl]
                            id_solnV_obs = pbm[num_pbm]["segment_v_obs"][1][numl]
                            
                            if no_site:
                                id_solnV=''
                                id_solnV_obs =''
                            
                            if len(dates_obs) == 0: #no obs for this solnV time segment
                                dates_obs = [12*'*', 12*'*']
                            fpbm.write(f"V- {code}{pt}{solnV} {dates[0]}  {dates[1]}   -->   {dates_obs[0]}  {dates_obs[1]}       {id_solnV:<50}  {id_solnV_obs}\n")      
                    if nump != len(site) -1:   
                        fpbm.write("\n")
                #fpbm.write("---------------------------------------------------------------------------------\n")
                fpbm.write(f"{82*'<'}\n\n")  