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
    A station: a node
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
        
    def __init__(self, snx=None, solns=None, df_sta=None):
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
        self.snx = None
        self.solns = None
        self.df_sta = None
        
        self.df_staId, self.df_sites = None, None
        
        
        if type(snx) != type(None): #init from snx object
            self.snx = snx
            #build id dataframes : station & sites (only base on DOMES 5 first chr)
            self.df_staId = self.generate_staId_from_snx()

        if type(solns) != type(None):
            self.solns = solns
            self.df_solns, self.df_v_disc = self.generate_staId_from_solns()
            
        if type(df_sta) != type(None):
            self.df_sta = df_sta
         
        if type(self.df_sta) != type(None) and type(self.solns) != type(None): ###init df_staId & df_sites
            #build StaId using in priority station in 'df_sta'
            self.df_staId = pd.merge(self.df_solns.reset_index(), self.df_sta[['code', 'pt', 'X', 'Y', 'Z', 'domes']], on=['code', 'pt'], how='right').set_index('staId')
            #set node name according to segment_v + site (domes[:5])
            self.df_staId['node_v_name'] = self.df_staId.apply(lambda row: self.get_node_sitename_linestring(row['domes'][:5], row['segment_v']), axis=1)
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
 
        
        self.df_sites = self.generate_sites_domes(self.df_staId)    
              
    def generate_staId_from_snx(self):
        """ DataFrame summarizing stations in self.snx file
            Columns : ['domes', 'code', 'pt', 'soln', 'datastart', 'dataend', '_staId_pytrf', 'X', 'Y', 'Z']
        """
        df_domes_snx = pd.DataFrame([(sta.domes, sta.code, sta.pt, soln.soln, soln.datastart, soln.dataend) for sta in self.snx.sta for soln in sta.soln], columns=["domes", "code", "pt", "soln", "datastart", "dataend"])
        df_domes_snx["_staId_pytrf"] = df_domes_snx["code"]+ df_domes_snx["pt"]+ df_domes_snx["soln"]
        df_domes_snx["staId"] = df_domes_snx.apply(lambda row: (row["code"]+ row["pt"]+ row["soln"]).replace(" ",""), axis=1) #no space
        df_domes_snx = df_domes_snx.set_index("staId")
        
        #coordinates
        coords = self.snx.get_xyz(df_domes_snx["code"].values.tolist(), pt=df_domes_snx["pt"].values.tolist(),  soln=df_domes_snx["soln"].values.tolist())
        df_domes_snx["X"], df_domes_snx["Y"], df_domes_snx["Z"] = coords[:,0], coords[:,1], coords[:,2]
        
        return df_domes_snx
    
    
    def generate_staId_from_solns(self):
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
                df_v_disc.loc[numV, 'soln'] = solV.soln
                
                ### mjd date
                if solV.start == '00:000:00000':
                    df_v_disc.loc[numV, 'mjd_datastart'] = -1 #-inf
                else:
                    df_v_disc.loc[numV, 'mjd_datastart'] = date.from_tsnx(solV.start).mjd
                    
                if solV.end == '00:000:00000':
                    df_v_disc.loc[numV, 'mjd_dataend'] = 999999 #+inf
                else:
                    df_v_disc.loc[numV, 'mjd_dataend'] = date.from_tsnx(solV.end).mjd
        
                numV+=1
                
                
        df_solns['_staId_pytrf'] = df_solns['code'] + df_solns['pt'] + df_solns['soln']
        df_solns['staId'] = df_solns.apply(lambda row: (row["_staId_pytrf"].replace(" ","")), axis=1)
        
        df_solns = df_solns.set_index("staId")
                
        return df_solns, df_v_disc
    
    
    def generate_sites_domes(self, df_domes_snx):
        """From df_staId, generate df_sites using domes 5chr"""
        
        if 'domes' not in df_domes_snx.columns or type(df_domes_snx) == type(None):
            return None
        
        #copy 
        df_domes_snx = df_domes_snx.copy()
        ### Sites according 5 domes character
        # Select the first 5 characters of the "domes" column
        df_domes_snx['group_key'] = df_domes_snx['domes'].str[:5]
        
        # Group by the first 5 characters and aggregate into a list with respective index values
        df_sites = df_domes_snx.groupby('group_key').agg(staId=('domes', lambda x: x.index.tolist())).reset_index()
        
        #### remove site without domes: '-----'
        df_sites = df_sites[df_sites['group_key'] != '-----'].reset_index(drop=True)
              
        return df_sites
        
      
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
        sub_complete_graphs = [nx.complete_graph(subgraph.nodes) for subgraph in subgraphs] #list of graphes
        
        # Accumulate all subgraphs into a single graph
        single_graph = nx.union_all(sub_complete_graphs)
        
        return single_graph, sub_complete_graphs

            
    
    def build_time_soln_graph(self):
        """
        Builds graph based on TIME relations btw stations SOLNS.
        
        Strategy:
        ---------
        1. Builds a graph based on solnV relations (G_time_seg)
           -node: segment solnV
           -edge: constraints
        
        There are 3 possibilites on each DOMES site (default: add segment solnV (node) without constraints (edges))
            -> same V solns -> same discontinuities (default: add nodes but not edges btw solnV segments)
            - no same solnV segment:
                -> 2 solnV segment of a station included in 1 solnV segment of another station: CONFLIT -> get station(s) name(s) to user (pbm_names) -> (default: add nodes but not edges btw solnV segments)
                -> 1 solnV segment of a station included in 1 solnV segment of another station: same V -> ADD CONSTRAINTS (edge) btw these 2 nodes.

        2. Builds a graph with stationsId (code/pt/solnP from self.df_staId) linked to their solnV node (1 station necessary linked to 1 node solnV <> 1 node solnV may be linked to 0--n stations)


        ==> Final G_all graph: nodes (staId + solnV), edges (constraints btw staId, neighbouring thanks to solnV nodes)
        """
        
        self.df_v_disc["segment"] = self.df_v_disc.apply(lambda row: LineString([(row['mjd_datastart'], 0), (row['mjd_dataend'], 0)]), axis=1)
        
        df = pd.merge(self.df_staId.reset_index(), self.df_v_disc[['code', 'pt','segment']], on=['code', 'pt'], how='left').set_index("staId")
        
        self.df_grouped = df.groupby('staId')['segment'].agg(list).to_frame()
        self.df_grouped["segment_multi"] = self.df_grouped["segment"].apply(lambda x: MultiLineString(x)) #dataframe with saId as index & list of segment
        
        num_pbm = 0
        list_pbm = []
        pbm_names = []
        
        # Create a graph using networkx
        G_time_seg = nx.Graph()
        for num_site, site in tqdm.tqdm(self.df_sites.iterrows(), desc='Building time graph...', total=len(self.df_sites)):
            
            sub = self.df_grouped.loc[site['staId']] #on site, all V discontinuities time segment
            nodes_seg_names = pd.unique([f'site_{site["group_key"]}_{list(line.coords)[0][0]}_{list(line.coords)[1][0]}' for line in sub['segment'].explode()])
            
            #add all temporal segments as nodes
            G_time_seg.add_nodes_from(nodes_seg_names, color='red') 
            
            ### add links btw nodes_seg_names?
            if sub["segment_multi"].nunique() == 1: #All values in V 'segment' are identical >>> same discontinuities, all right, no needs to link 'node_seg_names'
                #logging.info(f"Site {site['group_key']} OK.")
                pass
       
            else:
                #logging.warning(f"No consistent V discontinuities on site {num} >> {self.df_sites.loc[num, ['group_key', 'staId']].values}")
                list_multilinestrings_pbm = list(pd.unique(sub["segment_multi"]))
                
                dict_included, dict_equal, dict_included_idinput = self.check_inclusionML(list_multilinestrings_pbm)
                
                for sta, multiline in dict_included.items():
                    
                    #### inclusion trouble: 2 or + inclusion from station1 to station2 >>>> no more V discontinuities !!  flag but do nothing on links
                    if len(set(multiline)) != len(multiline):
                        #logging.warning("-----------------------")
                        # logging.warning(f"PBM {num_site} >> {self.df_sites.loc[num_site, ['group_key', 'staId']].values}")
                        # logging.warning(f" >>> {list_multilinestrings_pbm}, {sta}, {multiline}")
                        list_pbm.append(num_site)
                        num_pbm +=1
                        
                        indexes_of_duplicates = np.array([dict_included_idinput[sta][index] for index, item in enumerate(multiline) if multiline.count(item) > 1])
                        LineString_pbm1 = np.array(list_multilinestrings_pbm[sta].geoms)[indexes_of_duplicates]
                        multline_pbm1 = np.array(list_multilinestrings_pbm[sta])
                        
                        id_ml_pbms2 = np.array(list(set([ml2 for (ml2, l2) in multiline if multiline.count((ml2, l2)) > 1])))
                        id_li_pbms2 = np.array(list(set([l2 for (ml2, l2) in multiline if multiline.count((ml2, l2)) > 1])))
                        multline_pbm2 = np.array(list_multilinestrings_pbm)[np.array(id_ml_pbms2)][0]
                        
                        LineString_pbm2 = np.array(multline_pbm2.geoms)[id_li_pbms2]
                        
                        
                        list_staId_pbm1 = list(sub.loc[sub['segment_multi']==multline_pbm1].index)
                        list_staId_pbm1 = [(sta[:4], f" {sta[4]}") for sta in list_staId_pbm1]
                        c1, pt1 = zip(*list_staId_pbm1)
                        df_v_disc1 = self.df_v_disc[(self.df_v_disc['code'].isin(c1)) & (self.df_v_disc['pt'].isin(pt1))]
                        list_staId_pbm2 = list(sub.loc[sub['segment_multi']==multline_pbm2].index)
                        list_staId_pbm2 = [(sta[:4], f" {sta[4]}") for sta in list_staId_pbm2]
                        c2, pt2 = zip(*list_staId_pbm2)
                        
                        df_v_disc2 = self.df_v_disc[(self.df_v_disc['code'].isin(c2)) & (self.df_v_disc['pt'].isin(pt2))]
                        
                        #logging.warning(f" >>>{multline_pbm1}  {multline_pbm2}")
                        #logging.warning(f" >>>{list(sub.loc[sub['segment_multi']==multline_pbm1].index)} VS {list(sub.loc[sub['segment_multi']==multline_pbm2].index)}")
                        #logging.warning(f" >>>{LineString_pbm1} VS {LineString_pbm2}")
                        
                        # solns of "df_v_disc1" > refers to soln V (Velocity!) 
                        v_disc1 = df_v_disc1.loc[df_v_disc1['segment'].isin(LineString_pbm1)][['code','pt','soln']].values.tolist()
                        v_disc2 = df_v_disc2.loc[df_v_disc2['segment'].isin(LineString_pbm2)][['code','pt','soln']].values.tolist()
                        
                        logging.warning(f" >>>{v_disc1} VS {v_disc2}")
                        
        
                        pbm_names.append([[v_disc1, v_disc2],
                                          [[self.get_node_sitename_linestring(site['group_key'], lpb1) for lpb1 in  LineString_pbm1], [self.get_node_sitename_linestring(site['group_key'], lpb2) for lpb2 in  LineString_pbm2]]])
                    
                    ### 1 segment included in another sta segment -> link -> V constraints possible
                    elif len(multiline)!=0: 
                        # print(">>> possible inclusion") #possible to have several included segment for this current multline, but on different segments.
                        # print(f"site:{num_site}", sta, multiline)
                        
                        for iml, il in multiline:
                            
                            il1 = dict_included_idinput[sta][multiline.index((iml, il))]
                            line_obj1 = np.array(list_multilinestrings_pbm[sta].geoms)[il1]
                            node1 = self.get_node_sitename_linestring(site['group_key'], line_obj1)
                            
                            line_obj2 = np.array(list_multilinestrings_pbm[iml].geoms)[il]
                            node2 = self.get_node_sitename_linestring(site['group_key'], line_obj2)
                            
                            if (node1 not in G_time_seg.nodes) or (node2 not in G_time_seg.nodes):
                                logging.warning("[Add included time segment] Unknown node {node1} & {node2}")
                            
                            G_time_seg.add_edge(node1, node2)
                           

        print(f"Num sites: {len(self.df_sites)} ->num_pbm={num_pbm} {list_pbm}")
        
        #### add edges btw station id & its node_v_name
        G_all = G_time_seg.copy()
        G_all.add_edges_from(list(zip(list(self.df_staId['node_v_name']), list(self.df_staId.index))))
        
        return G_all, G_time_seg, pbm_names #WARNING: pbm names: code/pt/soln V (not P!)
        

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
        dict_equal : TYPE
            DESCRIPTION.
        dict_included_idinput : TYPE
            DESCRIPTION.

        """
        # Iterate over each MultiLineString
        dict_included = {}
        dict_included_idinput = {}
        dict_equal = {}
        for i, mls1 in enumerate(multilinestrings):
            ##print(f"\nMultiLineString {i}:")
            # Iterate over each LineString within the MultiLineString
            dict_included[i] = [] #list of tuple MultilineString, Linstring
            dict_included_idinput[i] = []
            dict_equal[i] = []
            for j, ls1 in enumerate(mls1.geoms):
                # Convert the LineString to a shapely LineString object
                inclusion_found = False
                # Check against other MultiLineStrings
                for k, mls2 in enumerate(multilinestrings):
                    if k != i:  # Skip self-comparison
                        # Iterate over each LineString within the other MultiLineString
                        for j2, ls2 in enumerate(mls2.geoms):
                            # Convert the LineString to a shapely LineString object
                            if ls1 == ls2:
                                ##print(f"LineString {j} of MultiLineString {i} is EQUAL LineString {j2} of MultiLineString {k}.")
                                dict_equal[i].append((j2,k))
                                inclusion_found = True
                                break
                                
                            # Check if ls1 is completely included in ls2
                            elif ls1.within(ls2):
                                ##print(f"LineString {j} of MultiLineString {i} is completely included in LineString {j2} of MultiLineString {k}.")
                                dict_included[i].append((k, j2)) #### LineString {j} of current MultiLineString {i} is completely included in LineString {j2} of MultiLineString {k}.
                                dict_included_idinput[i].append(j)
                                
                                inclusion_found = True
                                break
                        if inclusion_found:
                            break
                        
        return dict_included, dict_equal, dict_included_idinput

    def get_node_sitename_linestring(self, site_name, linestring):
        """ 
            From site name (DOMES 5chr) + LineString obj (composed of 2 points) --> builds the name of node 'segment_v
        '"""
        return f'site_{site_name}_{list(linestring.coords)[0][0]}_{list(linestring.coords)[1][0]}'
    
    
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