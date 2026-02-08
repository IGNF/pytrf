# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:42 2026

@author: loeva
"""

from pytrf.ts import ts, model
from pytrf.io import read_solns
from pytrf import date

class TimeSeriesManager:
    """Gère les séries temporelles et modèles"""
    
    def __init__(self):
        self.r = None  # Série temporelle
        self.m = None  # Modèle
        
    def load_time_series(self, file_path: str, auto_fit=True):
        """Charge une série temporelle"""
        self.r = ts.read(
            file_path,
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),
            dtrd=1,
            rotate=True
        )
        
        if auto_fit:
            self._initialize_base_model()
            self.fit_model() 
    
    def _initialize_base_model(self):
        """Initialise un modèle de base"""
        if self.r is None:
            return
        
        self.m = model(self.r)
        self.m.add_polynom(deg=0)
        self.m.add_polynom(deg=1)
        self.m.add_sine(per=365.25)
        self.m.add_sine(per=182.625)
        self.m.add_vw()
    
    def load_model_from_solns(self, discontinuity_path: str, station_code: str):
        """Charge un modèle depuis un fichier SOLNS"""
        if self.r is None:
            raise ValueError("No time series loaded")
        
        solns = read_solns(discontinuity_path)
        
        self.m = model.from_solns(
            self.r, 
            solns, 
            code=station_code,
            per=[365.25, 182.625],
            noise=['vw']
        )
    
    def fit_model(self):
        """Ajuste le modèle"""
        if self.m is None:
            raise ValueError("No model initialized")
        
        self.m.fit(finalize=True)
    
    # def fit_model_iterative(self, events=None):
    #     """Ajuste le modèle de manière itérative"""
    #     if self.m is None:
    #         raise ValueError("No model initialized")
        
    #     # Ajouter les discontinuités
    #     if events:
    #         for e in events:
    #             if e.get("pos"):
    #                 print('time pos', e["time"])
    #                 # m.add_jumps(e["time"], deg=[0])
    #             if e.get("vel"):
    #                 print('time vel', e["time"])
    #                 # m.add_jumps(e["time"], deg=[1])
        
    #     self.m.fit_iter(finalize=True)
    
    def fit_model_iterative(self, dates=None, pos_checked=None, vel_checked=None):
        """Ajuste le modèle de manière itérative
       
       Args:
           dates: numpy array of datetime64 (dates des événements)
           pos_checked: numpy array de bool (True si discontinuité de position)
           vel_checked: numpy array de bool (True si discontinuité de vitesse)
        """
        if self.m is None:
            raise ValueError("No model initialized")
       
        # Listes pour regrouper les dates par type
        pos_dates_mjd = []
        vel_dates_mjd = []
       
        # Time series start and end dates
        tmin = self.r.t[0]
        tmax = self.r.t[-1]
        
        print('t_min et t_max', tmin, tmax)
        
        # Full list of position and vel discontinuities
        if dates is not None and len(dates) > 0:
            for i in range(len(dates)):
                # convert datetime64 to MJD
                date_str = str(dates[i])
                print(date_str)
                mjd = date.from_tiso(date_str).mjd
                if pos_checked[i] and (mjd > tmin) and (mjd < tmax):
                    pos_dates_mjd.append(mjd)
                   
                if vel_checked[i] and (mjd > tmin) and (mjd < tmax):
                    vel_dates_mjd.append(mjd)
        if pos_dates_mjd:
            print(f'Adding {len(pos_dates_mjd)} position jump(s): {pos_dates_mjd}')
            self.m.add_jumps(deg=[0], t=pos_dates_mjd)
       
        if vel_dates_mjd:
            print(f'Adding {len(vel_dates_mjd)} velocity jump(s): {vel_dates_mjd}')
            self.m.add_jumps(deg=[1], t=vel_dates_mjd)
  
        self.m.fit_iter()
        print('le fit iter a fonctionnés')
   
   
    
    def clean_time_series(self, threshold: float):
        """Nettoie la série temporelle"""
        if self.r is None:
            raise ValueError("No time series loaded")
        
        ts.clean_sigmas(self.r, threshold)