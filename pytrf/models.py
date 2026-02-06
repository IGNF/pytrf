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
    
    def load_time_series(self, file_path: str):
        """Charge une série temporelle"""
        self.r = ts.read(
            file_path,
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),
            dtrd=1,
            rotate=True
        )
        
        self._initialize_base_model()
        self.fit_model()  # ✅ AJOUT : Ajuste automatiquement
    
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
    
    def fit_model_iterative(self, events=None):
        """Ajuste le modèle de manière itérative"""
        if self.m is None:
            raise ValueError("No model initialized")
        
        # Ajouter les discontinuités
        if events:
            for e in events:
                if e.get("pos"):
                    print('time pos', e["time"])
                    # m.add_jumps(e["time"], deg=[0])
                if e.get("vel"):
                    print('time vel', e["time"])
                    # m.add_jumps(e["time"], deg=[1])
        
        self.m.fit_iter(finalize=True)
    
    def clean_time_series(self, threshold: float):
        """Nettoie la série temporelle"""
        if self.r is None:
            raise ValueError("No time series loaded")
        
        ts.clean_sigmas(self.r, threshold)