# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:42 2026

@author: loeva
"""

from pytrf.ts import ts, model
from pytrf.io import read_solns
from pytrf import date

class TimeSeriesManager:
    """
    Manager for time series and models.
    
    This class handles the loading, processing, and modeling of time
    series data. It provides methods for loading time series from files,
    initializing and fitting models, managing discontinuities (jumps),
    and cleaning outliers.
    
    Attributes
    ----------
    r : pytrf.ts.ts or None
        Time series object containing the observational data
    m : pytrf.ts.model or None
        Model object representing the fitted model
    
    Examples
    --------
    >>> manager = TimeSeriesManager()
    >>> manager.load_time_series('station_igs.xyz')
    >>> manager.fit_model()
    """
    
    def __init__(self):
        self.r = None  # temporal serie
        self.m = None  # Model
        
    def load_time_series(self, file_path: str, auto_fit=True):
        """
        Load a time series from a file.
        
        Reads a GNSS time series file in XYZ format and performs initial
        cleaning of data with large uncertainties. Optionally initializes
        and fits a base model.
        
        Parameters
        ----------
        file_path : str
            Path to the time series file (typically in IGS XYZ format)
        auto_fit : bool, optional
            If True, automatically initialize and fit a base model after
            loading (default is True)
        """
        
        self.r = ts.read(
            file_path,
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            format=('t', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'cxy', 'cxz', 'cyz'),
            dtrd=1,
            rotate=True
        )
        self.r.clean_sigmas()
        
        if auto_fit:
            self._initialize_base_model()
            self.fit_model() 
    
    def _initialize_base_model(self):
        """
        Initialize a base model.
        
        Notes
        -----
        This is a private method automatically called by load_time_series
        when auto_fit=True. Does nothing if no time series is loaded.
        """
        
        if self.r is None:
            return
        
        self.m = model(self.r)
        self.m.add_polynom(deg=0)
        self.m.add_polynom(deg=1)
        self.m.add_sine(per=365.25)
        self.m.add_sine(per=182.625)
        self.m.add_vw()
    
    def load_model_from_solns(self, discontinuity_path: str, station_code: str):
        """
        Load a model from a SINEX discontinuities file.
        
        Reads discontinuities (position and velocity jumps) from a SINEX
        solutions file and creates a model incorporating these jumps along
        with standard components (polynomials, seasonal terms, noise).
        
        Parameters
        ----------
        discontinuity_path : str
            Path to the SINEX file containing discontinuity information
        station_code : str
            Four-character station code to extract from the SINEX file
        
        Raises
        ------
        ValueError
            If no time series has been loaded
        """
        
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
        """
        Fit the model, for further information go to ts.fit()
        """
        if self.m is None:
            raise ValueError("No model initialized")
        
        self.m.fit(finalize=True)
    
    
    def fit_model_iterative(self, dates=None, pos_checked=None, vel_checked=None):
        """
        Fit the model iteratively with discontinuity management.
        
        Performs an iterative fit that efficiently adds new jumps or removes
        existing ones based on the provided discontinuity flags. 
        
        Parameters
        ----------
        dates : array-like of numpy.datetime64, optional
            Dates of potential discontinuities
        pos_checked : array-like of bool, optional
            Boolean flags indicating which dates should have position jumps.
            Must have the same length as dates.
        vel_checked : array-like of bool, optional
            Boolean flags indicating which dates should have velocity jumps.
            Must have the same length as dates.
        
        Raises
        ------
        ValueError
            If no model has been initialized
        """
        if self.m is None:
            raise ValueError("No model initialized")
       
        # list to group date by type
        pos_dates_mjd = []
        vel_dates_mjd = []
       
        # Time series start and end dates
        tmin = self.r.t[0]
        tmax = self.r.t[-1]
        
        # List of position and vel discontinuities
        if dates is not None and len(dates) > 0:
            for i in range(len(dates)):
                
                # convert date in iso format to mjd
                date_str = str(dates[i])
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
        """
        Clean the time series by removing outliers.
        
        Parameters
        ----------
        threshold : float
            Threshold value for cleaning. Observations with sigmas greater
            than this value are removed.
        """
        if self.r is None:
            raise ValueError("No time series loaded")
        
        self.r.clean_sigmas(thr=threshold)