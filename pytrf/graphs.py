# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:31 2026

@author: loeva
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
from pytrf import date
from math import ceil, exp, log
from scipy import signal

try:
    from scipy.signal import gaussian
except:
    from scipy.signal.windows import gaussian

class TSGraph(FigureCanvas):
    """Graphique de série temporelle"""
    
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.subplots(nrows=3, ncols=1, sharex=False)
        super().__init__(fig)
        self.setParent(parent)
    
    def plot_data(self, r, discontinuities=None, model=None):
        """Trace les données"""
        x = TSGraph.mjd_to_datetime(r.t)
        y = r.y * 1e3
        labels = ['East[mm]', 'North[mm]', 'Up[mm]']
        
        for i in range(3):
            self.axes[i].clear()
            self.axes[i].errorbar(
                x, y[:, i], 
                yerr=np.sqrt(r.Q[:, i, i] * 1e6), 
                fmt='.',
                ms=1,
                mec='black',
                mfc='black',
                ecolor='black',
                elinewidth=0.5,
                zorder=10
            )
            
            self.axes[i].grid()
            self.axes[i].set_ylabel(labels[i], fontsize=8)
            self.axes[i].tick_params(axis='both', labelsize=7)
        
        # Tracer le modèle
        if model is not None:
            for d in range(min(3, model.nd)):
                self.axes[d].plot(x, model[d].yc * 1e3, 'r', linewidth=2, zorder=14)
                
                if model[d].sc is not None:
                    self.axes[d].fill_between(
                        x,
                        (model[d].yc - model[d].sc) * 1e3,
                        (model[d].yc + model[d].sc) * 1e3,
                        alpha=0.4,
                        zorder=4
                    )
        
        # Tracer les discontinuités
        if discontinuities:
            for ax in self.axes:
                for d in discontinuities:
                    if d[1] == 'antenna change':
                        color = 'blue'
                    elif d[1] == 'receptor change':
                        color = 'cyan'
                    elif d[1] == 'earthquake':
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    ax.axvline(
                        d[0],
                        color=color,
                        linestyle='-',
                        linewidth=1,
                        alpha=0.7,
                        zorder=20
                    )
        
        self.figure.tight_layout()
        self.draw()
    
    @staticmethod
    def mjd_to_datetime(mjd):
        """Convertit MJD en datetime"""
        x = np.datetime64('1858-11-17') + (mjd * np.timedelta64(1, 'D'))
        return x


class Graph(FigureCanvas):
    """Graphique générique (résidus, périodogramme)"""
    
    def __init__(self, parent=None, width=5, height=6, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.subplots(nrows=3, ncols=1, sharex=False)
        super().__init__(fig)
        self.setParent(parent)
    
    def plot_raw_res(self, m, thr_raw=None, tunit=None, dims=None, title=None):
        """Adapted from pytrf.ts.plot_res"""
        if m is None or m.r is None:
            return
    
        # Time
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            t = m.r.t
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            from pytrf import date
            t = np.array([date.from_mjd(d).ydec() for d in m.r.t])
        else:
            tunit = m.r.tunit
            t = m.r.t
    
        # Component name
        if dims is None:
            dims = m.r.dims
        if dims is None:
            dims = [f'Component {d+1}' for d in range(m.nd)]
        if isinstance(dims, str):
            dims = [dims]
    
        # Tracé
        for d in range(m.nd):
            ax = self.axes[d]
            ax.clear()
            ax.margins(0.01, 0.01)
            ax.grid(zorder=0)
    
            ax.set_ylabel(f"{dims[d]} residuals [{m.r.yunit}]")
            ax.errorbar(
                t,
                m[d].v,
                yerr=m[d].sv,
                fmt='.k',
                ecolor='gray',
                zorder=3
            )
    
            # Seuil
            if thr_raw is not None:
                thr = thr_raw * m[d].wrms
                ind = np.nonzero(np.abs(m[d].v) > thr)[0]
    
                if len(ind) > 0:
                    ax.errorbar(
                        t[ind],
                        m[d].v[ind],
                        yerr=m[d].sv[ind],
                        fmt='.r',
                        ecolor='orange',
                        zorder=4
                    )
    
                ax.plot([t[0], t[-1]], [thr, thr], '--r', linewidth=2)
                ax.plot([t[0], t[-1]], [-thr, -thr], '--r', linewidth=2)
    
        self.axes[-1].set_xlabel(f"Time [{tunit}]")
    
        if title:
            self.figure.suptitle(title)
    
        self.figure.tight_layout()
        self.draw()
        
    def plot_norm_res(self, m, thr_norm=None, tunit=None, dims=None, title=None):

        """
        Adapted from pytrf.ts.plot_normres()
        """
        if m is None or m.r is None:
             return
 
        # Time
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            t = m.r.t
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            t = np.array([date.from_mjd(d).ydec() for d in m.r.t])
        else:
            tunit = m.r.tunit
            t = m.r.t

        # Component names
        if (dims is None):
            dims = m.r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(m.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Loop over components
        for d in range(m.nd):
            ax = self.axes[d]
            ax.margins(0.01, 0.01)
            ax.grid(zorder=0)
            ax.set_ylabel(dims[d] + ' norm. res.')
            ax.plot(t, m[d].vn, '.k', zorder=3)
            if (thr_norm is not None):
                ind = np.nonzero(np.abs(m[d].vn) > thr_norm)[0]
                if (len(ind) > 0):
                    ax.plot(t[ind], m[d].vn[ind], '.r', zorder=4)
                ax.plot([t[0], t[-1]], [thr_norm, thr_norm], '--r', linewidth=2)
                ax.plot([t[0], t[-1]], [-thr_norm, -thr_norm], '--r', linewidth=2)
        ax.set_xlabel('Time ['+tunit+']')
        
        self.axes[-1].set_xlabel(f"Time [{tunit}]")
     
        if title:
             self.figure.suptitle(title)
     
        self.figure.tight_layout()
        self.draw()
        
    def plot_psd(self, m, smooth=1, figsize=None, tunit=None, dims=None, title=None, output=None, show=True):

        """
        Plot PSD of fit residuals and of noise model
        Adapted from pytrf.ts.plot_psd
        """

        if m is None or m.r is None:
             return
 

        # Time unit and frequencies
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            fr = m[0].fr
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            fr = 365.25 * m[0].fr
        else:
            tunit = m.r.tunit
            fr = m[0].fr

        # Frequency unit
        if (tunit == 'd'):
            funit = 'cpd'
        else:
            funit = 'cp'+tunit

        # Component names
        if (dims is None):
            dims = m.r.dims
        if (dims is None):
            dims = ['Component '+str(d+1) for d in range(m.nd)]
        if (isinstance(dims, str)):
            dims = [dims]

        # Loop over components
        for d in range(m.nd):
            ax = self.axes[d]
            ax.margins(0.01, 0.01)
            ax.set_ylabel(dims[d]+' res. PSD ['+m.r.yunit+'^2/'+funit+']')

            # Smoothed PSD of residuals
            pv = m[d].pv
            if (smooth is not None):
                w = gaussian(2*ceil(3*smooth)+1, smooth)
                pv = signal.convolve(pv, w/np.sum(w), mode='same')

            # Useful things
            fm = exp((log(fr[0])+log(fr[-1]))/2)
            pm = np.mean(m[d].pv)
            pmax = np.max([np.max(pv), np.max(m[d].pn)])
            pmin = np.min([np.min(pv), np.min(m[d].pn)])
            
            # WN, FN and RW
            ax.loglog(fr, pm*np.ones(len(fr)), color='#b0b0b0', linewidth=0.8, zorder=0)
            p = pm * (fm/fr)
            p[p > pmax] = np.inf
            p[p < pmin] = -np.inf
            ax.loglog(fr, p, color='#b0b0b0', linewidth=0.8, zorder=0)
            p = pm * (fm/fr)**2
            p[p > pmax] = np.inf
            p[p < pmin] = -np.inf
            ax.loglog(fr, p, color='#b0b0b0', linewidth=0.8, zorder=0)
            
            # Residuals and noise model spectra
            ax.loglog(fr, pv, 'k', zorder=3)
            ax.loglog(fr, m[d].pn, 'r', linewidth=2, zorder=4)
            if (m[d].spn is not None):
                ax.fill_between(fr, m[d].pn-m[d].spn, m[d].pn+m[d].spn, color='r', alpha=0.6, zorder=4)
            
            ## Debug
            #for n in m[d].n:
                #ax.loglog(fr, n.P, '--r', zorder=4)
            #for f in m[d].f:
                #if isinstance(f, sine):
                    #ax.loglog([365.25/f.per, 365.25/f.per], [pmin, pmax], '--r', zorder=0)
            
        ax.set_xlabel('Frequency ['+funit+']')
        self.axes[-1].set_xlabel(f"Time [{tunit}]")
     
        if title:
             self.figure.suptitle(title)
     
        self.figure.tight_layout()
        self.draw()
       