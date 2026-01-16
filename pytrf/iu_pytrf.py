# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:09:06 2026

@author: loeva
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 20:29:59 2025

@author: loeva
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QFileDialog,
    QTextEdit, QDialog, QMessageBox, QGridLayout, QTableWidget, QLineEdit,
    QTableWidgetItem, QCheckBox, QRadioButton
)
from PyQt6.QtCore import pyqtSignal
import sys

from platformdirs import user_cache_dir

import os

from pytrf.io import read_yaml, write_yaml, read_solns, get_sitelog, read_sitelog
from pytrf.ts import ts, model
from pytrf import date


from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.r = None #object of ts.read
        self.m = None
        
        # Cache utilisateur
        #self.cache_dir = user_cache_dir(appname="Pytrf",appauthor=None)
        #os.makedirs(self.cache_dir, exist_ok=True)
        """
        The temporary file is created at this place : 
        Linux : ~/.cache/Pytrf/
        Windows : C:\ Users\...\AppData\Local\Pytrf\Cache
        """

        #print("Cache dir:", self.cache_dir)
        
        # # Répertoire de données avec platformdirs
        # self.data_dir = user_data_dir("MonLogiciel", "MonNom")
        # os.makedirs(self.data_dir, exist_ok=True)
        # print("Répertoire de données :", self.data_dir)

    def initUI(self):
        self.setWindowTitle("Pytrf")
        #self.setGeometry(800, 100, 500, 300)

        # --- Central widget ---
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        pannels_layout = QHBoxLayout()
       
        # --- Top buttons ---
        button_layout = QHBoxLayout()

        btn_new = QPushButton("New Project")
        btn_open = QPushButton("Open Project")
        btn_modify = QPushButton("Modify Project")

        btn_new.clicked.connect(self.newProject)
        btn_open.clicked.connect(self.openProject)
        btn_modify.clicked.connect(self.modifyProject)

        button_layout.addWidget(btn_new)
        button_layout.addWidget(btn_open)
        button_layout.addWidget(btn_modify)
        button_layout.addStretch() 

        main_layout.addLayout(button_layout)
        

        # ---- Control pannel (left pannel)----
        
        self.control_panel = QTabWidget()

        # Create tabs
        det_model_tab = QWidget()
        sto_model_tab = QWidget()

        # --- Deterministic model tab ---
        det_model_layout = QVBoxLayout()
        det_model_tab.setLayout(det_model_layout)  
        
        self.date_table = Tabledate()
        det_model_layout.addWidget(self.date_table)
        
        btn_add_date = QPushButton('+')
        det_model_layout.addWidget(btn_add_date)
        btn_add_date.clicked.connect(self.add_date)
        
        
        btn_adjust_model = QPushButton('Adjust the model')
        btn_adjust_model.clicked.connect(self.adjust_model)
        btn_save = QPushButton('SAVE')
        btn_save.setStyleSheet("background-color : red")
        btns_layout = QHBoxLayout()
        btns_layout.addWidget(btn_adjust_model)
        btns_layout.addWidget(btn_save)
        det_model_layout.addLayout(btns_layout)
        
        # --- Stochastic model tab ---
        sto_model_layout = QVBoxLayout()
        sto_model_layout.addWidget(QLabel("holà"))
        sto_model_tab.setLayout(sto_model_layout)

        # Add tabs
        self.control_panel.addTab(det_model_tab, "Deterministic model")
        self.control_panel.addTab(sto_model_tab, "Stochastic model")
        
      
        
        pannels_layout.addWidget(self.control_panel)
        
        
        # ---- Middle pannel ----
        self.combobox_ts = ComboBoxTS()
        self.sbox_threshold = QDoubleSpinBox()
        self.sbox_threshold.setValue(5)
        self.btn_remove_pts = QPushButton("Remove points with large errors")
        self.btn_remove_pts.clicked.connect(self.remove_points)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.combobox_ts)
        
        # add matplotlib graph
        self.canvas = TSGraph(self, width=5, height=6, dpi=100)
        middle_layout.addWidget(NavigationToolbar(self.canvas, self)) 
        self.combobox_ts.currentIndexChanged.connect(self.update_ts_graph)
        middle_layout.addWidget(self.canvas)
        
        middle_layout.addWidget(self.text_edit)
        threshold_layout = QHBoxLayout()
        
        threshold_layout.addWidget(self.btn_remove_pts)
        threshold_layout.addWidget(self.sbox_threshold)      
        
        middle_layout.addLayout(threshold_layout)
        pannels_layout.addLayout(middle_layout)
        main_layout.addLayout(pannels_layout)
        
        
        # ---- Right pannel ----
        
        # self.right_pannel = QTabWidget()
        # # Create tabs
        # residuals_tab = QWidget()
        # periodogram_tab = QWidget()
        # model_tab = QWidget()
        # sitelog_tab = QWidget()
        # map_tab = QWidget()
        # # self.text_edit = QTextEdit()
        # # self.text_edit.setReadOnly(True)
        
        # #residual graph
        # residual_layout = QVBoxLayout()
        # self.residuals_tab.setLayout(residual_layout)  
        
        # self.residual_btn_layout = QHBoxLayout()
        # self.raw_residuals_btn = QRadioButton()
        # self.norm_residuals_btn = QRadioButton()
        # self.residual_btn_layout.addWidget(self.raw_residuals_btn)
        # self.residual_btn_layout.addWidget(self.norm_residuals_btn)
        
        
        # self.residual_graph = Graph(self, width=5, height=6, dpi=100)
        # self.residual_graph.addWidget(NavigationToolbar(self.residual_graph, self)) 
        
        # self.residual_btn_layout.addWidget(self.residual_graph)
        # residual_layout.addWidget(self.residual_graph)
        
        # #layouts
        # self.right_pannel.addTab(residuals_tab, 'Residuals')
        # self.right_pannel.addTab(periodogram_tab, 'Periodogram')
        # self.right_pannel.addTab(model_tab, 'Model')
        # self.right_pannel.addTab(sitelog_tab, 'Sitelog')
        # self.right_pannel.addTab(map_tab, 'Map and Links')
        
        # pannels_layout.addWidget(self.right_pannel)
                
        # ---- Right panel ----
        self.right_panel = QTabWidget()
        
        # Create tabs
        residuals_tab = QWidget()
        periodogram_tab = QWidget()
        model_tab = QWidget()
        sitelog_tab = QWidget()
        map_tab = QWidget()
        
        # ----- Residuals tab -----
        residual_layout = QVBoxLayout()
        residuals_tab.setLayout(residual_layout)
        
        # Radio buttons
        self.residual_btn_layout = QHBoxLayout()
        self.raw_residuals_btn = QRadioButton("Raw residuals")
        self.norm_residuals_btn = QRadioButton("Normalized residuals")
        
        self.residual_btn_layout.addWidget(self.raw_residuals_btn)
        self.residual_btn_layout.addWidget(self.norm_residuals_btn)
        self.residual_btn_layout.addStretch()
        
        residual_layout.addLayout(self.residual_btn_layout)
        
        # Graph
        self.residual_graph = Graph(self, width=5, height=6, dpi=80)
        toolbar = NavigationToolbar(self.residual_graph, self)
        
        residual_layout.addWidget(toolbar)
        residual_layout.addWidget(self.residual_graph)
        
        # ----- Periodogram tab -----
        periodogram_layout = QVBoxLayout()
        periodogram_tab.setLayout(periodogram_layout)
        
        self.periodogram_graph = Graph(self, width=5, height=6, dpi=80)
        toolbar = NavigationToolbar(self.periodogram_graph, self)
        
        periodogram_layout.addWidget(toolbar)
        periodogram_layout.addWidget(self.periodogram_graph)
        
        
        # Tabs
        self.right_panel.addTab(residuals_tab, 'Residuals')
        self.right_panel.addTab(periodogram_tab, 'Periodogram')
        self.right_panel.addTab(model_tab, 'Model')
        self.right_panel.addTab(sitelog_tab, 'Sitelog')
        self.right_panel.addTab(map_tab, 'Map and Links')
        
        pannels_layout.addWidget(self.right_panel)

    # --- Actions ---      
    def newProject(self):
        dialog =  DialogNewProject('new')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec() 
        
        
    def modifyProject(self):
        dialog =  DialogNewProject('modify')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec()  

    def openProject(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "YAML Files (*.yaml *.yml)")
       
        try:
            self.data = read_yaml(file_path)
            
            self.text_edit.setPlainText(str(self.data))
           
            if self.data.ts_path:
                self.combobox_ts.fill_combobox(self.data.ts_path)
                
            if self.data.sitelogs_path :
                self.name_sta = self.combobox_ts.currentText()[0:4]
                self.update_log_table(self.name_sta)  
            self.update_ts_graph()
            
           

        except Exception as e:
            print(e)
            self.text_edit.setPlainText("Erreur lecture YAML")     

    def update_log_table(self, station:str):
        
        self.date_table.setRowCount(0)
        log_file = read_yaml(self.data.sitelogs_path)
        sta_file = get_sitelog(station, log_file )
        data = read_sitelog(sta_file[0])
        ant, rec = data[0], data[1]
        
        for i in range(len(ant)):
            self.date_table.add_line(date.from_tsnx(ant[i].start), 'antenna change')
        
        for i in range(len(rec)):
            self.date_table.add_line(date.from_tsnx(rec[i].start), 'receptor change')
        
        
        
    # def update_ts_graph(self): 
    #     selected_name = self.combobox_ts.currentText()
    #     # if not selected_name or not data.ts_path : #self.ts_path:
    #     #     return

    #     ts_file_path = os.path.join(self.data.ts_path, selected_name)
    #     if os.path.isfile(ts_file_path):
    #         discontinuities = self.date_table.get_dates()
    #         #self.load_ts_graph(ts_file_path, discontinuities)
            
    #         r = ts.read(
    #             ts_file_path,
    #             usecols=(2,4,5,6,7,8,9,10,11,12),
    #             format=('t','x','y','z','sx','sy','sz','cxy','cxz','cyz'),
    #             dtrd=1,
    #             rotate=True
    #         )
    #         self.r = r
    #         self.canvas.plot_data(self.r, discontinuities, self.m)
     
    def update_ts_graph(self): 
        selected_name = self.combobox_ts.currentText()
        # if not selected_name or not data.ts_path : #self.ts_path:
        #     return
      
        ts_file_path = os.path.join(self.data.ts_path, selected_name)
        if os.path.isfile(ts_file_path):
            if self.data.discontinuity_path :   
                self.update_log_table(selected_name[0:4])
                discontinuities = self.date_table.get_dates()
                print(ts_file_path, selected_name[0:4] )
            else : 
                discontinuities = []
            
            r = ts.read(
                ts_file_path,
                usecols=(2,4,5,6,7,8,9,10,11,12),
                format=('t','x','y','z','sx','sy','sz','cxy','cxz','cyz'),
                dtrd=1,
                rotate=True
            )
            self.r = r
            self.canvas.plot_data(self.r, discontinuities, model = self.m)
               
            
    def remove_points(self):
        if self.r is None:
            QMessageBox.warning(self, "No series loaded", "Please load a time series first.")
            return
    
    
        threshold = self.sbox_threshold.value() 
        
        try:
            ts.clean_sigmas(self.r, threshold) 
            self.canvas.plot_data(self.r) 
        
        except Exception as e:
            QMessageBox.critical(self, "Cleaning error", str(e))

    def add_date(self):
        dialog = DialogNewDate(self)
        dialog.exec() 
        
        
    def adjust_model(self):
        if self.r is None:
            return
    
        if self.data.discontinuity_path:
            solns = read_solns(self.data.discontinuity_path)
            self.m = model.from_solns(
                self.r,
                solns,
                code=self.name_sta,
                per=[365.25, 182.625],
                noise=['vw']
            )
       
        else:
            self.m = model(self.r)
    
        self.m.fit(finalize=True)
        # self.residual_graph.plot_res(self.m)
        
        self.update_ts_graph()
    
    

class ComboBoxTS(QComboBox):
    
    def __init__(self, ts_path=None):
       super().__init__()
       if ts_path:
           self.fill_combobox(ts_path)

    def fill_combobox(self, ts_path: str):
        self.clear()

        if not ts_path or not os.path.isdir(ts_path):
            return

        for name in sorted(os.listdir(ts_path)):
            self.addItem(name)


class DialogNewProject(QDialog):
    tsPathChanged = pyqtSignal(str)
    
    def __init__(self, mode):
        super().__init__()
        
        self.setMinimumWidth(600)
        #cache
        # self.cache_dir = user_cache_dir(appname="Pytrf",appauthor=None)
        # os.makedirs(self.cache_dir, exist_ok=True)

        # self.cache_file = os.path.join(
        #     self.cache_dir,
        #     "dialog_new_project.yaml"
        # )
        
        #mode = 'modify' or 'new'
        self.mode = mode
        
        general_layout = QVBoxLayout()
        
        #message
        self.message = QLabel('Create a project')
        general_layout.addWidget(self.message)
        
        # Buttons
      
        btn_layout = QVBoxLayout()
        btn_load_yaml = QPushButton("Load existing project (YAML)")
        btn_ts = QPushButton("Add time serie folder")
        btn_model = QPushButton("Add model")
        btn_sitelogs = QPushButton("Add sitelogs")
        btn_discontinuity = QPushButton("Add discontinuity file")
        btn_psd = QPushButton("Add PSD file")
        
        # Connect buttons
        btn_load_yaml.clicked.connect(self.load_yaml_project)
        btn_ts.clicked.connect(self.load_ts)
        btn_model.clicked.connect(self.load_model)
        btn_sitelogs.clicked.connect(self.load_sitelogs)
        btn_discontinuity.clicked.connect(self.load_discontinuity)
        btn_psd.clicked.connect(self.load_psd)
        
        if self.mode == "new":
            btn_load_yaml.hide()
            self.setWindowTitle("New Project")
            
        elif self.mode == "modify":
            btn_load_yaml.show()
            self.message.setText("Modify existing project")
            self.setWindowTitle("Modify Project")
        
        # text
        self.label_project_path = QLabel()
        btn_layout.addSpacing(30)
        self.label_ts_path = QLabel()
        self.label_model_path = QLabel()
        self.label_sitelogs_path = QLabel()
        self.label_discontinuity_path = QLabel()
        self.label_psd_path = QLabel()       
    
        grid = QGridLayout()
        
        grid.addWidget(btn_load_yaml, 0, 0)
        grid.addWidget(self.label_project_path, 0, 1)
        
        grid.addWidget(QLabel(), 1, 0)
        grid.addWidget(QLabel(), 1, 1)        
        
        grid.addWidget(btn_ts, 2, 0)
        grid.addWidget(self.label_ts_path, 2, 1)
        
        grid.addWidget(btn_model, 3, 0)
        grid.addWidget(self.label_model_path, 3, 1)
        
        grid.addWidget(btn_sitelogs, 4, 0)
        grid.addWidget(self.label_sitelogs_path, 4, 1)

        grid.addWidget(btn_discontinuity, 5, 0)
        grid.addWidget(self.label_discontinuity_path, 5, 1)
        
        grid.addWidget(btn_psd, 6, 0)
        grid.addWidget(self.label_psd_path, 6, 1)
        
        general_layout.addLayout(grid)
        
        #last button
        if self.mode == "new":
            btn_create = QPushButton("Create")
            
        elif self.mode == "modify":
            btn_create = QPushButton("Modify")
            
        general_layout.addWidget(btn_create)
        btn_create.clicked.connect(self.create_project)
        
        self.setLayout(general_layout)

    # --- Actions ---
    
    def load_yaml_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open project YAML", "", "YAML Files (*.yaml *.yml)")
    
        if not file_path:
            return
    
        try:
            data = read_yaml(file_path)
    
            # fill fields
            self.label_project_path.setText(file_path)
            self.label_ts_path.setText(data.ts_path)
            self.label_model_path.setText(data.model_path)
            self.label_sitelogs_path.setText(data.sitelogs_path)
            self.label_discontinuity_path.setText(data.discontinuity_path)
            self.label_psd_path.setText(data.psd_path)
            
            self.tsPathChanged.emit(data.ts_path)
           
        except Exception as e:
            print("Erreur lecture YAML:", e)
            self.message.setText("Error loading YAML file")
        
    
    def load_ts(self):
        folder = QFileDialog.getExistingDirectory(self, "Select time series folder")
    
        if folder:
            self.label_ts_path.setText(folder)
            self.tsPathChanged.emit(folder)
    
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select model folder")
    
        if folder:
            self.label_model_path.setText(folder)
    
    def load_sitelogs(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select sitelog file", "", "OPT Files (*.opt)")
        
        if file:
            self.label_sitelogs_path.setText(file)
            
    
    def load_discontinuity(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select soln file", "", "SINEX Files (*.snx)")
    
        if file:
            self.label_discontinuity_path.setText(file)
            
    def load_psd(self):
         file, _ = QFileDialog.getOpenFileName(self, "Select PSD file", "", "SINEX Files (*.snx)")
     
         if file:
             self.label_psd_path.setText(file)
            
    def create_project(self):
        
        data = {
            'ts_path' : self.label_ts_path.text(),
            'model_path':self.label_model_path.text(),
            'sitelogs_path': self.label_sitelogs_path.text(),
            'discontinuity_path':self.label_discontinuity_path.text(),
            'psd_path':self.label_psd_path.text()
            }
        
        ts_path = self.label_ts_path.text()
        model_path = self.label_model_path.text()
        
        if not ts_path or not model_path:
            QMessageBox.critical(
                self,
                "Missing required files",
                "Time series and model folder are required."
            )
            return
            
            
        file_path, _ = QFileDialog.getSaveFileName(self, 
            "Save Project As",
            "new_project.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        
        if file_path: 
            write_yaml(data, file_path)
            
        self.close() #calls closeEvent() before closing the window   
        
        #self.accept() #close DialogNewProject()
    
    # def save_cache(self):
    #     data = {
    #         'ts_path': self.label_ts_path.text(),
    #         'model_path': self.label_model_path.text(),
    #         'sitelogs_path': self.label_sitelogs_path.text(),
    #         'discontinuity_path': self.label_discontinuity_path.text(),
    #         'psd_path': self.label_psd_path.text(),
    #     }
    #     write_yaml(data, self.cache_file)

    def closeEvent(self, event):
        #print('cache enregistré')
        #self.save_cache()
        event.accept()


class TSGraph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # 3 graphs
        self.axes = fig.subplots(nrows=3, ncols=1, sharex=False)
        super().__init__(fig)
        self.setParent(parent)
        
    
    def plot_data(self, r, discontinuities=None, model= None):
       x = TSGraph.mjd_to_datetime(r.t)
       y = r.y*1e3
       labels = ['East[mm]', 'North[mm]',' Up[mm]']
       
       for i in range(3):
           self.axes[i].clear()
           self.axes[i].errorbar(
                 x, y[:, i], 
                 yerr=np.sqrt(r.Q[:, i, i]*1e6), 
                 fmt='o',         # 'o' pour marker
                 ms=1,            # taille du marker
                 mec='black',     # contour marker
                 mfc='black',     # remplissage marker
                 ecolor='black',    # couleur des barres d'erreur
                 elinewidth=0.5,  # largeur des barres
                 zorder=10
             )
           
           self.axes[i].grid()
           self.axes[i].set_ylabel(labels[i], fontsize=8)
           self.axes[i].tick_params(axis='both', labelsize=7)
       # solns = read_solns(r'C:\Users\loeva\Documents\tests_pytrf\soln.snx')
       # m = model.from_solns(r, solns, code='CKSV',per=[365.25, 182.625],noise=['vw'])
       # m.fit(finalize=False)
       # print('heyyyyyyyyyy', m[0].f[0].t)
       
       if model is not None :
           for d in range(min(3, model.nd)):
             print('rrrrrrrrrrr',model[d].yc )
             self.axes[d].plot(x, model[d].yc * 1e3, 'r', linewidth=2, zorder=14)
    
             if model[d].sc is not None:
                 self.axes[d].fill_between(
                     x,
                     (model[d].yc - model[d].sc) * 1e3,
                     (model[d].yc + model[d].sc) * 1e3,
                     alpha=0.4,
                     zorder=4)  
      
       if discontinuities:
           for ax in self.axes:
               for d in discontinuities:
                   if d[1] == 'antenna change':
                       color = 'blue'
                   elif d[1] == 'receptor change':
                       color = 'cyan'
                   elif d[1] == 'earthquake':
                       color = 'orange'
                   else :
                       color = 'red'
                   
                   ax.axvline(
                     d[0],
                     color=color,
                     linestyle='-',
                     linewidth=1,
                     alpha=0.7,
                     zorder=20)
       
       # cmap = plt.cm.get_cmap('tab20')
       
       # if discontinuities:
       #     for ax in self.axes:
       #         for i, d in enumerate(discontinuities):
       #             color = cmap(i % 20)  #reapeat the colors
       #             ax.axvline(
       #                 d[0],
       #                 color=color,
       #                 linestyle='-',
       #                 linewidth=1,
       #                 alpha=0.7,
       #                 zorder=20)
       self.figure.tight_layout()
       self.draw()
        
    @staticmethod
    def mjd_to_datetime(mjd):
        """
        

        Parameters
        ----------
        mjd : TYPE np.array
            DESCRIPTION. days 

        Returns
        -------
        x : TYPE ,p.ndarray
            DESCRIPTION.

        """
        #x = datetime(1858, 11, 17) + timedelta(days=float(mjd))
        x = np.datetime64('1858-11-17') + (mjd * np.timedelta64(1, 'D'))
        return x

class Graph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # 3 graphs
        self.axes = fig.subplots(nrows=3, ncols=1, sharex=False)
        super().__init__(fig)
        self.setParent(parent)
     
    def plot_res(self, m, thr_raw=None, tunit=None, dims=None, title=None):
        """
        adapted from ts.plot_res
        """
        
        if m is None or m.r is None:
            return
    
        # ---- Time ----
        if (tunit is None) or (tunit == m.r.tunit):
            tunit = m.r.tunit
            t = m.r.t
        elif (m.r.tunit == 'd') and (tunit == 'y'):
            t = np.array([date.from_mjd(d).ydec() for d in m.r.t])
        else:
            tunit = m.r.tunit
            t = m.r.t
    
        # ---- Component names ----
        if dims is None:
            dims = m.r.dims
        if dims is None:
            dims = [f'Component {d+1}' for d in range(m.nd)]
        if isinstance(dims, str):
            dims = [dims]
    
        # ---- Plot ----
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
    
            # ---- Threshold ----
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


# --- discontinuity date list ---

class Tabledate(QTableWidget):
    
    def __init__(self, sitelog_path=None):
        super().__init__()
        
        self.setColumnCount(6)
        # for i in range(1,5):
        #     self.setColumnWidth(i,50)
        
        self.setColumnWidth(1,30)
        self.setColumnWidth(2,30)
        self.setColumnWidth(3,100)
        self.setColumnWidth(4,100)
        self.setHorizontalHeaderLabels(["Date", "Pos", "Vel", "Exp", "Log", "Info"])
        
        self.setColumnWidth(0,80)    
        
            
    def add_line(self, date_dis, info):
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem( str(date_dis) ))
        #for i in range(1,3):
        checkbox_pos = QCheckBox()
        checkbox_vel = QCheckBox()
        self.setCellWidget(row, 1, checkbox_pos)
        self.setCellWidget(row, 2, checkbox_vel)
        
     
        cell_exp = QWidget()
        cell_exp_layout = QHBoxLayout(cell_exp)
        cell_log = QWidget()
        cell_log_layout = QHBoxLayout(cell_log)
           
        for i in range(3):
            checkbox_exp = QCheckBox()
            checkbox_log = QCheckBox()
            cell_exp_layout.addWidget(checkbox_exp)
            cell_log_layout.addWidget(checkbox_log)

        self.setCellWidget(row, 3, cell_exp)
        self.setCellWidget(row, 4, cell_log)
     
        
        self.setItem(row, 5, QTableWidgetItem(info))

    def get_dates(self):
        dates = []

        for row in range(self.rowCount()):
            item = self.item(row, 0)
            info = self.item(row, 5).text()
            
            if item is None:
                continue

            txt = item.text().strip()
            try:
                dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
                dates.append([np.datetime64(dt), info])
            except ValueError:
                print(f"ignored date : {txt}")
        return dates


    # def get_dates(self):
    #     dates = {"pos": [], "vel": [], "exp": [], "log":[]}
    
    #     for row in range(self.rowCount()):
    #         item = self.item(row, 0)
    #         if item is None:
    #             continue
    
    #         try:
    #             t = np.datetime64(item.text())
    #         except Exception:
    #             print(f"ignored date : {t}")
    
    #         if self.cellWidget(row, 1).isChecked():
    #             dates["pos"].append(t)
    
    #         if self.cellWidget(row, 2).isChecked():
    #             dates["vel"].append(t)
    
    #     return dates


class DialogNewDate(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumWidth(400)
        self.setWindowTitle('Add Discontinuity')
        
        #text layout
        text_layout = QHBoxLayout()
        txt_date = QLabel('Date')
        txt_info = QLabel('Info')
        
        text_layout.addWidget(txt_date)
        text_layout.addWidget(txt_info)

        #line edit layout
        self.line_date = QLineEdit()
        self.line_date.setPlaceholderText("yyyy-mm-dd hh:mm:ss")
        self.line_info = QLineEdit()
        self.line_info.setPlaceholderText("antenna, earthquake, unknown...")
        lineedit_layout = QHBoxLayout()
        lineedit_layout.addWidget(self.line_date)
        lineedit_layout.addWidget(self.line_info)
        
        # button ok
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.validate)
        
        general_layout = QVBoxLayout()
        general_layout.addLayout(text_layout)
        general_layout.addLayout(lineedit_layout)
        general_layout.addWidget(btn_ok)
        
      
        self.setLayout(general_layout)
        
            
    def validate(self):
        date = self.line_date.text()
        info = self.line_info.text()
    
        if not date:
            QMessageBox.warning(self, "Error", "Enter a date and an information")
            return
    
        self.parent.date_table.add_line(date, info)
        self.parent.update_ts_graph()
        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.showMaximized()
    sys.exit(app.exec())
    
    """
    l'enregistrement dans un fichier cache ne fonctionne pas, surement à cause du \ Utilisateur'
    """

