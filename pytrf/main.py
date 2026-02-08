# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:48:32 2026

@author: loeva
"""
# -*- coding: utf-8 -*-
"""
Pytrf GUI - Application principale
Created on Fri Jan 16 14:09:06 2026
@author: loeva
"""
import sys
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QFileDialog, QMessageBox, QRadioButton,
    QScrollArea
)

from PyQt6.QtCore import Qt
# local imports
from widgets import ComboBoxTS, Tabledate, TableOutliner
from graphs import TSGraph, Graph
from dialogs import DialogNewProject, DialogNewDate
from models import TimeSeriesManager
from cache import CacheManager

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from pytrf.io import read_yaml, get_sitelog, read_sitelog
from pytrf import date
from pytrf.ts import model

class MyApp(QMainWindow):
    """Fenêtre principale de l'application"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
        
        #self.cache = CacheManager()
        
        self.ts_manager = TimeSeriesManager()
        
        # data of the project
        self.data = None
        self.name_sta = None
    
    def initUI(self):
        """Initialise l'interface utilisateur"""
        self.setWindowTitle("Pytrf")
        
        # central widget 
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        pannels_layout = QHBoxLayout()
       
        # === top buttons ===
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
        
        # === Left pannel ===
        control_panel = QTabWidget()
        
        # Determinist model tab
        
        #table
        det_model_tab = QWidget()
        det_model_layout = QVBoxLayout()
        det_model_tab.setLayout(det_model_layout)  
        
        self.date_table = Tabledate()
        det_model_layout.addWidget(self.date_table, stretch=1)
        
        btn_add_date = QPushButton('+')
        det_model_layout.addWidget(btn_add_date)
        btn_add_date.clicked.connect(self.add_date)
        
        #sinusoidal period
        sin_layout = QHBoxLayout()
        perdiod_layout = QVBoxLayout()
        del_period_layout = QVBoxLayout()
        
        del_btn = QPushButton('+')
        del_btn2 = QPushButton('-')
        del_btn3 = QPushButton('-')
        del_period_layout.addWidget(del_btn)
        del_period_layout.addWidget(del_btn2)
        del_period_layout.addWidget(del_btn3)
        
        sin_label0 = QLabel('Sinusoïdal period')
        sin_label1 = QLabel('365.25')
        sin_label2 = QLabel('182.625')
        
        perdiod_layout.addWidget(sin_label0)
        perdiod_layout.addWidget(sin_label1)
        perdiod_layout.addWidget(sin_label2)
        
        sin_layout.addLayout(perdiod_layout)
        sin_layout.addLayout(del_period_layout)
        det_model_layout.addLayout(sin_layout)      
        
        
        #outliner rejection
        outliner_layout = QVBoxLayout()
        outliner_rejection_title = QLabel('Outliner rejection')
        
        outliner_layout.addWidget(outliner_rejection_title)
        det_model_layout.addLayout(outliner_layout)      
        
        outliner_table = TableOutliner()
        det_model_layout.addWidget(outliner_table)      
        
        
        #last buttons
        btn_adjust_model = QPushButton('Adjust the model')
        btn_adjust_model.clicked.connect(self.adjust_model)
        btn_save = QPushButton('SAVE')
        btn_save.clicked.connect(self.save_model)
        btn_save.setStyleSheet("background-color : red")
        
        btns_layout = QHBoxLayout()
        btns_layout.addWidget(btn_adjust_model)
        btns_layout.addWidget(btn_save)
        det_model_layout.addLayout(btns_layout)
        
        # Stochastic model tab
        sto_model_tab = QWidget()
        sto_model_layout = QVBoxLayout()
        sto_model_layout.addWidget(QLabel("holà"))
        sto_model_tab.setLayout(sto_model_layout)
        
        control_panel.addTab(det_model_tab, "Deterministic model")
        control_panel.addTab(sto_model_tab, "Stochastic model")
        
        pannels_layout.addWidget(control_panel)
        
        # === Middle pannel ===
        self.combobox_ts = ComboBoxTS()
        self.sbox_threshold = QDoubleSpinBox()
        self.sbox_threshold.setValue(5)
        self.btn_remove_pts = QPushButton("Remove points with large errors")
        self.btn_remove_pts.clicked.connect(self.remove_points)
        
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.combobox_ts)
        
        self.canvas = TSGraph(self, width=5, height=6, dpi=100)
        middle_layout.addWidget(NavigationToolbar(self.canvas, self)) 
        self.combobox_ts.currentIndexChanged.connect(self.update_ts_graph)
        middle_layout.addWidget(self.canvas)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.btn_remove_pts)
        threshold_layout.addWidget(self.sbox_threshold)      
        middle_layout.addLayout(threshold_layout)
        
        pannels_layout.addLayout(middle_layout)
        main_layout.addLayout(pannels_layout)
        
        # === Right pannel ===
        self.right_panel = QTabWidget()
        
        # Residuals tab
        residuals_tab = QWidget()
        residual_layout = QVBoxLayout()
        residuals_tab.setLayout(residual_layout)
        
        self.residual_btn_layout = QHBoxLayout()
        self.raw_residuals_btn = QRadioButton("Raw residuals")
        self.raw_residuals_btn.setChecked(True)
        self.norm_residuals_btn = QRadioButton("Normalized residuals")
        
        self.raw_residuals_btn.toggled.connect(self.update_residual_graph)
        self.norm_residuals_btn.toggled.connect(self.update_residual_graph)
        
        self.residual_btn_layout.addWidget(self.raw_residuals_btn)
        self.residual_btn_layout.addWidget(self.norm_residuals_btn)
        self.residual_btn_layout.addStretch()
        
        residual_layout.addLayout(self.residual_btn_layout)
        
        self.residual_graph = Graph(self, width=5, height=6, dpi=80)
        toolbar = NavigationToolbar(self.residual_graph, self)
        
        residual_layout.addWidget(toolbar)
        residual_layout.addWidget(self.residual_graph)
        
        # Periodogram Tab
        periodogram_tab = QWidget()
        periodogram_layout = QVBoxLayout()
        periodogram_tab.setLayout(periodogram_layout)
        
        self.periodogram_graph = Graph(self, width=5, height=6, dpi=80)
        toolbar = NavigationToolbar(self.periodogram_graph, self)
        
        periodogram_layout.addWidget(toolbar)
        periodogram_layout.addWidget(self.periodogram_graph)
        
        # Model Tab
        #model_tab = QWidget()
        # model_layout = QHBoxLayout() 
        # model_tab.setLayout(model_layout) 
        # self.model_label = QLabel() 
        # self.model_label.setText('Rien à afficher') 
        # model_layout.addWidget(self.model_label)
        
        # Model Tab
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        model_tab.setLayout(model_layout)
        
        # Créer une zone de défilement
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Créer un widget conteneur pour le label
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_content_layout)
        
        # Créer le label
        self.model_label = QLabel()
        self.model_label.setText('Rien à afficher')
        self.model_label.setWordWrap(True)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Ajouter le label au conteneur
        scroll_content_layout.addWidget(self.model_label)
        scroll_content_layout.addStretch()
        
        # Ajouter le conteneur à la zone de défilement
        scroll_area.setWidget(scroll_content)
        
        # Ajouter la zone de défilement au layout
        model_layout.addWidget(scroll_area)

        
        # Sitelog Tab
        sitelog_tab = QWidget()
        
        # Map and links Tab
        map_tab = QWidget()
        
        self.right_panel.addTab(residuals_tab, 'Residuals')
        self.right_panel.addTab(periodogram_tab, 'Periodogram')
        self.right_panel.addTab(model_tab, 'Model')
        self.right_panel.addTab(sitelog_tab, 'Sitelog')
        self.right_panel.addTab(map_tab, 'Map and Links')
        
        pannels_layout.addWidget(self.right_panel)
    
    # === Actions ===
    
    def newProject(self):
        """Crée un nouveau projet"""
        dialog = DialogNewProject('new')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec() 
    
    def modifyProject(self):
        """Modifie un projet existant"""
        dialog = DialogNewProject('modify')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec()  
    
    def openProject(self):
        """Ouvre un projet"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "YAML Files (*.yaml *.yml)"
        )
       
        try:
            self.data = read_yaml(file_path)
            print('fichier lu')
            
                
            if self.data.ts_path:
                self.combobox_ts.fill_combobox(self.data.ts_path)
                self.name_sta = self.combobox_ts.currentText()[0:4]
                ts_file_path = os.path.join(
                    self.data.ts_path, 
                    self.name_sta + '_igs.xyz'
                )
           
                self.ts_manager.load_time_series(ts_file_path, auto_fit=False)
                print('je regarde si il y a un model')
                model_file_path = os.path.join(
                    self.data.model_path, 
                    self.name_sta + '.pkl'
                )
                print(f'model_file_path : {model_file_path}')
                
                if os.path.isfile(model_file_path):
                    print('model computed from pkl')
                    self.ts_manager.m = model.load(model_file_path)
                    self.ts_manager.r = self.ts_manager.m.r
                    #t = [date.from_mjd(d).ydec() for d in r.t]
                    #normalement pas besoin mais quand je ne le mets pas j'ai 'volume and kernel should have the same dimensionality'
                    self.ts_manager.fit_model()
                   
                else:
                    if self.data.discontinuity_path:
                        print('model computed from snx')
                        self.ts_manager.load_model_from_solns(
                            self.data.discontinuity_path, 
                            self.name_sta
                        )
                        
                    else:
                        print('chargement d un model par defaut')
                        self.ts_manager._initialize_base_model()
                    
                    self.ts_manager.fit_model()
                
                
                # if self.data.discontinuity_path:
                #     self.ts_manager.load_model_from_solns(
                #         self.data.discontinuity_path, 
                #         self.name_sta
                #     )
                
                # self.ts_manager.fit_model()
            #instead of update_ts_graph
            if self.data.discontinuity_path:   
                self.update_log_table(self.name_sta)
                discontinuities = self.date_table.get_dates()
            else: 
                discontinuities = []
            
            self.canvas.plot_data(
                self.ts_manager.r, 
                discontinuities, 
                model=self.ts_manager.m
            )
            
            self.update_residual_graph()
            self.update_psd()
            self.model_label.setText(str(self.ts_manager.m))
            
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Error loading project: {e}")
    
    def update_ts_graph(self): 
        """Met à jour le graphique de série temporelle"""
        selected_name = self.combobox_ts.currentText()
        self.name_sta = selected_name[0:4]
        
        if not selected_name or not self.data or not self.data.ts_path:
            return
        
        ts_file_path = os.path.join(self.data.ts_path, selected_name)
        
        if os.path.isfile(ts_file_path):
            self.ts_manager.load_time_series(ts_file_path)
            
            if self.data.discontinuity_path:   
                self.update_log_table(selected_name[0:4])
                discontinuities = self.date_table.get_dates()
            else: 
                discontinuities = []
            
            self.canvas.plot_data(
                self.ts_manager.r, 
                discontinuities, 
                model=self.ts_manager.m
            )
    def update_residual_graph(self):
        """Update graph depending on the radio button checked"""
        if self.ts_manager.m is None:
            return
        
        if self.raw_residuals_btn.isChecked():
            self.residual_graph.plot_raw_res(self.ts_manager.m)
        else:
            self.residual_graph.plot_norm_res(self.ts_manager.m)
    
    def update_psd(self):
        if self.ts_manager.m is None:
            return
        
        self.periodogram_graph.plot_psd(self.ts_manager.m)
        
    def update_log_table(self, station: str):
       """Met à jour le tableau des logs"""
       # Sauvegarder l'état actuel du tableau
       saved_rows = []
       for row in range(self.date_table.rowCount()):
           date_item = self.date_table.item(row, 0)
           info_item = self.date_table.item(row, 1)
           
           if date_item is None or info_item is None:
               continue
           
           pos_checked = self.date_table.cellWidget(row, 2).isChecked()
           vel_checked = self.date_table.cellWidget(row, 3).isChecked()
           
           saved_rows.append([
               date_item.text(),
               info_item.text(),
               pos_checked,
               vel_checked
           ])
       
       # Réinitialiser le tableau
       self.date_table.setRowCount(0)
       
       log_file = read_yaml(self.data.sitelogs_path)
       sta_file = get_sitelog(station, log_file)
       data = read_sitelog(sta_file[0])
       ant, rec = data[1], data[0]
       
       chg_pos = []
       chg_vel = []
       
       if self.ts_manager.m:
           chg_pos_mjd = self.ts_manager.m[0].f[0].t
           chg_vel_mjd = self.ts_manager.m[0].f[1].t
           
           for i in range(len(chg_pos_mjd)):
               d_1 = date.from_mjd(chg_pos_mjd[i])
               chg_pos.append(str(d_1))
               
           for i in range(len(chg_vel_mjd)):
               d_2 = date.from_mjd(chg_vel_mjd[i])
               chg_vel.append(str(d_2))
       
       # Liste des dates du sitelog
       sitelog_dates = []
       
       # Ajouter les antennes
       for i in range(len(ant)):
           date_str = str(date.from_tsnx(ant[i].start))
           info_str = 'antenna change'
           sitelog_dates.append([date_str, info_str])
           
           self.date_table.add_line(date_str, info_str)
           last_row = self.date_table.rowCount() - 1
           
           # Chercher si cette date était sauvegardée
           found = False
           for saved_row in saved_rows:
               if saved_row[0] == date_str and saved_row[1] == info_str:
                   # Restaurer l'état sauvegardé
                   if saved_row[2]:  # pos
                       self.date_table.cellWidget(last_row, 2).setChecked(True)
                   if saved_row[3]:  # vel
                       self.date_table.cellWidget(last_row, 3).setChecked(True)
                   found = True
                   break
           
           if not found:
               # Utiliser l'ancienne logique du modèle
               if date_str in chg_pos:
                   self.date_table.cellWidget(last_row, 2).setChecked(True)
               if date_str in chg_vel:
                   self.date_table.cellWidget(last_row, 3).setChecked(True)
       
       # Ajouter les récepteurs
       for i in range(len(rec)):
           date_str = str(date.from_tsnx(rec[i].start))
           info_str = 'receptor change'
           sitelog_dates.append([date_str, info_str])
           
           self.date_table.add_line(date_str, info_str)
           last_row = self.date_table.rowCount() - 1
           
           # Chercher si cette date était sauvegardée
           found = False
           for saved_row in saved_rows:
               if saved_row[0] == date_str and saved_row[1] == info_str:
                   # Restaurer l'état sauvegardé
                   if saved_row[2]:  # pos
                       self.date_table.cellWidget(last_row, 2).setChecked(True)
                   if saved_row[3]:  # vel
                       self.date_table.cellWidget(last_row, 3).setChecked(True)
                   found = True
                   break
           
           if not found:
               # Utiliser l'ancienne logique du modèle
               if date_str in chg_pos:
                   self.date_table.cellWidget(last_row, 2).setChecked(True)
               if date_str in chg_vel:
                   self.date_table.cellWidget(last_row, 3).setChecked(True)
       
       # Restaurer les lignes qui ne sont pas dans le sitelog (ajoutées manuellement)
       for saved_row in saved_rows:
           # Vérifier si cette ligne est dans le sitelog
           is_sitelog = False
           for sitelog_row in sitelog_dates:
               if saved_row[0] == sitelog_row[0] and saved_row[1] == sitelog_row[1]:
                   is_sitelog = True
                   break
           
           # Si pas dans le sitelog, c'est une ligne manuelle à restaurer
           if not is_sitelog:
               self.date_table.add_line(saved_row[0], saved_row[1])
               last_row = self.date_table.rowCount() - 1
               
               if saved_row[2]:  # pos
                   self.date_table.cellWidget(last_row, 2).setChecked(True)
               if saved_row[3]:  # vel
                   self.date_table.cellWidget(last_row, 3).setChecked(True)
   
    def remove_points(self):
        """Supprime les points aberrants"""
        if self.ts_manager.r is None:
            QMessageBox.warning(
                self, 
                "No series loaded", 
                "Please load a time series first."
            )
            return
        
        threshold = self.sbox_threshold.value() 
        
        try:
            self.ts_manager.clean_time_series(threshold)
            self.update_ts_graph()
        except Exception as e:
            QMessageBox.critical(self, "Cleaning error", str(e))
    
    def add_date(self):
        """Ajoute une date de discontinuité"""
        dialog = DialogNewDate(self)
        dialog.exec() 
    
    def adjust_model(self):
        """Ajuste le modèle"""
        if self.ts_manager.r is None:
            return
        
        #events = self.date_table.get_model_events()
        dates, infos, pos_checked, vel_checked = self.date_table.get_model_events()
  
        try:
            self.ts_manager.fit_model_iterative(dates, pos_checked, vel_checked)
           
            
            if self.data.discontinuity_path:   
                discontinuities = self.date_table.get_dates()
            else: 
                discontinuities = []
            print('avant le plot')
            self.canvas.plot_data(
                self.ts_manager.r, 
                discontinuities, 
                model=self.ts_manager.m
            )
            
            # update graph
            if self.raw_residuals_btn.isChecked():
                self.residual_graph.plot_raw_res(self.ts_manager.m)
            else:
                self.residual_graph.plot_norm_res(self.ts_manager.m)
            
            self.periodogram_graph.plot_psd(self.ts_manager.m)
            self.model_label.setText(str(self.ts_manager.m))
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))
            

    def save_model(self):
        """SAve model in pickel file"""
        if self.ts_manager.m is None:
            QMessageBox.warning(
                self,
                "No model to save",
                "Please fit a model before saving."
            )
            return
        
        # open file explorer
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            f"{self.name_sta}.pkl" if self.name_sta else "model.pkl",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if not file_path:
            # save aborted 
            return
        
        try:
            # save model
            self.ts_manager.m.clean()
            self.ts_manager.m.dump(file_path)
            QMessageBox.information(
                self,
                "Success",
                f"Model successfully saved to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save error",
                f"Error saving model:\n{str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.showMaximized()
    sys.exit(app.exec())
