# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 20:29:59 2025

@author: loeva
"""
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QFileDialog,
    QTextEdit, QDialog, QMessageBox, QGridLayout
)
from PyQt6.QtCore import pyqtSignal
import sys

from platformdirs import user_cache_dir
import os
from pytrf.io import read_yaml, write_yaml

import matplotlib

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Cache utilisateur
        self.cache_dir = user_cache_dir(
            appname="Pytrf",
            appauthor=None
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        """
        The temporary file is created at this place : 
        Linux : ~/.cache/Pytrf/
        Windows : C:\ Users\...\AppData\Local\Pytrf\Cache
        """

        print("Cache dir:", self.cache_dir)
        
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

        btn_new.clicked.connect(self.action_new)
        btn_open.clicked.connect(self.open_yaml_file)
        btn_modify.clicked.connect(self.action_modify)

        button_layout.addWidget(btn_new)
        button_layout.addWidget(btn_open)
        button_layout.addWidget(btn_modify)
        button_layout.addStretch() 

        main_layout.addLayout(button_layout)
        

        # ---- Control pannel ----
        
        self.control_panel = QTabWidget()

        # Create tabs
        det_model_tab = QWidget()
        sto_model_tab = QWidget()

        # Deterministic model tab
        det_model_layout = QVBoxLayout()
        det_model_layout.addWidget(QLabel("holà"))
        det_model_tab.setLayout(det_model_layout)

        # Stochastic model tab
        sto_model_layout = QVBoxLayout()
        sto_model_layout.addWidget(QLabel("holà"))
        sto_model_tab.setLayout(sto_model_layout)

        # Add tabs
        self.control_panel.addTab(det_model_tab, "Deterministic model")
        self.control_panel.addTab(sto_model_tab, "Stochastic model")

        pannels_layout.addWidget(self.control_panel)
        
        
        # ---- Middle pannel ----
        self.combobox_ts = ComboBoxTS()
        self.btn_remove_pts = QPushButton("Remove points with large errors")
        self.sbox_seuil = QDoubleSpinBox()
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.combobox_ts)
        middle_layout.addWidget(self.text_edit)
        middle_layout.addWidget(self.btn_remove_pts)
        middle_layout.addWidget(self.sbox_seuil)
        
        pannels_layout.addLayout(middle_layout)
        main_layout.addLayout(pannels_layout)
        
        
        # ---- Right pannel ----
        
        self.right_pannel = QTabWidget()
        # Create tabs
        residuals_tab = QWidget()
        periodogram_tab = QWidget()
        model_tab = QWidget()
        sitelog_tab = QWidget()
        map_tab = QWidget()
        # self.text_edit = QTextEdit()
        # self.text_edit.setReadOnly(True)
        
        self.right_pannel.addTab(residuals_tab, 'Residuals')
        self.right_pannel.addTab(periodogram_tab, 'Periodogram')
        self.right_pannel.addTab(model_tab, 'Model')
        self.right_pannel.addTab(sitelog_tab, 'Sitelog')
        self.right_pannel.addTab(map_tab, 'Map and Links')
        
        pannels_layout.addWidget(self.right_pannel)
        

    # --- Actions ---      
    def action_new(self):
        dialog =  DialogNewProject('new')
        dialog.exec() 
        
        
    def action_modify(self):
        dialog =  DialogNewProject('modify')
        dialog.exec()  

    def open_yaml_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "YAML Files (*.yaml *.yml)")
       
        try:
            data = read_yaml(file_path)
            self.text_edit.setPlainText(str(data))
            self.combobox_ts.fill_combobox(data.ts_path)
        except Exception as e:
            print(e)
            self.text_edit.setPlainText("Erreur lecture YAML")         
            
    


class ComboBoxTS(QComboBox):
    
    def __init__(self, ts_path=None):
       super().__init__()
       if ts_path:
           self.fill_combobox(ts_path)

    # def fill_combobox(self):
    #     ts_name = os.listdir(DialogNewProject.label_ts_path)
        
    #     for i in ts_name : 
    #         self.addItem(i)

    def fill_combobox(self, ts_path: str):
        self.clear()

        if not ts_path or not os.path.isdir(ts_path):
            return

        for name in sorted(os.listdir(ts_path)):
            self.addItem(name)


class DialogNewProject(QDialog):
    
    def __init__(self, mode):
        super().__init__()
        self.setMinimumWidth(600)
        
        #cache
        self.cache_dir = user_cache_dir("Pytrf", None)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            "dialog_new_project.yaml"
        )
        
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
           
        except Exception as e:
            print("Erreur lecture YAML:", e)
            self.message.setText("Error loading YAML file")
        
    
    def load_ts(self):
        folder = QFileDialog.getExistingDirectory(self, "Select time series folder")
    
        if folder:
            self.label_ts_path.setText(folder)
    
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select model folder")
    
        if folder:
            self.label_model_path.setText(folder)
    
    def load_sitelogs(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select sitelog file", "", "YAML Files (*.yaml *.yml)")
    
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
    
    def save_cache(self):
        data = {
            'ts_path': self.label_ts_path.text(),
            'model_path': self.label_model_path.text(),
            'sitelogs_path': self.label_sitelogs_path.text(),
            'discontinuity_path': self.label_discontinuity_path.text(),
            'psd_path': self.label_psd_path.text(),
        }
        write_yaml(data, self.cache_file)

    def closeEvent(self, event):
        print('cache enregistré')
        self.save_cache()
        event.accept()


    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.showMaximized()
    sys.exit(app.exec())
    
    """
    l'enregistrement dans un fichier cache ne fonctionne pas, surement à cause du \ Utilisateur'
    """

