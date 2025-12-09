# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 20:29:59 2025

@author: loeva
"""
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QPushButton, QDoubleSpinBox, QFileDialog,
    QTextEdit, QDialog
)
import sys

from platformdirs import user_config_dir, user_data_dir
import os
from pytrf.io import read_yaml
import yaml

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Répertoire de données avec platformdirs
        self.data_dir = user_data_dir("MonLogiciel", "MonNom")
        os.makedirs(self.data_dir, exist_ok=True)
        print("Répertoire de données :", self.data_dir)

    def initUI(self):
        self.setWindowTitle("Example of PyQt6 App")
        self.setGeometry(800, 100, 500, 300)

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
        dialog =  DialogNewProject()
        result = dialog.exec()
        
    def action_modify(self):
        print("Modify project")

    def open_yaml_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier YAML", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    print('file_path', file_path)
                    data = read_yaml(file_path)
                    self.text_edit.setPlainText(str(data))
                    print("Contenu YAML :", data)
                except :
                    self.text_edit.setPlainText("Erreur lecture YAML")         
    
            

class ComboBoxTS(QComboBox):
    def __init__(self):
        super().__init__()
        self.fill_combobox()

    def fill_combobox(self):
        self.addItem("One")
        self.addItem("Two")
        self.addItem("Three")

class DialogNewProject(QDialog):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("New Project")
        self.setMinimumWidth(600)

        general_layout = QVBoxLayout()
        
        #message
        self.message = QLabel('Create a project')
        general_layout.addWidget(self.message)
        
        # Buttons
        btn_and_path_layout = QHBoxLayout()
        
        btn_layout = QVBoxLayout()
        btn_ts = QPushButton("Add time serie folder")
        btn_model = QPushButton("Add model")
        btn_sitelogs = QPushButton("Add sitelogs")
        btn_discontinuity = QPushButton("Add discontinuity file")
        btn_psd = QPushButton("Add PSD file")
        
        btn_layout.addWidget(btn_ts)
        btn_layout.addWidget(btn_model)
        btn_layout.addWidget(btn_sitelogs)
        btn_layout.addWidget(btn_discontinuity)
        btn_layout.addWidget(btn_psd)
        
        # Connect buttons
        btn_ts.clicked.connect(self.load_ts)
        btn_model.clicked.connect(self.load_model)
        btn_sitelogs.clicked.connect(self.load_sitelogs)
        btn_discontinuity.clicked.connect(self.load_discontinuity)
        btn_psd.clicked.connect(self.load_psd)
        
        btn_and_path_layout.addLayout(btn_layout)
        
        # text
        text_layout = QVBoxLayout()
        self.label_ts_path = QLabel()
        self.label_model_path = QLabel()
        self.label_sitelogs_path = QLabel()
        self.label_discontinuity_path = QLabel()
        self.label_psd_path = QLabel()       
        
        text_layout.addWidget(self.label_ts_path)
        text_layout.addWidget(self.label_model_path)
        text_layout.addWidget(self.label_sitelogs_path)
        text_layout.addWidget(self.label_discontinuity_path)
        text_layout.addWidget(self.label_psd_path)
        
        btn_and_path_layout.addLayout(text_layout)
        
        general_layout.addLayout(btn_and_path_layout)
        
        #last button
        btn_create = QPushButton("Create")
        general_layout.addWidget(btn_create)
        btn_create.clicked.connect(self.create_project)
        
        self.setLayout(general_layout)

      
        
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
        
        
        file_path, _ = QFileDialog.getSaveFileName(self, 
            "Save Project As",
            "new_project.yaml",
            "YAML Files (*.yaml *.yml)"
        )
    
        if file_path: 
            with open(file_path, "w", encoding="utf-8") as file:
                yaml.dump(data, file, allow_unicode=True, sort_keys=False)
         
        self.accept() #close DialogNewProject()
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())
    

