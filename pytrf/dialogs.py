# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:20 2026

@author: loeva
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QGridLayout, QMessageBox,
    QLineEdit
)
from PyQt6.QtCore import pyqtSignal
from pytrf.io import read_yaml, write_yaml


class DialogNewProject(QDialog):
    """Dialogue pour créer/modifier un projet"""
    
    tsPathChanged = pyqtSignal(str)
    
    def __init__(self, mode):
        super().__init__()
        
        self.setMinimumWidth(600)
        self.mode = mode  # 'new' ou 'modify'
        
        general_layout = QVBoxLayout()
        
        # Message
        self.message = QLabel('Create a project')
        general_layout.addWidget(self.message)
        
        # Boutons
        btn_load_yaml = QPushButton("Load existing project (YAML)")
        btn_ts = QPushButton("Add time serie folder")
        btn_model = QPushButton("Add model")
        btn_sitelogs = QPushButton("Add sitelogs")
        btn_discontinuity = QPushButton("Add discontinuity file")
        btn_psd = QPushButton("Add PSD file")
        
        # Connexions
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
        
        # Labels pour afficher les chemins
        self.label_project_path = QLabel()
        self.label_ts_path = QLabel()
        self.label_model_path = QLabel()
        self.label_sitelogs_path = QLabel()
        self.label_discontinuity_path = QLabel()
        self.label_psd_path = QLabel()
        
        # Grille
        grid = QGridLayout()
        grid.addWidget(btn_load_yaml, 0, 0)
        grid.addWidget(self.label_project_path, 0, 1)
        grid.addWidget(QLabel(), 1, 0)
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
        
        # Bouton créer/modifier
        if self.mode == "new":
            btn_create = QPushButton("Create")
        elif self.mode == "modify":
            btn_create = QPushButton("Modify")
        
        general_layout.addWidget(btn_create)
        btn_create.clicked.connect(self.create_project)
        
        self.setLayout(general_layout)
    
    def load_yaml_project(self):
        """Charge un projet YAML existant"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open project YAML", "", "YAML Files (*.yaml *.yml)"
        )
        
        if not file_path:
            return
        
        try:
            data = read_yaml(file_path)
            
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
        """Sélectionne le dossier de séries temporelles"""
        folder = QFileDialog.getExistingDirectory(self, "Select time series folder")
        if folder:
            self.label_ts_path.setText(folder)
            self.tsPathChanged.emit(folder)
    
    def load_model(self):
        """Sélectionne le dossier de modèle"""
        folder = QFileDialog.getExistingDirectory(self, "Select model folder")
        if folder:
            self.label_model_path.setText(folder)
    
    def load_sitelogs(self):
        """Sélectionne le fichier sitelog"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select sitelog file", "", "OPT Files (*.opt)"
        )
        if file:
            self.label_sitelogs_path.setText(file)
    
    def load_discontinuity(self):
        """Sélectionne le fichier de discontinuités"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select soln file", "", "SINEX Files (*.snx)"
        )
        if file:
            self.label_discontinuity_path.setText(file)
    
    def load_psd(self):
        """Sélectionne le fichier PSD"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select PSD file", "", "SINEX Files (*.snx)"
        )
        if file:
            self.label_psd_path.setText(file)
    
    def create_project(self):
        """Crée/sauvegarde le projet"""
        data = {
            'ts_path': self.label_ts_path.text(),
            'model_path': self.label_model_path.text(),
            'sitelogs_path': self.label_sitelogs_path.text(),
            'discontinuity_path': self.label_discontinuity_path.text(),
            'psd_path': self.label_psd_path.text()
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
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Project As",
            "new_project.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        
        if file_path: 
            write_yaml(data, file_path)
        
        self.close()


class DialogNewDate(QDialog):
    """Dialogue pour ajouter une discontinuité"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumWidth(400)
        self.setWindowTitle('Add Discontinuity')
        
        # Layout textes
        text_layout = QHBoxLayout()
        txt_date = QLabel('Date')
        txt_info = QLabel('Info')
        text_layout.addWidget(txt_date)
        text_layout.addWidget(txt_info)
        
        # Layout lignes de saisie
        self.line_date = QLineEdit()
        self.line_date.setPlaceholderText("yyyy-mm-dd hh:mm:ss")
        self.line_info = QLineEdit()
        self.line_info.setPlaceholderText("antenna, earthquake, unknown...")
        
        lineedit_layout = QHBoxLayout()
        lineedit_layout.addWidget(self.line_date)
        lineedit_layout.addWidget(self.line_info)
        
        # Bouton OK
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.validate)
        
        # Layout général
        general_layout = QVBoxLayout()
        general_layout.addLayout(text_layout)
        general_layout.addLayout(lineedit_layout)
        general_layout.addWidget(btn_ok)
        
        self.setLayout(general_layout)
    
    def validate(self):
        """Valide et ajoute la date"""
        date = self.line_date.text()
        info = self.line_info.text()
    
        if not date:
            QMessageBox.warning(self, "Error", "Enter a date and an information")
            return
    
        self.parent.date_table.add_line(date, info)
        
        self.line_date.clear()
        self.line_info.clear()
        
        self.accept()
    
    def closeEvent(self, event):
        self.line_date.deleteLater()
        self.line_info.deleteLater()
        super().closeEvent(event)