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
    """
    Dialog for creating or modifying a project.
    
    This dialog allows creating a new project or modifying an existing one
    by specifying paths to various required files and folders (time series,
    models, sitelogs, etc.).
    
    Parameters
    ----------
    mode : str
        Dialog operation mode. Accepted values:
        - 'new' : Create a new project
        - 'modify' : Modify an existing project
    
    Attributes
    ----------
    mode : str
        Current operation mode
    message : QLabel
        Label displaying a message to the user
    label_project_path : QLabel
        Path of the loaded project file
    label_ts_path : QLabel
        Path to the time series folder
    label_model_path : QLabel
        Path to the models folder
    label_sitelogs_path : QLabel
        Path to the sitelogs file
    label_discontinuity_path : QLabel
        Path to the discontinuities file
    label_psd_path : QLabel
        Path to the PSD file
    
    Signals
    -------
    tsPathChanged : pyqtSignal(str)
        Signal emitted when the time series path changes
    
    """
    
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
        
        # Button create/modify
        if self.mode == "new":
            btn_create = QPushButton("Create")
        elif self.mode == "modify":
            btn_create = QPushButton("Modify")
        
        general_layout.addWidget(btn_create)
        btn_create.clicked.connect(self.create_project)
        
        self.setLayout(general_layout)
    
    def load_yaml_project(self):
        """
        Load an existing YAML project.
        
        Opens a file dialog allowing the user to select a project YAML file.
        The project information is then loaded and displayed in the corresponding labels.
        
        Raises
        ------
        Exception
            If an error occurs while reading the YAML file, an error
            message is displayed in the message label.
        
        Notes
        -----
        Emits the tsPathChanged signal with the time series path
        if loading succeeds
        """
        
        
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
        """
        Opens a file dialog allowing the user to select the folder containing
        time series files. The selected path is displayed in the corresponding
        label.
        
        Notes
        -----
        Emits the tsPathChanged signal with the selected path.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select time series folder")
        if folder:
            self.label_ts_path.setText(folder)
            self.tsPathChanged.emit(folder)
    
    def load_model(self):
        """
        Select the model folder.
        
        Opens a file dialog allowing the user to select the folder containing
        model files. The selected path is displayed in the corresponding label.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select model folder")
        if folder:
            self.label_model_path.setText(folder)
    
    def load_sitelogs(self):
        """
        Select the sitelog file.
        
        Opens a file dialog allowing the user to select a sitelog file (.opt).
        The selected path is displayed in the corresponding label.
        """
        file, _ = QFileDialog.getOpenFileName(
            self, "Select sitelog file", "", "OPT Files (*.opt)"
        )
        if file:
            self.label_sitelogs_path.setText(file)
    
    def load_discontinuity(self):
        """
       Select the discontinuities file.
       
       Opens a file dialog allowing the user to select a discontinuities
       file in SINEX format (.snx). The selected path is displayed in
       the corresponding label.
       """
        file, _ = QFileDialog.getOpenFileName(
            self, "Select soln file", "", "SINEX Files (*.snx)"
        )
        if file:
            self.label_discontinuity_path.setText(file)
    
    def load_psd(self):
        """
       Select the PSD file.
       
       Opens a file dialog allowing the user to select a PSD file in
       SINEX format (.snx). The selected path is displayed in the corresponding label.
       """
        file, _ = QFileDialog.getOpenFileName(
            self, "Select PSD file", "", "SINEX Files (*.snx)"
        )
        if file:
            self.label_psd_path.setText(file)
    
    def create_project(self):
        """
        Create or save the project.
        
        Collects all file and folder paths specified in the interface and
        saves them to a YAML project file. Validates that mandatory fields
        (time series and model folder) are filled before saving.
        
        Returns
        -------
        None
        
        Notes
        -----
        Displays an error dialog if mandatory fields are not filled.
        Closes the dialog after successful save.
        """
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
    """
    Dialog for manually adding a discontinuity.
    
    This dialog allows the user to add a new discontinuity by specifying
    a date and a description (info).
    
    Parameters
    ----------
    parent : QWidget
        Parent widget, typically the main window containing the
        discontinuities table (date_table)
    
    Attributes
    ----------
    parent : QWidget
        Reference to the parent widget
    line_date : QLineEdit
        Input field for the date (format: yyyy-mm-dd hh:mm:ss)
    line_info : QLineEdit
        Input field for the discontinuity description
    
    """
    
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
        """
        Validate and add the date to the table.
        
        Checks that input fields are not empty, then adds a new row to
        the parent's discontinuities table with the entered date and
        information. Fields are then cleared to allow adding another
        discontinuity.
        
        Returns
        -------
        None
        
        Notes
        -----
        Displays a warning if the date field is empty. Calls accept()
        to close the dialog after successful addition.
        """
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
        """
        Handle dialog closing.
        
        Cleans up resources by explicitly deleting QLineEdit widgets
        before closing the dialog.
        
        Parameters
        ----------
        event : QCloseEvent
            Qt close event
        """
        self.line_date.deleteLater()
        self.line_info.deleteLater()
        super().closeEvent(event)