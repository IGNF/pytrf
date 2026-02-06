# -*- coding: utf-8 -*-
"""
Widgets réutilisables
"""
from PyQt6.QtWidgets import (
    QComboBox, QTableWidget, QTableWidgetItem, 
    QCheckBox, QHBoxLayout, QWidget, QHeaderView
)
import os
import numpy as np
from datetime import datetime


class ComboBoxTS(QComboBox):
    """Combobox pour sélectionner une série temporelle"""
    
    def __init__(self, ts_path=None):
        super().__init__()
        if ts_path:
            self.fill_combobox(ts_path)
    
    def fill_combobox(self, ts_path: str):
        """Remplit la combobox avec les fichiers du dossier"""
        self.blockSignals(True)
        self.clear()
        if not ts_path or not os.path.isdir(ts_path):
            self.blockSignals(False) 
            return
        
        for name in sorted(os.listdir(ts_path)):
            self.addItem(name)
        self.blockSignals(False)


class Tabledate(QTableWidget):
    """Tableau pour gérer les dates de discontinuités"""
    
    def __init__(self, sitelog_path=None):
        super().__init__()
        
        self.setColumnCount(6)
        self.setColumnWidth(0, 80)  
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 30)
        self.setColumnWidth(3, 30)
        self.setColumnWidth(4, 100)
        self.setHorizontalHeaderLabels(["Date", "Info", "Pos", "Vel", "Exp", "Log"])
    
    def add_line(self, date_dis, info):
        """Ajoute une ligne au tableau"""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(date_dis)))
        self.setItem(row, 1, QTableWidgetItem(info))
        
        checkbox_pos = QCheckBox()
        checkbox_vel = QCheckBox()
        self.setCellWidget(row, 2, checkbox_pos)
        self.setCellWidget(row, 3, checkbox_vel)
        
        cell_exp = QWidget()
        cell_exp_layout = QHBoxLayout(cell_exp)
        cell_log = QWidget()
        cell_log_layout = QHBoxLayout(cell_log)
           
        for i in range(3):
            checkbox_exp = QCheckBox()
            checkbox_log = QCheckBox()
            cell_exp_layout.addWidget(checkbox_exp)
            cell_log_layout.addWidget(checkbox_log)
        
        self.setCellWidget(row, 4, cell_exp)
        self.setCellWidget(row, 5, cell_log)
    
    def get_dates(self):
        """Récupère les dates pour tracer les lignes verticales"""
        dates = []
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            info = self.item(row, 1).text()
            
            if item is None:
                continue
            txt = item.text().strip()
            try:
                dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
                dates.append([np.datetime64(dt), info])
            except ValueError:
                print(f"ignored date : {txt}")
        return dates
    
    def get_model_events(self):
        """Récupère les événements pour le modèle"""
        events = []
    
        for row in range(self.rowCount()):
            date_item = self.item(row, 0)
            info_item = self.item(row, 1)
    
            if date_item is None:
                continue
    
            pos = self.cellWidget(row, 2).isChecked()
            vel = self.cellWidget(row, 3).isChecked()
            
            if not (pos or vel):
                continue
             
            t = np.datetime64(date_item.text())
    
            events.append({
                "time": t,
                "info": info_item.text(),
                "pos": pos,
                "vel": vel,
            })
        
        return events

class TableOutliner(QTableWidget):
    """Tableau pour gérer les dates de discontinuités"""
    
    def __init__(self, sitelog_path=None):
        super().__init__()
    
        self.setRowCount(4)
        self.setColumnCount(5)
    
        labels = [
            "Normalised Residuals",
            "X times WRMS",
            "Median ± X times MAD",
            "Raw Residuals"
        ]
    
        for row, text in enumerate(labels):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags())
            self.setItem(row, 1, item)
    
        for row in range(4):
            checkbox = QCheckBox()
            checkbox.setStyleSheet("margin-left:8px;") 
            self.setCellWidget(row, 0, checkbox)
    
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 150)
        
        for col in range(2, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
    
            
            