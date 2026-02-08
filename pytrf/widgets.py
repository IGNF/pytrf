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
    """
    Custom QComboBox for selecting time series files from a directory.
    
    Parameters
    ----------
    ts_path : str, optional
        Path to the directory containing time series files. If provided, the combobox 
        is automatically populated with files from this directory.
    """
    
    def __init__(self, ts_path=None):
        """
        Initialize the ComboBoxTS widget.
        
        Parameters
        ----------
        ts_path : str, optional
            Path to the directory containing time series files.
        """
        
        super().__init__()
        if ts_path:
            self.fill_combobox(ts_path)
    
    def fill_combobox(self, ts_path: str):
        """
        Populate the combobox with sorted filenames from the specified directory.
        
        Parameters
        ----------
        ts_path : str
            Path to the directory to scan for files. If the path is None or invalid,
            the combobox is cleared.
        
        Notes
        -----
        Blocks signals during population to prevent unwanted triggers
        Files are sorted alphabetically
        Clears existing items before populating

        """
        self.blockSignals(True)
        self.clear()
        if not ts_path or not os.path.isdir(ts_path):
            self.blockSignals(False) 
            return
        
        for name in sorted(os.listdir(ts_path)):
            self.addItem(name)
        self.blockSignals(False)


class Tabledate(QTableWidget):
    """
    Table widget for managing discontinuity dates with associated flags and checkboxes.
    """
    
    def __init__(self):
        """
        Initialize the Tabledate widget with predefined columns and headers.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.setColumnCount(6)
        self.setColumnWidth(0, 80)  
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 30)
        self.setColumnWidth(3, 30)
        self.setColumnWidth(4, 100)
        self.setHorizontalHeaderLabels(["Date", "Info", "Pos", "Vel", "Exp", "Log"])
    
    def add_line(self, date_dis, info):
        """
        Add a new row to the table with the specified date and information.
        
        Parameters
        ----------
        date_dis : str or datetime
            Discontinuity date to be displayed in the first column.
        
        info : str
            Information or description associated with the discontinuity.
        
        Returns
        -------
        None
        
        Notes
        -----
        Creates Position (Pos) and Velocity (Vel) checkboxes
        Vel checkbox is initially disabled and only enabled when Pos is checked
        """
        
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(date_dis)))
        self.setItem(row, 1, QTableWidgetItem(info))
        
        checkbox_pos = QCheckBox()
        checkbox_vel = QCheckBox()
        
        checkbox_vel.setEnabled(False)
        # Connect signal of checkbox_pos to activate/desactivate vel
        checkbox_pos.stateChanged.connect(
            lambda state, r=row: self._on_pos_changed(r, state)
        )
        
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
        """
        Retrieve all dates from the table with their associated information and checkbox states.
        
        Parameters
        ----------
        None
        
        Returns
        --------
        dates : list of list
            List where each element contains:
            - [0] : numpy.datetime64
            Parsed date
            - [1] : str
            Information text
            - [2] : bool
            Position checkbox state
            - [3] : bool
            Velocity checkbox state
            
        Notes
        ------
        Dates must be in format: "YYYY-MM-DD HH:MM:SS"
        """
        dates = []
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            info_item = self.item(row, 1)
            
            if item is None or info_item is None:
                continue
            
            txt = item.text().strip()
            info = info_item.text()
            
            # get checkbox status
            pos_widget = self.cellWidget(row, 2)
            vel_widget = self.cellWidget(row, 3)
            
            pos_checked = pos_widget.isChecked() if pos_widget else False
            vel_checked = vel_widget.isChecked() if vel_widget else False
            
            try:
                dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
                dates.append([np.datetime64(dt), info, pos_checked, vel_checked])  # ← 4 éléments
            except ValueError:
                print(f"ignored date : {txt}")
        return dates
    
    def _on_pos_changed(self, row, state):
        """
        Internal callback to handle Position checkbox state changes.
        
        Parameters
        ----------
        row : int
            Row index of the changed checkbox.
        state : int
            Qt CheckState value (2 for checked, 0 for unchecked).
        
        Returns
        --------
        None
        """
        checkbox_vel = self.cellWidget(row, 3)
        if checkbox_vel:
            if state == 2:  # Qt.CheckState.Checked
                checkbox_vel.setEnabled(True)
            else:
                checkbox_vel.setEnabled(False)
                checkbox_vel.setChecked(False)
    
    
    def get_model_events(self):
        """
        Retrieve discontinuity events formatted for model processing.
        
        Parameters
        ----------
        None
        
        Returns
        --------
        events : list of arrays
            List containing 4 elements:
            - [0] : numpy.ndarray of datetime64
            Array of event dates (empty array if no events)
            - [1] : list of str
            List of information strings
            - [2] : numpy.ndarray of bool
            Array of position flags
            - [3] : numpy.ndarray of bool
            Array of velocity flags
            
        Notes
        ------
        Only includes rows where at least one checkbox (Pos or Vel) is checked
        Returns empty arrays if no events are selected
        """
        
        dates = []
        infos = []
        pos_flags = []
        vel_flags = []
    
        for row in range(self.rowCount()):
            date_item = self.item(row, 0)
            info_item = self.item(row, 1)
    
            if date_item is None:
                continue
    
            pos = self.cellWidget(row, 2).isChecked()
            vel = self.cellWidget(row, 3).isChecked()
            
            if not (pos or vel):
                continue
             
            dates.append(np.datetime64(date_item.text()))
            infos.append(info_item.text())
            pos_flags.append(pos)
            vel_flags.append(vel)
        
        events = [
            np.array(dates) if dates else np.array([], dtype='datetime64'),
            infos,
            np.array(pos_flags, dtype=bool) if pos_flags else np.array([], dtype=bool),
            np.array(vel_flags, dtype=bool) if vel_flags else np.array([], dtype=bool)
        ]
        print('events', events)
        return events

class TableOutliner(QTableWidget):
    """
    Table widget for configuring outlier detection methods and parameters.
    """
    
    def __init__(self, sitelog_path=None):
        """
        Initialize the TableOutliner widget with predefined outlier detection methods.
        
        Parameters
        ----------
        sitelog_path : str, optional
            Path to sitelog file (reserved for future use).
        
        Returns
        --------
        None
        """
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
    
            
            