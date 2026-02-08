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
        """
        Initialize the main application widget
    
        This constructor initializes the user interface and sets up the time 
        series manager and project-related attributes
        """
        super().__init__()
        self.initUI()
        
        
        #self.cache = CacheManager() #not finished
        
        self.ts_manager = TimeSeriesManager()
        
        # data of the project
        self.data = None
        self.name_sta = None
    
    def initUI(self):
        """"
        Initialize the graphical user interface
    
        This method builds and organizes all graphical components of the main
        window, including menus, control panels, plots, and interaction widgets. 
        It sets up the layout structure, initializes widgets, and
        connects user actions to their corresponding callbacks.
    
        The interface is divided into three main panels:
        - Left panel: deterministic and stochastic model configuration
        - Middle panel: time series visualization and data filtering
        - Right panel: residuals, periodogram, model display, and auxiliary views
    
        """
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
        
        #date table
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
        sto_model_layout.addWidget(QLabel("To implement"))
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
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        model_tab.setLayout(model_layout)
        
        # create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
      
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_content_layout)
        
        # Create label
        self.model_label = QLabel()
        self.model_label.setText('No model, no informations')
        self.model_label.setWordWrap(True)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        scroll_content_layout.addWidget(self.model_label)
        scroll_content_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        model_layout.addWidget(scroll_area)

        # Sitelog Tab
        sitelog_tab = QWidget()
        #to do...
        
        # Map and links Tab
        map_tab = QWidget()
        #to do...
        
        self.right_panel.addTab(residuals_tab, 'Residuals')
        self.right_panel.addTab(periodogram_tab, 'Periodogram')
        self.right_panel.addTab(model_tab, 'Model')
        self.right_panel.addTab(sitelog_tab, 'Sitelog')
        self.right_panel.addTab(map_tab, 'Map and Links')
        
        pannels_layout.addWidget(self.right_panel)
    
    # === Actions ===
    
    def newProject(self):
        """
        Create a new project

        This method opens the new project dialog, allowing the user to define a 
        new project configuration. When the time series path is updated, the 
        time series selector is refreshed accordingly.
        """
        
        dialog = DialogNewProject('new')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec() 
    
    def modifyProject(self):
        """
        Modify an existing project
    
        This method opens the project modification dialog, allowing the user to 
        update the configuration of the current project. When the time series
        path is changed, the time series selector is refreshed accordingly.
        """
        
        dialog = DialogNewProject('modify')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec()  
    
    def openProject(self):
        """
        Open an existing project
    
        This method opens a file selection dialog to load a project 
        configuration from a YAML file. It initializes the project data, loads 
        the associated time series, and restores the corresponding model if available.
    
        Depending on the project content, the model is either loaded from a 
        previously saved file, reconstructed from discontinuity solutions,
        or initialized with a default configuration. The method then updates
        all graphical components, including time series plots, residuals,
        periodogram, and model display.
    
        In case of an error during loading, an error message is displayed to the user.
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "YAML Files (*.yaml *.yml)"
        )
       
        try:
            self.data = read_yaml(file_path)
            print('file read')
            
                
            if self.data.ts_path:
                self.combobox_ts.fill_combobox(self.data.ts_path)
                self.name_sta = self.combobox_ts.currentText()[0:4]
                ts_file_path = os.path.join(
                    self.data.ts_path, 
                    self.name_sta + '_igs.xyz'
                )
           
                self.ts_manager.load_time_series(ts_file_path, auto_fit=False)
                model_file_path = os.path.join(
                    self.data.model_path, 
                    self.name_sta + '.pkl'
                )
                
                if os.path.isfile(model_file_path):
                    print('model computed from pkl')
                    self.ts_manager.m = model.load(model_file_path)
                    self.ts_manager.r = self.ts_manager.m.r
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
                        print('model created by default')
                        self.ts_manager._initialize_base_model()
                    
                    self.ts_manager.fit_model()
                
                
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
        """
        Update the time series plot
    
        This method updates the displayed time series according to the
        currently selected item in the time series combobox. It loads the
        corresponding data file, applies discontinuities if available, and
        refreshes the main time series plot.
    
        If no valid project or time series path is defined, the method
        returns without action.
    
        """
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
        """
        Update the residuals plot
    
        This method updates the residuals graph based on the selected
        display mode. Raw or normalized residuals are plotted depending
        on the state of the corresponding radio buttons.
         
        If no model is available, the method returns without action.
         
        """
        
        if self.ts_manager.m is None:
            return
        
        if self.raw_residuals_btn.isChecked():
            self.residual_graph.plot_raw_res(self.ts_manager.m)
        else:
            self.residual_graph.plot_norm_res(self.ts_manager.m)
    
    def update_psd(self):
        """
        Update the periodogram plot
    
        This method updates the PSD plot using the current model.
        If no model is available, the method returns
        without action.
    
        """
        if self.ts_manager.m is None:
            return
        
        self.periodogram_graph.plot_psd(self.ts_manager.m)
        
    def update_log_table(self, station: str):
        """
        Update the sitelog table (not optimized)
        
        This method updates the table of discontinuities using information
        extracted from the station sitelog file. Existing user selections
        are preserved whenever possible. Antenna and receiver changes are
        automatically added, and their association with position and
        velocity offsets is restored from the current model or previous
        user input.
        
        Parameters
        ----------
        station : str
            Station identifier used to select the corresponding sitelog
            information.
        
        """
        # save current state of the table
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
        
        # reinitialize table to update it
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
       
      
        sitelog_dates = []
        
        # add antenna changes
        for i in range(len(ant)):
            date_str = str(date.from_tsnx(ant[i].start))
            info_str = 'antenna change'
            sitelog_dates.append([date_str, info_str])
            
            self.date_table.add_line(date_str, info_str)
            last_row = self.date_table.rowCount() - 1
            
            # look if this date was saved
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
                if date_str in chg_pos:
                    self.date_table.cellWidget(last_row, 2).setChecked(True)
                if date_str in chg_vel:
                    self.date_table.cellWidget(last_row, 3).setChecked(True)

        # add receptor change
        for i in range(len(rec)):
            date_str = str(date.from_tsnx(rec[i].start))
            info_str = 'receptor change'
            sitelog_dates.append([date_str, info_str])
            
            self.date_table.add_line(date_str, info_str)
            last_row = self.date_table.rowCount() - 1
            
            # look if this date was saved
            found = False
            for saved_row in saved_rows:
                if saved_row[0] == date_str and saved_row[1] == info_str:
                    # Restore previous state
                    if saved_row[2]:  # pos
                        self.date_table.cellWidget(last_row, 2).setChecked(True)
                    if saved_row[3]:  # vel
                        self.date_table.cellWidget(last_row, 3).setChecked(True)
                    found = True
                    break

            if not found:
                if date_str in chg_pos:
                    self.date_table.cellWidget(last_row, 2).setChecked(True)
                if date_str in chg_vel:
                    self.date_table.cellWidget(last_row, 3).setChecked(True)
       
        # Restore added lines
        for saved_row in saved_rows:
            is_sitelog = False
            for sitelog_row in sitelog_dates:
                if saved_row[0] == sitelog_row[0] and saved_row[1] == sitelog_row[1]:
                    is_sitelog = True
                    break
           
            if not is_sitelog:
                self.date_table.add_line(saved_row[0], saved_row[1])
                last_row = self.date_table.rowCount() - 1
                
                if saved_row[2]:  # pos
                    self.date_table.cellWidget(last_row, 2).setChecked(True)
                if saved_row[3]:  # vel
                    self.date_table.cellWidget(last_row, 3).setChecked(True)
   
    # def remove_points(self):
    #     if self.ts_manager.r is None:
    #         QMessageBox.warning(
    #             self, 
    #             "No series loaded", 
    #             "Please load a time series first."
    #         )
    #         return
        
    #     threshold = self.sbox_threshold.value() 
        
    #     try:
    #         self.ts_manager.clean_time_series(threshold)
    #         self.update_ts_graph()
    #     except Exception as e:
    #         QMessageBox.critical(self, "Cleaning error", str(e))
    
    def remove_points(self):
        """
        (not working)
        Remove points with large error
        
        This method removes observations with large errors using a user-defined 
        threshold (by default 5). After cleaning, the model is re-fitted and all 
        relevant graphical components (time series, residuals, periodogram, and model 
        display) are updated.
        
        If no time series is loaded, a warning message is displayed.
        """
        if self.ts_manager.r is None:
            QMessageBox.warning(
                self, 
                "No series loaded", 
                "Please load a time series first."
            )
        
        threshold = self.sbox_threshold.value() 

        try:
            self.ts_manager.clean_time_series(threshold)
            
            # fit model after cleaning
            if self.ts_manager.m is not None:
                self.ts_manager.fit_model()
            # update ts_graph without re-loading the time serie
            if self.data and self.data.discontinuity_path:   
                discontinuities = self.date_table.get_dates()
            else: 
                discontinuities = []
            self.canvas.plot_data(
                self.ts_manager.r, 
                discontinuities, 
                model=self.ts_manager.m
            )
            
            # update residuals graph and psd graph
            if self.ts_manager.m is not None:
                if self.raw_residuals_btn.isChecked():
                    self.residual_graph.plot_raw_res(self.ts_manager.m)
                else:
                    self.residual_graph.plot_norm_res(self.ts_manager.m)
                self.periodogram_graph.plot_psd(self.ts_manager.m)
                self.model_label.setText(str(self.ts_manager.m))
            
        except Exception as e:
            QMessageBox.critical(self, "Cleaning error", str(e))

    
    def add_date(self):
        """
        Add a date in the table

        This method opens a dialog allowing the user to add a new
        date to the deterministic model configuration.
        """
        dialog = DialogNewDate(self)
        dialog.exec() 
    
    def adjust_model(self):
        """
        Fit the deterministic model
        
        This method fits or refines the deterministic model using the information 
        defined in the date table. The model is adjusted iteratively and all 
        graphical components are updated accordingly.
        
        If no time series is loaded, the method returns without action.
        """
        
        if self.ts_manager.r is None:
            return
        
        dates, infos, pos_checked, vel_checked = self.date_table.get_model_events()
  
        try:
            self.ts_manager.fit_model_iterative(dates, pos_checked, vel_checked)
           
            
            if self.data.discontinuity_path:   
                discontinuities = self.date_table.get_dates()
            else: 
                discontinuities = []
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
            
            # #save model in cache
            # if self.name_sta:
            #     self.cache.save_model(self.ts_manager.m, self.name_sta)
                
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))
            

    def save_model(self):
        """
        Save the current model to a file
         
        This method saves the currently fitted model to a pickle file
        selected by the user. If no model is available, a warning message
        is displayed.
        """
        
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
