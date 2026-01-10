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
    QTableWidgetItem, QCheckBox
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


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.ts_path = None
        self.r = None #object of ts.read
        
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

        btn_new.clicked.connect(self.action_new)
        btn_open.clicked.connect(self.open_yaml_file)
        btn_modify.clicked.connect(self.action_modify)

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
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec() 
        
        
    def action_modify(self):
        dialog =  DialogNewProject('modify')
        dialog.tsPathChanged.connect(self.combobox_ts.fill_combobox)
        dialog.exec()  

    def open_yaml_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "YAML Files (*.yaml *.yml)")
       
        try:
            data = read_yaml(file_path)
            self.text_edit.setPlainText(str(data))
            #print(data.__dict__)
            #fill combobox and draw first ts graph
            # if hasattr(data, "ts_path"):
            #     self.combobox_ts.fill_combobox(data.ts_path)
            #     self.ts_path = data.ts_path
            if data.ts_path:
                self.combobox_ts.fill_combobox(data.ts_path)
                self.ts_path = data.ts_path
            
                if self.combobox_ts.count() > 0:
                    self.update_ts_graph()
                    
            if data.sitelogs_path : 
                log_file = read_yaml(data.sitelogs_path)
                sta_file = get_sitelog('cksv', log_file )
                data = read_sitelog(sta_file[0])
                ant, rec = data[0], data[1]
                
                for i in range(len(ant)):
                    self.date_table.add_line(date.from_tsnx(ant[i].start), 'antenna change')
                
                for i in range(len(rec)):
                    self.date_table.add_line(date.from_tsnx(rec[i].start), 'receptor change')
                


        except Exception as e:
            print(e)
            self.text_edit.setPlainText("Erreur lecture YAML")     
    
    def load_ts_graph(self, ts_file_path):
        r = ts.read(
            ts_file_path,
            usecols=(2,4,5,6,7,8,9,10,11,12),
            format=('t','x','y','z','sx','sy','sz','cxy','cxz','cyz'),
            dtrd=1,
            rotate=True
        )
        self.r = r
        self.canvas.plot_data(r)
        
    
    def update_ts_graph(self):
        selected_name = self.combobox_ts.currentText()
        if not selected_name or not self.ts_path:
            return

        ts_file_path = os.path.join(self.ts_path, selected_name)
        if os.path.isfile(ts_file_path):
            self.load_ts_graph(ts_file_path)
            
            
    def remove_points(self):
        if self.r is None:
            QMessageBox.warning(self, "No series loaded", "Please load a time series first.")
            return
    
    
        threshold = self.sbox_threshold.value() 
        print(threshold)
        
        try:
            ts.clean_sigmas(self.r, threshold) 
            self.canvas.plot_data(self.r) 
        
        except Exception as e:
            QMessageBox.critical(self, "Cleaning error", str(e))

    def add_date(self):
        dialog = DialogNewDate(self)
        dialog.exec() 


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
    
    def plot_data(self, r):
       x = TSGraph.mjd_to_datetime(r.t)
       y = r.y*1e3
       labels = ['East[mm]', 'North[mm]',' Up[mm]']
       
       for i in range(3):
           self.axes[i].clear()
           self.axes[i].grid()
           #self.axes[i].scatter(x, y[:, i], s=3, c='black', zorder=10)
           #self.axes[i].errorbar(x, y[:,i], yerr = r.Q[:,i,i])
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
           self.axes[i].set_ylabel(labels[i], fontsize=8)
           self.axes[i].tick_params(axis='both', labelsize=7)
       solns = read_solns(r'C:\Users\loeva\Documents\tests_pytrf\soln.snx')
       m = model.from_solns(r, solns, code='CKSV',per=[365.25, 182.625],noise=['vw'])
       m.fit(finalize=False)
       for d in range(min(3, m.nd)):
        self.axes[d].plot(x, m[d].yc * 1e3, 'r', linewidth=2, zorder=14)

        if m[d].sc is not None:
            self.axes[d].fill_between(
                x,
                (m[d].yc - m[d].sc) * 1e3,
                (m[d].yc + m[d].sc) * 1e3,
                alpha=0.4,
                zorder=4
            )  
           
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

    

# --- discontinuity date list ---

class Tabledate(QTableWidget):
    
    def __init__(self, sitelog_path=None):
        super().__init__()
        
        self.setColumnCount(6)
        for i in range(1,5):
            self.setColumnWidth(i,50)
        self.setHorizontalHeaderLabels(["Date", "Position", "Velocity", "Exp", "Log", "Info"])

        
        if sitelog_path :
            self.fill_table(sitelog_path)
            
    def add_line(self, date_dis, info):
        row = self.rowCount()
        self.insertRow(row)
        print('hola', type(date_dis))
        
        self.setItem(row, 0, QTableWidgetItem( str(date_dis) ))
            
            
            
        for i in range(1,5):
            checkbox = QCheckBox()
            self.setCellWidget(row, i, checkbox)
        self.setItem(row, 5, QTableWidgetItem(info))



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
        self.line_date.setPlaceholderText("dd-mm-yyyy hh:mm:ss")
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
        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.showMaximized()
    sys.exit(app.exec())
    
    """
    l'enregistrement dans un fichier cache ne fonctionne pas, surement à cause du \ Utilisateur'
    """

