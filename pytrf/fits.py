'''
fits: pytrf GUI for fitting time series

'''

# External imports
#-----------------
import sys
import matplotlib.pyplot as pp
pp.rcParams['font.family'] = 'monospace'
pp.rcParams['font.size'] = 12
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QComboBox, QTabWidget, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QFileDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QLineEdit, QSizePolicy, QScrollArea, QPushButton, QSpinBox, QMessageBox
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from traceback import print_exc

# Internal imports
#-----------------

# Current application settings
#-----------------------------
settings = QSettings('fits', 'IGN')



# Create and launch fits application
#-----------------------------------
def main():
    
    '''
    Create and launch tsfit application
    
    '''

    # Create the app
    app = QApplication(sys.argv)
        
    # Create the window and display it
    w = mainWindow()
    w.show()

    # Execute the app
    app.exec_()


    
# fits mainWindow class
#----------------------
class mainWindow(QMainWindow):
  
    '''
    fits Main Window class
    
    '''
    
    # fits mainWindow initialization
    #-------------------------------
    def __init__(self):
      
        '''
        fits mainWindow initialization

        '''

        # Initialize window as QMainWindow
        QMainWindow.__init__(self)

        # Ancillary windows
        #------------------
        
        # Read options window
        self.readOptionsWin = readOptionsWindow()
        self.readOptionsWin.OKSignal.connect(self.getReadOptions)        
        
        # Actions
        #--------
        
        # New working directory
        self.newWorkDirAction = QAction('New working directory', self)
        self.newWorkDirAction.setShortcut('Ctrl+N')
        self.newWorkDirAction.setStatusTip('Create new working directory (Ctrl+N)')
        self.newWorkDirAction.triggered.connect(self.newWorkDir)
        
        # Open working directory
        self.openWorkDirAction = QAction('Open working directory', self)
        self.openWorkDirAction.setShortcut('Ctrl+O')
        self.openWorkDirAction.triggered.connect(self.openWorkDir)
        
        # Load input discontinuity list
        self.loadSolnAction = QAction('Load discontinuity list', self)
        self.loadSolnAction.setShortcut('Ctrl+D')
        self.loadSolnAction.triggered.connect(self.loadSoln)

        # Load PSD models
        self.loadPSDAction = QAction('Load PSD models', self)
        self.loadPSDAction.setShortcut('Ctrl+P')
        self.loadPSDAction.triggered.connect(self.loadPSD)

        # Load site log source list
        self.loadLogSourceAction = QAction('Load site log source list', self)
        self.loadLogSourceAction.setShortcut('Ctrl+L')
        self.loadLogSourceAction.triggered.connect(self.loadLogSource)
        
        # Load CMT catalog
        self.loadCMTAction = QAction('Load CMT catalog', self)
        self.loadCMTAction.setShortcut('Ctrl+Q')
        self.loadCMTAction.triggered.connect(self.loadCMT)        
        
        # Configure
        self.configureAction = QAction('Configure', self)
        self.configureAction.setShortcut('Ctrl+C')
        self.configureAction.setStatusTip('Configure fits (Ctrl+C)')
        self.configureAction.triggered.connect(self.configure)
        
        # Previous station
        self.previousStationAction = QAction('Previous', self)
        self.previousStationAction.setToolTip('Previous station (Ctrl+Up)')
        self.previousStationAction.setShortcut('Ctrl+Up')
        self.previousStationAction.triggered.connect(self.previousStation)

        # Next station
        self.nextStationAction = QAction('Next', self)
        self.nextStationAction.setToolTip('Next station (Ctrl+Down)')
        self.nextStationAction.setShortcut('Ctrl+Down')
        self.nextStationAction.triggered.connect(self.nextStation)

        # Main widgets
        #-------------
        
        # Menu bar
        menu = self.menuBar()
        fileMenu = menu.addMenu('&File')
        fileMenu.addAction(self.newWorkDirAction)
        fileMenu.addAction(self.openWorkDirAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.loadSolnAction)
        fileMenu.addAction(self.loadPSDAction)
        fileMenu.addAction(self.loadLogSourceAction)
        fileMenu.addAction(self.loadCMTAction)
        
        settingsMenu = menu.addMenu('&Settings')
        settingsMenu.addAction(self.configureAction)
        
        # Tool bar
        toolbar = self.addToolBar('')

        # Station list combobox
        self.stationList = QComboBox()
        self.stationList.activated.connect(self.changeCurrentStation)
        toolbar.addWidget(self.stationList)

        # Action buttons
        toolbar.addSeparator()
        toolbar.addAction(self.previousStationAction)
        toolbar.addAction(self.nextStationAction)

        # Central widget = tab container
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Status bar
        self.statusBar().showMessage('Welcome!')
        #self.statusBar().messageChanged.connect(self.updateStatus)

        # Plots tab
        #----------
        
        # Left figure
        self.leftFigure = pp.figure(1)
        self.leftCanvas = FigureCanvas(self.leftFigure)
        #self.leftCanvas.mpl_connect('button_press_event',self.figureClickedEvent)
        leftToolbar = NavigationToolbar(self.leftCanvas, self)
        
        # Right figure
        self.rightFigure = pp.figure(2)
        self.rightCanvas = FigureCanvas(self.rightFigure)
        #self.rightCanvas.mpl_connect('button_prights_event',self.figureClickedEvent)
        rightToolbar = NavigationToolbar(self.rightCanvas, self)

        # Control panel
        panel = QWidget()
        
        # Plots tab
        tab = QWidget()
        layout = QGridLayout()
        layout.addWidget(leftToolbar, 0, 0)
        layout.addWidget(rightToolbar, 0, 1)
        layout.addWidget(self.leftCanvas, 1, 0)
        layout.addWidget(self.rightCanvas, 1, 1)
        layout.addWidget(panel, 1, 2)
        tab.setLayout(layout)
        tabs.addTab(tab, 'Plots')
        
        # Settings
        #---------
        
        # Get default settings
        self.work_dir = settings.value('work_dir', None)




    # Close event
    #------------
    def closeEvent(self, event):
        
        '''
        Close event
        
        '''
        
        self.readOptionsWin.hide()
        event.accept()



    # Create new working directory
    #-----------------------------
    def newWorkDir(self):

        '''
        Create new working directory
        
        '''

        # Select new working directory
        dir = QFileDialog.getExistingDirectory(None, 'Select new working directory:', self.work_dir, QFileDialog.ShowDirsOnly)
        
        # Select input data files
        files = QFileDialog.getOpenFileNames(None, 'Select input data files:', dir)[0]

        # Configure and show read options window
        self.readOptionsWin.dir = dir
        self.readOptionsWin.editorLabel.setText('Content of '+files[0]+':')
        self.readOptionsWin.editor.setPlainText(open(files[0], encoding='ISO-8859-1').read())
        self.readOptionsWin.showMaximized()
        
        

    # Open existing working directory
    #--------------------------------
    def openWorkDir(self):

        '''
        Open existing working directory
        
        '''

        pass



    # Load input discontinuity list
    #------------------------------
    def loadSoln(self):

        '''
        Load input discontinuity list
        
        '''

        pass



    # Load post-seismic deformation models
    #-------------------------------------
    def loadPSD(self):

        '''
        Load post-seismic deformation models
        
        '''

        pass



    # Load site log source list
    #--------------------------
    def loadLogSource(self):

        '''
        Load site log source list
        
        '''

        pass



    # Load CMT catalog
    #-----------------
    def loadCMT(self):

        '''
        Load CMT catalog
        
        '''

        pass



    # Configure fits
    #---------------
    def configure(self):

        '''
        Configure fits
        
        '''

        pass



    # Go to previous station
    #-------------------
    def previousStation(self):

        '''
        Go to previous station
        
        '''

        pass



    # Go to next station
    #-------------------
    def nextStation(self):

        '''
        Go to next station
        
        '''

        pass



    # Change current station
    #-----------------------
    def changeCurrentStation(self):

        '''
        Change current station
        
        '''

        pass



    # Get read options
    #-----------------
    def getReadOptions(self):

        '''
        Get read options
        
        '''

        print('Get read options')



# fits readOptionsWindow class
#-----------------------------
class readOptionsWindow(QScrollArea):
  
    '''
    fits readOptionsWindow class
    
    '''
    
    # Signals
    #--------
    
    OKSignal = pyqtSignal()



    
    # fits readOptionsWindow initialization
    #--------------------------------------
    def __init__(self):
      
        '''
        fits readOptionsWindow initialization

        '''

        # Initialize window as QScrollArea
        QScrollArea.__init__(self)
        self.setWidgetResizable(True)
        self.setWindowTitle('Set reading and preprocessing options:')
        
        # Main widgets
        #-------------
        
        # Main widget
        mainWidget = QWidget()
        
        # Editor label
        self.editorLabel = QLabel()
        
        # Text editor
        self.editor = editor()
        self.editor.setMinimumHeight(180)
        
        # Known format label
        knownFormatLabel = QLabel('Is this format already known to pytrf?')
        
        # Known format combobox
        self.knownFormatBox = QComboBox()
        self.knownFormatBox.addItems(['No', 'pytrf .ts format', 'JPL .series/.resid format', 'NGL .txyz2 format', 'NGL .tenv3 format'])
        self.knownFormatBox.setEditable(True)
        self.knownFormatBox.setFixedSize(160, 25)        
        self.knownFormatBox.currentIndexChanged.connect(self.knownFormatBoxChange)
        
        # Format panel
        formatPanel = QWidget()

        # Columns label
        columnsLabel = QLabel('Column indices and contents (please use zero-based indexing):')

        # Columns panel
        columnsPanel = QWidget()

        # Bottom panel
        bottomPanel = QWidget()

        # OK button
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.OKClicked)

        # Format panel widgets
        #---------------------

        # Header lines label
        headerLinesLabel = QLabel('Nb. header lines:')
        headerLinesLabel.setFixedWidth(140)

        # Header lines combobox
        self.headerLinesBox = QSpinBox()
        self.headerLinesBox.setMinimum(0)
        self.headerLinesBox.setMaximum(10000)
        self.headerLinesBox.setFixedSize(80, 25)

        # Separator label
        separatorLabel = QLabel('Column separator:')
        separatorLabel.setFixedWidth(140)

        # Separator combobox
        self.separatorBox = QComboBox()
        self.separatorBox.addItems(['space', ',', ';'])
        self.separatorBox.setEditable(True)
        self.separatorBox.setFixedSize(80, 25)

        # Comment label
        commentLabel = QLabel('Comment indicator:')
        commentLabel.setFixedWidth(140)

        # Comment combobox
        self.commentBox = QComboBox()
        self.commentBox.addItems(['#'])
        self.commentBox.setEditable(True)
        self.commentBox.setFixedSize(80, 25)

        # Number of dimensions label
        ndimLabel = QLabel('Nb. dimensions:')
        ndimLabel.setFixedWidth(140)

        # Number of dimensions combobox
        self.ndimBox = QComboBox()
        self.ndimBox.addItems(['1', '2', '3'])
        self.ndimBox.setFixedSize(80, 25)
        self.ndimBox.setCurrentIndex(2)
        self.ndimBox.currentIndexChanged.connect(self.ndimBoxChange)

        # Rotation label
        rotateLabel = QLabel('XYZ -> ENH ?')
        rotateLabel.setFixedWidth(140)
        
        # Rotation combobox
        self.rotateBox = QComboBox()
        self.rotateBox.addItems(['Yes', 'No'])
        self.rotateBox.setFixedSize(80, 25)
        self.rotateBox.currentIndexChanged.connect(self.rotateBoxChange)

        # Dimension 1 name label
        dim1Label = QLabel('Dim. 1 name:')
        dim1Label.setFixedWidth(140)

        # Dimension 1 name combobox
        self.dim1Box = QComboBox()
        self.dim1Box.addItems(['East', 'X'])
        self.dim1Box.setEditable(True)
        self.dim1Box.setFixedSize(80, 25)
        self.dim1Box.setEnabled(False)
        
        # Dimension 2 name label
        dim2Label = QLabel('Dim. 2 name:')
        dim2Label.setFixedWidth(140)

        # Dimension 2 name combobox
        self.dim2Box = QComboBox()
        self.dim2Box.addItems(['North', 'Y'])
        self.dim2Box.setEditable(True)
        self.dim2Box.setFixedSize(80, 25)
        self.dim2Box.setEnabled(False)
        
        # Dimension 3 name label
        dim3Label = QLabel('Dim. 3 name:')
        dim3Label.setFixedWidth(140)

        # Dimension 3 name combobox
        self.dim3Box = QComboBox()
        self.dim3Box.addItems(['Up', 'Z'])
        self.dim3Box.setEditable(True)
        self.dim3Box.setFixedSize(80, 25)
        self.dim3Box.setEnabled(False)
        
        # Format panel layout
        #--------------------
        
        layout = QGridLayout()
 
        layout.addWidget(headerLinesLabel, 0, 0)
        layout.addWidget(self.headerLinesBox, 1, 0)

        layout.addWidget(separatorLabel, 0, 1)
        layout.addWidget(self.separatorBox, 1, 1)

        layout.addWidget(commentLabel, 0, 2)
        layout.addWidget(self.commentBox, 1, 2)

        layout.addWidget(ndimLabel, 0, 3)
        layout.addWidget(self.ndimBox, 1, 3)

        layout.addWidget(rotateLabel, 0, 4)
        layout.addWidget(self.rotateBox, 1, 4)

        layout.addWidget(dim1Label, 0, 5)
        layout.addWidget(self.dim1Box, 1, 5)

        layout.addWidget(dim2Label, 0, 6)
        layout.addWidget(self.dim2Box, 1, 6)

        layout.addWidget(dim3Label, 0, 7)
        layout.addWidget(self.dim3Box, 1, 7)

        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        formatPanel.setLayout(layout)
        
        # Columns panel widgets
        #----------------------

        # Time column label
        timeColumnLabel = QLabel('Column index of time:')
        
        # Time column combobox
        self.timeColumnBox = QComboBox()
        self.timeColumnBox.addItems([str(i) for i in range(10)])
        self.timeColumnBox.setEditable(True)
        self.timeColumnBox.setFixedSize(80, 25)

        # Time unit label
        timeUnitLabel = QLabel('Time unit:')

        # Time unit combobox
        self.timeUnitBox = QComboBox()
        self.timeUnitBox.addItems(['MJD'])
        self.timeUnitBox.setEditable(True)
        self.timeUnitBox.setFixedSize(80, 25)
        self.timeUnitBox.currentTextChanged.connect(self.timeUnitBoxChange)

        # Integration intervals column label
        TColumnLabel = QLabel('Column index of integration intervals:')
        
        # Integration intervals column combobox
        self.TColumnBox = QComboBox()
        self.TColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.TColumnBox.setEditable(True)
        self.TColumnBox.setFixedSize(80, 25)
        self.TColumnBox.setCurrentIndex(0)
        self.TColumnBox.currentTextChanged.connect(self.TColumnBoxChange)

        # Integration intervals label
        TLabel = QLabel('Constant interval:')

        # Integration intervals validator
        self.TBox = QLineEdit('1')
        self.TBox.setFixedSize(80, 25)

        # Integration intervals unit label
        self.TUnitLabel = QLabel('[MJD]')

        # 1st dimension column label
        dim1ColumnLabel = QLabel('Column index of dimension 1 values:')
        
        # 1st dimension column combobox
        self.dim1ColumnBox = QComboBox()
        self.dim1ColumnBox.addItems([str(i) for i in range(10)])
        self.dim1ColumnBox.setEditable(True)
        self.dim1ColumnBox.setFixedSize(80, 25)
        self.dim1ColumnBox.setCurrentIndex(1)

        # Time series unit label
        seriesUnitLabel = QLabel('Time series unit:')

        # Time series unit combobox
        self.seriesUnitBox = QComboBox()
        self.seriesUnitBox.addItems(['m', 'mm'])
        self.seriesUnitBox.setEditable(True)
        self.seriesUnitBox.setFixedSize(80, 25)
        self.seriesUnitBox.currentTextChanged.connect(self.seriesUnitBoxChange)

        # 2nd dimension column label
        dim2ColumnLabel = QLabel('Column index of dimension 2 values:')
        
        # 2nd dimension column combobox
        self.dim2ColumnBox = QComboBox()
        self.dim2ColumnBox.addItems([str(i) for i in range(10)])
        self.dim2ColumnBox.setEditable(True)
        self.dim2ColumnBox.setFixedSize(80, 25)
        self.dim2ColumnBox.setCurrentIndex(2)

        # 3rd dimension column label
        dim3ColumnLabel = QLabel('Column index of dimension 3 values:')
        
        # 3rd dimension column combobox
        self.dim3ColumnBox = QComboBox()
        self.dim3ColumnBox.addItems([str(i) for i in range(10)])
        self.dim3ColumnBox.setEditable(True)
        self.dim3ColumnBox.setFixedSize(80, 25)
        self.dim3ColumnBox.setCurrentIndex(3)

        # 1st dimension error column label
        dim1ErrorColumnLabel = QLabel('Column index of dimension 1 errors:')
        
        # 1st dimension error column combobox
        self.dim1ErrorColumnBox = QComboBox()
        self.dim1ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim1ErrorColumnBox.setEditable(True)
        self.dim1ErrorColumnBox.setFixedSize(80, 25)
        self.dim1ErrorColumnBox.setCurrentIndex(5)

        # Error type label
        errorTypeLabel = QLabel('Type of errors:')
        
        # Error type combobox
        self.errorTypeBox = QComboBox()
        self.errorTypeBox.addItems(['std', 'var'])
        self.errorTypeBox.setFixedSize(80, 25)
        self.errorTypeBox.setCurrentIndex(0)
        self.errorTypeBox.currentTextChanged.connect(self.errorTypeBoxChange)

        # Error unit label
        self.errorUnitLabel = QLabel('[m]')

        # 2nd dimension error column label
        dim2ErrorColumnLabel = QLabel('Column index of dimension 2 errors:')

        # 2nd dimension error column combobox
        self.dim2ErrorColumnBox = QComboBox()
        self.dim2ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim2ErrorColumnBox.setEditable(True)
        self.dim2ErrorColumnBox.setFixedSize(80, 25)
        self.dim2ErrorColumnBox.setCurrentIndex(6)

        # 3rd dimension error column label
        dim3ErrorColumnLabel = QLabel('Column index of dimension 3 errors:')

        # 3rd dimension error column combobox
        self.dim3ErrorColumnBox = QComboBox()
        self.dim3ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim3ErrorColumnBox.setEditable(True)
        self.dim3ErrorColumnBox.setFixedSize(80, 25)
        self.dim3ErrorColumnBox.setCurrentIndex(7)

        # 1st/2nd dimensions co-error column label
        dim12ErrorColumnLabel = QLabel('Column index of dim. 1/2 co-errors:')
        
        # 1st/2nd dimensions co-error column combobox
        self.dim12ErrorColumnBox = QComboBox()
        self.dim12ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim12ErrorColumnBox.setEditable(True)
        self.dim12ErrorColumnBox.setFixedSize(80, 25)
        self.dim12ErrorColumnBox.setCurrentIndex(8)

        # Co-error type label
        coerrorTypeLabel = QLabel('Type of co-errors:')
        
        # Co-error type combobox
        self.coerrorTypeBox = QComboBox()
        self.coerrorTypeBox.addItems(['corr', 'cov'])
        self.coerrorTypeBox.setFixedSize(80, 25)
        self.coerrorTypeBox.setCurrentIndex(0)
        self.coerrorTypeBox.currentTextChanged.connect(self.coerrorTypeBoxChange)

        # Co-error unit label
        self.coerrorUnitLabel = QLabel('[]')

        # 1st/3rd dimensions co-error column label
        dim13ErrorColumnLabel = QLabel('Column index of dim. 1/3 co-errors:')
        
        # 1st/3rd dimensions co-error column combobox
        self.dim13ErrorColumnBox = QComboBox()
        self.dim13ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim13ErrorColumnBox.setEditable(True)
        self.dim13ErrorColumnBox.setFixedSize(80, 25)
        self.dim13ErrorColumnBox.setCurrentIndex(9)

        # 2nd/3rd dimensions co-error column label
        dim23ErrorColumnLabel = QLabel('Column index of dim. 2/3 co-errors:')
        
        # 2nd/3rd dimensions co-error column combobox
        self.dim23ErrorColumnBox = QComboBox()
        self.dim23ErrorColumnBox.addItems(['None'] + [str(i) for i in range(10)])
        self.dim23ErrorColumnBox.setEditable(True)
        self.dim23ErrorColumnBox.setFixedSize(80, 25)
        self.dim23ErrorColumnBox.setCurrentIndex(10)

        # Columns panel layout
        #---------------------
        
        layout = QGridLayout()
 
        layout.addWidget(timeColumnLabel, 0, 0)
        layout.addWidget(self.timeColumnBox, 0, 1)
        layout.addWidget(timeUnitLabel, 0, 2)
        layout.addWidget(self.timeUnitBox, 0, 3)
        
        layout.addWidget(TColumnLabel, 1, 0)
        layout.addWidget(self.TColumnBox, 1, 1)
        layout.addWidget(TLabel, 1, 2)
        layout.addWidget(self.TBox, 1, 3)
        layout.addWidget(self.TUnitLabel, 1, 4)

        layout.addWidget(dim1ColumnLabel, 2, 0)
        layout.addWidget(self.dim1ColumnBox, 2, 1)
        layout.addWidget(seriesUnitLabel, 2, 2)
        layout.addWidget(self.seriesUnitBox, 2, 3)
        
        layout.addWidget(dim2ColumnLabel, 3, 0)
        layout.addWidget(self.dim2ColumnBox, 3, 1)
        
        layout.addWidget(dim3ColumnLabel, 4, 0)
        layout.addWidget(self.dim3ColumnBox, 4, 1)

        layout.addWidget(dim1ErrorColumnLabel, 5, 0)
        layout.addWidget(self.dim1ErrorColumnBox, 5, 1)
        layout.addWidget(errorTypeLabel, 5, 2)
        layout.addWidget(self.errorTypeBox, 5, 3)
        layout.addWidget(self.errorUnitLabel, 5, 4)

        layout.addWidget(dim2ErrorColumnLabel, 6, 0)
        layout.addWidget(self.dim2ErrorColumnBox, 6, 1)
        
        layout.addWidget(dim3ErrorColumnLabel, 7, 0)
        layout.addWidget(self.dim3ErrorColumnBox, 7, 1)
        
        layout.addWidget(dim12ErrorColumnLabel, 8, 0)
        layout.addWidget(self.dim12ErrorColumnBox, 8, 1)
        layout.addWidget(coerrorTypeLabel, 8, 2)
        layout.addWidget(self.coerrorTypeBox, 8, 3)
        layout.addWidget(self.coerrorUnitLabel, 8, 4)

        layout.addWidget(dim13ErrorColumnLabel, 9, 0)
        layout.addWidget(self.dim13ErrorColumnBox, 9, 1)

        layout.addWidget(dim23ErrorColumnLabel, 10, 0)
        layout.addWidget(self.dim23ErrorColumnBox, 10, 1)

        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        columnsPanel.setLayout(layout)
        columnsPanel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Bottom panel widgets
        #---------------------
        
        # Detrending degree label
        dtrdLabel = QLabel('Degree of detrending polynomial:')
        dtrdLabel.setFixedWidth(280)
        
        # Detrending degree combobox
        self.dtrdBox = QComboBox()
        self.dtrdBox.addItems(['None', '0', '1'])
        self.dtrdBox.setFixedSize(80, 25)
        self.dtrdBox.setCurrentIndex(2)
        
        # Figure time unit label
        figureTimeUnitLabel = QLabel('Time unit in figures:')
        figureTimeUnitLabel.setFixedWidth(280)
        
        # Figure time unit combobox
        self.figureTimeUnitBox = QComboBox()
        self.figureTimeUnitBox.addItems(['Decimal year', 'MJD'])
        self.figureTimeUnitBox.setFixedSize(120, 25)
        
        # Approximate XYZ coordinates label
        XYZLabel = QLabel('Approximate XYZ coord. (for station map):')
        XYZLabel.setFixedWidth(280)
        
        # Approximate XYZ panel
        XYZPanel = QWidget()
        
        # Approximate XYZ coordinates line edit
        self.XYZEdit = QLineEdit()
        self.XYZEdit.setEnabled(False)
        self.XYZEdit.setFixedSize(240, 25)
        
        # Approximate XYZ coordinates button
        self.XYZButton = QPushButton('...')
        self.XYZButton.setFixedSize(30, 25)
        self.XYZButton.setEnabled(False)
        self.XYZButton.clicked.connect(self.getXYZfile)

        # Approximate XYZ panel layout
        layout = QHBoxLayout()
        layout.addWidget(self.XYZEdit)
        layout.addWidget(self.XYZButton)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        XYZPanel.setLayout(layout)

        # Bottom panel layout
        #--------------------

        layout = QGridLayout()

        layout.addWidget(dtrdLabel, 0, 0)
        layout.addWidget(self.dtrdBox, 1, 0)
        
        layout.addWidget(figureTimeUnitLabel, 0, 1)
        layout.addWidget(self.figureTimeUnitBox, 1, 1)

        layout.addWidget(XYZLabel, 0, 2)
        layout.addWidget(XYZPanel, 1, 2)

        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        bottomPanel.setLayout(layout)
        bottomPanel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Main widget layout
        #-------------------
        
        layout = QVBoxLayout()
        
        layout.addWidget(self.editorLabel)
        layout.addWidget(self.editor)

        layout.addWidget(QLabel())
        layout.addWidget(knownFormatLabel)
        layout.addWidget(self.knownFormatBox)

        layout.addWidget(QLabel())
        layout.addWidget(formatPanel)
                
        layout.addWidget(QLabel())
        layout.addWidget(columnsLabel)
        layout.addWidget(columnsPanel)
        
        layout.addWidget(QLabel())
        layout.addWidget(bottomPanel)

        layout.addWidget(buttonBox)
        
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        mainWidget.setLayout(layout)
        mainWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set main widget as scroll area widget
        #--------------------------------------
        
        self.setWidget(mainWidget)
        


    # Change in known format box
    #---------------------------
    def knownFormatBoxChange(self):
      
        '''
        Change in known format box

        '''
        
        # Enable/disable lots of boxes
        if (self.knownFormatBox.currentText() != 'No'):
            self.headerLinesBox.setEnabled(False)
            self.separatorBox.setEnabled(False)
            self.commentBox.setEnabled(False)
            self.ndimBox.setEnabled(False)
            self.dim1Box.setEnabled(False)
            self.dim2Box.setEnabled(False)
            self.dim3Box.setEnabled(False)
            self.timeColumnBox.setEnabled(False)
            self.timeUnitBox.setEnabled(False)
            self.TColumnBox.setEnabled(False)
            self.TBox.setEnabled(False)
            self.dim1ColumnBox.setEnabled(False)
            self.seriesUnitBox.setEnabled(False)
            self.dim2ColumnBox.setEnabled(False)
            self.dim3ColumnBox.setEnabled(False)
            self.dim1ErrorColumnBox.setEnabled(False)
            self.errorTypeBox.setEnabled(False)
            self.dim2ErrorColumnBox.setEnabled(False)
            self.dim3ErrorColumnBox.setEnabled(False)
            self.dim12ErrorColumnBox.setEnabled(False)
            self.coerrorTypeBox.setEnabled(False)
            self.dim13ErrorColumnBox.setEnabled(False)
            self.dim23ErrorColumnBox.setEnabled(False)
            self.rotateBox.setEnabled(False)
        else:
            self.headerLinesBox.setEnabled(True)
            self.separatorBox.setEnabled(True)
            self.commentBox.setEnabled(True)
            self.ndimBox.setEnabled(True)
            self.dim1Box.setEnabled(True)
            self.dim2Box.setEnabled(True)
            self.dim3Box.setEnabled(True)
            self.timeColumnBox.setEnabled(True)
            self.timeUnitBox.setEnabled(True)
            self.TColumnBox.setEnabled(True)
            self.TBox.setEnabled(True)
            self.dim1ColumnBox.setEnabled(True)
            self.seriesUnitBox.setEnabled(True)
            self.dim2ColumnBox.setEnabled(True)
            self.dim3ColumnBox.setEnabled(True)
            self.dim1ErrorColumnBox.setEnabled(True)
            self.errorTypeBox.setEnabled(True)
            self.dim2ErrorColumnBox.setEnabled(True)
            self.dim3ErrorColumnBox.setEnabled(True)
            self.dim12ErrorColumnBox.setEnabled(True)
            self.coerrorTypeBox.setEnabled(True)
            self.dim13ErrorColumnBox.setEnabled(True)
            self.dim23ErrorColumnBox.setEnabled(True)
            self.rotateBox.setEnabled(True)

        # Enable/disable approximate XYZ button and edit
        if (self.knownFormatBox.currentText() in ['pytrf .ts format', 'NGL .txyz2 format']):
            self.XYZButton.setEnabled(False)
        elif (self.knownFormatBox.currentText() == 'No') and (self.ndimBox.currentText() == '3') and (self.rotateBox.currentText == 'False'):
            self.XYZButton.setEnabled(True)
        else:
            self.XYZButton.setEnabled(False)
            
        # Enable/disable figure time unit box
        if (self.knownFormatBox.currentText() != 'No'):
            self.figureTimeUnitBox.setEnabled(True)
        elif (self.timeUnitBox.currentText() == 'MJD'):
            self.figureTimeUnitBox.setEnabled(True)
        else:
            self.figureTimeUnitBox.setEnabled(False)
            


    # Change in number of dimensions box
    #-----------------------------------
    def ndimBoxChange(self):
      
        '''
        Change in number of dimensions box

        '''
        
        # Change rotate box value
        if (int(self.ndimBox.currentText()) < 3):
            self.rotateBox.setCurrentIndex(1)
        
        # Enable/disable 2nd and 3rd dimension boxes, rotation box and approximate XYZ button
        if (int(self.ndimBox.currentText()) < 3):
            self.dim3ColumnBox.setEnabled(False)
            self.dim3ErrorColumnBox.setEnabled(False)
            self.dim13ErrorColumnBox.setEnabled(False)
            self.dim23ErrorColumnBox.setEnabled(False)
            self.dim3Box.setEnabled(False)
            self.rotateBox.setEnabled(False)
            self.XYZButton.setEnabled(False)
        else:
            self.dim3ColumnBox.setEnabled(True)
            self.dim3ErrorColumnBox.setEnabled(True)
            self.dim13ErrorColumnBox.setEnabled(True)
            self.dim23ErrorColumnBox.setEnabled(True)
            self.dim3Box.setEnabled(True)
            self.rotateBox.setEnabled(True)
            self.XYZButton.setEnabled(True)
            
        if (int(self.ndimBox.currentText()) < 2):
            self.dim2ColumnBox.setEnabled(False)
            self.dim2ErrorColumnBox.setEnabled(False)
            self.dim12ErrorColumnBox.setEnabled(False)
            self.coerrorTypeBox.setEnabled(False)
            self.dim2Box.setEnabled(False)
        else:
            self.dim2ColumnBox.setEnabled(True)
            self.dim2ErrorColumnBox.setEnabled(True)
            self.dim12ErrorColumnBox.setEnabled(True)
            self.coerrorTypeBox.setEnabled(True)
            self.dim2Box.setEnabled(True)
            


    # Change in rotation box
    #-----------------------
    def rotateBoxChange(self):
      
        '''
        Change in rotation box

        '''
        
        # Enable/disable dimension name boxes and approximate XYZ button
        if (self.rotateBox.currentText() == 'Yes'):
            self.dim1Box.setCurrentIndex(0)
            self.dim1Box.setEnabled(False)
            self.dim2Box.setCurrentIndex(0)
            self.dim2Box.setEnabled(False)
            self.dim3Box.setCurrentIndex(0)
            self.dim3Box.setEnabled(False)
            self.XYZButton.setEnabled(False)
        else:
            #self.dim1Box.setCurrentIndex(1)
            self.dim1Box.setEnabled(True)
            #self.dim2Box.setCurrentIndex(1)
            self.dim2Box.setEnabled(True)
            #self.dim3Box.setCurrentIndex(1)
            self.dim3Box.setEnabled(True)
            self.XYZButton.setEnabled(True)
            
            

    # Change in integration intervals column combobox
    #------------------------------------------------
    def TColumnBoxChange(self):
      
        '''
        Change in integration intervals column combobox

        '''
        
        # Enable/disable constant integration interval box
        if (self.TColumnBox.currentText() == 'None'):
            self.TBox.setEnabled(True)
        else:
            self.TBox.setEnabled(False)



    # Change in time unit combobox
    #-----------------------------
    def timeUnitBoxChange(self):
      
        '''
        Change in time unit combobox

        '''
        
        # Change TUnitLabel
        self.TUnitLabel.setText('['+self.timeUnitBox.currentText()+']')

        # Enable/disable figure time unit box
        if (self.timeUnitBox.currentText() == 'MJD'):
            self.figureTimeUnitBox.setEnabled(True)
        else:
            self.figureTimeUnitBox.setEnabled(False)
        
        

    # Change in series unit combobox
    #-----------------------------
    def seriesUnitBoxChange(self):
      
        '''
        Change in series unit combobox

        '''
        
        # Change errorUnitLabel
        if (self.errorTypeBox.currentText() == 'std'):
            self.errorUnitLabel.setText('['+self.seriesUnitBox.currentText()+']')
        else:
            self.errorUnitLabel.setText('['+self.seriesUnitBox.currentText()+'^2]')

        # Change coerrorUnitLabel
        if (self.coerrorTypeBox.currentText() == 'corr'):
            self.coerrorUnitLabel.setText('[]')
        else:
            self.coerrorUnitLabel.setText('['+self.seriesUnitBox.currentText()+'^2]')



    # Change in error type combobox
    #------------------------------
    def errorTypeBoxChange(self):
      
        '''
        Change in error type combobox

        '''
        
        # Change errorUnitLabel
        if (self.errorTypeBox.currentText() == 'std'):
            self.errorUnitLabel.setText('['+self.seriesUnitBox.currentText()+']')
        else:
            self.errorUnitLabel.setText('['+self.seriesUnitBox.currentText()+'^2]')



    # Change in co-error type combobox
    #---------------------------------
    def coerrorTypeBoxChange(self):
      
        '''
        Change in co-error type combobox

        '''
        
        # Change coerrorUnitLabel
        if (self.coerrorTypeBox.currentText() == 'corr'):
            self.coerrorUnitLabel.setText('[]')
        else:
            self.coerrorUnitLabel.setText('['+self.seriesUnitBox.currentText()+'^2]')



    # Get name of file with approximate XYZ coordinates
    #--------------------------------------------------
    def getXYZfile(self):
      
        '''
        Get name of file with approximate XYZ coordinates
        
        '''
        
        file = QFileDialog.getOpenFileName(None, 'Select file with approximate XYZ coordinates:', self.dir)[0]
        self.XYZEdit.setText(file)



    # OK button clicked
    #------------------
    def OKClicked(self):
      
        '''
        OK button clicked
        
        '''
        
        # Get reading and preprocessing options
        #--------------------------------------
        
        try:
        
            # Known file format?
            self.known_format = None
            if (self.knownFormatBox.currentText() == 'pytrf .ts format'):
                self.known_format = 'ts'
            elif (self.knownFormatBox.currentText() == 'JPL .series/.resid format'):
                self.known_format = 'jpl'
            elif (self.knownFormatBox.currentText() == 'NGL .txyz2 format'):
                self.known_format = 'txyz2'
            elif (self.knownFormatBox.currentText() == 'NGL .tenv3 format'):
                self.known_format = 'tenv3'

            # If not, get reading options
            if (self.known_format is None):

                # Nb. header lines
                self.skiprows = self.headerLinesBox.value()

                # Column separator
                self.delimiter = None
                if not(self.separatorBox.currentText().strip() in ['', 'space']):
                    self.delimiter = self.separatorBox.currentText().strip()

                # Comment indicator
                self.comments = self.commentBox.currentText().strip()
                
                # Number of dimensions
                self.ndim = int(self.ndimBox.currentText())

                # XYZ->ENH rotation
                if (self.rotateBox.currentText() == 'Yes'):
                    self.rotate = True
                else:
                    self.rotate = False
                    
                # Dimension names
                if (self.ndim == 1):
                    self.dnames = self.dim1Box.currentText()
                elif (self.ndim == 2):
                    self.dnames = [self.dim1Box.currentText(), self.dim2Box.currentText()]
                else:
                    self.dnames = [self.dim1Box.currentText(), self.dim2Box.currentText(), self.dim3Box.currentText()]
                
                # Column indices and contents
                self.format = ['t']
                self.usecols = [int(self.timeColumnBox.currentText())]
            
                if (self.TColumnBox.currentText() != 'None'):
                    self.format.append('T')
                    self.usecols.append(int(self.TColumnBox.currentText()))
                    
                self.format.append('x')
                self.usecols.append(int(self.dim1ColumnBox.currentText()))
                
                if (self.dim1ErrorColumnBox.currentText() != 'None'):
                    if (self.errorTypeBox.currentText() == 'std'):
                        self.format.append('sx')
                    else:
                        self.format.append('qx')
                    self.usecols.append(int(self.dim1ErrorColumnBox.currentText()))
                    
                if (self.ndim > 1):
                    self.format.append('y')
                    self.usecols.append(int(self.dim2ColumnBox.currentText()))
                    
                    if (self.dim2ErrorColumnBox.currentText() != 'None'):
                        if (self.errorTypeBox.currentText() == 'std'):
                            self.format.append('sy')
                        else:
                            self.format.append('qy')
                        self.usecols.append(int(self.dim2ErrorColumnBox.currentText()))
                        
                    if (self.dim12ErrorColumnBox.currentText() != 'None'):
                        if (self.coerrorTypeBox.currentText() == 'corr'):
                            self.format.append('cxy')
                        else:
                            self.format.append('qxy')
                        self.usecols.append(int(self.dim12ErrorColumnBox.currentText()))
                    
                if (self.ndim > 2):
                    self.format.append('z')
                    self.usecols.append(int(self.dim3ColumnBox.currentText()))
                    
                    if (self.dim3ErrorColumnBox.currentText() != 'None'):
                        if (self.errorTypeBox.currentText() == 'std'):
                            self.format.append('sz')
                        else:
                            self.format.append('qz')
                        self.usecols.append(int(self.dim3ErrorColumnBox.currentText()))

                    if (self.dim13ErrorColumnBox.currentText() != 'None'):
                        if (self.coerrorTypeBox.currentText() == 'corr'):
                            self.format.append('cxz')
                        else:
                            self.format.append('qxz')
                        self.usecols.append(int(self.dim13ErrorColumnBox.currentText()))

                    if (self.dim23ErrorColumnBox.currentText() != 'None'):
                        if (self.coerrorTypeBox.currentText() == 'corr'):
                            self.format.append('cyz')
                        else:
                            self.format.append('qyz')
                        self.usecols.append(int(self.dim23ErrorColumnBox.currentText()))
                
                # Time unit
                if (self.timeUnitBox.currentText().strip() == 'MJD'):
                    self.tunit = 'mjd'
                else:
                    self.tunit = self.timeUnitBox.currentText().strip()

                # Constant integration interval
                if (self.TBox.isEnabled()):
                    self.T = float(self.TBox.text())
                else:
                    self.T = None
                
                # Time series unit
                self.yunit = self.seriesUnitBox.currentText().strip()

            # Degree of detrending polynomial
            if (self.dtrdBox.currentText() == 'None'):
                self.dtrd = None
            else:
                self.dtrd = int(self.dtrdBox.currentText())
                
            # Time unit for figures
            self.fig_tunit = None
            if (self.figureTimeUnitBox.isEnabled()):
                if (self.figureTimeUnitBox.currentText() == 'Decimal year'):
                    self.fig_tunit = 'y'
                
        except:
            
            # Print error
            print_exc()

            # Show notification
            msgBox = QMessageBox()
            msgBox.setText('Incorrect options')
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.exec()

            # Exit
            return
            

        # Emit OK signal and hide
        #------------------------
        
        self.OKSignal.emit()
        self.hide()



# fits editor class
#------------------
class editor(QPlainTextEdit):
    
    '''
    fits editor class
    
    '''

    # editor initialization
    #----------------------
    def __init__(self):
        
        '''
        editor initialization
        
        '''                  
        
        super(editor, self).__init__()

        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setFont(QFont('monospace'))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.number_bar = numberBar(self)
            
        
    # editor resize event
    #--------------------
    def resizeEvent(self, *e):
        
        '''
        editor resize event
        
        '''
                
        cr = self.contentsRect()
        rec = QRect(cr.left(), cr.top(), self.number_bar.getWidth(), cr.height())
        self.number_bar.setGeometry(rec)
        
        QPlainTextEdit.resizeEvent(self, *e)



    
# fits numberBar class
#---------------------
class numberBar(QWidget):

    '''
    fits numberBar class
    
    '''

    # numberBar initialization
    #-------------------------
    def __init__(self, editor):
        
        '''
        numberBar initialization
        
        '''
        
        QWidget.__init__(self, editor)
        
        self.editor = editor
        self.editor.blockCountChanged.connect(self.updateWidth)
        self.editor.updateRequest.connect(self.updateContents)
        self.font = QFont()
        self.numberBarColor = QColor("#e8e8e8")
                    
    # numberBar paint event
    #----------------------
    def paintEvent(self, event):
        
        '''
        numberBar paint event
        
        '''
        
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.numberBarColor)
            
        block = self.editor.firstVisibleBlock()

        # Iterate over all visible text blocks in the document
        while block.isValid():
            blockNumber = block.blockNumber()
            block_top = self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top()

            # Check if the position of the block is outside of the visible area
            if not block.isVisible() or block_top >= event.rect().bottom():
                break

            # We want the line number for the selected line to be bold.
            if blockNumber == self.editor.textCursor().blockNumber():
                self.font.setBold(True)
                painter.setPen(QColor("#000000"))
            else:
                self.font.setBold(False)
                painter.setPen(QColor("#717171"))
            painter.setFont(self.font)
            
            # Draw the line number right justified at the position of the line.
            paint_rect = QRect(0, block_top, self.width(), self.editor.fontMetrics().height())
            painter.drawText(paint_rect, Qt.AlignRight, str(blockNumber+1))

            block = block.next()

        painter.end()
        
        QWidget.paintEvent(self, event)


    # Get numberBar width
    #--------------------
    def getWidth(self):
        
        '''
        Get numberBar width
        
        '''
        
        count = self.editor.blockCount()
        width = self.fontMetrics().width(str(count)) + 10
        return width      
    
    # Update numberBar width
    #-----------------------
    def updateWidth(self):
        
        '''
        Update numberBar width
        
        '''
        
        width = self.getWidth()
        if self.width() != width:
            self.setFixedWidth(width)
            self.editor.setViewportMargins(width, 0, 0, 0);

    # Update numberBar contents
    #--------------------------
    def updateContents(self, rect, scroll):

        '''
        Update numberBar contents
        
        '''

        if scroll:
            self.scroll(0, scroll)
        else:
            self.update(0, rect.y(), self.width(), rect.height())
        
        if rect.contains(self.editor.viewport().rect()):   
            fontSize = self.editor.currentCharFormat().font().pointSize()
            self.font.setPointSize(fontSize)
            self.font.setStyle(QFont.StyleNormal)
            self.updateWidth()
            
