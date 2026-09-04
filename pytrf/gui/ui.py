"""
pytrf GUI widgets
"""

import datetime
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from matplotlib import pyplot
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2, _Mode
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDoubleValidator, QFont, QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QItemDelegate,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pytrf import date
from pytrf.gui.map import StationMap
from pytrf.gui.state import (
    Direction,
    Discontinuity,
    DiscontinuityElement,
    DiscontinuityType,
    GenericOption,
    ModelAndSNXSyncStates,
    ModelFitConfig,
    ModelFitConfigOption,
    NoiseComponent,
    Project,
    ProjectConfiguration,
    SinusoidalPeriod,
    Station,
    StochasticModel,
)
from pytrf.gui.tab import DetachableTabView

MONOSPACE_FONT = QFont("monospace")


class TimeseriesToolbarMode(str, Enum):
    JUMP = "jump"

    def __str__(self):
        return self.value


class TimeseriesToolbar(NavigationToolbar2QT):
    add_jump_ev = pyqtSignal(float)
    _tunit = "d"

    def __init__(self, canvas, parent=None, coordinates=True):
        self.toolitems = NavigationToolbar2QT.toolitems + [
            (None, None, None, None),
            ("Add jump", "Add jump", "", "add_jump"),
        ]
        super().__init__(canvas, parent, coordinates)
        add_jump_action: QAction = self._actions["add_jump"]
        add_jump_action.setCheckable(True)
        add_jump_action.setIcon(QIcon())

    def _zoom_pan_handler(self, event: MouseEvent):
        if self.mode == TimeseriesToolbarMode.JUMP:
            if event.name == "button_release_event":
                self.mode = _Mode.NONE
                self._update_buttons_checked()
                return
            if event.name != "button_press_event" or event.xdata is None:
                return
            value = event.xdata
            if self._tunit == "y":
                value = date.from_ydec(event.xdata).mjd
            self.add_jump_ev.emit(value)
            return

        return super()._zoom_pan_handler(event)

    def _update_buttons_checked(self):
        self._actions["add_jump"].setChecked(self.mode.name == "JUMP")

        super()._update_buttons_checked()

    def add_jump(self):
        if self.mode == TimeseriesToolbarMode.JUMP:
            self.mode = _Mode.NONE
        else:
            self.mode = TimeseriesToolbarMode.JUMP
        self._update_buttons_checked()

    def set_tunit(self, tunit: str):
        self._tunit = tunit


def UpdatableCanvasWithToolbar(
    Toolbar: type[NavigationToolbar2] = NavigationToolbar2QT,
):
    class _UpdatableCanvasWithToolbar(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)

            layout = QVBoxLayout()
            self.setLayout(layout)
            layout.setSizeConstraints(
                QLayout.SizeConstraint.SetMaximumSize,
                QLayout.SizeConstraint.SetMaximumSize,
            )

            self.canvas = FigureCanvasQTAgg()
            self.canvas.setParent(self)
            self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            self.toolbar = Toolbar(self.canvas, self)

            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)

        def _connect(self):
            self.toolbar._id_press = self.canvas.mpl_connect(
                "button_press_event", self.toolbar._zoom_pan_handler
            )
            self.toolbar._id_release = self.canvas.mpl_connect(
                "button_release_event", self.toolbar._zoom_pan_handler
            )
            self.toolbar._id_drag = self.canvas.mpl_connect(
                "motion_notify_event", self.toolbar.mouse_move
            )

        def _disconnect(self):
            self.canvas.mpl_disconnect(self.toolbar._id_press)
            self.canvas.mpl_disconnect(self.toolbar._id_release)
            self.canvas.mpl_disconnect(self.toolbar._id_drag)

        def set_figure(self, figure: Figure):
            self.canvas.flush_events()
            self._disconnect()
            self.toolbar.release_pan(None)
            if self.toolbar._zoom_info is not None:
                self.toolbar._cleanup_post_zoom()
            self.toolbar.update()

            pyplot.close(self.canvas.figure)

            figure.set_size_inches(
                self.canvas.width() / figure.dpi, self.canvas.height() / figure.dpi
            )
            figure.set_canvas(self.canvas)

            self.canvas.figure = figure

            self.canvas.drawRectangle(None)
            self.canvas.draw()
            NavigationToolbar2.__init__(
                self.toolbar, self.canvas
            )  # Reset toolbar state
            for action_name, widget in self.toolbar._actions.items():
                widget.triggered.disconnect()
                widget.triggered.connect(getattr(self.toolbar, action_name))

            self.toolbar._update_buttons_checked()

    return _UpdatableCanvasWithToolbar


class TimeseriesAndModelViewer(UpdatableCanvasWithToolbar(TimeseriesToolbar)):  # type: ignore[misc]
    station: Station | None = None
    add_discontinuity = pyqtSignal()

    def __init__(self, selector: "DiscontinuitySelector", parent=None):
        super().__init__(parent)
        self.toolbar.add_jump_ev.connect(self._handle_add_jump)
        self._selector = selector

    def _handle_add_jump(self, t: float):
        assert self.station is not None
        assert t is not None

        dialog = AddJumpDialog(datetime.datetime.fromisoformat(date.from_mjd(t).tiso()))
        result = dialog.exec()

        if result != QDialog.DialogCode.Accepted:
            return

        self.station.add_discontinuity(
            Discontinuity(dialog.date, dialog.description, dialog.type)
        )
        self.add_discontinuity.emit()

    def set_station(self, station: Station | None):
        self.station = station
        self.on_model_updated()

    def on_model_updated(self):
        if self.station is None:
            return

        m = self.station.model()
        figure: Figure = m.plot_fit_figure(tunit="y")
        for direction in Direction.all():
            for discontinuity in self._selector.filter(
                self.station.deterministic_model().discontinuities.values(), direction
            ):
                t = date.from_mjd(discontinuity.date).ydec()
                color = discontinuity.color_graph()
                figure.axes[direction.direction_to_index()].axvline(
                    t,
                    color=color,
                    linestyle=("dashed" if discontinuity.is_empty() else "solid"),
                )

        self.set_figure(figure)
        self.toolbar.set_tunit("y")


class TimeseriesSelector(QWidget):
    project: Project | None = None
    station_change = pyqtSignal(Station)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.combobox = QComboBox(self)

        self.previous_button = QPushButton(
            QIcon.fromTheme(QIcon.ThemeIcon.GoPrevious), None, self
        )
        self.next_button = QPushButton(
            QIcon.fromTheme(QIcon.ThemeIcon.GoNext), None, self
        )
        self.previous_button.setDisabled(True)
        self.next_button.setDisabled(True)
        layout.addWidget(self.combobox)
        layout.addWidget(self.previous_button)
        layout.addWidget(self.next_button)

        self.combobox.currentIndexChanged.connect(self._handle_update)

        self.previous_button.clicked.connect(self._previous)
        self.next_button.clicked.connect(self._next)

    def set_project(self, project: Project):
        self.project = project
        self._update_combobox()
        self._update_buttons()

    def _handle_update(self, i: int):
        self._update_buttons()
        if self.project is None:
            return
        selected_station: Station | None = self.combobox.itemData(i)
        for station in self.project.stations():
            if station == selected_station:
                station.set_active()
            else:
                station.set_inactive()
        if selected_station is None:
            return
        self.station_change.emit(selected_station)

    def _update_combobox(self):
        self.combobox.clear()
        if self.project == None:
            return

        for station in self.project.stations():
            self.combobox.addItem(station.name, station)

    def _update_buttons(self):
        if self.project == None:
            self.previous_button.setDisabled(True)
            self.next_button.setDisabled(True)
            return
        index = self.combobox.currentIndex()

        self.previous_button.setDisabled(index == 0)
        self.next_button.setDisabled((index + 1) >= len(self.project.stations()))

    def _previous(self):
        if self.project == None:
            return
        self.combobox.setCurrentIndex(max(0, self.combobox.currentIndex() - 1))

    def _next(self):
        if self.project == None:
            return
        self.combobox.setCurrentIndex(
            min(len(self.project.stations()) - 1, self.combobox.currentIndex() + 1)
        )


class ModelSyncInfo(QWidget):
    _project = None
    _station = None
    updated_model = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.label = QLabel("-")
        self.button = QPushButton("Update")

        self.button.clicked.connect(self._update_click)

        layout.addWidget(self.label, 2)
        layout.addWidget(self.button, 1)

    def set_project(self, project: Project):
        self._project = project

    def _update_click(self):
        if self._station is None:
            return
        self._station.sync_model_with_snx()
        self.label.setText("Model synced. Press 'Adjust model' to see the changes.")
        self.button.setDisabled(True)
        self.button.setText("-")
        self.updated_model.emit()

    def check_station(self, station: Station):
        state = station.model_and_snx_sync_state()
        update = self._project.config.update_model_with_snx_data
        self.button.setDisabled(True)
        self.button.setText("-")
        if update:
            station.sync_model_with_snx()
            if state:
                station.update_model()
        else:
            if state:
                self.button.setDisabled(False)
                self.button.setText("Update")
            self._station = station

        message = []

        if ModelAndSNXSyncStates.SOLN_OUT_OF_SYNC in state:
            message.append(
                "Model updated with SOLN data"
                if update
                else "Model is out of sync with SOLN data"
            )

        if ModelAndSNXSyncStates.PSD_OUT_OF_SYNC in state:
            message.append(
                "Model updated with PSD data"
                if update
                else "Model is out of sync with PSD data"
            )

        if len(message) == 0:
            message.append("Model is in sync with PSD and SOLN data")

        self.label.setText(", ".join(message))


class AddJumpDialog(QDialog):
    date: float
    type: DiscontinuityType
    description: str

    def __init__(self, initial_date=None, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        input_layouts = QVBoxLayout()
        self._date_input = QDateTimeEdit(initial_date or datetime.datetime.now(), self)
        self._date_input.setCalendarPopup(True)
        self._type_input = QComboBox()
        self._type_input.addItems([str(d) for d in DiscontinuityType.all()])
        self._description_input = QLineEdit(self)
        self._description_input.setPlaceholderText("Description")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        input_layouts.addWidget(self._date_input)
        input_layouts.addWidget(self._type_input)
        input_layouts.addWidget(self._description_input)
        layout.addLayout(input_layouts)
        layout.addLayout(buttons_layout)

    def accept(self):
        if not self._date_input.hasAcceptableInput():
            return

        self.date = date.from_tiso(
            self._date_input.dateTime().toPyDateTime().isoformat()
        ).mjd
        self.type = DiscontinuityType.from_str(self._type_input.currentText())
        self.description = self._description_input.text()

        return super().accept()


class JumpListItemDelegate(QItemDelegate):
    onItemEdited = pyqtSignal(QTableWidgetItem)

    def __init__(self, table: QTableWidget, parent=None):
        super().__init__(parent)
        self.table = table

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        itm = self.table.itemFromIndex(index)

        def handle(v):
            if editor != v:
                return
            try:
                self.closeEditor.disconnect(conn)
            except Exception:
                pass
            self.onItemEdited.emit(itm)

        conn = self.closeEditor.connect(handle)

        return editor


class DeterministicModelJumpWidget(QWidget):
    station: Station | None = None
    _connections: list

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = []
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = QTableWidget(self)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "Date",
                "Type",
                "Position",
                "Velocity",
                "Exp",
                "Log",
                "Description",
                "Remove",
            ]
        )
        self.item_delegate = JumpListItemDelegate(self.table)
        self.table.setItemDelegate(self.item_delegate)

        self.add_jump = QPushButton(
            QIcon.fromTheme(QIcon.ThemeIcon.ListAdd), "Add jump", self
        )
        layout.addWidget(self.table)
        layout.addWidget(self.add_jump)

        self.add_jump.clicked.connect(self._add_jump)

    def set_station(self, station: Station | None):
        self.station = station
        self.refresh()

    def refresh(self):
        for signal, conn in self._connections:
            try:
                signal.disconnect(conn)
            except Exception as e:
                traceback.print_exception(e)
        self._connections = []

        self.table.clearContents()
        self.table.setRowCount(0)

        if self.station is None:
            return
        discontinuities = self.station.deterministic_model().discontinuities.values()

        for discontinuity in sorted(discontinuities, key=lambda d: d.date):
            self._add_row(discontinuity)

    def _add_row(self, discontinuity: Discontinuity):
        assert self.station is not None
        assert discontinuity.date in self.station.deterministic_model().discontinuities

        row = self.table.rowCount()

        self.table.insertRow(row)

        fg_color, bg_color = discontinuity.color_display()

        fg = QColor.fromString(fg_color)
        bg = QColor.fromString(bg_color)

        date_widget = QTableWidgetItem(date.from_mjd(discontinuity.date).tiso())
        date_widget.setFont(MONOSPACE_FONT)
        date_widget.setForeground(fg)
        date_widget.setBackground(bg)
        date_widget.setToolTip(discontinuity.tooltip)
        self.table.setItem(row, 0, date_widget)

        type_widget = QLabel(str(discontinuity.type))
        self.table.setCellWidget(row, 1, type_widget)

        description_widget = QTableWidgetItem(discontinuity.description)
        self.table.setItem(row, 6, description_widget)

        def item_changed(item):
            if item == date_widget:
                try:
                    discontinuity.set_date(date.from_tiso(date_widget.text()).mjd)
                except Exception as e:
                    traceback.print_exception(e)
                    date_widget.setText(date.from_mjd(discontinuity.date).tiso())
                return
            if item == description_widget:
                discontinuity.description = description_widget.text()

        self._connect(self.item_delegate.onItemEdited, item_changed)

        position_checkbox = QCheckBox()
        position_checkbox.setChecked(
            DiscontinuityElement.position in discontinuity.elements
        )

        self._connect(
            position_checkbox.checkStateChanged,
            lambda state: (
                discontinuity.elements.add(DiscontinuityElement.position)
                if state == Qt.CheckState.Checked
                else discontinuity.elements.discard(DiscontinuityElement.position)
            ),
        )

        velocity_checkbox = QCheckBox()
        velocity_checkbox.setChecked(
            DiscontinuityElement.velocity in discontinuity.elements
        )

        self._connect(
            velocity_checkbox.checkStateChanged,
            lambda state: (
                discontinuity.elements.add(DiscontinuityElement.velocity)
                if state == Qt.CheckState.Checked
                else discontinuity.elements.discard(DiscontinuityElement.velocity)
            ),
        )

        remove_button = QPushButton(QIcon.fromTheme(QIcon.ThemeIcon.ListRemove), None)

        log_checkboxes = QWidget()
        log_layout = QHBoxLayout()
        log_checkboxes.setLayout(log_layout)

        exp_checkboxes = QWidget()
        exp_layout = QHBoxLayout()
        exp_checkboxes.setLayout(exp_layout)

        for direction in Direction.all():
            log_box = QCheckBox(direction.name)
            log_box.setChecked(direction in discontinuity.log_components)
            self._connect(
                log_box.checkStateChanged,
                lambda state, direction=direction: (
                    discontinuity.log_components.add(direction)
                    if state == Qt.CheckState.Checked
                    else discontinuity.log_components.discard(direction)
                ),
            )
            log_layout.addWidget(log_box)

            exp_box = QCheckBox(direction.name)
            exp_box.setChecked(direction in discontinuity.exp_components)
            self._connect(
                exp_box.checkStateChanged,
                lambda state, direction=direction: (
                    discontinuity.exp_components.add(direction)
                    if state == Qt.CheckState.Checked
                    else discontinuity.exp_components.discard(direction)
                ),
            )

            exp_layout.addWidget(exp_box)

        def _remove():
            self.station.remove_discontinuity(discontinuity)
            self.table.removeRow(self.table.row(date_widget))

        self._connect(remove_button.clicked, _remove)

        self.table.setCellWidget(row, 2, position_checkbox)
        self.table.setCellWidget(row, 3, velocity_checkbox)
        self.table.setCellWidget(row, 4, log_checkboxes)
        self.table.setCellWidget(row, 5, exp_checkboxes)
        self.table.setCellWidget(row, 7, remove_button)

        self.table.resizeColumnsToContents()

    def _connect(self, signal, cb):
        self._connections.append((signal, signal.connect(cb)))

    def _add_jump(self):
        assert self.station is not None
        dialog = AddJumpDialog()
        result = dialog.exec()

        if result != QDialog.DialogCode.Accepted:
            return

        self.station.add_discontinuity(
            Discontinuity(dialog.date, dialog.description, dialog.type)
        )
        self.refresh()


class SineModelListWidget(QWidget):
    station: Station | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        add_sine_container = QWidget(self)
        add_sine_layout = QHBoxLayout()
        add_sine_container.setLayout(add_sine_layout)

        add_sine_button = QPushButton(
            QIcon.fromTheme(QIcon.ThemeIcon.ListAdd), "Add", self
        )
        self.add_sine_input = QLineEdit(self)
        self.add_sine_input.setValidator(QDoubleValidator(self))

        self.add_sine_input.setPlaceholderText("Period")
        add_sine_layout.addWidget(self.add_sine_input)
        add_sine_layout.addWidget(add_sine_button)

        self.sine_list = QTableWidget(self)
        self.sine_list.setColumnCount(2)
        self.sine_list.setHorizontalHeaderLabels(["Period", "Remove"])

        layout.addWidget(add_sine_container)
        layout.addWidget(self.sine_list)

        add_sine_button.clicked.connect(self._add_sine)
        self.add_sine_input.returnPressed.connect(self._add_sine)

    def set_station(self, station: Station | None):
        self.station = station
        self._update()

    def _add_sine(self):
        if self.station is None:
            return

        if not self.add_sine_input.hasAcceptableInput():
            return
        try:
            value = self.add_sine_input.text().replace(",", ".")

            self.station.deterministic_model().sinusoidal_periods.add(
                SinusoidalPeriod(float(value))
            )
            self._update()
            self.add_sine_input.clear()
        except Exception as e:
            traceback.print_exception(e)

    def _update(self):
        self.sine_list.clearContents()
        self.sine_list.setRowCount(0)

        if self.station is None:
            return

        assert self.station.active()

        for period in sorted(
            self.station.deterministic_model().sinusoidal_periods,
            key=lambda v: v.value,
            reverse=True,
        ):
            self._add_row(period)

    def _add_row(self, period: SinusoidalPeriod):
        row = self.sine_list.rowCount()

        self.sine_list.insertRow(row)

        period_value = QLabel(str(period.value))

        self.sine_list.setCellWidget(
            row,
            0,
            period_value,
        )
        remove_button = QPushButton(QIcon.fromTheme(QIcon.ThemeIcon.ListRemove), None)

        def _remove():
            self.station.deterministic_model().sinusoidal_periods.remove(period)
            self._update()

        remove_button.clicked.connect(_remove)

        self.sine_list.setCellWidget(row, 1, remove_button)


class SaveModelWidget(QWidget):
    station: Station | None = None
    before_adjust = pyqtSignal()
    after_adjust = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        adjust_model = QPushButton("Adjust model")
        self.save_model = QPushButton("Save model")
        unfix_psd = QPushButton("Unfix PSD")
        save_psd = QPushButton("Save PSD")

        adjust_model.clicked.connect(self._adjust_model)
        self.save_model.clicked.connect(self._save_model)

        unfix_psd.clicked.connect(self._unfix_psd)
        save_psd.clicked.connect(self._save_psd)

        layout.addWidget(adjust_model)
        layout.addWidget(self.save_model)
        layout.addWidget(unfix_psd)
        layout.addWidget(save_psd)

    def _unfix_psd(self):
        if self.station is None:
            return
        self.station.unfix_psd()

    def _save_psd(self):
        if self.station is None:
            return

        self.station.save_psd()

    def _adjust_model(self):
        if self.station is None:
            return
        self.before_adjust.emit()
        self.station.update_model()
        self.after_adjust.emit()
        self._update_save_button_text()

    def _save_model(self):
        if self.station is None:
            return
        self.station.save()
        self._update_save_button_text()

    def _update_save_button_text(self):
        if self.station is None:
            return

        if self.station.has_unsaved_changes():
            self.save_model.setText("Save model*")
        else:
            self.save_model.setText("Save model")

    def set_station(self, station: Station | None):
        self.station = station
        self._update_save_button_text()


class OptionWidget[T](QWidget):
    def __init__(self, option: GenericOption[T], parent=None):
        super().__init__(parent)
        self._option = option

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.entry = QLineEdit(
            None if option.default_value is None else str(option.default_value)
        )
        label = QLabel(option.name)
        layout.addWidget(self.entry)
        layout.addWidget(label)

    def value(self):
        v = self.entry.text()
        try:
            return self._option.t(v)
        except Exception as e:
            traceback.print_exception(e)

            return None

    def inject(self, target: T):
        self._option.inject(self.value(), target)

    def set_value(self, instance: T):
        v = self._option.get(instance)
        if v is None:
            self.entry.setText(None)
        else:
            self.entry.setText(str(v))


class StochasticModelComponentWidget(QWidget):
    _options: list[OptionWidget]
    _component: type[NoiseComponent]
    _instance: NoiseComponent | None = None

    def __init__(self, component: type[NoiseComponent], parent=None):
        super().__init__(parent)
        self._options = []
        self._component = component

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.state = QCheckBox(component.name)

        self.state.checkStateChanged.connect(self._update_option)

        layout.addWidget(self.state)
        for option in component.OPTIONS:
            widget = OptionWidget(option)
            self._options.append(widget)
            layout.addWidget(widget)

    def set_instance(self, instance: NoiseComponent):
        assert isinstance(instance, self._component)
        self._instance = instance

        self.state.setChecked(instance.enabled())

        for option in self._options:
            option.set_value(instance)

    def _update_option(self, state: Qt.CheckState):
        if self._instance is None:
            return

        self._instance.set_enabled(state == Qt.CheckState.Checked)

    def inject(self):
        if self._instance is None:
            return
        if not self._instance.enabled():
            return

        for widget in self._options:
            widget.inject(self._instance)


class StochasticModelWidget(QWidget):
    _components: list[StochasticModelComponentWidget]
    _station: Station | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._components = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        for component in StochasticModel.NOISE_COMPONENTS:
            widget = StochasticModelComponentWidget(component)
            self._components.append(widget)
            layout.addWidget(widget)

    def set_station(self, station: Station | None):
        self._station = station

        if station is None:
            return

        stochastic_model = station.stochastic_model()
        assert len(self._components) == len(stochastic_model.components)

        for i in range(len(self._components)):
            component = stochastic_model.components[i]
            widget = self._components[i]
            widget.set_instance(component)

    def inject(self):
        assert self._station is not None

        for component in self._components:
            component.inject()


class OutlierRejectionEntryWidget(QWidget):
    _option_instance: ModelFitConfigOption | None = None
    _option_widgets: list[OptionWidget]
    _option_ctr: type[ModelFitConfigOption]

    def __init__(self, option: type[ModelFitConfigOption], parent=None):
        super().__init__(parent)
        self._option_ctr = option
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.option_state = QCheckBox(option.name)

        self.option_state.checkStateChanged.connect(self._update_option)

        layout.addWidget(self.option_state)
        self._option_widgets = []
        for o in option.OPTIONS:
            widget = OptionWidget(o, self)
            self._option_widgets.append(widget)
            layout.addWidget(widget)

    def set_option_instance(self, option_instance: ModelFitConfigOption):
        assert isinstance(option_instance, self._option_ctr)
        self._option_instance = option_instance

        self.option_state.setChecked(option_instance.enabled())

        for widget in self._option_widgets:
            widget.set_value(option_instance)

    def _update_option(self, state: Qt.CheckState):
        if self._option_instance is None:
            return

        self._option_instance.set_enabled(state == Qt.CheckState.Checked)

    def inject(self):
        if self._option_instance is None:
            return
        if not self._option_instance.enabled():
            return

        for widget in self._option_widgets:
            widget.inject(self._option_instance)


class OutlierRejectionWidget(QWidget):
    _entries: list[OutlierRejectionEntryWidget]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._entries = []

        description = QLabel("Outlier rejection")
        self._layout.addWidget(description)

        for option in ModelFitConfig.OPTIONS:
            widget = OutlierRejectionEntryWidget(option, self)
            self._layout.addWidget(widget)
            self._entries.append(widget)

    def set_station(self, station: Station | None):
        if station is None:
            return

        options = station.fit_config().options()
        assert len(options) == len(self._entries)

        for i in range(len(options)):
            self._entries[i].set_option_instance(options[i])

    def inject(self):
        for option in self._entries:
            option.inject()


class Tab(QWidget):
    name: str

    def set_station(self, station: Station):
        raise NotImplementedError()


class LazyTabs(DetachableTabView):
    _station: Station | None = None
    _update_states: dict[Tab, Station | None]

    def __init__(self, tabs: list[Tab], parent=None):
        super().__init__(parent)
        self._tabs = tabs
        self._update_states = {}
        for tab in tabs:
            self.addTab(tab, tab.name)
            self._update_states[tab] = None

        self.visible_tabs_changed.connect(self._tab_changed)
        self.setMovable(True)

    def _tab_changed(self, tabs: set[QWidget]):
        if self._station is None:
            return

        for tab in tabs:
            if not isinstance(tab, Tab):
                continue
            if self._update_states[tab] == self._station:
                continue
            tab.set_station(self._station)
            self._update_states[tab] = self._station

    def set_station(self, station: Station | None):
        self._station = station
        self._tab_changed(self._visible_tabs)

    def refresh(self):
        for key in self._update_states:
            self._update_states[key] = None
        self._tab_changed(self._visible_tabs)


UpdatableCanvasWithDefaultToolbar = UpdatableCanvasWithToolbar()


class ResidualsTab(Tab):
    name = "Residuals"
    _station: Station | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        plot_type_layout = QHBoxLayout()
        raw_residuals = QRadioButton("Raw redisuals")
        raw_residuals.setChecked(True)
        raw_residuals.toggled.connect(self._update_plot)
        self.normalised_residuals = QRadioButton("Normalised residuals")

        plot_type_layout.addWidget(raw_residuals)
        plot_type_layout.addWidget(self.normalised_residuals)
        plot_type_layout.addStretch(1)

        self._canvas = UpdatableCanvasWithDefaultToolbar()
        layout.addLayout(plot_type_layout)
        layout.addWidget(self._canvas)

    def set_station(self, station):
        self._station = station
        self._update_plot()

    def _update_plot(self):
        assert self._station is not None

        model = self._station.model()

        if self.normalised_residuals.isChecked():
            self._canvas.set_figure(model.plot_normres_figure(tunit="y"))
        else:
            self._canvas.set_figure(model.plot_res_figure(tunit="y"))


class PeriodogramTab(Tab):
    name = "Periodogram"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self._canvas = UpdatableCanvasWithDefaultToolbar()
        layout.addWidget(self._canvas)

    def set_station(self, station):
        model = station.model()

        self._canvas.set_figure(model.plot_psd_figure(tunit="y"))


class ModelTab(Tab):
    name = "Model"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)

        scroll_layout = QHBoxLayout()
        container = QWidget()
        container.setLayout(scroll_layout)
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)

        self._model_content = QLabel()
        self._model_content.setFont(MONOSPACE_FONT)

        scroll_layout.addWidget(self._model_content)
        layout.addWidget(scroll_area)

    def set_station(self, station):
        try:
            self._model_content.setText(str(station.model()))
        except Exception as e:
            print(e)
            self._model_content.setText("Unable to get model content")


class SiteLogTab(Tab):
    name = "Sitelog"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        container = QWidget()
        scroll_layout = QHBoxLayout()
        container.setLayout(scroll_layout)
        self._sitelog_content = QLabel()
        self._sitelog_content.setFont(MONOSPACE_FONT)

        scroll_layout.addWidget(self._sitelog_content)

        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

    def set_station(self, station):
        content = station.sitelog_raw_content()

        if content is None:
            self._sitelog_content.setText("Sitelog not found for this station")
        else:
            self._sitelog_content.setText(content)


class LinksTab(Tab):
    name = "Links and Map"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.webigs_link = QLabel(
            '<a href="https://webigs-rf.ign.fr">https://webigs-rf.ign.fr</a>'
        )
        self.webigs_link.setTextFormat(Qt.TextFormat.RichText)
        self.webigs_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.webigs_link.setOpenExternalLinks(True)

        self.unr_link = QLabel(
            '<a href="https://geodesy.unr.edu">https://geodesy.unr.edu</a>'
        )
        self.unr_link.setTextFormat(Qt.TextFormat.RichText)
        self.unr_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.unr_link.setOpenExternalLinks(True)

        self.map_widget = StationMap()

        layout.addWidget(self.map_widget, 2)
        layout.addWidget(self.webigs_link, 1)
        layout.addWidget(self.unr_link, 1)

    def set_station(self, station):
        pos = station.position_geographic(False)
        if pos is not None:
            self.map_widget.setHidden(False)
            self.map_widget.set_view(pos[:2], 10)
        else:
            self.map_widget.setHidden(True)
        self.webigs_link.setText(
            f'<a href="https://webigs-rf.ign.fr/stations/{station.name}">https://webigs-rf.ign.fr/stations/{station.name}</a>'
        )
        self.unr_link.setText(
            f'<a href="https://geodesy.unr.edu/NGLStationPages/stations/{station.name}.sta">https://geodesy.unr.edu/NGLStationPages/stations/{station.name}.sta</a>'
        )


class ViewTab(Tab):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.view = QWebEngineView()
        layout.addWidget(self.view)


class WebIGSViewTab(ViewTab):
    name = "WebIGS Page"

    def set_station(self, station):
        self.view.load(QUrl(f"https://webigs-rf.ign.fr/stations/{station.name}"))


class UNRViewTab(ViewTab):
    name = "UNR Page"

    def set_station(self, station):
        self.view.load(
            QUrl(f"https://geodesy.unr.edu/NGLStationPages/stations/{station.name}.sta")
        )


class NotesTab(Tab):
    name = "Notes"
    _station: Station | None = None

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.data = QPlainTextEdit()
        self.data.textChanged.connect(self._change_text)

        save = QPushButton("Save notes")
        save.clicked.connect(self._save)

        layout.addWidget(self.data, 5)
        layout.addWidget(save, 1)

    def _save(self):
        if not self.data.isEnabled() or self._station is None:
            return
        self._station.write_notes()

    def _change_text(self):
        if not self.data.isEnabled() or self._station is None:
            return
        self._station.set_notes(self.data.toPlainText())

    def set_station(self, station):
        self._station = station
        notes = station.notes()
        if notes is None:
            self.data.setPlainText("Please configure notes path to use this feature.")
            self.data.setDisabled(True)
            return
        self.data.setDisabled(False)

        self.data.setPlainText(notes)


class RemovePointWithError(QWidget):
    _station: Station | None = None
    _sigmas: float = 5

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        update_button = QPushButton("Remove points with large errors")
        self.value_input = QLineEdit("Thresold")
        self.value_input.setText(str(self._sigmas))
        self.value_input.setValidator(QDoubleValidator())
        self.value_input.textChanged.connect(self._update_value)

        update_button.clicked.connect(self._update_ts)

        layout.addWidget(update_button)
        layout.addWidget(self.value_input)

    def set_station(self, station: Station | None):
        self._station = station

    def _update_value(self):
        text = self.value_input.text()
        if text == "":
            return

        self._sigmas = float(text)

    def _update_ts(self):
        if self._station is None:
            return
        self._station.clean_ts(self._sigmas)


class DiscontinuitySelector(QWidget):
    _current_selection: str = "Enabled"
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        label = QLabel("Discontinuity to show in graph")
        combo = QComboBox()
        combo.addItems(
            [
                "Enabled",
                "All",
                "Antenna change",
                "Earthquake",
                "Receptor change",
                "Other",
            ]
        )
        combo.currentTextChanged.connect(self._update_selection)
        layout.addWidget(label)
        layout.addWidget(combo)

    def _update_selection(self, value: str):
        self._current_selection = value
        self.selection_changed.emit()

    def filter(
        self, discontinuities: Iterable[Discontinuity], dim: Direction
    ) -> Iterable[Discontinuity]:
        if self._current_selection == "All":
            return discontinuities
        if self._current_selection == "Antenna change":
            return (
                discontinuity
                for discontinuity in discontinuities
                if discontinuity.type == DiscontinuityType.antenna_change
            )
        if self._current_selection == "Receptor change":
            return (
                discontinuity
                for discontinuity in discontinuities
                if discontinuity.type == DiscontinuityType.receiver_change
            )

        if self._current_selection == "Enabled":
            return (
                discontinuity
                for discontinuity in discontinuities
                if dim in discontinuity.exp_components
                or dim in discontinuity.log_components
                or len(discontinuity.elements) > 0
            )

        if self._current_selection == "Earthquake":
            return (
                discontinuity
                for discontinuity in discontinuities
                if discontinuity.type == DiscontinuityType.earthquake
            )
        return (
            discontinuity
            for discontinuity in discontinuities
            if discontinuity.type == DiscontinuityType.other
        )


@dataclass
class Filter:
    mode: Literal["file", "directory"] = "directory"
    file_types: str | None = None


class ProjectEditConfigurationDialogEntry(QWidget):
    def __init__(
        self,
        config: ProjectConfiguration,
        key: str,
        description: str,
        type: Filter | None = None,
        parent=None,
    ):
        super().__init__(parent)

        self._config = config
        self._key = key
        self._filter = type or Filter()

        layout = QHBoxLayout()
        self.setLayout(layout)

        label = QLabel(description)
        self.text_input = QLineEdit(getattr(config, key, None))
        self.choose_file = QPushButton("...")

        self.choose_file.clicked.connect(self._picker_dialog)
        self.text_input.textChanged.connect(self._input_updated)

        layout.addWidget(label)
        layout.addWidget(self.text_input)
        layout.addWidget(self.choose_file)

    def _picker_dialog(self):
        path = ""
        if self._filter.mode == "file":
            path, _ = QFileDialog.getOpenFileName(
                None, None, None, self._filter.file_types, None
            )
        else:
            path = QFileDialog.getExistingDirectory()

        if path != "":
            self.text_input.setText(path)

    def _input_updated(self, value: str):
        setattr(self._config, self._key, value)


class ProjectEditConfigurationDialog(QDialog):
    _options: list[ProjectEditConfigurationDialogEntry]

    def __init__(self, config: ProjectConfiguration, parent=None):
        super().__init__(parent)

        self._config = config
        self._options = []

        message = "Create a new project"
        if config.validate():
            message = "Edit project configuration"
        self.setWindowTitle(message)

        layout = QVBoxLayout()
        self.setLayout(layout)

        ts_path_option = ProjectEditConfigurationDialogEntry(
            config, "ts_path", "Timeseries directory*"
        )
        model_path_option = ProjectEditConfigurationDialogEntry(
            config, "model_path", "Model directory*"
        )
        discontinuity_path_option = ProjectEditConfigurationDialogEntry(
            config,
            "discontinuity_path",
            "Discontinuity file",
            Filter("file", "Discontinuity file (*.snx)"),
        )
        psd_path_option = ProjectEditConfigurationDialogEntry(
            config,
            "psd_path",
            "Post-seismic deformation models",
            Filter("file", "Post-seismic deformation models (*.snx)"),
        )
        sitelogs_path_option = ProjectEditConfigurationDialogEntry(
            config,
            "sitelogs_path",
            "Sitelogs config",
            Filter("file", "Sitelogs config (*.opt)"),
        )
        notes_path_option = ProjectEditConfigurationDialogEntry(
            config, "notes_path", "Notes directory"
        )

        update_model_with_snx_data = QCheckBox(
            "Update the model with soln and psd if needed."
        )
        update_model_with_snx_data.setChecked(config.update_model_with_snx_data)

        layout.addWidget(ts_path_option)
        layout.addWidget(model_path_option)
        layout.addWidget(discontinuity_path_option)
        layout.addWidget(psd_path_option)
        layout.addWidget(sitelogs_path_option)
        layout.addWidget(notes_path_option)
        layout.addWidget(update_model_with_snx_data)

        def _update_check_model_and_snx_data():
            config.update_model_with_snx_data = update_model_with_snx_data.isChecked()

        update_model_with_snx_data.checkStateChanged.connect(
            _update_check_model_and_snx_data
        )

        self._options.append(ts_path_option)
        self._options.append(model_path_option)
        self._options.append(discontinuity_path_option)
        self._options.append(psd_path_option)
        self._options.append(sitelogs_path_option)
        self._options.append(notes_path_option)

        self.status_label = QLabel(None)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )

        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout.addWidget(buttons)

    def accept(self):
        if not self._config.validate():
            self.status_label.setText(
                "Configuration invalid. (fields with * are required)"
            )
            return

        return super().accept()


class ProjectManagementWidget(QMenuBar):
    project_changed = pyqtSignal(Project)
    _project: Project | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setNativeMenuBar(True)

        new_project_action = self.addAction("New Project", "CTRL+n")
        open_project_action = self.addAction("Open Project", "CTRL+o")
        edit_project_action = self.addAction("Edit Project", "CTRL+e")

        new_project_action.triggered.connect(self._new_project)
        open_project_action.triggered.connect(self._open_project)
        edit_project_action.triggered.connect(self._edit_project)

    def set_project(self, project: Project):
        self._project = project

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Open project", None, "YAML Files (*.yaml)"
        )
        if path == "":
            return
        self._project = Project(ProjectConfiguration.load(path))
        self.project_changed.emit(self._project)

    def _edit_project(self):
        if self._project is None:
            return

        config_copy = self._project.config.copy()
        dialog = ProjectEditConfigurationDialog(config_copy)
        r = dialog.exec()
        if r == 0:
            return

        config_copy.save()
        self._project = Project(config_copy)
        self.project_changed.emit(self._project)

    def _new_project(self):
        config = ProjectConfiguration(None, None, None, None, None, None)
        dialog = ProjectEditConfigurationDialog(config)
        r = dialog.exec()
        if r == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "Save project configuration", "project.yaml", None, None
        )
        if path == "":
            return
        config.file_path = path
        config.save()

        self._project = Project(config)
        self.project_changed.emit(self._project)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Pytrf UI")

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 3)
        main_layout.addLayout(right_layout, 2)

        deterministic_container = QWidget()
        deterministic_container_layout = QVBoxLayout()
        deterministic_container.setLayout(deterministic_container_layout)

        deterministic_model_widget = DeterministicModelJumpWidget()
        sine_model_widget = SineModelListWidget()
        outliner_rejection_widget = OutlierRejectionWidget()
        deterministic_container_layout.addWidget(deterministic_model_widget, 2)
        deterministic_container_layout.addWidget(sine_model_widget, 1)
        deterministic_container_layout.addWidget(outliner_rejection_widget, 1)

        stochastic_model_widget = StochasticModelWidget()

        left_tabs = QTabWidget()

        left_tabs.addTab(deterministic_container, "Deterministic model")
        left_tabs.addTab(stochastic_model_widget, "Stochastic model")

        save_widget = SaveModelWidget()

        left_layout.addWidget(left_tabs)
        left_layout.addWidget(save_widget)

        discontinuity_selector = DiscontinuitySelector()
        discontinuity_selector.selection_changed.connect(
            lambda: model_viewer_widget.on_model_updated()
        )

        timeseries_selector_widget = TimeseriesSelector()
        model_sync_info = ModelSyncInfo()  # Must be the first to be executed on events
        model_viewer_widget = TimeseriesAndModelViewer(discontinuity_selector)
        remove_points_widget = RemovePointWithError()

        bottom_center = QHBoxLayout()
        bottom_center.addWidget(remove_points_widget)
        bottom_center.addWidget(discontinuity_selector)

        center_layout.addWidget(timeseries_selector_widget)
        center_layout.addWidget(model_sync_info)
        center_layout.addWidget(model_viewer_widget, 3)
        center_layout.addLayout(bottom_center)

        links_tab = LinksTab()

        station_infos = LazyTabs(
            [
                ResidualsTab(),
                PeriodogramTab(),
                ModelTab(),
                SiteLogTab(),
                links_tab,
                WebIGSViewTab(),
                UNRViewTab(),
                NotesTab(),
            ]
        )

        right_layout.addWidget(station_infos)

        def refresh_discontinuities():
            deterministic_model_widget.refresh()
            model_viewer_widget.on_model_updated()

        model_viewer_widget.add_discontinuity.connect(refresh_discontinuities)

        def update_view(station: Station | None):
            if station is not None:
                model_sync_info.check_station(station)

            model_viewer_widget.set_station(station)
            remove_points_widget.set_station(station)

            deterministic_model_widget.set_station(station)
            sine_model_widget.set_station(station)
            outliner_rejection_widget.set_station(station)
            save_widget.set_station(station)

            stochastic_model_widget.set_station(station)

            station_infos.set_station(station)

        def inject_before_adjust():
            outliner_rejection_widget.inject()
            stochastic_model_widget.inject()

        def update_on_adjust():
            model_viewer_widget.on_model_updated()
            station_infos.refresh()

        def set_project(project: Project):
            model_sync_info.set_project(project)
            timeseries_selector_widget.set_project(project)

            links_tab.map_widget.reset_map()

            for station in project.stations():
                lat, lng, _ = station.position_geographic(False)

                links_tab.map_widget.add_marker((lat, lng), station.name)

        model_sync_info.updated_model.connect(refresh_discontinuities)

        save_widget.before_adjust.connect(inject_before_adjust)
        save_widget.after_adjust.connect(update_on_adjust)

        timeseries_selector_widget.station_change.connect(update_view)

        project_management = ProjectManagementWidget(self)
        project_management.project_changed.connect(set_project)

        if initial_path is not None:
            project = Project(ProjectConfiguration.load(initial_path))
            set_project(project)
            project_management.set_project(project)

        self.setCentralWidget(main_widget)
        self.setMenuWidget(project_management)
