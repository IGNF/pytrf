from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QTabBar,
    QTabWidget,
    QWidget,
)

DRAG_FORMAT = "application/vnd.pytrf.tab"


class DetachableTabWindow(QMainWindow):
    def __init__(self, parent: "DetachableTabView"):
        super().__init__(parent)
        self.tab = DetachableTabView(self)
        self.tab._main = parent
        self.setCentralWidget(self.tab)

    def closeEvent(self, event):
        self.tab.move_all_tabs_to_main()

        return super().closeEvent(event)


class DetachableTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


class DetachableTabView(QTabWidget):
    visible_tabs_changed = pyqtSignal(set)
    all_tabs: set[QWidget]

    _main: "None | DetachableTabView" = None
    _visible_tabs: set[QWidget]
    _last_tab: QWidget | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_tabs = set()
        self._visible_tabs = set()

        self.setAcceptDrops(True)

        tabBar = DetachableTabBar(self)
        self.setTabBar(tabBar)

        tabBar.detach_tab.connect(self._on_tab_detach)

        self.currentChanged.connect(self._update_current_tab)

    def _on_tab_detach(self):
        i = self.currentIndex()
        widget = self.currentWidget()

        drag = QDrag(self)
        data = QMimeData()
        data.setData(DRAG_FORMAT, b"\x01")
        drag.setMimeData(data)
        action = drag.exec()

        if action == Qt.DropAction.IgnoreAction:
            text = self.tabText(i)
            self.removeTab(i)
            window = DetachableTabWindow(self)
            window.setWindowTitle(self.window().windowTitle())
            window.tab.addTab(widget, text)
            window.tab.setMovable(self.isMovable())
            window.show()

            if self.count() <= 0 and self._main is not None:
                self.window().close()
                return

    def dragEnterEvent(self, event: QDragEnterEvent | None):
        if event is None:
            return

        mime = event.mimeData()
        if mime is None:
            return
        if mime.hasFormat(DRAG_FORMAT):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None):
        if event is None:
            return

        source = event.source()
        if source is None or not isinstance(source, DetachableTabView):
            return

        i = source.currentIndex()
        text = source.tabText(i)
        widget = source.currentWidget()

        source.removeTab(i)
        source.update()
        self.setCurrentIndex(self.addTab(widget, text))

        event.acceptProposedAction()

        if source.count() == 0 and source._main is not None:
            source.window().close()

    def move_all_tabs_to_main(self):
        if self._main is None:
            return

        while self.count() > 0:
            tab_widget = self.widget(0)
            tab_text = self.tabText(0)

            self._main.addTab(tab_widget, tab_text)
            self.removeTab(0)

    def addTab(self, widget, icon=None, label=None):
        target = self
        if self._main is not None:
            target = self._main
        target._all_tabs.add(widget)

        if label is None:
            return super().addTab(widget, icon)
        return super().addTab(widget, icon, label)

    def _update_current_tab(self, i):
        target = self
        if self._main is not None:
            target = self._main
        target._visible_tabs.discard(self._last_tab)

        if i < 0:
            target.visible_tabs_changed.emit(target._visible_tabs)
            return

        self._last_tab = self.currentWidget()
        target._visible_tabs.add(self._last_tab)
        target.visible_tabs_changed.emit(target._visible_tabs)


class DetachableTabBar(QTabBar):
    detach_tab = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mouseMoveEvent(self, event: QMouseEvent | None):
        if event is None:
            return super().mouseMoveEvent(event)

        if not self.geometry().contains(event.pos()):
            self.detach_tab.emit()

        return super().mouseMoveEvent(event)
