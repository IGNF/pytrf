"""
Leaflet map inside QT
"""

import json
import os

from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

RES_DIRECTORY = os.path.dirname(__file__) + "/res"
PAGE = RES_DIRECTORY + "/index.html"


def _noop(*_, **__):
    # noop
    pass


class StationMap(QWebEngineView):
    _ready = False
    _ops: list[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self._ops = []
        self.loadFinished.connect(self._setup_map)
        self.load(QUrl.fromLocalFile(PAGE))

    def _run_js_without_waiting(self, code: str) -> None:
        if not self._ready:
            self._ops.append("void " + code)
            return
        self.page().runJavaScript("void " + code, _noop)

    def _setup_map(self):
        self._run_js_without_waiting(
            """
            L.tileLayer(
                'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                {
                maxZoom: 19,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                }
            ).addTo(map);
            """,
        )
        self._ready = True
        for op in self._ops:
            self.page().runJavaScript(op, _noop)
        self._ops = []

    def set_view(self, latlng: tuple[float, float], zoom: int) -> None:
        self._run_js_without_waiting(
            f"map.setView({json.dumps(latlng)}, {json.dumps(zoom)})"
        )

    def add_marker(self, latlng: tuple[float, float], name: str) -> None:
        self._run_js_without_waiting(
            f"L.marker({json.dumps(latlng)}, {json.dumps({'title': name})}).bindPopup({json.dumps(name)}).addTo(map);"
        )

    def reset_map(self) -> None:
        if not self._ready:
            return

        self._run_js_without_waiting("""
            map.eachLayer((layer) => {
                if (!layer instanceof L.TileLayer) {
                    layer.remove()
                }
            })
        """)
