import signal
import sys

from PyQt6.QtWidgets import QApplication

from pytrf.gui.ui import MainWindow


def main():
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    initial_path = None
    if len(app.arguments()) > 1:
        initial_path = app.arguments()[1]

    window = MainWindow(initial_path)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
