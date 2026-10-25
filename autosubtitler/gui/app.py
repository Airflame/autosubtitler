"""GUI entry point."""

from __future__ import annotations

import sys
from pathlib import Path


def use_own_taskbar_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("autosubtitler.app")
    except (AttributeError, OSError):
        pass


def main() -> int:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .main_window import ICON_PATH, MainWindow

    use_own_taskbar_identity()
    app = QApplication(sys.argv)
    app.setApplicationName("Autosubtitler")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window.add_paths([Path(a) for a in sys.argv[1:]])  # e.g. files dropped onto the launcher
    window.show()
    return app.exec()
