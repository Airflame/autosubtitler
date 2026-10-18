"""GUI entry point."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Autosubtitler")
    window = MainWindow()
    window.add_paths([Path(a) for a in sys.argv[1:]])  # e.g. files dropped onto the launcher
    window.show()
    return app.exec()
