"""Background thread that runs the pipeline over the job queue."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..pipeline import Callbacks, Pipeline, Settings
from ..transcriber import Cancelled


class Worker(QThread):
    log = Signal(str)
    file_started = Signal(int)
    file_progress = Signal(int, float, str)
    file_done = Signal(int, str, list)      # row, detected language code, output paths
    file_failed = Signal(int, str)
    file_cancelled = Signal(int)

    def __init__(self, pipeline: Pipeline, jobs: list[tuple[int, Path]], settings: Settings):
        super().__init__()
        self.pipeline = pipeline
        self.jobs = jobs
        self.settings = settings
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        for row, path in self.jobs:
            if self._cancel.is_set():
                break
            self.file_started.emit(row)
            cb = Callbacks(
                log=self.log.emit,
                progress=lambda f, stage, r=row: self.file_progress.emit(r, f, stage),
                is_cancelled=self._cancel.is_set,
            )
            try:
                result = self.pipeline.process(path, self.settings, cb)
                self.file_done.emit(row, result.language, [str(p) for p in result.outputs])
            except Cancelled:
                self.file_cancelled.emit(row)
                self.log.emit("Cancelled.")
                break
            except Exception as e:
                self.log.emit(traceback.format_exc())
                self.file_failed.emit(row, str(e) or type(e).__name__)
        if self.settings.low_vram:
            self.pipeline.release()
