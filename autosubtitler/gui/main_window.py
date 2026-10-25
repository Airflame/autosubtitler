"""Main application window."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from .worker import Worker
from .. import languages, model_store
from ..pipeline import MEDIA_EXTENSIONS, ORIGINAL, Pipeline, Settings, default_models_dir
from ..transcriber import WHISPER_MODELS
from ..translator import NLLB_MODELS

COL_FILE, COL_STATUS, COL_LANG, COL_OUTPUT = range(4)
PATH_ROLE = Qt.UserRole
ICON_PATH = Path(__file__).resolve().parent.parent / "resources" / "icon.png"


class FunctionThread(QThread):
    """Runs a callable off the UI thread and reports its result or error."""
    finished_ok = Signal(object)
    failed = Signal(str)
    message = Signal(str)
    progress = Signal(float)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e) or type(e).__name__)


def _probe_gpu() -> str | None:
    from .. import cuda_setup
    cuda_setup.setup()
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} ({props.total_memory / 2**30:.1f} GB)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Autosubtitler")
        self.resize(1250, 720)
        self.setAcceptDrops(True)
        self.qsettings = QSettings("Autosubtitler", "Autosubtitler")
        self.pipeline = Pipeline()
        self.worker: Worker | None = None
        self.helper: FunctionThread | None = None
        self.gpu_thread: FunctionThread | None = None

        self._build_ui()
        self._load_settings()
        self._update_buttons()
        self._probe_gpu()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # File queue
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Language", "Subtitles"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_LANG, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_OUTPUT, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self._open_output)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.setToolTip("Drop video or audio files here. Double-click a finished file to show its subtitles.")

        self.add_files_btn = QPushButton("Add files…")
        self.add_folder_btn = QPushButton("Add folder…")
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear")
        self.add_files_btn.clicked.connect(self._add_files_dialog)
        self.add_folder_btn.clicked.connect(self._add_folder_dialog)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear)
        file_buttons = QHBoxLayout()
        for b in (self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            file_buttons.addWidget(b)
        file_buttons.addStretch()

        queue_box = QGroupBox("Media files (drag && drop supported)")
        qv = QVBoxLayout(queue_box)
        qv.addWidget(self.table)
        qv.addLayout(file_buttons)

        # Options
        self.tabs = QTabWidget()
        self.tabs.addTab(self._subtitle_tab(), "Subtitles")
        self.tabs.addTab(self._models_tab(), "Models && performance")

        # Progress + log
        self.current_label = QLabel("Idle")
        self.file_bar = QProgressBar()
        self.total_bar = QProgressBar()
        self.total_bar.setFormat("Overall: %v / %m files")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)

        self.start_btn = QPushButton("Generate subtitles")
        self.start_btn.setDefault(True)
        self.start_btn.setMinimumHeight(34)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row = QHBoxLayout()
        run_row.addWidget(self.current_label, 1)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(self.start_btn)

        # Left: queue, progress, log. Right: options.
        left = QVBoxLayout()
        left.addWidget(queue_box, 3)
        left.addLayout(run_row)
        left.addWidget(self.file_bar)
        left.addWidget(self.total_bar)
        left.addWidget(self.log_view, 1)

        right = QVBoxLayout()
        right.addWidget(self.tabs)
        right.addStretch()
        self.tabs.setFixedWidth(500)

        columns = QHBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right)
        root.addLayout(columns)

        quit_action = QAction("Quit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.addAction(quit_action)

    def _subtitle_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Auto-detect", None)
        for lang in languages.source_languages():
            self.source_combo.addItem(lang.name, lang.code)

        self.target_combo = QComboBox()
        self.target_combo.addItem("Same as spoken language (no translation)", ORIGINAL)
        for lang in languages.target_languages():
            self.target_combo.addItem(lang.name, lang.code)
        for combo in (self.source_combo, self.target_combo):
            combo.setEditable(True)  # type to search
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.completer().setFilterMode(Qt.MatchContains)
            combo.completer().setCompletionMode(combo.completer().CompletionMode.PopupCompletion)

        self.keep_original_chk = QCheckBox("Also save subtitles in the spoken language")
        self.overwrite_chk = QCheckBox("Overwrite existing subtitle files")

        self.output_mode = QComboBox()
        self.output_mode.addItems(["Next to the media file", "Custom folder"])
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Choose a folder…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._pick_dir(self.output_dir_edit))
        self.output_mode.currentIndexChanged.connect(self._update_output_widgets)
        self.output_browse_btn = browse
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, 1)
        out_row.addWidget(browse)

        form.addRow("Spoken language:", self.source_combo)
        form.addRow("Subtitle language:", self.target_combo)
        form.addRow("Save to:", self.output_mode)
        form.addRow("", out_row)
        form.addRow("", self.keep_original_chk)
        form.addRow("", self.overwrite_chk)
        return w

    def _models_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(WHISPER_MODELS)
        self.whisper_combo.setToolTip("large-v3-turbo: best speed/quality balance. "
                                      "large-v3: most accurate. Smaller models are faster but less accurate.")
        self.nllb_combo = QComboBox()
        for label, model_id in NLLB_MODELS.items():
            self.nllb_combo.addItem(label, model_id)

        self.device_combo = QComboBox()
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU (slow)", "cpu")
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "float16", "int8_float16", "int8", "float32"])
        self.compute_combo.setToolTip("Whisper precision. 'auto' = float16 on GPU, int8 on CPU. "
                                      "int8_float16 uses less VRAM.")
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setToolTip("Higher = slightly more accurate, slower.")
        self.line_spin = QSpinBox()
        self.line_spin.setRange(20, 80)
        self.line_spin.setSuffix(" characters")

        self.vad_chk = QCheckBox("Skip silence and music")
        self.vad_chk.setToolTip("Voice activity detection: only speech is transcribed, which avoids "
                                "hallucinated text during music or silence.")
        self.context_chk = QCheckBox("Translate whole sentences (better quality)")
        self.context_chk.setToolTip("Sentences spanning several subtitles are translated together, "
                                    "then split back across the original timings.")
        self.low_vram_chk = QCheckBox("Low VRAM mode")
        self.low_vram_chk.setToolTip("Unload each model right after use. Slower for batches, "
                                     "but lets large models fit on smaller GPUs.")
        self.download_chk = QCheckBox("Allow model downloads")
        self.download_chk.setToolTip("Only needed once per model. Uncheck for strict offline operation.")

        self.models_dir_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._pick_dir(self.models_dir_edit))
        browse.setToolTip("Use models from another folder (nothing is moved).")
        move = QPushButton("Move…")
        move.clicked.connect(self._move_models)
        move.setToolTip("Move the downloaded models to a new folder and use it from now on.")
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.models_dir_edit, 1)
        dir_row.addWidget(browse)
        dir_row.addWidget(move)

        self.download_btn = QPushButton("Download selected models now")
        self.download_btn.clicked.connect(self._download_models)
        self.gpu_label = QLabel("Checking GPU…")

        form.addRow("Speech model:", self.whisper_combo)
        form.addRow("Translation model:", self.nllb_combo)
        form.addRow("Device:", self.device_combo)
        form.addRow("GPU:", self.gpu_label)
        form.addRow("Whisper precision:", self.compute_combo)
        form.addRow("Beam size:", self.beam_spin)
        form.addRow("Max line length:", self.line_spin)
        form.addRow("", self.vad_chk)
        form.addRow("", self.context_chk)
        form.addRow("", self.low_vram_chk)
        form.addRow("", self.download_chk)
        form.addRow("Models folder:", dir_row)
        form.addRow("", self.download_btn)
        return w

    # ------------------------------------------------------------ settings
    def _load_settings(self) -> None:
        q = self.qsettings
        d = Settings()

        def set_data(combo: QComboBox, value) -> None:
            i = combo.findData(value)
            if i >= 0:
                combo.setCurrentIndex(i)

        set_data(self.source_combo, q.value("source", None) or None)
        set_data(self.target_combo, q.value("target", d.target_language))
        self.keep_original_chk.setChecked(q.value("keep_original", d.also_save_original, bool))
        self.overwrite_chk.setChecked(q.value("overwrite", d.overwrite, bool))
        self.output_mode.setCurrentIndex(q.value("output_mode", 0, int))
        self.output_dir_edit.setText(q.value("output_dir", ""))
        self.whisper_combo.setCurrentText(q.value("whisper", d.whisper_model))
        set_data(self.nllb_combo, q.value("nllb", d.nllb_model))
        set_data(self.device_combo, q.value("device", d.device))
        self.compute_combo.setCurrentText(q.value("compute", d.compute_type))
        self.beam_spin.setValue(q.value("beam", d.beam_size, int))
        self.line_spin.setValue(q.value("line_chars", d.max_line_chars, int))
        self.vad_chk.setChecked(q.value("vad", d.vad_filter, bool))
        self.context_chk.setChecked(q.value("context", d.sentence_context, bool))
        self.low_vram_chk.setChecked(q.value("low_vram", d.low_vram, bool))
        self.download_chk.setChecked(q.value("allow_download", d.allow_download, bool))
        self.models_dir_edit.setText(q.value("models_dir", str(default_models_dir())))
        self._update_output_widgets()

    def _save_settings(self) -> None:
        q = self.qsettings
        q.setValue("source", self.source_combo.currentData() or "")
        q.setValue("target", self.target_combo.currentData())
        q.setValue("keep_original", self.keep_original_chk.isChecked())
        q.setValue("overwrite", self.overwrite_chk.isChecked())
        q.setValue("output_mode", self.output_mode.currentIndex())
        q.setValue("output_dir", self.output_dir_edit.text())
        q.setValue("whisper", self.whisper_combo.currentText())
        q.setValue("nllb", self.nllb_combo.currentData())
        q.setValue("device", self.device_combo.currentData())
        q.setValue("compute", self.compute_combo.currentText())
        q.setValue("beam", self.beam_spin.value())
        q.setValue("line_chars", self.line_spin.value())
        q.setValue("vad", self.vad_chk.isChecked())
        q.setValue("context", self.context_chk.isChecked())
        q.setValue("low_vram", self.low_vram_chk.isChecked())
        q.setValue("allow_download", self.download_chk.isChecked())
        q.setValue("models_dir", self.models_dir_edit.text())

    def _settings(self) -> Settings:
        custom = self.output_mode.currentIndex() == 1 and self.output_dir_edit.text().strip()
        return Settings(
            source_language=self.source_combo.currentData(),
            target_language=self.target_combo.currentData(),
            also_save_original=self.keep_original_chk.isChecked(),
            output_dir=Path(self.output_dir_edit.text().strip()) if custom else None,
            overwrite=self.overwrite_chk.isChecked(),
            whisper_model=self.whisper_combo.currentText(),
            device=self.device_combo.currentData(),
            compute_type=self.compute_combo.currentText(),
            beam_size=self.beam_spin.value(),
            vad_filter=self.vad_chk.isChecked(),
            nllb_model=self.nllb_combo.currentData(),
            sentence_context=self.context_chk.isChecked(),
            low_vram=self.low_vram_chk.isChecked(),
            max_line_chars=self.line_spin.value(),
            models_dir=Path(self.models_dir_edit.text().strip() or default_models_dir()),
            allow_download=self.download_chk.isChecked(),
        )

    # --------------------------------------------------------------- files
    def add_paths(self, paths: list[Path]) -> None:
        existing = {self.table.item(r, COL_FILE).data(PATH_ROLE) for r in range(self.table.rowCount())}
        for p in paths:
            if p.is_dir():
                self.add_paths(sorted(c for c in p.rglob("*") if c.suffix.lower() in MEDIA_EXTENSIONS))
                continue
            if p.suffix.lower() not in MEDIA_EXTENSIONS or str(p) in existing:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(p.name)
            item.setData(PATH_ROLE, str(p))
            item.setToolTip(str(p))
            self.table.setItem(row, COL_FILE, item)
            for col, text in ((COL_STATUS, "Queued"), (COL_LANG, ""), (COL_OUTPUT, "")):
                self.table.setItem(row, col, QTableWidgetItem(text))
            existing.add(str(p))
        self._update_buttons()

    def _add_files_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(MEDIA_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add media files", self.qsettings.value("last_dir", ""),
                                                f"Media files ({exts});;All files (*)")
        if files:
            self.qsettings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_paths([Path(f) for f in files])

    def _add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder", self.qsettings.value("last_dir", ""))
        if folder:
            self.qsettings.setValue("last_dir", folder)
            self.add_paths([Path(folder)])

    def _remove_selected(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
        self._update_buttons()

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self._update_buttons()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and not self._busy():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_paths([Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()])

    def _open_output(self, index) -> None:
        item = self.table.item(index.row(), COL_OUTPUT)
        outputs = item.data(PATH_ROLE) if item else None
        target = outputs[-1] if outputs else self.table.item(index.row(), COL_FILE).data(PATH_ROLE)
        subprocess.Popen(["explorer", "/select,", str(Path(target))])

    def _pick_dir(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
        if folder:
            edit.setText(folder)

    def _update_output_widgets(self) -> None:
        custom = self.output_mode.currentIndex() == 1
        self.output_dir_edit.setEnabled(custom)
        self.output_browse_btn.setEnabled(custom)

    # ----------------------------------------------------------------- run
    def _busy(self) -> bool:
        return any(t is not None and t.isRunning() for t in (self.worker, self.helper))

    def _update_buttons(self) -> None:
        busy = self._busy()
        has_rows = self.table.rowCount() > 0
        self.start_btn.setEnabled(not busy and has_rows)
        self.cancel_btn.setEnabled(self.worker is not None and self.worker.isRunning())
        self.download_btn.setEnabled(not busy)
        for b in (self.add_files_btn, self.add_folder_btn):
            b.setEnabled(not busy)
        self.remove_btn.setEnabled(not busy and bool(self.table.selectedIndexes()))
        self.clear_btn.setEnabled(not busy and has_rows)
        self.tabs.setEnabled(not busy)

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _set_status(self, row: int, text: str) -> None:
        self.table.item(row, COL_STATUS).setText(text)

    def _start(self) -> None:
        s = self._settings()
        if self.output_mode.currentIndex() == 1 and s.output_dir is None:
            QMessageBox.warning(self, "Autosubtitler", "Choose an output folder first.")
            return
        jobs = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, COL_STATUS).text() != "Done":
                jobs.append((row, Path(self.table.item(row, COL_FILE).data(PATH_ROLE))))
                self._set_status(row, "Queued")
        if not jobs:
            QMessageBox.information(self, "Autosubtitler",
                                    "All files are already done. Add more files or clear the list.")
            return
        self._save_settings()
        self.total_bar.setRange(0, len(jobs))
        self.total_bar.setValue(0)
        self.file_bar.setValue(0)

        self.worker = Worker(self.pipeline, jobs, s)
        self.worker.log.connect(self._log)
        self.worker.file_started.connect(self._on_started)
        self.worker.file_progress.connect(self._on_progress)
        self.worker.file_done.connect(self._on_done)
        self.worker.file_failed.connect(self._on_failed)
        self.worker.file_cancelled.connect(lambda r: self._set_status(r, "Cancelled"))
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._update_buttons()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.current_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _on_started(self, row: int) -> None:
        name = self.table.item(row, COL_FILE).text()
        self.current_label.setText(f"Processing: {name}")
        self._set_status(row, "Starting…")
        self.file_bar.setValue(0)
        self.table.scrollToItem(self.table.item(row, COL_FILE))

    def _on_progress(self, row: int, fraction: float, stage: str) -> None:
        pct = int(fraction * 100)
        self.file_bar.setValue(pct)
        self._set_status(row, f"{stage} {pct}%")

    def _on_done(self, row: int, lang: str, outputs: list) -> None:
        self._set_status(row, "Done")
        if lang:
            self.table.item(row, COL_LANG).setText(languages.get(lang).name)
        out = self.table.item(row, COL_OUTPUT)
        out.setText(", ".join(Path(p).name for p in outputs))
        out.setToolTip("\n".join(outputs))
        out.setData(PATH_ROLE, outputs)
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_failed(self, row: int, message: str) -> None:
        self._set_status(row, "Failed")
        self.table.item(row, COL_OUTPUT).setText(message)
        self.table.item(row, COL_OUTPUT).setToolTip(message)
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_finished(self) -> None:
        self.current_label.setText("Idle")
        self._log("Finished.")
        self._update_buttons()

    # ------------------------------------------------------- model helpers
    def _download_models(self) -> None:
        s = self._settings()
        s.allow_download = True
        self._save_settings()
        self._log(f"Downloading models to {s.models_dir} (this can take a while)…")

        def job():
            from ..transcriber import Transcriber
            from ..translator import Translator
            Transcriber(s.whisper_model, "cpu", "int8", s.models_dir, True, self._log_threadsafe).close()
            Translator(s.nllb_model, "cpu", s.models_dir, True, self._log_threadsafe).close()
            return "Models are downloaded. Autosubtitler can now run fully offline."

        self._run_helper(job)

    def _move_models(self) -> None:
        src = Path(self.models_dir_edit.text().strip() or default_models_dir())
        entries = model_store.installed(src)
        if not entries:
            QMessageBox.information(self, "Move models", f"No downloaded models found in\n{src}")
            return
        folder = QFileDialog.getExistingDirectory(self, "Move models to", str(src.parent))
        if not folder:
            return
        dst = Path(folder)
        size_gb = model_store.size_of(entries) / 2**30
        names = ", ".join(e.name for e in entries)
        if QMessageBox.question(
                self, "Move models",
                f"Move {size_gb:.1f} GB of models ({names})\n\nfrom: {src}\nto: {dst}\n\n"
                "Moving to another drive copies the files first and can take a few minutes.") \
                != QMessageBox.Yes:
            return

        self.pipeline.release()  # loaded models keep files open, which blocks moving on Windows
        self._log(f"Moving models to {dst}…")
        self.current_label.setText("Moving models…")
        self.file_bar.setValue(0)

        def job():
            model_store.move_models(src, dst, self._progress_threadsafe)
            return dst

        self._run_helper(job, on_done=self._on_models_moved, on_failed=self._on_move_failed)

    def _on_models_moved(self, dst: Path) -> None:
        self.models_dir_edit.setText(str(dst))
        self._save_settings()
        self.current_label.setText("Idle")
        self._log(f"Models moved. Models folder is now {dst}")

    def _on_move_failed(self, err: str) -> None:
        self.current_label.setText("Idle")
        self.file_bar.setValue(0)
        QMessageBox.warning(self, "Move models", err)

    def _log_threadsafe(self, text: str) -> None:
        if self.helper:
            self.helper.message.emit(text)

    def _progress_threadsafe(self, fraction: float) -> None:
        if self.helper:
            self.helper.progress.emit(fraction)

    def _run_helper(self, fn, on_done=None, on_failed=None) -> None:
        self.helper = FunctionThread(fn)
        self.helper.finished_ok.connect(on_done or (lambda msg: self._log(str(msg))))
        self.helper.message.connect(self._log)
        self.helper.progress.connect(lambda f: self.file_bar.setValue(int(f * 100)))
        self.helper.failed.connect(lambda err: self._log(f"Error: {err}"))
        if on_failed:
            self.helper.failed.connect(on_failed)
        self.helper.finished.connect(self._update_buttons)
        self.helper.start()
        self._update_buttons()

    def _probe_gpu(self) -> None:
        self.gpu_thread = FunctionThread(_probe_gpu)
        self.gpu_thread.finished_ok.connect(self._on_gpu)
        self.gpu_thread.failed.connect(lambda err: self.gpu_label.setText(f"Unavailable ({err})"))
        self.gpu_thread.start()

    def _on_gpu(self, name: str | None) -> None:
        if name:
            self.gpu_label.setText(name)
            self._log(f"GPU ready: {name}")
        else:
            self.gpu_label.setText("No CUDA GPU detected – using CPU")
            self.device_combo.setCurrentIndex(self.device_combo.findData("cpu"))
            self._log("No CUDA GPU detected; switched to CPU (much slower).")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Autosubtitler", "Subtitles are still being generated. Quit anyway?") \
                    != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(10000)
        self._save_settings()
        event.accept()
