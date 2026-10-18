"""Speech recognition and language detection with faster-whisper."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import cuda_setup
from .segmenter import Word
from .srt import Cue

WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny", "distil-large-v3"]


class Cancelled(Exception):
    pass


@dataclass
class Transcript:
    language: str
    language_probability: float
    duration: float
    words: list[Word]
    segments: list[Cue]  # raw Whisper segments, used when word timing is unavailable
    translated_to_english: bool = False


class Transcriber:
    def __init__(self, model: str, device: str, compute_type: str, models_dir: Path,
                 allow_download: bool, log: Callable[[str], None] = print):
        cuda_setup.setup()
        from faster_whisper import WhisperModel

        self.key = (model, device, compute_type)
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        root = str(models_dir / "whisper")
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type,
                                      download_root=root, local_files_only=True)
        except Exception:
            if not allow_download:
                raise RuntimeError(
                    f"Whisper model '{model}' is not downloaded and model downloads are disabled.")
            log(f"Downloading Whisper model '{model}' (one-time)...")
            self.model = WhisperModel(model, device=device, compute_type=compute_type,
                                      download_root=root)

    def transcribe(self, media: Path, language: str | None, task: str, beam_size: int,
                   vad: bool, progress: Callable[[float], None],
                   is_cancelled: Callable[[], bool]) -> Transcript:
        segments, info = self.model.transcribe(
            str(media),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad,
            word_timestamps=True,
            condition_on_previous_text=False,  # avoids repetition loops on long media
            language_detection_segments=4,
        )
        words: list[Word] = []
        raw: list[Cue] = []
        for seg in segments:
            if is_cancelled():
                raise Cancelled()
            raw.append(Cue(seg.start, seg.end, seg.text.strip()))
            for w in seg.words or []:
                words.append(Word(w.start, w.end, w.word))
            if info.duration:
                progress(min(1.0, seg.end / info.duration))
        progress(1.0)
        return Transcript(info.language, info.language_probability, info.duration,
                          words, raw, translated_to_english=(task == "translate"))

    def close(self) -> None:
        del self.model
        gc.collect()
