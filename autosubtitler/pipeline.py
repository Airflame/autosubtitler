"""End-to-end processing of one media file: transcribe -> segment -> translate -> .srt"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import languages
from .segmenter import SegmentOptions, build_cues, distribute, sentence_groups, wrap
from .srt import Cue, write_srt
from .transcriber import Transcriber, Transcript
from .translator import Translator

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv", ".m4v", ".mpg", ".mpeg", ".ts",
    ".m2ts", ".3gp", ".ogv", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mka",
}

ORIGINAL = "original"  # target language value meaning "no translation"


def default_models_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Autosubtitler" / "models"


@dataclass
class Settings:
    source_language: str | None = None   # None = auto-detect
    target_language: str = "en"          # Whisper code, or ORIGINAL
    also_save_original: bool = False
    output_dir: Path | None = None       # None = next to the media file
    overwrite: bool = True

    whisper_model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True

    nllb_model: str = "facebook/nllb-200-distilled-1.3B"
    sentence_context: bool = True
    low_vram: bool = False

    max_line_chars: int = 42
    models_dir: Path = field(default_factory=default_models_dir)
    allow_download: bool = True


@dataclass
class Result:
    language: str
    language_probability: float
    outputs: list[Path]


@dataclass
class Callbacks:
    log: Callable[[str], None] = print
    progress: Callable[[float, str], None] = lambda fraction, stage: None
    is_cancelled: Callable[[], bool] = lambda: False


class Pipeline:
    """Holds loaded models between files so a batch doesn't reload them each time."""

    def __init__(self) -> None:
        self._transcriber: Transcriber | None = None
        self._translator: Translator | None = None

    # -- model management -------------------------------------------------
    def transcriber(self, settings: Settings, callbacks: Callbacks) -> Transcriber:
        key = (settings.whisper_model, settings.device, settings.compute_type)
        if self._transcriber is None or self._transcriber.key != key:
            self.release_transcriber()
            callbacks.log(f"Loading Whisper '{settings.whisper_model}' on {settings.device}...")
            self._transcriber = Transcriber(settings.whisper_model, settings.device, settings.compute_type,
                                            settings.models_dir, settings.allow_download, callbacks.log)
        return self._transcriber

    def translator(self, settings: Settings, callbacks: Callbacks) -> Translator:
        key = (settings.nllb_model, settings.device)
        if self._translator is None or self._translator.key != key:
            self.release_translator()
            callbacks.log(f"Loading translator '{settings.nllb_model}' on {settings.device}...")
            self._translator = Translator(settings.nllb_model, settings.device, settings.models_dir,
                                          settings.allow_download, callbacks.log)
        return self._translator

    def release_transcriber(self) -> None:
        if self._transcriber:
            self._transcriber.close()
            self._transcriber = None

    def release_translator(self) -> None:
        if self._translator:
            self._translator.close()
            self._translator = None

    def release(self) -> None:
        self.release_transcriber()
        self.release_translator()

    def process(self, media: Path, settings: Settings, callbacks: Callbacks) -> Result:
        out_dir = settings.output_dir or media.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        translate = settings.target_language != ORIGINAL

        target_path = None
        if translate:
            target_path = out_dir / f"{media.stem}.{languages.get(settings.target_language).iso}.srt"
            if target_path.exists() and not settings.overwrite:
                callbacks.log(f"Skipping, already exists: {target_path.name}")
                return Result("", 0.0, [target_path])

        # Stage 1: speech recognition (0-50% of the progress bar, 0-100% if no translation)
        progress_span = 0.5 if translate else 1.0
        whisper = self.transcriber(settings, callbacks)
        callbacks.log(f"Transcribing {media.name}...")
        transcript = whisper.transcribe(media, settings.source_language, "transcribe", settings.beam_size, settings.vad_filter,
                                        lambda p: callbacks.progress(p * progress_span, "Transcribing"), callbacks.is_cancelled)
        src = languages.get(transcript.language)
        if settings.source_language is None:
            callbacks.log(f"Detected language: {src.name} ({transcript.language_probability:.0%} confidence)")

        outputs: list[Path] = []
        opts = SegmentOptions(max_line_chars=settings.max_line_chars if src.spaced else 20)
        original_cues = self._cues(transcript, opts, src.spaced)

        if not translate or settings.also_save_original:
            path = out_dir / f"{media.stem}.{src.iso}.srt"
            write_srt(original_cues, path)
            outputs.append(path)
            callbacks.log(f"Saved {path.name}")

        if translate:
            tgt = languages.get(settings.target_language)
            if tgt.code == src.code:
                write_srt(original_cues, target_path)
                callbacks.log(f"Source is already {tgt.name}; saved transcript as {target_path.name}")
            else:
                cues, src_nllb = original_cues, src.nllb
                if src_nllb is None:
                    # NLLB can't read this language: let Whisper translate to English first.
                    callbacks.log(f"{src.name} isn't supported by the translator; "
                           "using Whisper's English translation as an intermediate step.")
                    tr_en = whisper.transcribe(media, transcript.language, "translate", settings.beam_size,
                                               settings.vad_filter, lambda p: None, callbacks.is_cancelled)
                    cues = self._cues(tr_en, SegmentOptions(max_line_chars=settings.max_line_chars), True)
                    src_nllb = languages.get("en").nllb
                if settings.low_vram:
                    self.release_transcriber()
                if tgt.code == "en" and src_nllb == languages.get("en").nllb:
                    translated = cues
                else:
                    translated = self._translate(cues, src_nllb, tgt, settings, callbacks, progress_span)
                write_srt(translated, target_path)
                callbacks.log(f"Saved {target_path.name}")
                if settings.low_vram:
                    self.release_translator()
            if target_path not in outputs:
                outputs.append(target_path)

        callbacks.progress(1.0, "Done")
        return Result(transcript.language, transcript.language_probability, outputs)

    @staticmethod
    def _cues(tr: Transcript, opts: SegmentOptions, spaced: bool) -> list[Cue]:
        if tr.words:
            return build_cues(tr.words, opts, spaced)
        return [Cue(c.start, c.end, wrap(c.text, opts.max_line_chars, spaced)) for c in tr.segments]

    def _translate(self, cues: list[Cue], src_nllb: str, tgt: languages.Language, settings: Settings,
                   callbacks: Callbacks, progress_start: float) -> list[Cue]:
        translator = self.translator(settings, callbacks)
        callbacks.log(f"Translating {len(cues)} subtitles into {tgt.name}...")
        flat = [" ".join(c.text.split()) for c in cues]
        groups = sentence_groups(cues) if settings.sentence_context else [[i] for i in range(len(cues))]
        texts = [" ".join(flat[i] for i in g) for g in groups]

        results = translator.translate(
            texts, src_nllb, tgt.nllb,
            progress=lambda p: callbacks.progress(progress_start + p * (1 - progress_start), "Translating"),
            is_cancelled=callbacks.is_cancelled)

        max_chars = settings.max_line_chars if tgt.spaced else 20
        out: list[Cue] = []
        for group, text in zip(groups, results):
            parts = distribute(text, [len(flat[i]) for i in group], tgt.spaced)
            for i, part in zip(group, parts):
                out.append(Cue(cues[i].start, cues[i].end, wrap(part, max_chars, tgt.spaced)))
        return out
