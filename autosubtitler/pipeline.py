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
    def transcriber(self, s: Settings, cb: Callbacks) -> Transcriber:
        key = (s.whisper_model, s.device, s.compute_type)
        if self._transcriber is None or self._transcriber.key != key:
            self.release_transcriber()
            cb.log(f"Loading Whisper '{s.whisper_model}' on {s.device}...")
            self._transcriber = Transcriber(s.whisper_model, s.device, s.compute_type,
                                            s.models_dir, s.allow_download, cb.log)
        return self._transcriber

    def translator(self, s: Settings, cb: Callbacks) -> Translator:
        key = (s.nllb_model, s.device)
        if self._translator is None or self._translator.key != key:
            self.release_translator()
            cb.log(f"Loading translator '{s.nllb_model}' on {s.device}...")
            self._translator = Translator(s.nllb_model, s.device, s.models_dir,
                                          s.allow_download, cb.log)
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

    def process(self, media: Path, s: Settings, cb: Callbacks) -> Result:
        out_dir = s.output_dir or media.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        translate = s.target_language != ORIGINAL

        target_path = None
        if translate:
            target_path = out_dir / f"{media.stem}.{languages.get(s.target_language).iso}.srt"
            if target_path.exists() and not s.overwrite:
                cb.log(f"Skipping, already exists: {target_path.name}")
                return Result("", 0.0, [target_path])

        # Stage 1: speech recognition (0-60% of the progress bar, 0-100% if no translation)
        span = 0.6 if translate else 1.0
        whisper = self.transcriber(s, cb)
        cb.log(f"Transcribing {media.name}...")
        tr = whisper.transcribe(media, s.source_language, "transcribe", s.beam_size, s.vad_filter,
                                lambda p: cb.progress(p * span, "Transcribing"), cb.is_cancelled)
        src = languages.get(tr.language)
        if s.source_language is None:
            cb.log(f"Detected language: {src.name} ({tr.language_probability:.0%} confidence)")

        outputs: list[Path] = []
        opts = SegmentOptions(max_line_chars=s.max_line_chars if src.spaced else 20)
        original_cues = self._cues(tr, opts, src.spaced)

        if not translate or s.also_save_original:
            path = out_dir / f"{media.stem}.{src.iso}.srt"
            write_srt(original_cues, path)
            outputs.append(path)
            cb.log(f"Saved {path.name}")

        if translate:
            tgt = languages.get(s.target_language)
            if tgt.code == src.code:
                write_srt(original_cues, target_path)
                cb.log(f"Source is already {tgt.name}; saved transcript as {target_path.name}")
            else:
                cues, src_nllb = original_cues, src.nllb
                if src_nllb is None:
                    # NLLB can't read this language: let Whisper translate to English first.
                    cb.log(f"{src.name} isn't supported by the translator; "
                           "using Whisper's English translation as an intermediate step.")
                    tr_en = whisper.transcribe(media, tr.language, "translate", s.beam_size,
                                               s.vad_filter, lambda p: None, cb.is_cancelled)
                    cues = self._cues(tr_en, SegmentOptions(max_line_chars=s.max_line_chars), True)
                    src_nllb = languages.get("en").nllb
                if s.low_vram:
                    self.release_transcriber()
                if tgt.code == "en" and src_nllb == languages.get("en").nllb:
                    translated = cues
                else:
                    translated = self._translate(cues, src_nllb, tgt, s, cb, span)
                write_srt(translated, target_path)
                cb.log(f"Saved {target_path.name}")
                if s.low_vram:
                    self.release_translator()
            if target_path not in outputs:
                outputs.append(target_path)

        cb.progress(1.0, "Done")
        return Result(tr.language, tr.language_probability, outputs)

    @staticmethod
    def _cues(tr: Transcript, opts: SegmentOptions, spaced: bool) -> list[Cue]:
        if tr.words:
            return build_cues(tr.words, opts, spaced)
        return [Cue(c.start, c.end, wrap(c.text, opts.max_line_chars, spaced)) for c in tr.segments]

    def _translate(self, cues: list[Cue], src_nllb: str, tgt: languages.Language, s: Settings,
                   cb: Callbacks, start: float) -> list[Cue]:
        translator = self.translator(s, cb)
        cb.log(f"Translating {len(cues)} subtitles into {tgt.name}...")
        flat = [" ".join(c.text.split()) for c in cues]
        groups = sentence_groups(cues) if s.sentence_context else [[i] for i in range(len(cues))]
        texts = [" ".join(flat[i] for i in g) for g in groups]

        results = translator.translate(
            texts, src_nllb, tgt.nllb,
            progress=lambda p: cb.progress(start + p * (1 - start), "Translating"),
            is_cancelled=cb.is_cancelled)

        max_chars = s.max_line_chars if tgt.spaced else 20
        out: list[Cue] = []
        for group, text in zip(groups, results):
            parts = distribute(text, [len(flat[i]) for i in group], tgt.spaced)
            for i, part in zip(group, parts):
                out.append(Cue(cues[i].start, cues[i].end, wrap(part, max_chars, tgt.spaced)))
        return out
