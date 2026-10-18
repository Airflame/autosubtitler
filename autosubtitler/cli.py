"""Command-line interface: python -m autosubtitler.cli movie.mkv --to pl"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import languages
from .pipeline import ORIGINAL, Callbacks, Pipeline, Settings
from .transcriber import WHISPER_MODELS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autosubtitler", description="Generate translated .srt subtitles offline.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--to", default="en", help=f"target language code, or '{ORIGINAL}' for no translation")
    p.add_argument("--from", dest="source", default=None, help="source language code (default: auto-detect)")
    p.add_argument("--whisper", default="large-v3-turbo", choices=WHISPER_MODELS)
    p.add_argument("--nllb", default=Settings.nllb_model)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--keep-original", action="store_true", help="also save the untranslated subtitles")
    p.add_argument("--offline", action="store_true", help="never download models")
    a = p.parse_args(argv)

    for code in (a.to, a.source):
        if code and code != ORIGINAL and code not in languages.LANGUAGES:
            p.error(f"unknown language code: {code}")

    s = Settings(source_language=a.source, target_language=a.to, whisper_model=a.whisper,
                 nllb_model=a.nllb, device=a.device, output_dir=a.output_dir,
                 also_save_original=a.keep_original, allow_download=not a.offline)

    def progress(fraction: float, stage: str) -> None:
        print(f"\r  {stage:<12} {fraction:6.1%}", end="", file=sys.stderr, flush=True)

    pipeline = Pipeline()
    cb = Callbacks(log=lambda m: print(f"\n{m}" if m else "", file=sys.stderr), progress=progress)
    failed = 0
    for f in a.files:
        try:
            result = pipeline.process(f, s, cb)
            for out in result.outputs:
                print(out)
        except Exception as e:  # keep going with the rest of the batch
            failed += 1
            print(f"\nERROR {f}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
