"""Minimal SRT model and writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = []
    for i, cue in enumerate((c for c in cues if c.text.strip()), start=1):
        blocks.append(
            f"{i}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text.strip()}\n"
        )
    return "\n".join(blocks)


def write_srt(cues: list[Cue], path: Path) -> None:
    # UTF-8 with BOM: the most widely recognised encoding by Windows players and TVs.
    path.write_text(to_srt(cues), encoding="utf-8-sig")
