"""Inspecting and relocating the downloaded model cache.

The models folder contains one subfolder per backend (see MODEL_SUBDIRS). Only
those subfolders are ever moved, so pointing the models folder at a directory
that also holds unrelated files is safe.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

MODEL_SUBDIRS = ("whisper", "nllb")
_CHUNK = 16 * 1024 * 1024


class MoveError(Exception):
    """A move that was refused or failed; the message is meant for the user."""


def installed(models_dir: Path) -> list[Path]:
    """The model subfolders present in models_dir."""
    return [models_dir / name for name in MODEL_SUBDIRS if (models_dir / name).is_dir()]


def size_of(paths: list[Path]) -> int:
    return sum(f.stat().st_size for p in paths for f in p.rglob("*") if f.is_file())


def move_models(src: Path, dst: Path, progress: Callable[[float], None] = lambda f: None) -> None:
    """Move the model subfolders of src into dst.

    Same drive: renames, which is instant. Different drive: copies everything,
    and deletes the originals only once the whole copy succeeded.
    """
    src, dst = src.resolve(), dst.resolve()
    entries = installed(src)
    if not entries:
        raise MoveError(f"No downloaded models found in {src}.")
    if dst == src:
        raise MoveError("The models are already in that folder.")
    if dst.is_relative_to(src):
        raise MoveError("The new location can't be inside the current models folder.")
    conflicts = [dst / e.name for e in entries if (dst / e.name).exists()]
    if conflicts:
        raise MoveError(f"{conflicts[0]} already exists. If it holds models, "
                        "select that folder with Browse… instead.")

    dst.mkdir(parents=True, exist_ok=True)
    if _same_device(src, dst):
        _rename_all(entries, dst)
    else:
        total = size_of(entries)
        if shutil.disk_usage(dst).free < total:
            raise MoveError(f"Not enough free space in {dst} ({total / 2**30:.1f} GB needed).")
        _copy_all(entries, dst, total, progress)
        for e in entries:
            shutil.rmtree(e)
    progress(1.0)

    try:
        src.rmdir()  # only succeeds if nothing else is left in it
    except OSError:
        pass


def _same_device(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def _rename_all(entries: list[Path], dst: Path) -> None:
    moved: list[Path] = []
    try:
        for e in entries:
            os.replace(e, dst / e.name)
            moved.append(e)
    except OSError as err:
        for e in moved:  # put back what was already moved
            os.replace(dst / e.name, e)
        raise MoveError(f"Could not move {e.name}: {err}. Close any program using the models and retry.")


def _copy_all(entries: list[Path], dst: Path, total: int, progress: Callable[[float], None]) -> None:
    done = 0
    created: list[Path] = []
    try:
        for e in entries:
            created.append(dst / e.name)
            for f in e.rglob("*"):
                target = dst / f.relative_to(e.parent)
                if f.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "rb") as fin, open(target, "wb") as fout:
                    while chunk := fin.read(_CHUNK):
                        fout.write(chunk)
                        done += len(chunk)
                        progress(done / total if total else 1.0)
                shutil.copystat(f, target)
    except OSError as err:
        for c in created:
            shutil.rmtree(c, ignore_errors=True)
        raise MoveError(f"Copy failed: {err}. The models were left in their original location.")
