"""Make the CUDA runtime DLLs visible to CTranslate2 (used by faster-whisper).

On Windows CTranslate2 needs cuBLAS and cuDNN DLLs. PyTorch's CUDA wheels ship
them in ``torch/lib``, and the optional ``nvidia-*-cu12`` pip packages ship them
in ``site-packages/nvidia/*/bin``. Both locations are added to the DLL search
path before faster-whisper is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False


def setup() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    _done = True

    dirs: list[Path] = []
    try:
        import torch  # noqa: F401  (importing torch also loads its bundled CUDA DLLs)
        dirs.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            dirs.extend(p for p in nvidia.glob("*/bin") if p.is_dir())

    for d in dirs:
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"


def cuda_available() -> bool:
    setup()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
