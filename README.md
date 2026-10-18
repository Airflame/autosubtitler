# Autosubtitler

Offline desktop app that listens to a video or audio file, detects the spoken
language, and writes an `.srt` subtitle file in the language you choose.
Everything runs locally with free models. After the one-time model download,
no internet connection is needed.

| Stage | Model | Notes |
|---|---|---|
| Speech recognition and language detection | [Whisper](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ~99 spoken languages, word-level timestamps, voice activity detection |
| Translation | Meta [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | ~200 languages, runs on the GPU in fp16 |
| GUI | PySide6 (Qt) | batch queue, drag and drop, progress, cancel |

Media is decoded with PyAV's bundled FFmpeg, so most video and audio formats
work (`mkv`, `mp4`, `avi`, `mov`, `webm`, `mp3`, `flac`, …) without installing
FFmpeg.

## Requirements

- Windows 10/11, Python 3.10–3.12
- An NVIDIA GPU with a recent driver (CPU works too, but slowly)
- Disk space: ~6 GB for the Python packages, plus 1.6–9 GB for models

## Install and run

```bat
install.bat          :: one-time: creates .venv, installs PyTorch (CUDA) and the dependencies
Autosubtitler.bat    :: starts the app (you can also drop media files onto it)
```

You can also run it manually with `.venv\Scripts\python -m autosubtitler`.

The first run downloads the selected models to
`%LOCALAPPDATA%\Autosubtitler\models`. To download them ahead of time, click
**Models & performance → Download selected models now**. After that, turn off
**Allow downloading models** for strict offline operation.

## Usage

1. Add files (button, folder, or drag and drop).
2. Leave **Spoken language** on *Auto-detect*, or pick it to skip detection.
3. Choose the **Subtitle language**.
4. Click **Generate subtitles**.

Output is written next to the media as `<name>.<lang>.srt`, for example
`Movie.pl.srt`. VLC, MPC-HC, mpv, Plex, Jellyfin and Kodi pick this name up
automatically. Double-click a finished row to show the file in Explorer.

### Command line

```bat
.venv\Scripts\python -m autosubtitler.cli "D:\Movies\film.mkv" --to pl
.venv\Scripts\python -m autosubtitler.cli *.mp4 --to en --keep-original --offline
```

## Choosing models (8 GB VRAM guide)

| Whisper | VRAM (fp16) | Notes |
|---|---|---|
| `large-v3-turbo` (default) | ~2 GB | near large-v3 accuracy, much faster |
| `large-v3` | ~4 GB | most accurate |
| `medium` / `small` | 1–2 GB | faster, less accurate |

| NLLB | VRAM (fp16) | Download |
|---|---|---|
| distilled 600M | ~1.5 GB | 2.5 GB |
| distilled 1.3B (default) | ~3 GB | 5.5 GB |
| 3.3B | ~7 GB | 17.6 GB, enable *Low VRAM mode* |

## How it works

1. Whisper transcribes the audio with word timestamps. It detects the language
   from several parts of the file, not only the first 30 seconds.
2. The words are regrouped into subtitle cues: up to 2 lines of about 42
   characters, at most 6 seconds on screen, with breaks at sentence ends,
   pauses and commas.
3. Cues that make up one sentence are translated together for context. The
   translation is then split back across the original cue timings.
4. If NLLB doesn't support the spoken language (Latin, Breton, Hawaiian),
   Whisper translates it to English first and NLLB continues from there.

## Tests

```bat
.venv\Scripts\python -m pytest
```
