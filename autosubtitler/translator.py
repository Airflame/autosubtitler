"""Offline machine translation with Meta's NLLB-200 models."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from . import cuda_setup
from .transcriber import Cancelled

NLLB_MODELS = {
    "NLLB-200 distilled 1.3B (recommended)": "facebook/nllb-200-distilled-1.3B",
    "NLLB-200 distilled 600M (fast, lower quality)": "facebook/nllb-200-distilled-600M",
    "NLLB-200 3.3B (best, needs ~8 GB VRAM)": "facebook/nllb-200-3.3B",
}


class Translator:
    def __init__(self, model_id: str, device: str, models_dir: Path, allow_download: bool,
                 log: Callable[[str], None] = print):
        cuda_setup.setup()
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.key = (model_id, device)
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        cache = str(models_dir / "nllb")

        def load(local_only: bool):
            tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, local_files_only=local_only)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, cache_dir=cache, local_files_only=local_only, dtype=dtype)
            return tok, mdl

        try:
            self.tokenizer, model = load(local_only=True)
        except Exception:
            if not allow_download:
                raise RuntimeError(
                    f"Translation model '{model_id}' is not downloaded and model downloads are disabled.")
            log(f"Downloading translation model '{model_id}' (one-time, several GB)...")
            self.tokenizer, model = load(local_only=False)
        self.model = model.to(self.device).eval()

    def translate(self, texts: list[str], src: str, tgt: str, beam_size: int = 4,
                  batch_size: int = 16, progress: Callable[[float], None] = lambda p: None,
                  is_cancelled: Callable[[], bool] = lambda: False) -> list[str]:
        import torch

        self.tokenizer.src_lang = src
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt)
        # Translate longest texts together: less padding, faster batches.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        results: list[str] = [""] * len(texts)

        for start in range(0, len(order), batch_size):
            if is_cancelled():
                raise Cancelled()
            idx = order[start:start + batch_size]
            batch = [texts[i] for i in idx]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=512).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    forced_bos_token_id=tgt_id,
                    num_beams=beam_size,
                    max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 16),
                    no_repeat_ngram_size=4,
                )
            for i, text in zip(idx, self.tokenizer.batch_decode(out, skip_special_tokens=True)):
                results[i] = text.strip()
            progress(min(1.0, (start + len(idx)) / len(order)))
        return results

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
