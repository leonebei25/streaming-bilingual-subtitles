"""Punctuation restoration through sherpa-onnx.

The streaming zipformer was trained on unpunctuated, uppercased text, so its
output has no sentence marks. The CT-Transformer punctuation model restores
`,.?!` from context, which gives the segmenter real sentence boundaries and lets
`prettify_english` capitalise correctly.

Model (Chinese + English, ~279 MB):
  https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/
  sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

Extract it and pass the resulting `model.onnx` as --punct-model. If the file or
the package is missing the error says exactly what to do; nothing else changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class PunctRestorer:
    def __init__(self, model_path: str, num_threads: int = 1, provider: str = "cpu"):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "punctuation restoration needs sherpa-onnx: `pip install sherpa-onnx`."
            ) from exc

        path = Path(model_path)
        if path.is_dir():
            matches = sorted(path.glob("*.onnx"))
            if not matches:
                raise RuntimeError(f"{path}: no .onnx file inside")
            path = matches[0]
        if not path.exists():
            raise RuntimeError(
                f"punctuation model not found: {path}\n"
                "Download and extract:\n"
                "  https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "punctuation-models/"
                "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2\n"
                "then pass --punct-model <extracted>/model.onnx"
            )

        self._sherpa = sherpa_onnx
        config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=str(path),
                num_threads=int(num_threads),
                provider=provider,
            )
        )
        self._punct = sherpa_onnx.OfflinePunctuation(config)
        self.model_path = str(path)

    def restore(self, text: Optional[str]) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        try:
            return self._punct.add_punctuation(t).strip()
        except Exception:
            # Restoration is an enhancement: never lose the transcript over it.
            return t
