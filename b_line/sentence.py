"""Sentence segmentation for recognizer output.

The streaming zipformer emits ALL-CAPS text with no punctuation, so sentence
marks have to come from somewhere else. Two sources, in order of preference:

1. a punctuation-restoration model (b_line.punct) inserts . ? ! ; when the text
   already has terminal punctuation this module simply splits on it;
2. otherwise fall back to discourse markers ("so", "but", "because", ...) and a
   length cap -- cruder, but no model and no download.

Either way, split_sentences() returns sentence-sized pieces. Pieces that are
too short are merged into the previous one so a stray "Well," does not become
its own caption.
"""
from __future__ import annotations

import re
from typing import List

TERMINAL = ".!?…。！？"

# Clause-initial cues that usually start a new sentence in lecture English.
# Matched case-insensitively; "so that", "also" etc. are excluded by \b.
_MARKERS = re.compile(
    r"\b(?:and then|and so|but|because|however|actually|you know|i mean|"
    r"which is why|that's why|so|now|well)\b",
    re.IGNORECASE,
)


def split_sentences(text: str, min_chars: int = 18, max_chars: int = 96) -> List[str]:
    """Split one finalized ASR slice into sentence-sized pieces."""
    t = " ".join((text or "").split())
    if not t:
        return []

    if any(ch in t for ch in TERMINAL):
        parts = _split_on_punctuation(t)
    else:
        parts = _split_on_markers(t, min_chars)

    parts = _merge_short(parts, min_chars)
    parts = _enforce_cap(parts, max_chars)
    return [p.strip() for p in parts if p.strip()]


def _split_on_punctuation(text: str) -> List[str]:
    out, start = [], 0
    for i, ch in enumerate(text):
        if ch in TERMINAL:
            # keep the mark, and absorb closing quotes/brackets that follow it
            end = i + 1
            while end < len(text) and text[end] in "\"'”’)）":
                end += 1
            out.append(text[start:end])
            start = end
    if text[start:].strip():
        out.append(text[start:])
    return out


def _split_on_markers(text: str, min_chars: int) -> List[str]:
    cuts = []
    for m in _MARKERS.finditer(text):
        if m.start() - (cuts[-1] if cuts else 0) >= min_chars:
            cuts.append(m.start())
    if not cuts:
        return [text]
    out, start = [], 0
    for c in cuts:
        out.append(text[start:c])
        start = c
    out.append(text[start:])
    return out


def _merge_short(parts: List[str], min_chars: int) -> List[str]:
    """Glue a marker-cut fragment back onto its neighbour.

    A piece that ends with sentence punctuation is a real sentence no matter how
    short ("Yes.", "Right."), so it is never merged. Only punctuation-free
    fragments are, because they were cut by a marker, not by the language.
    """
    merged: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and len(merged[-1]) < min_chars and not _ends_sentence(merged[-1]):
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[-1]) < min_chars and not _ends_sentence(merged[-1]):
        merged[-2] = f"{merged[-2]} {merged[-1]}"
        merged.pop()
    return merged


def _ends_sentence(p: str) -> bool:
    return bool(p) and p[-1] in TERMINAL


def _enforce_cap(parts: List[str], max_chars: int) -> List[str]:
    if max_chars <= 0:
        return parts
    out: List[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(p[:cut])
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out
