"""Evaluation metrics for the B-line."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import SubtitleEvent


def edit_distance(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,              # deletion
                cur[j - 1] + 1,           # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> Optional[float]:
    ref = reference.split()
    if not ref:
        return None
    hyp = hypothesis.split()
    return edit_distance(ref, hyp) / len(ref)


def _char_ngrams(s: str, n: int) -> Counter:
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(reference: str, hypothesis: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF). Suitable for Chinese translations."""
    if not reference and not hypothesis:
        return 1.0
    p_total = r_total = 0.0
    for n in range(1, max_n + 1):
        r = _char_ngrams(reference, n)
        h = _char_ngrams(hypothesis, n)
        matches = sum((r & h).values())
        r_total += matches / max(1, sum(r.values()))
        p_total += matches / max(1, sum(h.values()))
    p = p_total / max_n
    r = r_total / max_n
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def evaluate(
    events: List[SubtitleEvent],
    references: Optional[List[Tuple[float, float, str, Optional[str]]]] = None,
    duration_sec: Optional[float] = None,
    processing_sec: Optional[float] = None,
    realtime: bool = False,
) -> dict:
    """Assemble B-line metrics.

    'references' are (t_start, t_end, text_en, text_zh) tuples in the same order
    as the emitted segments. WER and timing error align by order.
    """
    out: dict = {}
    subs = [e for e in events if not e.gap_flag]

    if references is not None and subs:
        ref_text = " ".join(r[2] for r in references)
        hyp_text = " ".join(e.text_en for e in subs)
        out["wer"] = wer(ref_text, hyp_text)

        errs = [abs(e.t_start - r[0]) * 1000.0 for e, r in zip(subs, references)]
        if errs:
            out["timing_error_ms_mean"] = float(np.mean(errs))
            out["timing_error_ms_max"] = float(np.max(errs))

        if any(len(r) == 4 and r[3] for r in references):
            zh_ref = "".join(r[3] for r in references if len(r) == 4 and r[3])
            zh_hyp = "".join(e.text_zh for e in subs)
            out["chrf"] = chrf(zh_ref, zh_hyp)

    if realtime:
        lats = [e.emitted_at - e.t_start for e in subs if e.emitted_at is not None]
        if lats:
            out["latency_p50_ms"] = float(np.percentile(lats, 50) * 1000.0)
            out["latency_p95_ms"] = float(np.percentile(lats, 95) * 1000.0)

    if duration_sec and processing_sec is not None and duration_sec > 0:
        out["rtf"] = processing_sec / duration_sec

    out["n_segments"] = len(subs)
    out["n_gaps"] = sum(1 for e in events if e.gap_flag)
    return out
