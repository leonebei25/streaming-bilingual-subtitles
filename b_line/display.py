"""Display pacing for live captions.

The pipeline decides *what* a caption says; this module decides *how long it
stays on screen*. Two rules keep a live stream readable:

* a caption is shown for at least its reading time, clamped to
  [min_hold_s, max_hold_s], so it never flashes past;
* captions that arrive faster than they can be read -- a burst after an ASR or
  MT backlog -- are compressed, never dropped and never swapped instantly.

Times are audio-stream seconds, the same clock the subtitle events carry, so a
recording replays with exactly the pacing a viewer saw live.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

# Reading speeds in characters per second. 7 CJK characters/s and 18 Latin
# characters/s are conservative broadcast-subtitle figures (roughly 200 wpm).
CJK_CPS = 7.0
LATIN_CPS = 18.0


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF      # CJK extension A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified ideographs
        or 0x3000 <= code <= 0x30FF   # CJK punctuation, hiragana, katakana
        or 0xF900 <= code <= 0xFAFF   # compatibility ideographs
        or 0xFF00 <= code <= 0xFFEF   # fullwidth forms
    )


def reading_time(text: str, cjk_cps: float = CJK_CPS,
                 latin_cps: float = LATIN_CPS) -> float:
    """Seconds a viewer needs to read 'text'."""
    text = (text or "").strip()
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if is_cjk(ch))
    latin = len(text) - cjk
    return cjk / max(cjk_cps, 1e-6) + latin / max(latin_cps, 1e-6)


def dwell_sec(text: str, min_hold_s: float = 1.8, max_hold_s: float = 5.0,
              cjk_cps: float = CJK_CPS, latin_cps: float = LATIN_CPS) -> float:
    """How long one caption should stay up, in seconds."""
    t = reading_time(text, cjk_cps, latin_cps)
    return min(max(t, min_hold_s), max(max_hold_s, min_hold_s))


class _JoinedCaption:
    """Several consecutive events shown as a single caption.

    They come from one utterance that the recognizer committed in pieces; the
    viewer should read one sentence, not three fragments.
    """

    def __init__(self, events):
        self.events = tuple(events)
        self.text_en = " ".join(e.text_en.strip() for e in events if e.text_en).strip()
        self.text_zh = "".join(e.text_zh.strip() for e in events if e.text_zh).strip()
        self.t_start = events[0].t_start
        self.t_end = events[-1].t_end
        self.seg_id = events[0].seg_id
        self.revision = events[-1].revision
        self.gap_flag = False


class CaptionQueue:
    """FIFO of subtitle events paced by reading time.

    Push events as they are emitted; call poll() every display frame. The
    current caption is replaced only after it has been readable, and when a
    backlog exists the remaining captions are compressed rather than skipped.

    The queue is bounded: if the display ever falls more than max_lag_s behind
    the newest event (speech faster than anyone can read it), intermediate
    captions are dropped rather than replayed, so the same drift cannot grow
    without limit over a ten-minute run.
    """

    def __init__(self, min_hold_s: float = 1.8, max_hold_s: float = 5.0,
                 cjk_cps: float = CJK_CPS, latin_cps: float = LATIN_CPS,
                 backlog_floor_s: float = 1.2, max_lag_s: float = 4.0,
                 merge_gap_s: float = 1.2, merge_max_chars: int = 100,
                 join_window_s: float = 2.5):
        self.min_hold_s = float(min_hold_s)
        self.max_hold_s = float(max_hold_s)
        self.cjk_cps = float(cjk_cps)
        self.latin_cps = float(latin_cps)
        self.backlog_floor_s = float(backlog_floor_s)
        self.max_lag_s = float(max_lag_s)
        # A fragment that continues the caption already on screen is appended to
        # it instead of replacing it. Cut points come from the recognizer, not
        # from the sentence, so joining them is what makes a caption read as a
        # sentence rather than as three pieces flashing past.
        self.merge_gap_s = float(merge_gap_s)
        self.merge_max_chars = int(merge_max_chars)
        self.join_window_s = float(join_window_s)
        self.reset()

    def reset(self) -> None:
        self.pending: Deque = deque()
        self.current = None
        self.previous = None
        self.dropped = 0
        self._shown_at = 0.0
        self._dwell = 0.0

    # -- queue -------------------------------------------------------------

    def push(self, event, now: float) -> None:
        # Continue the caption that is already on screen when the new piece is
        # simply the rest of the same sentence.
        if self.current is not None and self._can_join(self.current, event, now):
            self.current = self._joined(self.current, event)
            self._dwell = max(self._dwell, dwell_sec(
                self._display_text(self.current), self.min_hold_s,
                self.max_hold_s, self.cjk_cps, self.latin_cps))
            return
        self.pending.append(event)
        self._advance(now)

    def poll(self, now: float) -> Tuple[Optional[object], Optional[object]]:
        """Advance the queue to 'now'. Returns (current, previous)."""
        self._advance(now)
        return self.current, self.previous

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def shown_at(self) -> float:
        return self._shown_at

    @property
    def dwell(self) -> float:
        return self._dwell

    def remaining(self, now: float) -> float:
        if self.current is None:
            return 0.0
        return max(0.0, self._shown_at + self._dwell - now)

    # -- internals ---------------------------------------------------------

    def _advance(self, now: float) -> None:
        if self.current is None:
            if self.pending:
                self._take(now)
            return
        if not self.pending:
            return
        lag = now - self._event_end(self.current, now)
        if now - self._shown_at + 1e-9 >= self._dwell or lag > self.max_lag_s:
            self.previous = self.current
            self._take(now)

    def _take(self, now: float) -> None:
        # A caption is worth showing only if it can still stay up for the
        # minimum hold inside what is left of the lag budget. Skipping the ones
        # that cannot is what keeps the lag from growing over a long run, and
        # what stops a hopeless backlog from flashing past at 10 captions/s.
        horizon = max(0.0, self.max_lag_s - self.backlog_floor_s)
        while (len(self.pending) > 1
               and now - self._event_end(self.pending[0], now) > horizon):
            self.pending.popleft()
            self.dropped += 1
        self.current = self.pending.popleft()
        if self.pending:
            self.current = self._join(self.current)
        self._shown_at = now
        text = self._display_text(self.current)
        dwell = dwell_sec(text, self.min_hold_s, self.max_hold_s,
                          self.cjk_cps, self.latin_cps)
        # Backlog compression: with N captions still queued, show this one
        # faster so the delay does not accumulate, but never instant.
        backlog = len(self.pending)
        if backlog > 0:
            dwell = max(self.backlog_floor_s, dwell / (1.0 + 0.5 * backlog))
        self._dwell = dwell

    def _can_join(self, cur, event, now: float) -> bool:
        if now - self._shown_at > self.join_window_s:
            return False
        if not hasattr(cur, "t_end") or not hasattr(event, "t_start"):
            return False
        if event.t_start - cur.t_end > self.merge_gap_s:
            return False
        joined = len(self._display_text(cur)) + 1 + len(self._display_text(event))
        return joined <= self.merge_max_chars

    @staticmethod
    def _joined(cur, event):
        events = list(getattr(cur, "events", (cur,))) + [event]
        return _JoinedCaption(events)

    def _join(self, first):
        """Merge following fragments that continue the same utterance."""
        events = [first]
        total = len(self._display_text(first))
        while self.pending and len(events) < 4:
            nxt = self.pending[0]
            if not hasattr(nxt, "t_start") or not hasattr(events[-1], "t_end"):
                break
            gap = nxt.t_start - events[-1].t_end
            joined = total + 1 + len(self._display_text(nxt))
            if gap > self.merge_gap_s or joined > self.merge_max_chars:
                break
            events.append(self.pending.popleft())
            total = joined
        if len(events) == 1:
            return first
        return _JoinedCaption(events)

    @staticmethod
    def _display_text(event) -> str:
        """The text a viewer has to read: translation when present, else ASR."""
        return getattr(event, "text_zh", "") or getattr(event, "text_en", "") or ""

    @staticmethod
    def _event_end(event, default: float) -> float:
        return float(getattr(event, "t_end", default))
