#!/usr/bin/env python3
"""Live two-window demo: video on one window, scrolling transcript on another.

The video's own audio track is decoded to 16 kHz with ffmpeg and replayed at
wall-clock rate through the normal pipeline while the video is paced against the
same clock. Two OpenCV windows: the video itself and a transcript panel that
keeps every recognized sentence for the whole session and scrolls like a normal
text box. Silent playback -- the audio drives the recogniser, it is not played
out.

The model has to be built *before* the audio source, because the source starts
the wall clock when it is constructed.

Examples:
  python scripts/live_demo.py --video video_test/vedio_1.mp4 \
      --asr sherpa --asr-model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26

  # already have a 16 kHz WAV, skip the ffmpeg step
  python scripts/live_demo.py --video ... --wav video_test/vedio_1_16k.wav ...

  # no display (useful on a headless box or for a smoke test)
  python scripts/live_demo.py --video ... --headless --max-seconds 10

Keys: ESC or q quits. Mouse wheel, arrows, Page Up/Down, Home/End and
j/k/u/d/g/G scroll the transcript; scrolling up pauses following, End resumes.
"""
from __future__ import annotations

import argparse
import atexit
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from a_line.enhance import BypassEnhancer, SpectralSubtractionEnhancer
from a_line.enhance.source import EnhancedFrameSource
from a_line.link import LinkConfig, LinkSource
from a_line.video import find_cjk_font
from b_line.asr import ScriptedASR, SherpaASR
from b_line.config import BLineConfig
from b_line.display import CJK_CPS, LATIN_CPS, CaptionQueue
from b_line.mt import IdentityMT, OpusMT
from b_line.pipeline import SubtitlePipeline
from b_line.sources import WavFrameSource

MIN_HOLD_S = 1.8
MAX_HOLD_S = 6.5
ROLL_CHARS = 240   # longest unfinished sentence shown on the live line


class TimedASR:
    """Pass-through wrapper that reports recogniser throughput.

    If one 320 ms chunk takes longer than 320 ms, the pipeline cannot keep up
    and subtitles fall further behind every second; this makes that visible.
    """

    def __init__(self, inner, quiet=True):
        self.inner = inner
        self.quiet = quiet
        self.calls = 0
        self.total = 0.0
        self.worst = 0.0
        # Latest cumulative transcript, used to show words while they are spoken
        # instead of waiting for the sentence to end.
        self.partial = ""

    def reset(self):
        self.inner.reset()
        self.partial = ""

    def _record(self, t0):
        dt = time.perf_counter() - t0
        self.calls += 1
        self.total += dt
        self.worst = max(self.worst, dt)
        if not self.quiet and self.calls % 10 == 0:
            print(f"[asr] calls={self.calls} mean={1000 * self.total / self.calls:.0f}ms "
                  f"max={1000 * self.worst:.0f}ms")

    def stream(self, samples, start_sec, end_sec):
        t0 = time.perf_counter()
        update = self.inner.stream(samples, start_sec, end_sec)
        self._record(t0)
        self.partial = getattr(update, "text", "") or ""
        return update

    def finalize(self):
        t0 = time.perf_counter()
        update = self.inner.finalize()
        self._record(t0)
        self.partial = getattr(update, "text", "") or ""
        return update

    def force_finalize(self):
        """Delegate the stall safety net; a wrapper that swallows it leaves the
        pipeline unable to emit anything when the recognizer never endpoints."""
        t0 = time.perf_counter()
        update = self.inner.force_finalize()
        self._record(t0)
        self.partial = getattr(update, "text", "") or ""
        return update

    def __getattr__(self, name):
        # Any other backend capability must pass through, not disappear.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def prettify_english(text: str) -> str:
    """Normalise the recogniser's all-caps output for display.

    The sherpa zipformer was trained on uppercased text, so it has no case and
    no punctuation. Lower everything and capitalise sentence starts; this does
    not change the string's length, so it can be applied to any text safely.
    """
    t = (text or "").strip()
    if not t:
        return t
    t = t.lower()
    out = []
    capitalise = True
    for ch in t:
        if capitalise and ch.isalpha():
            out.append(ch.upper())
            capitalise = False
        else:
            out.append(ch)
        if ch in ".!?":
            capitalise = True
    t = "".join(out)
    return re.sub(r"\bi\b", "I", t)


def parse_loss(s: str):
    if not s:
        return ()
    out = []
    for part in s.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return tuple(out)


def load_script(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(u["t0"], u["t1"], u["en"], u.get("conf", 0.95)) for u in data["utterances"]]


def extract_audio(video: str) -> str:
    """Decode the video's audio track to a 16 kHz mono WAV next to a temp dir.

    Using ffmpeg on the same file the video window plays removes any question
    of the two inputs being out of step.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg not found on PATH; install ffmpeg or pass --wav explicitly"
        )
    tmp = tempfile.mkdtemp(prefix="live_demo_")
    atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    out = str(Path(tmp) / "audio_16k.wav")
    print(f"extracting audio from {video} ...")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", video, "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not Path(out).exists() or Path(out).stat().st_size < 1024:
        raise SystemExit(f"ffmpeg failed to extract audio:\n{proc.stderr.strip()}")
    return out


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", required=True, help="video file to play")
    ap.add_argument("--wav", help="16 kHz PCM WAV; extracted from --video with ffmpeg "
                                  "when omitted")
    ap.add_argument("--script", help="transcript JSON for --asr scripted")

    ap.add_argument("--asr", choices=["scripted", "sherpa"], default="sherpa")
    ap.add_argument("--asr-model-dir")
    ap.add_argument("--asr-provider", default="cpu")
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--no-endpoint", action="store_true",
                    help="disable sherpa's own endpoint detector (it is on by "
                         "default); sentences then come only from forced cuts")
    ap.add_argument("--rule1-ms", type=float, default=2400.0,
                    help="endpoint rule1: trailing silence that ends an "
                         "utterance even with no decoded speech (default 2400)")
    ap.add_argument("--rule2-ms", type=float, default=1200.0,
                    help="endpoint rule2: trailing silence after decoded speech "
                         "(default 1200); raise it if a breath splits a sentence")
    ap.add_argument("--rule3-sec", type=float, default=20.0,
                    help="endpoint rule3: hard cap on one utterance (default 20 s)")

    ap.add_argument("--mt", choices=["auto", "none", "identity", "opus"],
                    default="auto",
                    help="auto (default) = opus when models/opus-mt-en-zh exists, "
                         "else English-only; none = English-only; identity = no "
                         "translation text")
    ap.add_argument("--mt-model", default=None,
                    help="local model directory or hub id (default: "
                         "models/opus-mt-en-zh when present, else "
                         "Helsinki-NLP/opus-mt-en-zh)")
    ap.add_argument("--mt-device", default=None)
    ap.add_argument("--mt-context", action="store_true")
    ap.add_argument("--punct-model", default="",
                    help="punctuation model: a directory or a .onnx file. A dir "
                         "containing bpe.vocab uses the online English CNN-BiLSTM; "
                         "anything else uses the offline zh+en CT-Transformer")
    ap.add_argument("--semantic-cuts", action="store_true",
                    help="commit a caption as soon as a stable sentence end appears "
                         "inside the draft (needs --punct-model)")
    ap.add_argument("--semantic-lookahead", type=int, default=8,
                    help="words that must remain after a candidate cut (default 8)")
    ap.add_argument("--semantic-confirm", type=int, default=2,
                    help="consecutive runs that must agree on the cut (default 2)")
    ap.add_argument("--semantic-min-chars", type=int, default=24,
                    help="do not look for a cut below this draft size (default 24)")
    ap.add_argument("--force-pause-ms", type=float, default=400.0,
                    help="when a draft must be forced, back off to the latest "
                         "pause at least this long (default 400 ms)")
    ap.add_argument("--force-min-chars", type=int, default=20,
                    help="text that must precede a forced pause cut (default 20)")
    ap.add_argument("--no-rule-punct", action="store_true",
                    help="leave the ASR text exactly as emitted: no sentence-end "
                         "full stop either")

    ap.add_argument("--enhance", choices=["none", "spectral", "wiener"], default="none")
    ap.add_argument("--enhance-block-ms", type=float, default=320.0)

    ap.add_argument("--link", action="store_true")
    ap.add_argument("--link-loss", type=float, default=0.0)
    ap.add_argument("--link-jitter-ms", type=float, default=0.0)
    ap.add_argument("--link-buffer-ms", type=float, default=120.0)
    ap.add_argument("--link-burst", default="")
    ap.add_argument("--link-seed", type=int, default=0)

    ap.add_argument("--font", help="CJK font path")
    ap.add_argument("--debug", action="store_true",
                    help="print ASR throughput and event lag every 2 s")
    ap.add_argument("--show-draft", action="store_true",
                    help="deprecated and ignored: the live words are now part of "
                         "the caption itself")
    ap.add_argument("--no-partial", action="store_true",
                    help="deprecated alias: drafts are off unless --show-draft")
    ap.add_argument("--vad-ratio", type=float, default=40.0,
                    help="speech needs energy >= ratio * noise floor (default 40)")
    ap.add_argument("--vad-hangover-ms", type=float, default=500.0,
                    help="VAD silence before its speech run resets; no longer a "
                         "sentence split (default 500)")
    ap.add_argument("--vad-zcr", type=float, default=0.4)
    ap.add_argument("--chunk-ms", type=float, default=160.0)
    ap.add_argument("--stability-ms", type=float, default=120.0)
    ap.add_argument("--max-seg-chars", type=int, default=0,
                    help="force a caption once a draft reaches this many characters "
                         "(default 0 = no character cap); length is then bounded by "
                         "--max-seg-seconds and sherpa's rule3")
    ap.add_argument("--max-seg-seconds", type=float, default=15.0,
                    help="force a caption once a draft has been growing this long "
                         "(default 15 s); longer keeps a sentence together")
    ap.add_argument("--roll-chars", type=int, default=ROLL_CHARS,
                    help="characters of the unfinished sentence kept visible in the "
                         "live line before the oldest words scroll off (default 240; "
                         "0 = keep all of it)")
    ap.add_argument("--sentence-gap-ms", type=float, default=800.0,
                    help="with --split-on-pauses: word gap that counts as a "
                         "sentence boundary (default 800 ms)")
    ap.add_argument("--sentence-min-chars", type=int, default=60,
                    help="with --split-on-pauses: shortest caption worth "
                         "cutting at a pause (default 60)")
    ap.add_argument("--split-on-pauses", action="store_true",
                    help="cut a committed draft at the pauses inside it, so one "
                         "caption may hold fewer words than the draft line showed; "
                         "off by default: one draft run = one caption")
    ap.add_argument("--min-hold-ms", type=float, default=MIN_HOLD_S * 1000.0,
                    help="a caption stays at least this long, even if short")
    ap.add_argument("--max-hold-ms", type=float, default=MAX_HOLD_S * 1000.0,
                    help="a caption never stays longer than this before the next "
                         "one takes over")
    ap.add_argument("--max-lag-ms", type=float, default=4000.0,
                    help="if the display falls further behind than this, skip "
                         "ahead instead of replaying every caption")
    ap.add_argument("--read-zh-cps", type=float, default=CJK_CPS,
                    help="Chinese reading speed, characters per second")
    ap.add_argument("--read-en-cps", type=float, default=LATIN_CPS,
                    help="Latin reading speed, characters per second")
    ap.add_argument("--headless", action="store_true", help="do not open windows")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many seconds of wall time (0 = whole video)")
    return ap


def rolling_text(settled: str, live: str, limit: int = 100) -> str:
    """One caption's text while it is still being spoken.

    'settled' is normally empty: each caption is its own segment and starts
    from a blank line, so a finished sentence never bleeds into the next one.
    'live' grows word by word; past 'limit' characters the oldest words drop
    off the front, at a word boundary.
    """
    settled = (settled or "").strip()
    live = (live or "").strip()
    text = f"{settled} {live}".strip() if settled else live
    if limit <= 0 or len(text) <= limit:
        return text
    tail = text[-limit:]
    sp = tail.find(" ")
    if 0 < sp < 15:
        tail = tail[sp + 1:]
    return tail


def wrap_text(draw, text: str, font, max_width: int):
    """Wrap by words when there are spaces, by characters for Chinese."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split(" ") if " " in text else list(text)
    sep = " " if " " in text else ""
    lines, current = [], ""
    for tok in tokens:
        trial = tok if not current else f"{current}{sep}{tok}"
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = tok
    if current:
        lines.append(current)
    return lines


class TranscriptPanel:
    """A scrolling transcript window that keeps every recognized sentence.

    This is a text box, not a one-line subtitle strip. Each finalized sentence
    is appended as a block -- timestamp, English text, and the Chinese
    translation underneath when translation is enabled -- and stays there for
    the rest of the session.

    Text never appears as a block: every character is revealed in turn, a few
    per frame, so words land one after another the way they are spoken. The
    sentence being spoken now is the green line at the bottom and is typed out
    as the recognizer emits it; a sentence that arrives in one piece (a backend
    that returns no partials) is typed out too, not pasted in.

    The view follows the newest line by default. Scrolling up pauses the
    follow: lines keep being recorded, but the view stays where the reader put
    it. Scrolling back to the bottom (or pressing End) resumes following.
    Mouse wheel, arrow keys, Page Up/Down, Home/End, and j/k/u/d/g/G scroll.
    """

    PAD = 12
    BLOCK_GAP = 14

    # Characters revealed per second. Speech runs at roughly 12-16 characters
    # per second, so the base rate keeps up with a normal voice; a burst that
    # arrives late is drained over CATCHUP_S instead of being pasted at once.
    BASE_CPS = 30.0
    CATCHUP_S = 1.2

    # Arrow keys report different codes on GTK, Qt and Windows builds, so keep
    # every common value; letters are a fallback that works everywhere.
    _UP = {82, 65362, 2490368, 63232, ord("k")}
    _DOWN = {84, 65364, 2621440, 63233, ord("j")}
    _PAGE_UP = {85, 65365, 2162688, 63276, ord("u")}
    _PAGE_DOWN = {86, 65366, 2228224, 63277, ord("d")}
    _HOME = {80, 65360, 2359296, 63273, ord("g")}
    _END = {81, 65367, 2293760, 63275, ord("G")}

    def __init__(self, width: int, height: int, font_path):
        self.font_path = font_path
        self.entries = []       # [{"t", "en", "zh", "shown", "zh_shown"}]
        self.draft = ""
        self.follow = True
        self.offset = 0.0
        self.width = max(80, int(width))
        self.height = max(60, int(height))
        self._runs = []         # (text, font, x, y, colour, stroke)
        self._total_h = 0.0
        self._dirty = True
        self._key = None
        self._now = 0.0
        self._last_frame = None
        self._draft_shown = 0.0
        self._make_fonts()
        self._image = np.full((self.height, self.width, 3), 12, np.uint8)

    # -- fonts and geometry ------------------------------------------------

    def _make_fonts(self) -> None:
        w = self.width
        # Large, outlined letters: the same look the single-caption subtitle
        # panel had, because that is what reads well on a video at a distance.
        zh_size = max(16, w // 40)
        en_size = max(18, w // 34)
        small_size = max(11, w // 78)
        self.zh_size, self.en_size, self.small_size = zh_size, en_size, small_size
        self.zh_lh = zh_size + 10
        self.en_lh = en_size + 10
        self._cjk = bool(self.font_path)
        if self._cjk:
            self._zh = ImageFont.truetype(self.font_path, zh_size)
            self._en = ImageFont.truetype(self.font_path, en_size)
            self._small = ImageFont.truetype(self.font_path, small_size)
        else:
            default = ImageFont.load_default()
            self._zh = self._en = self._small = default
            self.zh_lh = self.en_lh = 18
            self.small_size = 11

    def resize(self, width: int, height: int) -> None:
        width, height = max(80, int(width)), max(60, int(height))
        if (width, height) == (self.width, self.height):
            return
        at_bottom = self.follow
        self.width, self.height = width, height
        self._make_fonts()
        self._dirty = True
        self._key = None
        if at_bottom:
            self.follow = True

    # -- content -----------------------------------------------------------

    def add_event(self, t_start: float, text_en: str, text_zh: str = "",
                  now: float = 0.0, complete: bool = True) -> None:
        """Append one finalized sentence.

        'complete' means the viewer has already seen this text grow on the live
        line, so it can appear immediately; otherwise it is typed out from the
        first character, so a backend that returns no partials still shows the
        words one after another instead of pasting a block.
        """
        text_en = (text_en or "").strip()
        text_zh = (text_zh or "").strip()
        if not (text_en or text_zh):
            return
        self.entries.append({
            "t": float(t_start or 0.0),
            "en": text_en,
            "zh": text_zh,
            "shown": float(len(text_en)) if complete else 0.0,
            "zh_shown": float(len(text_zh)) if complete else 0.0,
            "_lines": None,
            "_lines_w": None,
        })
        self._dirty = True
        self._key = None

    def set_draft(self, text: str, now: float = 0.0) -> None:
        """The sentence being spoken right now.

        Growth keeps the reveal position: the new words are typed on from where
        the previous ones stopped. A revision that is not an extension keeps
        whatever is already visible; a different sentence starts over.
        """
        text = (text or "").strip()
        if text == self.draft:
            return
        if text.startswith(self.draft):
            pass                                    # same sentence, longer
        elif self.draft.startswith(text) or self.draft.endswith(text):
            self._draft_shown = min(self._draft_shown, float(len(text)))
        else:
            # A revision of the same sentence (a capital letter moving, a word
            # rewritten) keeps the reader's position as long as most of the
            # visible prefix is unchanged; only a different sentence restarts.
            common = 0
            for a, b in zip(text, self.draft):
                if a != b:
                    break
                common += 1
            if common >= max(4, 0.6 * self._draft_shown):
                self._draft_shown = min(self._draft_shown, float(common))
            else:
                self._draft_shown = 0.0
        self.draft = text
        self._dirty = True
        self._key = None

    # -- typewriter reveal -------------------------------------------------

    @classmethod
    def _reveal_step(cls, shown: float, target: int, dt: float) -> float:
        """How many characters are visible after 'dt' seconds."""
        if shown >= target:
            return float(target)
        speed = max(cls.BASE_CPS, (target - shown) / cls.CATCHUP_S)
        return min(float(target), shown + speed * dt)

    def _advance(self, now: float) -> bool:
        """Move every reveal forward by one frame. True when anything moved."""
        if self._last_frame is None:
            dt = 0.0
        else:
            # A resize or a stall must not dump a whole burst at once.
            dt = min(0.25, max(0.0, now - self._last_frame))
        self._last_frame = now

        moved = False
        before = int(self._draft_shown)
        self._draft_shown = self._reveal_step(self._draft_shown, len(self.draft), dt)
        moved = moved or int(self._draft_shown) != before

        for e in self.entries:
            if e["shown"] < len(e["en"]):
                before = int(e["shown"])
                e["shown"] = self._reveal_step(e["shown"], len(e["en"]), dt)
                moved = moved or int(e["shown"]) != before
            elif e["zh_shown"] < len(e["zh"]) and self._cjk:
                before = int(e["zh_shown"])
                e["zh_shown"] = self._reveal_step(
                    e["zh_shown"], len(e["zh"]), dt)
                moved = moved or int(e["zh_shown"]) != before

        # Once typing has caught up the caches are valid again; the entry keeps
        # its wrapped lines keyed by width (see _ensure_layout).
        return moved

    # -- scrolling ---------------------------------------------------------

    @staticmethod
    def _stamp_text(t: float) -> str:
        t = max(0, int(round(float(t))))
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _max_offset(self) -> float:
        self._ensure_layout()
        return max(0.0, self._total_h - self.height)

    def scroll_by(self, dy: float) -> None:
        max_off = self._max_offset()
        self.offset = min(max(0.0, self.offset + float(dy)), max_off)
        self.follow = self.offset >= max_off - 1.0
        self._key = None

    def go_top(self) -> None:
        self.offset = 0.0
        self.follow = False
        self._key = None

    def go_bottom(self) -> None:
        self.offset = self._max_offset()
        self.follow = True
        self._key = None

    def on_mouse(self, event, x, y, flags, _param=None) -> None:
        if event != cv2.EVENT_MOUSEWHEEL:
            return
        delta = flags
        try:
            delta = cv2.getMouseWheelDelta(flags)
        except (AttributeError, cv2.error):
            pass
        # One notch is 120; half a line-height per unit reads at a sane speed.
        self.scroll_by(-delta * self.en_lh / 240.0)

    def handle_key(self, key) -> None:
        if key is None or key < 0:
            return
        if key in self._UP:
            self.scroll_by(-2 * self.en_lh)
        elif key in self._DOWN:
            self.scroll_by(2 * self.en_lh)
        elif key in self._PAGE_UP:
            self.scroll_by(-(self.height - 2 * self.en_lh))
        elif key in self._PAGE_DOWN:
            self.scroll_by(self.height - 2 * self.en_lh)
        elif key in self._HOME:
            self.go_top()
        elif key in self._END:
            self.go_bottom()

    # -- layout and rendering ---------------------------------------------

    def _ensure_layout(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        scratch = Image.new("RGB", (8, 8))
        draw = ImageDraw.Draw(scratch)
        runs = []
        max_w = self.width - 2 * self.PAD - 12   # room for the scrollbar
        y = float(self.PAD)

        for entry in self.entries:
            stamp = self._stamp_text(entry["t"])
            stamp_w = draw.textlength(stamp + "  ", font=self._small)
            en_full = entry["shown"] >= len(entry["en"])
            zh_full = entry["zh_shown"] >= len(entry["zh"])
            if (en_full and zh_full and entry["_lines"] is not None
                    and entry["_lines_w"] == self.width):
                en_lines, zh_lines = entry["_lines"]
            else:
                # While typing, wrap the revealed prefix: the last word grows
                # one letter at a time and may reflow to the next line, which
                # is what makes it read as typing rather than as a paste.
                en_text = entry["en"][:int(entry["shown"])]
                en_lines = wrap_text(draw, en_text, self._en, max_w - stamp_w)
                zh_text = entry["zh"][:int(entry["zh_shown"])]
                zh_lines = (wrap_text(draw, zh_text, self._zh, max_w)
                            if zh_text and self._cjk else [])
                if en_full and zh_full:
                    entry["_lines"] = (en_lines, zh_lines)
                    entry["_lines_w"] = self.width
            runs.append((stamp, self._small, self.PAD, y, (125, 125, 125), 0))
            for i, line in enumerate(en_lines or [""]):
                runs.append((line, self._en, self.PAD + stamp_w,
                             y + i * self.en_lh, (245, 245, 245), 3))
            y += max(1, len(en_lines)) * self.en_lh
            if zh_lines:
                y += 3
                for line in zh_lines:
                    runs.append((line, self._zh, self.PAD + stamp_w, y,
                                 (150, 205, 255), 2))
                    y += self.zh_lh
            y += self.BLOCK_GAP

        if self.draft:
            # The sentence being spoken right now, revealed from its first
            # character at typing speed. It is the same text that will be
            # committed when the recognizer ends the sentence.
            marker = "▌ "
            mark_w = draw.textlength(marker, font=self._small)
            draft_text = self.draft[:int(self._draft_shown)]
            draft_lines = wrap_text(draw, draft_text, self._en, max_w - mark_w)
            runs.append((marker, self._small, self.PAD, y, (120, 210, 130), 0))
            for i, line in enumerate(draft_lines):
                runs.append((line, self._en, self.PAD + mark_w,
                             y + i * self.en_lh, (175, 225, 180), 3))
            y += max(1, len(draft_lines)) * self.en_lh + self.BLOCK_GAP
        elif not self.entries:
            hint = "等待识别结果 …" if self._cjk else "waiting for speech ..."
            runs.append((hint, self._zh, self.PAD, y, (110, 110, 110), 0))
            y += self.zh_lh + self.BLOCK_GAP

        self._runs = runs
        self._total_h = y + self.PAD
        self._clamp()

    def _clamp(self) -> None:
        max_off = max(0.0, self._total_h - self.height)
        if self.follow:
            self.offset = max_off
        else:
            self.offset = min(max(0.0, self.offset), max_off)
            if self.offset >= max_off - 1.0:
                self.follow = True

    def render(self, now: float = None):
        """One frame at audio-clock time 'now' (seconds); the reveal advances."""
        if now is None:
            now = self._now
        self._now = float(now)
        if self._advance(self._now):
            self._dirty = True

        self._ensure_layout()
        key = (round(self.offset), len(self.entries), self.draft,
               self.width, self.height, self.follow,
               int(self._draft_shown),
               tuple(int(e["shown"]) for e in self.entries))
        if key == self._key:
            return self._image
        self._key = key

        img = Image.new("RGB", (self.width, self.height), (12, 12, 12))
        draw = ImageDraw.Draw(img)
        off = self.offset
        for text, font, x, y, colour, stroke in self._runs:
            top = y - off
            if top < -self.zh_lh - 4 or top > self.height:
                continue
            draw.text((x, top), text, font=font, fill=colour,
                      stroke_width=stroke, stroke_fill=(0, 0, 0))

        self._draw_scrollbar(draw)
        if not self.follow:
            self._draw_paused_badge(draw)

        self._image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return self._image

    def _draw_scrollbar(self, draw) -> None:
        if self._total_h <= self.height + 1:
            return
        x = self.width - 6
        top, bottom = self.PAD, self.height - self.PAD
        draw.rectangle((x, top, x + 3, bottom), fill=(38, 38, 38))
        track = max(24.0, (bottom - top) * self.height / self._total_h)
        max_off = max(1.0, self._total_h - self.height)
        pos = (bottom - top - track) * (self.offset / max_off)
        draw.rectangle((x, top + pos, x + 3, top + pos + track),
                       fill=(135, 135, 135))

    def _draw_paused_badge(self, draw) -> None:
        label = ("↑ 翻看中 · End 恢复跟随" if self._cjk
                 else "PAUSED · End to resume")
        tw = draw.textlength(label, font=self._small)
        x0 = max(4.0, self.width - tw - 20)
        draw.rectangle((x0, 4, self.width - 6, self.small_size + 12),
                       fill=(34, 34, 34))
        draw.text((x0 + 6, 7), label, font=self._small, fill=(232, 194, 120))


class NullMT:
    """No translation: the English line is the subtitle."""

    def translate(self, text_en, context=()):
        return "", 1.0


def _local_mt_dir() -> Path:
    """models/opus-mt-en-zh next to the repository, independent of the cwd."""
    return Path(__file__).resolve().parent.parent / "models" / "opus-mt-en-zh"


def build_mt(args):
    choice = args.mt
    model = args.mt_model

    if choice == "auto":
        local = _local_mt_dir()
        if model or local.is_dir():
            choice = "opus"
            model = model or str(local)
        else:
            print("translation: off -- no local model at models/opus-mt-en-zh; "
                  "the Chinese line needs --mt opus --mt-model <dir>")
            return NullMT()

    if choice == "none":
        print("translation: off -- the transcript will be English-only")
        return NullMT()
    if choice == "identity":
        print("translation: identity -- no Chinese text will be produced")
        return IdentityMT()

    model = model or "Helsinki-NLP/opus-mt-en-zh"
    try:
        mt = OpusMT(
            model_name=model,
            device=args.mt_device,
            use_context=args.mt_context,
        )
    except Exception as exc:
        print(f"translation: off -- OpusMT could not load ({exc}). "
              "The transcript will be English-only.")
        return NullMT()

    device = getattr(mt, "_device", "?")
    print(f"translation: opus {model} on {device}; each finalized English "
          "line gets its Chinese translation underneath")
    if device == "cpu":
        print("  note: translating on CPU blocks the pipeline for the length of "
              "each sentence; use --mt-device cuda when torch has CUDA")
    return mt


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.semantic_cuts and not args.punct_model:
        raise SystemExit("--semantic-cuts needs --punct-model <dir or model.onnx>")
    if args.wav and not Path(args.wav).exists():
        raise SystemExit(
            f"--wav {args.wav} does not exist.\n"
            f"Either omit --wav (the audio track is extracted from {args.video} "
            "with ffmpeg), or create it once:\n"
            f"  ffmpeg -y -i {args.video} -vn -ac 1 -ar 16000 -c:a pcm_s16le {args.wav}"
        )
    if not args.wav:
        args.wav = extract_audio(args.video)
    cfg = BLineConfig(
        chunk_ms=args.chunk_ms,
        stability_delay_ms=args.stability_ms,
        vad_energy_ratio=args.vad_ratio,
        vad_hangover_ms=args.vad_hangover_ms,
        vad_zcr_max=args.vad_zcr,
        max_seg_chars=args.max_seg_chars,
        max_seg_seconds=args.max_seg_seconds,
        sentence_gap_ms=args.sentence_gap_ms,
        sentence_min_chars=args.sentence_min_chars,
        split_on_pauses=args.split_on_pauses,
        semantic_cuts=args.semantic_cuts,
        semantic_lookahead_words=args.semantic_lookahead,
        semantic_confirm=args.semantic_confirm,
        semantic_min_chars=args.semantic_min_chars,
        force_pause_ms=args.force_pause_ms,
        force_min_chars=args.force_min_chars,
        enable_endpoint_detection=not args.no_endpoint,
        rule1_min_trailing_silence=args.rule1_ms / 1000.0,
        rule2_min_trailing_silence=args.rule2_ms / 1000.0,
        rule3_min_utterance_length=args.rule3_sec,
    )

    # Models first: the source starts the wall clock when it is constructed.
    if args.asr == "scripted":
        entries = load_script(args.script) if args.script else []
        if not entries:
            raise SystemExit("--asr scripted needs --script for this demo")
        asr = ScriptedASR(entries, cfg)
    else:
        if not args.asr_model_dir:
            raise SystemExit("--asr sherpa requires --asr-model-dir")
        asr = SherpaASR(
            model_dir=args.asr_model_dir,
            cfg=cfg,
            num_threads=args.asr_threads,
            provider=args.asr_provider,
        )
    asr = TimedASR(asr, quiet=not args.debug)
    if args.asr == "sherpa":
        print(f"endpoint: {'on' if cfg.enable_endpoint_detection else 'off'} "
              f"(rule1 {cfg.rule1_min_trailing_silence * 1000:.0f} ms, "
              f"rule2 {cfg.rule2_min_trailing_silence * 1000:.0f} ms, "
              f"rule3 {cfg.rule3_min_utterance_length:.0f} s)")
    mt = build_mt(args)
    punct = None
    if args.punct_model:
        from b_line.punct import PunctRestorer
        punct = PunctRestorer(args.punct_model)
        cfg.punct_model = punct.describe()
    if args.semantic_cuts:
        print(f"semantic cuts: on (lookahead {cfg.semantic_lookahead_words} words, "
              f"confirm {cfg.semantic_confirm}, min {cfg.semantic_min_chars} chars)")
    # The recognizer emits no punctuation. With no punctuation model installed,
    # each committed sentence gets a full stop and nothing else -- commas
    # guessed from pauses read as wrong. Install --punct-model for real marks.
    cfg.rule_punctuation = punct is None and not args.no_rule_punct

    if args.enhance == "none":
        enhancer = BypassEnhancer()
    else:
        enhancer = SpectralSubtractionEnhancer(
            sample_rate=cfg.sample_rate, method=args.enhance
        )

    source = WavFrameSource(args.wav, cfg, realtime=True)
    if args.link:
        source = LinkSource(
            source,
            LinkConfig(
                jitter_ms=args.link_jitter_ms,
                loss_rate=args.link_loss,
                burst_windows=parse_loss(args.link_burst),
                jitter_buffer_ms=args.link_buffer_ms,
                seed=args.link_seed,
            ),
            cfg.sample_rate,
        )
    if args.enhance != "none":
        source = EnhancedFrameSource(
            source, enhancer, cfg.sample_rate, block_ms=args.enhance_block_ms
        )

    pipe = SubtitlePipeline(cfg, asr, mt, punct=punct)
    failure = []

    def run_pipeline():
        try:
            pipe.run(source)
        except BaseException as exc:  # surfaced in the main loop
            failure.append(exc)

    t0 = time.monotonic()
    worker = threading.Thread(target=run_pipeline, daemon=True)
    worker.start()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    video_dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
    audio_dur = float(getattr(source, "duration_sec", 0.0) or 0.0)
    if video_dur and audio_dur and abs(video_dur - audio_dur) > 1.0:
        print(f"warning: video is {video_dur:.1f}s but audio is {audio_dur:.1f}s")

    font_path = find_cjk_font(args.font)
    # A transcript needs room: about 60% of the video height, within reason.
    panel_h = max(240, min(460, int(height * 0.6)))
    panel = TranscriptPanel(width, panel_h, font_path)
    if not args.headless:
        try:
            # WINDOW_AUTOSIZE (the imshow default) locks the window to the image
            # size; WINDOW_NORMAL lets the user drag it freely.
            cv2.namedWindow("video", cv2.WINDOW_NORMAL)
            cv2.namedWindow("transcript", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("video", width, height)
            cv2.resizeWindow("transcript", width, panel_h)
            # Wheel events arrive per window, so the panel owns the callback.
            cv2.setMouseCallback("transcript", panel.on_mouse)
        except cv2.error:
            print("warning: this OpenCV build has no GUI backend; running headless")
            args.headless = True
    if not args.headless and font_path is None:
        print("warning: no CJK font found; the transcript window will show English only")
    print(f"video {width}x{height} @ {fps:.2f} fps, audio clock starts now")

    captions = CaptionQueue(
        min_hold_s=args.min_hold_ms / 1000.0,
        max_hold_s=args.max_hold_ms / 1000.0,
        cjk_cps=args.read_zh_cps,
        latin_cps=args.read_en_cps,
        max_lag_s=args.max_lag_ms / 1000.0,
    )
    frame_idx = 0
    next_frame = t0
    printed = 0
    image = panel.render(0.0)
    draft_seen_at = -1e9   # last audio time the live line had text
    interrupted = False
    display = not args.headless
    last_debug = -1.0

    while True:
        now = time.monotonic()
        if failure:
            print("\n[pipeline] the worker thread died -- no captions can be produced:")
            traceback.print_exception(
                type(failure[0]), failure[0], failure[0].__traceback__
            )
            break
        if args.max_seconds and now - t0 >= args.max_seconds:
            break
        if now < next_frame:
            time.sleep(min(next_frame - now, 0.02))
            if display:
                key_code = cv2.waitKey(1)
                if key_code in (27, ord("q")):
                    interrupted = True
                    break
                panel.handle_key(key_code)
            continue

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        next_frame = t0 + frame_idx / fps

        # If decoding fell behind, drop frames instead of accumulating delay.
        behind = (time.monotonic() - t0) * fps - frame_idx
        while behind > 1.5:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            next_frame = t0 + frame_idx / fps
            behind -= 1

        now_audio = time.monotonic() - t0
        events = pipe.events
        while printed < len(events):
            event = events[printed]
            printed += 1
            if not event.gap_flag and event.text_en:
                en_text = prettify_english(event.text_en)
                print(f"[{event.t_start:6.2f}-{event.t_end:6.2f}s] {en_text}")
                if event.text_zh:
                    print(f"          {event.text_zh}")
                # The transcript keeps every sentence for the whole session;
                # the queue below still drives the latency figures in --debug.
                # If the live line has been showing this sentence, the viewer
                # already read it word by word: show it at once. If the ASR
                # backend returns no partials there was nothing to read, so it
                # is typed out character by character instead of pasted.
                panel.add_event(event.t_start, en_text, event.text_zh,
                                now=now_audio,
                                complete=(now_audio - draft_seen_at) < 3.0)
                captions.push(event, now_audio)
        current, previous = captions.poll(now_audio)

        if args.debug and now_audio - last_debug >= 2.0:
            last_debug = now_audio
            if current is None:
                print(f"[dbg] t={now_audio:6.2f}s events={len(events)} "
                      f"vad_speech={pipe.vad.speech} seg_start={pipe.vad.seg_start} "
                      f"last=none ep={getattr(pipe.asr, 'endpoint_count', 0)} "
                      f"forced={getattr(pipe.asr, 'force_count', 0)} "
                      f"cuts={getattr(pipe, 'semantic_cut_count', 0)} "
                      f"pcuts={getattr(pipe, 'pause_cut_count', 0)}")
            else:
                print(f"[dbg] t={now_audio:6.2f}s events={len(events)} "
                      f"vad_speech={pipe.vad.speech} seg_start={pipe.vad.seg_start} "
                      f"last=[{current.t_start:6.2f},{current.t_end:6.2f}] "
                      f"lag={now_audio - current.t_end:+.2f}s "
                      f"hold={captions.remaining(now_audio):.2f}s "
                      f"queued={captions.pending_count} "
                      f"words={'yes' if getattr(pipe, 'words_available', False) else 'NO'} "
                      f"ep={getattr(pipe.asr, 'endpoint_count', 0)} "
                      f"forced={getattr(pipe.asr, 'force_count', 0)} "
                      f"cuts={getattr(pipe, 'semantic_cut_count', 0)} "
                      f"pcuts={getattr(pipe, 'pause_cut_count', 0)} "
                      f"draft={len(getattr(pipe, 'live_text', '') or '')}")

        if display:
            try:
                rect = cv2.getWindowImageRect("transcript")
            except cv2.error:
                rect = None
            if rect and rect[2] > 0 and rect[3] > 0:
                panel.resize(rect[2], rect[3])

            # The unfinished sentence is the green line at the bottom; it is
            # typed out character by character, so the words land one after
            # another instead of a block appearing at the end of the sentence.
            live = ""
            if not args.no_partial:
                live = (getattr(pipe, "live_text", "") or "").strip()
            if live:
                draft_seen_at = now_audio
            panel.set_draft(prettify_english(rolling_text("", live, args.roll_chars))
                            if live else "", now=now_audio)
            image = panel.render(now_audio)
            try:
                cv2.imshow("transcript", image)
                cv2.putText(frame, f"t={now_audio:6.2f}s", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2,
                            cv2.LINE_AA)
                cv2.imshow("video", frame)
                key_code = cv2.waitKey(1)
            except cv2.error:
                print("warning: this OpenCV build has no GUI backend; "
                      "continuing headless (install opencv-python, not -headless)")
                display = False
                continue
            if key_code in (27, ord("q")):
                interrupted = True
                break
            panel.handle_key(key_code)
            for name in ("video", "transcript"):
                try:
                    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                        interrupted = True
                        break
                except cv2.error:
                    interrupted = True
                    break
            if interrupted:
                break

    cap.release()
    if display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    worker.join(timeout=5.0)
    if failure:
        raise failure[0]
    print(f"\nstopped after {frame_idx / fps:.1f}s of video, "
          f"{len(pipe.events)} events "
          f"({sum(1 for e in pipe.events if e.gap_flag)} gaps)"
          + (", interrupted" if interrupted else ""))
    if args.asr == "sherpa":
        print(f"endpoints={getattr(pipe.asr, 'endpoint_count', 0)} "
              f"forced={getattr(pipe.asr, 'force_count', 0)} "
              f"streams={getattr(pipe.asr, 'stream_count', 0)} "
              f"semantic_cuts={getattr(pipe, 'semantic_cut_count', 0)} "
              f"pause_cuts={getattr(pipe, 'pause_cut_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
