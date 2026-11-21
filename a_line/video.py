"""Burn subtitle events into a video.

The renderer places each subtitle by its timestamp: an event is shown while
`t_start <= t < t_end`. Gap events are skipped, so a packet loss leaves an
empty stretch rather than shifting later subtitles. Chinese text needs a CJK
font; if none is found the renderer falls back to English-only OpenCV text and
says so in the returned stats.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONT_CANDIDATES = [
    os.environ.get("SUBTITLE_FONT", ""),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def find_cjk_font(preferred: Optional[str] = None) -> Optional[str]:
    for candidate in ([preferred] if preferred else []) + _CJK_FONT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _as_dict(event: Any) -> Dict[str, Any]:
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return dict(event)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
          max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _active_events(events: Iterable[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
    active = [
        e for e in events
        if not e.get("gap_flag") and e.get("t_start", 0.0) <= t < e.get("t_end", 0.0)
    ]
    active.sort(key=lambda e: e.get("t_start", 0.0))
    return active


def render_subtitles(
    video_in: str,
    events: Iterable[Any],
    video_out: str,
    font_path: Optional[str] = None,
    font_size: int = 30,
    position: float = 0.78,
) -> Dict[str, Any]:
    """Render events onto a video and return simple frame statistics."""
    events = [_as_dict(e) for e in events]
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video writer: {video_out}")

    resolved_font = find_cjk_font(font_path)
    font = ImageFont.truetype(resolved_font, font_size) if resolved_font else None
    max_text_width = int(width * 0.92)

    frames_in = frames_out = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_in += 1
        t = idx / fps
        active = _active_events(events, t)

        if active:
            current = active[-1]
            en = (current.get("text_en") or "").strip()
            zh = (current.get("text_zh") or "").strip()
            if font is not None and (en or zh):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).convert("RGBA")
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                lines: List[tuple] = []
                for text, color in ((en, (255, 255, 255, 255)),
                                    (zh, (255, 235, 120, 255))):
                    if text:
                        for line in _wrap(draw, text, font, max_text_width):
                            lines.append((line, color))
                if lines:
                    line_h = font_size + 8
                    block_h = line_h * len(lines) + 16
                    top = int(height * position) - block_h // 2
                    top = max(8, min(top, height - block_h - 8))
                    draw.rectangle(
                        [(0, top - 8), (width, top + block_h)],
                        fill=(0, 0, 0, 150),
                    )
                    y = top
                    for line, color in lines:
                        draw.text(
                            (12, y), line, font=font, fill=color,
                            stroke_width=2, stroke_fill=(0, 0, 0, 255),
                        )
                        y += line_h
                frame = cv2.cvtColor(
                    np.array(Image.alpha_composite(image, overlay).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
            elif en:
                # No CJK font: keep the English line so the demo still runs.
                cv2.putText(frame, en, (max(10, width // 40),
                                        int(height * position)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)

        writer.write(frame)
        frames_out += 1
        idx += 1

    cap.release()
    writer.release()
    return {
        "frames_in": frames_in,
        "frames_out": frames_out,
        "frames_expected": total,
        "dropped": max(0, total - frames_out) if total else 0,
        "fps": fps,
        "font": resolved_font,
        "cjk_supported": resolved_font is not None,
    }


def make_test_video(
    path: str,
    seconds: float = 10.0,
    fps: float = 10.0,
    size: tuple = (640, 360),
) -> str:
    """Create a simple timestamped test video for the desktop smoke run."""
    width, height = size
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    n = int(round(seconds * fps))
    for i in range(n):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(20, 90, width, dtype=np.uint8)[None, :]
        frame[:, :, 1] = np.linspace(10, 60, height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = 30
        cv2.putText(frame, f"test video  t={i / fps:5.1f}s", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2,
                    cv2.LINE_AA)
        cv2.line(frame, (0, int(height * 0.78)), (width, int(height * 0.78)),
                 (120, 120, 120), 1)
        writer.write(frame)
    writer.release()
    return str(out_path)
