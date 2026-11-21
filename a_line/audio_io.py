"""WAV input/output and format normalisation for the A-line.

The rest of the project only accepts 16 kHz mono (or stereo) 16-bit PCM.
This module is the one place that reads arbitrary PCM WAV files and converts
them to that format, using a polyphase anti-aliasing resampler rather than
naive decimation. It deliberately does not depend on ffmpeg.

Supported input: PCM WAV with 8/16/24/32-bit samples, 1-2 channels (or more
when downmixing to mono). Float WAV is not supported by Python's `wave` module;
convert it first or use ffmpeg.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

TARGET_RATE = 16000


@dataclass
class WavInfo:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bits": self.sample_width * 8,
            "frames": self.frames,
            "duration_sec": round(self.duration_sec, 4),
        }


def probe_wav(path: str) -> WavInfo:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.getnframes()
    return WavInfo(
        path=str(path),
        sample_rate=rate,
        channels=channels,
        sample_width=width,
        frames=frames,
        duration_sec=frames / float(rate) if rate else 0.0,
    )


def read_wav(path: str) -> Tuple[np.ndarray, WavInfo]:
    """Read a PCM WAV into float32 in [-1, 1].

    Returns shape (frames,) for mono and (frames, channels) otherwise.
    """
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)

    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported PCM sample width {width * 8} bits")

    if channels > 1:
        x = x.reshape(-1, channels)
    info = WavInfo(
        path=str(path),
        sample_rate=rate,
        channels=channels,
        sample_width=width,
        frames=frames,
        duration_sec=frames / float(rate) if rate else 0.0,
    )
    return x.astype(np.float32, copy=False), info


def write_wav(path: str, samples: np.ndarray, sample_rate: int = TARGET_RATE) -> None:
    """Write float32 samples as 16-bit PCM WAV.

    1-D input is mono; 2-D input is (frames, channels).
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim not in (1, 2):
        raise ValueError(f"expected 1-D or 2-D samples, got shape {x.shape}")

    channels = 1 if x.ndim == 1 else x.shape[1]
    pcm = np.clip(np.rint(np.clip(x, -1.0, 1.0) * 32767.0), -32768, 32767)
    pcm = pcm.astype("<i2")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.reshape(-1).tobytes())


def _resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return x
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "resampling needs scipy. Install it with "
            "`pip install scipy` (it is in requirements-desktop.txt)."
        ) from exc

    g = gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g
    if x.ndim == 1:
        return resample_poly(x, up, down).astype(np.float32)
    # resample each channel independently, keeping (frames, channels)
    return np.stack(
        [resample_poly(x[:, c], up, down) for c in range(x.shape[1])],
        axis=1,
    ).astype(np.float32)


def convert_wav(
    src: str,
    dst: str,
    mono: bool = True,
    target_rate: int = TARGET_RATE,
    normalize: bool = False,
) -> dict:
    """Convert a PCM WAV to `target_rate`, 16-bit PCM, optionally mono.

    Returns a summary dictionary suitable for logging or JSON output.
    """
    if Path(src).resolve() == Path(dst).resolve():
        raise ValueError("source and destination are the same file")

    x, info = read_wav(src)
    channels = info.channels

    if mono and channels > 1:
        x = x.mean(axis=1)
        channels = 1
    elif not mono and channels not in (1, 2):
        raise ValueError(
            f"{src}: --keep-stereo only supports 1 or 2 channels, got {channels}"
        )

    y = _resample(x, info.sample_rate, int(target_rate))

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if normalize and peak > 0.0:
        y = y * (0.99 / peak)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
    clipped = int(np.sum(np.abs(y) > 1.0))

    write_wav(dst, y, target_rate)
    out_frames = int(y.shape[0])
    return {
        "source": info.to_dict(),
        "output": {
            "path": str(dst),
            "sample_rate": int(target_rate),
            "channels": int(channels),
            "sample_width_bits": 16,
            "frames": out_frames,
            "duration_sec": round(out_frames / float(target_rate), 4),
        },
        "resample_ratio": round(target_rate / float(info.sample_rate), 6),
        "peak_before_write": round(peak, 6),
        "clipped_samples": clipped,
        "normalized": bool(normalize),
    }


def is_target_format(path: str) -> bool:
    info = probe_wav(path)
    return (
        info.sample_rate == TARGET_RATE
        and info.channels in (1, 2)
        and info.sample_width == 2
    )


def default_output_name(src: str) -> str:
    p = Path(src)
    return str(p.with_name(f"{p.stem}_16k.wav"))


def _cli(argv=None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        description="Convert PCM WAV files to 16 kHz mono 16-bit PCM."
    )
    ap.add_argument("input", help="input .wav file or a directory")
    ap.add_argument("-o", "--output", help="output file (single-file mode)")
    ap.add_argument("--out-dir", help="output directory (batch mode)")
    ap.add_argument("--keep-stereo", action="store_true",
                    help="keep 2 channels instead of downmixing to mono")
    ap.add_argument("--normalize", action="store_true",
                    help="scale the peak to 0.99 before writing")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into subdirectories in batch mode")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"error: no such file or directory: {src}", file=sys.stderr)
        return 2

    jobs = []
    if src.is_dir():
        out_dir = Path(args.out_dir) if args.out_dir else src / "16k"
        pattern = "**/*.wav" if args.recursive else "*.wav"
        files = sorted(p for p in src.glob(pattern) if p.is_file())
        # Never re-convert files produced by a previous run into the same tree.
        resolved_out = out_dir.resolve()
        files = [p for p in files if resolved_out not in p.resolve().parents]
        if not files:
            print(f"error: no .wav files under {src}", file=sys.stderr)
            return 2
        for p in files:
            rel = p.relative_to(src)
            jobs.append((p, out_dir / rel.parent / f"{p.stem}_16k.wav"))
    else:
        if args.output:
            out = Path(args.output)
        elif args.out_dir:
            out = Path(args.out_dir) / f"{src.stem}_16k.wav"
        else:
            out = Path(default_output_name(str(src)))
        jobs.append((src, out))

    failures = 0
    for in_path, out_path in jobs:
        try:
            report = convert_wav(
                str(in_path),
                str(out_path),
                mono=not args.keep_stereo,
                normalize=args.normalize,
            )
            if not args.quiet:
                print(json.dumps(report, ensure_ascii=False))
        except Exception as exc:
            failures += 1
            print(f"error: {in_path}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(f"converted {len(jobs) - failures}/{len(jobs)} file(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
