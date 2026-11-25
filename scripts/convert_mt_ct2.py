#!/usr/bin/env python3
"""Convert a MarianMT / Opus checkpoint to CTranslate2 format.

Run this once, on a machine that has torch and network access for the first
install -- normally the Windows desktop. The output directory is portable:
copy it to the Jetson and only ``ctranslate2`` + ``transformers`` (tokenizer)
are needed there. No torch, no GPU build.

Examples:
  # int8, the deployment default
  python scripts/convert_mt_ct2.py --model models/opus-mt-en-zh \
      --out models/opus-mt-en-zh-ct2

  # keep full precision (bigger, slightly better BLEU)
  python scripts/convert_mt_ct2.py --model models/opus-mt-en-zh \
      --out models/opus-mt-en-zh-ct2-f32 --quantization float32

  # convert and immediately translate two probe sentences
  python scripts/convert_mt_ct2.py --model models/opus-mt-en-zh \
      --out models/opus-mt-en-zh-ct2 --check "Welcome to digital signal processing."

Notes:
  * int8 is ~4x smaller than the fp32 torch checkpoint and several times
    faster on the Orin Nano CPU. English-Chinese caption quality loss is
    small; compare a few sentences with --check before shipping.
  * The converter needs the tokenizer files in the output directory, or the
    runtime cannot turn text into tokens. They are copied automatically.
"""
from __future__ import annotations

import argparse
import inspect
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Files MarianTokenizer needs to reload from the converted directory.
TOKENIZER_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "source.spm",
    "target.spm",
]


def _dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="models/opus-mt-en-zh",
                    help="source Hugging Face directory (default: "
                         "models/opus-mt-en-zh)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <model>-ct2)")
    ap.add_argument("--quantization", default="int8",
                    choices=["int8", "int8_float32", "int8_float16",
                             "int8_bfloat16", "int16", "float16", "bfloat16",
                             "float32"],
                    help="weight quantization (default: int8)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out directory")
    ap.add_argument("--check", action="append", metavar="SENTENCE",
                    help="translate this sentence after converting "
                         "(repeatable)")
    args = ap.parse_args(argv)

    try:
        import ctranslate2
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        raise SystemExit(
            "ctranslate2 is not installed.\n"
            "  pip install ctranslate2 transformers sentencepiece torch"
        )

    source = Path(args.model)
    if not source.is_dir():
        raise SystemExit(
            f"no model directory at {source}. Download Opus MT first:\n"
            "  $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
            "  huggingface-cli download Helsinki-NLP/opus-mt-en-zh "
            "--local-dir models/opus-mt-en-zh"
        )
    out = Path(args.out or (str(source) + "-ct2"))
    if out.exists():
        # A directory without model.bin is the residue of a failed conversion;
        # remove it silently instead of demanding --force.
        partial = not (out / "model.bin").is_file()
        if not args.force and not partial:
            raise SystemExit(
                f"{out} already exists; pass --force to overwrite it"
            )
        shutil.rmtree(out)

    copy_files = [name for name in TOKENIZER_FILES if (source / name).is_file()]
    missing = [n for n in ("vocab.json", "source.spm", "target.spm")
               if n not in copy_files]
    if missing:
        raise SystemExit(
            f"{source} does not look like a Marian directory (missing "
            f"{', '.join(missing)})"
        )

    print(f"ctranslate2 {ctranslate2.__version__}")
    print(f"converting {source} -> {out}  quantization={args.quantization}")

    # copy_files moved between the constructor and convert() across
    # CTranslate2 releases (4.9 wants it on TransformersConverter.__init__).
    # Place it wherever this installed version expects it.
    ctor_kwargs = {}
    convert_kwargs = {"quantization": args.quantization, "force": True}
    try:
        ctor_params = inspect.signature(TransformersConverter).parameters
        convert_params = inspect.signature(TransformersConverter.convert).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic builds
        ctor_params = convert_params = {}
    if "copy_files" in ctor_params:
        ctor_kwargs["copy_files"] = copy_files
    elif "copy_files" in convert_params:
        convert_kwargs["copy_files"] = copy_files
    else:
        print("warning: this ctranslate2 cannot copy tokenizer files; copying "
              "them by hand after conversion")

    t0 = time.perf_counter()
    try:
        converter = TransformersConverter(str(source), **ctor_kwargs)
        converter.convert(str(out), **convert_kwargs)
    except BaseException:
        # A half-written directory would make the next run trip over "already
        # exists" and, worse, look like a usable model.
        shutil.rmtree(out, ignore_errors=True)
        raise
    if not ctor_kwargs and "copy_files" not in convert_kwargs:
        for name in copy_files:
            shutil.copy2(source / name, out / name)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s  "
          f"({_dir_size_mb(source):.0f} MB -> {_dir_size_mb(out):.0f} MB)")

    model_bin = out / "model.bin"
    if not model_bin.is_file():
        raise SystemExit(f"conversion finished but {model_bin} is missing")
    print(f"weights: {model_bin}")
    print("copy to the Jetson with, e.g.:")
    print(f"  scp -r {out} <user>@<jetson>:~/<project>/models/")

    if args.check:
        from b_line.mt import CT2MT

        print("\nprobe translations:")
        try:
            mt = CT2MT(
                model_dir=str(out),
                device="cpu",
                compute_type=args.quantization,
                beam_size=4,
            )
        except Exception as exc:
            raise SystemExit(f"converted model did not load: {exc}")
        for sentence in args.check:
            t = time.perf_counter()
            zh, conf = mt.translate(sentence)
            dt = time.perf_counter() - t
            print(f"  {dt * 1000:6.0f} ms  conf={conf:.3f}")
            print(f"    en: {sentence}")
            print(f"    zh: {zh or '<empty>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
