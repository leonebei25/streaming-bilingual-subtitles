"""CTranslate2 MT backend without ctranslate2 or transformers installed.

The backend is a thin adapter: tokenize with the Marian tokenizer, hand the
tokens to CTranslate2, decode the hypothesis. The tests pin the three things
that actually break in deployment files:

* the options (device, compute type, beam, threads) reach the Translator;
* tokenization and decoding use the tokenizer that ships with the model;
* a missing converted model or missing package produces an actionable message
  instead of a bare traceback.

A fake Translator records its constructor arguments and returns a fixed
hypothesis, so none of this needs the real model or a GPU.
"""
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from b_line.mt import CT2MT


class _FakeTranslationResult:
    def __init__(self, hypotheses, scores):
        self.hypotheses = hypotheses
        self.scores = scores


class _FakeTranslator:
    last = None

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls = []
        _FakeTranslator.last = self

    def translate_batch(self, batch, **kwargs):
        self.calls.append({"batch": [list(row) for row in batch], **kwargs})
        return [_FakeTranslationResult([["▁你好", "，", "世界", "</s>"]], [-0.5])
                for _ in batch]


class _FakeTokenizer:
    last = None

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        inst = cls()
        inst.name = name
        inst.kwargs = kwargs
        _FakeTokenizer.last = inst
        return inst

    def encode(self, text):
        # One opaque "token" per input string keeps assertions readable and
        # still exercises the id -> token -> id round trip.
        return [text]

    def convert_ids_to_tokens(self, ids):
        return list(ids)

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def decode(self, tokens, skip_special_tokens=True):
        out = "".join(str(t) for t in tokens if t != "</s>")
        return out.replace("▁", "").strip()


def _fake_modules(translator_cls=None, tokenizer_cls=None):
    ct2 = types.ModuleType("ctranslate2")
    ct2.__version__ = "9.9.9"
    ct2.Translator = translator_cls or _FakeTranslator
    tf = types.ModuleType("transformers")
    tf.MarianTokenizer = tokenizer_cls or _FakeTokenizer
    return {"ctranslate2": ct2, "transformers": tf}


class CT2MTTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmp.name)
        (self.model_dir / "model.bin").write_bytes(b"not really a model")

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, **kwargs):
        with mock.patch.dict(sys.modules, _fake_modules()):
            mt = CT2MT(model_dir=str(self.model_dir), **kwargs)
        return mt

    def test_options_reach_the_translator(self):
        mt = self._build(device="cpu", compute_type="int8", beam_size=4,
                         intra_threads=6, inter_threads=1)
        tr = _FakeTranslator.last
        self.assertEqual(tr.model_path, str(self.model_dir))
        self.assertEqual(tr.kwargs["device"], "cpu")
        self.assertEqual(tr.kwargs["compute_type"], "int8")
        self.assertEqual(tr.kwargs["intra_threads"], 6)
        self.assertEqual(mt.beam_size, 4)
        self.assertEqual(mt.compute_type, "int8")

    def test_tokenizer_comes_from_the_model_directory(self):
        self._build()
        tok = _FakeTokenizer.last
        self.assertEqual(tok.name, str(self.model_dir))
        self.assertTrue(tok.kwargs.get("local_files_only"))

    def test_translate_decodes_and_reports_confidence(self):
        mt = self._build()
        zh, conf = mt.translate("Hello world.")
        self.assertEqual(zh, "你好，世界")
        self.assertAlmostEqual(conf, math.exp(-0.5), places=6)
        call = _FakeTranslator.last.calls[0]
        self.assertEqual(call["batch"], [["Hello world."]])
        self.assertEqual(call["beam_size"], 4)
        self.assertEqual(call["max_decoding_length"], 256)

    def test_empty_text_does_not_call_the_translator(self):
        mt = self._build()
        self.assertEqual(mt.translate("   "), ("", 1.0))
        self.assertEqual(_FakeTranslator.last.calls, [])

    def test_context_prepends_the_previous_sentence(self):
        mt = self._build(use_context=True)
        mt.translate("second sentence", context=("first sentence.",))
        batch = _FakeTranslator.last.calls[0]["batch"]
        self.assertEqual(batch, [["first sentence. second sentence"]])

    def test_context_ignored_when_disabled(self):
        mt = self._build(use_context=False)
        mt.translate("second sentence", context=("first sentence.",))
        batch = _FakeTranslator.last.calls[0]["batch"]
        self.assertEqual(batch, [["second sentence"]])

    def test_beam_size_zero_is_clamped(self):
        mt = self._build(beam_size=0)
        self.assertEqual(mt.beam_size, 1)
        mt.translate("x")
        self.assertEqual(_FakeTranslator.last.calls[0]["beam_size"], 1)

    def test_missing_model_directory_names_the_converter(self):
        with mock.patch.dict(sys.modules, _fake_modules()):
            with self.assertRaises(RuntimeError) as ctx:
                CT2MT(model_dir=str(self.model_dir / "nope"))
        self.assertIn("convert_mt_ct2.py", str(ctx.exception))

    def test_missing_ctranslate2_names_the_install_command(self):
        modules = {"ctranslate2": None, "transformers": None}
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as ctx:
                CT2MT(model_dir=str(self.model_dir))
        message = str(ctx.exception)
        self.assertIn("pip install ctranslate2", message)
        self.assertIn("torch is not required", message)


if __name__ == "__main__":
    unittest.main()
