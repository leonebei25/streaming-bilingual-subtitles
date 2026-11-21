"""Punctuation model family detection, without sherpa-onnx installed.

A fake sherpa_onnx records which model family was constructed and returns a
tagged string, so the tests can check directory detection, int8 preference and
the fallback path.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

from b_line.punct import PunctRestorer


def _install_fake(fail: bool = False):
    module = types.ModuleType("sherpa_onnx")
    module.calls = []
    module.inputs = []

    class OfflinePunctuationModelConfig:
        def __init__(self, **kw):
            self.kw = kw

    class OfflinePunctuationConfig:
        # pybind takes the model config positionally, never as a keyword.
        def __init__(self, model_config, /):
            self.model = model_config

    class OfflinePunctuation:
        def __init__(self, config):
            self.config = config
            module.calls.append(("offline", config))

        def add_punctuation(self, text):
            if fail:
                raise RuntimeError("boom")
            return "OFF:" + text + "."

    class OnlinePunctuationModelConfig:
        def __init__(self, **kw):
            self.kw = kw

    class OnlinePunctuationConfig:
        def __init__(self, model_config, /):
            self.model = model_config

    class OnlinePunctuation:
        def __init__(self, config):
            self.config = config
            module.calls.append(("online", config))

        # The C++ API exposes only this name; the wrapper must accept it.
        def add_punctuation_with_case(self, text):
            module.inputs.append(text)
            if fail:
                raise RuntimeError("boom")
            return "ON:" + text + "."

    module.OfflinePunctuationModelConfig = OfflinePunctuationModelConfig
    module.OfflinePunctuationConfig = OfflinePunctuationConfig
    module.OfflinePunctuation = OfflinePunctuation
    module.OnlinePunctuationModelConfig = OnlinePunctuationModelConfig
    module.OnlinePunctuationConfig = OnlinePunctuationConfig
    module.OnlinePunctuation = OnlinePunctuation
    sys.modules["sherpa_onnx"] = module
    return module


class TestPunctRestorer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._saved = sys.modules.get("sherpa_onnx")
        self.addCleanup(self._restore_module)

    def _restore_module(self):
        if self._saved is None:
            sys.modules.pop("sherpa_onnx", None)
        else:
            sys.modules["sherpa_onnx"] = self._saved

    def _online_dir(self) -> Path:
        d = self.root / "online"
        d.mkdir()
        (d / "model.int8.onnx").write_bytes(b"x")
        (d / "model.onnx").write_bytes(b"x")
        (d / "bpe.vocab").write_text("a\n", encoding="utf-8")
        return d

    def _offline_dir(self) -> Path:
        d = self.root / "offline"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"x")
        return d

    def test_bpe_vocab_selects_the_online_model(self):
        module = _install_fake()
        d = self._online_dir()
        restorer = PunctRestorer(d)
        self.assertEqual(restorer.kind, "online")
        self.assertEqual(Path(restorer.model_path).name, "model.onnx",
                         "the full-precision model must beat model.int8.onnx")
        self.assertEqual(restorer.bpe_vocab, str(d / "bpe.vocab"))
        kind, config = module.calls[-1]
        self.assertEqual(kind, "online")
        self.assertEqual(config.model.kw["cnn_bilstm"], str(d / "model.onnx"))
        self.assertEqual(config.model.kw["bpe_vocab"], str(d / "bpe.vocab"))
        self.assertEqual(restorer.restore("HELLO"), "ON:hello.",
                         "the online BPE vocabulary is lower-case only")
        self.assertEqual(module.inputs[-1], "hello")

    def test_plain_onnx_dir_is_offline(self):
        module = _install_fake()
        d = self._offline_dir()
        restorer = PunctRestorer(d)
        self.assertEqual(restorer.kind, "offline")
        kind, config = module.calls[-1]
        self.assertEqual(kind, "offline")
        self.assertEqual(config.model.kw["ct_transformer"], str(d / "model.onnx"))
        self.assertEqual(restorer.restore("HELLO"), "OFF:HELLO.")

    def test_single_file_is_offline(self):
        _install_fake()
        model = self._offline_dir() / "model.onnx"
        restorer = PunctRestorer(model)
        self.assertEqual(restorer.kind, "offline")
        self.assertEqual(restorer.model_path, str(model))

    def test_online_mode_requires_bpe_vocab(self):
        _install_fake()
        d = self._offline_dir()
        with self.assertRaises(RuntimeError) as ctx:
            PunctRestorer(d, mode="online")
        self.assertIn("bpe.vocab", str(ctx.exception))

    def test_missing_path_names_both_models(self):
        _install_fake()
        with self.assertRaises(RuntimeError) as ctx:
            PunctRestorer(self.root / "nope")
        message = str(ctx.exception)
        self.assertIn("online-punct-en", message)
        self.assertIn("punct-ct-transformer", message)

    def test_failed_inference_falls_back_to_the_input(self):
        _install_fake(fail=True)
        restorer = PunctRestorer(self._offline_dir())
        self.assertEqual(restorer.restore("HELLO"), "HELLO")

    def test_apply_propagates_errors_for_diagnostics(self):
        _install_fake(fail=True)
        restorer = PunctRestorer(self._online_dir())
        with self.assertRaises(RuntimeError):
            restorer.apply("HELLO")
        self.assertEqual(restorer.restore("HELLO"), "HELLO",
                         "restore() must still degrade to the raw text")

    def test_dropped_spaces_are_repaired(self):
        _install_fake()
        restorer = PunctRestorer(self._online_dir())
        restorer._apply = lambda text: "so now wefilter before sampling."
        self.assertEqual(restorer.apply("so now we filter before sampling"),
                         "so now we filter before sampling.")

    def test_merged_word_and_punctuation_is_split(self):
        _install_fake()
        restorer = PunctRestorer(self._online_dir())
        restorer._apply = lambda text: "welcome back,everyone to the course."
        self.assertEqual(restorer.apply("welcome back everyone to the course"),
                         "welcome back, everyone to the course.")

    def test_equal_word_counts_are_left_alone(self):
        _install_fake()
        restorer = PunctRestorer(self._online_dir())
        restorer._apply = lambda text: "So now, we sample."
        self.assertEqual(restorer.apply("so now we sample"), "So now, we sample.")

    def test_describe_carries_the_family(self):
        _install_fake()
        restorer = PunctRestorer(self._online_dir())
        self.assertEqual(restorer.describe(), "online:model.onnx")


if __name__ == "__main__":
    unittest.main()
