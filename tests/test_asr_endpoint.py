"""Sherpa endpoint wiring, without needing sherpa-onnx installed.

A fake recognizer records how it was constructed and mimics the streaming
contract: text accumulates on the current stream, and `is_endpoint` is a flag
the test flips. That is enough to check the two things that used to be wrong:
endpoint detection was never enabled, and each utterance must start on a fresh
stream (reset() keeps context tokens without their timestamps).
"""
import sys
import types
import unittest

import numpy as np

from b_line.asr import SherpaASR
from b_line.config import BLineConfig


class _FakeStream:
    def __init__(self):
        self.text = ""
        self.endpoint = False

    def accept_waveform(self, sample_rate, samples):
        pass


class _FakeResult:
    def __init__(self, text, tokens, timestamps):
        self.text = text
        self.tokens = tokens
        self.timestamps = timestamps


class _FakeRecognizer:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.streams = []
        self.reset_calls = 0
        _FakeRecognizer.last = self

    def create_stream(self):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return stream.text

    def get_result_all(self, stream):
        words = stream.text.split() if stream.text else []
        return _FakeResult(
            stream.text,
            ["▁" + w for w in words],
            [0.2 * i for i in range(len(words))],
        )

    def is_endpoint(self, stream):
        return stream.endpoint

    def reset(self, stream):
        self.reset_calls += 1


def _install_fake_sherpa():
    module = types.ModuleType("sherpa_onnx")

    class OnlineRecognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            return _FakeRecognizer(**kwargs)

    module.OnlineRecognizer = OnlineRecognizer
    sys.modules["sherpa_onnx"] = module


def _audio():
    return np.zeros(160, dtype=np.float32)


class TestEndpointWiring(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("sherpa_onnx")
        _install_fake_sherpa()
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("sherpa_onnx", None)
        else:
            sys.modules["sherpa_onnx"] = self._saved

    def _make(self, **cfg_kw):
        cfg = BLineConfig(**cfg_kw)
        asr = SherpaASR(tokens="tokens.txt", encoder="e.onnx",
                        decoder="d.onnx", joiner="j.onnx", cfg=cfg)
        self.addCleanup(lambda: sys.modules.pop("sherpa_onnx", None))
        return asr, _FakeRecognizer.last

    def test_endpoint_is_on_by_default(self):
        _, rec = self._make()
        self.assertTrue(rec.kwargs["enable_endpoint_detection"])
        self.assertAlmostEqual(rec.kwargs["rule1_min_trailing_silence"], 2.4)
        self.assertAlmostEqual(rec.kwargs["rule2_min_trailing_silence"], 1.2)
        self.assertAlmostEqual(rec.kwargs["rule3_min_utterance_length"], 20.0)

    def test_endpoint_can_be_disabled(self):
        _, rec = self._make(enable_endpoint_detection=False)
        self.assertFalse(rec.kwargs["enable_endpoint_detection"])

    def test_rules_come_from_config(self):
        _, rec = self._make(rule2_min_trailing_silence=0.9,
                            rule3_min_utterance_length=12.0)
        self.assertAlmostEqual(rec.kwargs["rule2_min_trailing_silence"], 0.9)
        self.assertAlmostEqual(rec.kwargs["rule3_min_utterance_length"], 12.0)

    def test_endpoint_commits_and_opens_a_new_stream(self):
        asr, rec = self._make()
        stream = rec.streams[-1]
        stream.text = "HELLO WORLD"
        update = asr.stream(_audio(), 0.0, 1.0)
        self.assertEqual(update.stable_chars, 0)
        self.assertEqual(update.text, "HELLO WORLD")

        stream.endpoint = True
        update = asr.stream(_audio(), 1.0, 1.16)
        self.assertEqual(update.text, "HELLO WORLD")
        self.assertEqual(update.stable_chars, len("HELLO WORLD"))
        self.assertEqual(update.words,
                         [("HELLO", 0.0), ("WORLD", 0.2)])
        self.assertEqual(asr.endpoint_count, 1)
        self.assertEqual(len(rec.streams), 2, "endpoint must start a new stream")
        self.assertEqual(rec.reset_calls, 0,
                         "reset() would echo the context tokens as text")

        # The next utterance starts clean: no repeated "WORLD" prefix.
        rec.streams[-1].text = "NEXT ONE"
        update = asr.stream(_audio(), 1.16, 2.0)
        self.assertEqual(update.text, "HELLO WORLD NEXT ONE")
        self.assertEqual(update.stable_chars, len("HELLO WORLD"))

    def test_no_endpoint_means_no_commit(self):
        asr, rec = self._make(enable_endpoint_detection=False)
        rec.streams[-1].text = "HELLO"
        rec.streams[-1].endpoint = True     # must be ignored
        update = asr.stream(_audio(), 0.0, 1.0)
        self.assertEqual(update.stable_chars, 0)
        self.assertEqual(asr.endpoint_count, 0)
        self.assertEqual(len(rec.streams), 1)

    def test_force_finalize_also_opens_a_new_stream(self):
        asr, rec = self._make()
        rec.streams[-1].text = "HELLO"
        asr.stream(_audio(), 0.0, 1.0)
        update = asr.force_finalize()
        self.assertEqual(update.text, "HELLO")
        self.assertEqual(update.stable_chars, len("HELLO"))
        self.assertEqual(asr.force_count, 1)
        self.assertEqual(len(rec.streams), 2)
        self.assertEqual(rec.reset_calls, 0)

    def test_word_times_are_absolute_across_streams(self):
        asr, rec = self._make()
        rec.streams[-1].text = "HELLO WORLD"
        update = asr.stream(_audio(), 10.0, 10.16)
        self.assertEqual(update.words, [("HELLO", 10.0), ("WORLD", 10.2)])

        rec.streams[-1].endpoint = True
        asr.stream(_audio(), 10.16, 10.32)          # endpoint; folds current
        rec.streams[-1].text = "NEXT ONE"
        update = asr.stream(_audio(), 10.32, 10.48)  # new stream, new origin
        self.assertEqual(update.words,
                         [("HELLO", 10.0), ("WORLD", 10.2),
                          ("NEXT", 10.32), ("ONE", 10.52)])

    def test_reset_clears_counters(self):
        asr, rec = self._make()
        rec.streams[-1].text = "HELLO"
        asr.stream(_audio(), 0.0, 1.0)
        asr.force_finalize()
        self.assertEqual(asr.force_count, 1)
        streams_before = len(rec.streams)
        asr.reset()
        self.assertEqual(asr.force_count, 0)
        self.assertEqual(asr.endpoint_count, 0)
        self.assertEqual(asr.stream_count, 1)
        self.assertEqual(len(rec.streams), streams_before + 1)


if __name__ == "__main__":
    unittest.main()
