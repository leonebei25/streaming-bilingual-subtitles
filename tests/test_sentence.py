import unittest

from b_line.config import BLineConfig
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline
from b_line.asr import ScriptedASR
from b_line.sentence import split_sentences
from b_line.types import Segment


class TestSplitSentences(unittest.TestCase):
    def test_splits_on_punctuation(self):
        self.assertEqual(
            split_sentences("Hello world. How are you? Fine!"),
            ["Hello world.", "How are you?", "Fine!"],
        )

    def test_keeps_closing_quote_with_sentence(self):
        parts = split_sentences('He said "go home." Then he left.')
        self.assertEqual(parts[0], 'He said "go home."')

    def test_marker_split_without_punctuation(self):
        text = ("the fourier transform decomposes a signal into sinusoids so the "
                "sampling theorem tells us how fast we must sample")
        parts = split_sentences(text, min_chars=18, max_chars=200)
        self.assertGreater(len(parts), 1)
        self.assertTrue(any(p.lower().startswith("so ") for p in parts))

    def test_short_fragment_is_merged(self):
        # the trailing marker fragment has no punctuation, so it is glued back
        parts = split_sentences("the cat sat on the mat and then it stayed",
                                min_chars=25, max_chars=200)
        self.assertEqual(len(parts), 1)

    def test_short_punctuated_sentence_is_kept(self):
        parts = split_sentences("Yes. This is a much longer sentence that stays.",
                                min_chars=18)
        self.assertEqual(parts, ["Yes.", "This is a much longer sentence that stays."])

    def test_cap_splits_long_text(self):
        parts = split_sentences("word " * 60, min_chars=18, max_chars=96)
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLessEqual(len(p), 96)

    def test_plain_sentence_is_untouched(self):
        text = "the cat sat on the mat"
        self.assertEqual(split_sentences(text, min_chars=18, max_chars=96), [text])

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


class TestPipelineSentenceEvents(unittest.TestCase):
    def test_one_slice_becomes_one_event_per_sentence(self):
        cfg = BLineConfig()
        pipe = SubtitlePipeline(cfg, ScriptedASR([], cfg), IdentityMT())
        pipe._emit(Segment(0, 0.0, 4.0, "First sentence. Second sentence.", 1.0))

        self.assertEqual(len(pipe.events), 2)
        first, second = pipe.events
        self.assertEqual(first.text_en, "First sentence.")
        self.assertEqual(second.text_en, "Second sentence.")
        self.assertAlmostEqual(first.t_start, 0.0)
        self.assertAlmostEqual(first.t_end, second.t_start)
        self.assertAlmostEqual(second.t_end, 4.0)
        self.assertNotEqual(first.seg_id, second.seg_id, "display keys must differ")


if __name__ == "__main__":
    unittest.main()
