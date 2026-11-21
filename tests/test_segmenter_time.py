import unittest

from b_line.config import BLineConfig
from b_line.segmenter import StablePrefixSegmenter
from b_line.types import ASRUpdate


def upd(text, stable_chars, conf=1.0):
    return ASRUpdate(text=text, stable_chars=stable_chars, confidence=conf)


class TestAsrDrivenSegmentation(unittest.TestCase):
    def test_finalized_text_commits_whole_sentence(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        text = "hello this is six minute english"
        seg.update(upd(text, len(text)), 3.0)
        out = seg.pop()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text_en, text)
        self.assertAlmostEqual(out[0].t_start, 0.0)
        self.assertAlmostEqual(out[0].t_end, 3.0)

    def test_partial_is_draft_not_committed(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        # sentence A finalized, sentence B only a partial
        text = "sentence a is done sentence b is"
        seg.update(upd(text, len("sentence a is done")), 2.0)
        out = seg.pop()
        self.assertEqual(len(out), 1, "only the finalized sentence may commit")
        self.assertEqual(out[0].text_en, "sentence a is done")
        self.assertEqual(seg.partial, "sentence b is")

    def test_two_finalized_sentences_commit_in_order(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        first = "first sentence"
        seg.update(upd(first, len(first)), 1.0)
        out = seg.pop()
        seg.note_speech(True, 1.5, 1.5)
        second = "second sentence"
        both = f"{first} {second}"
        seg.update(upd(both, len(both)), 3.0)
        out.extend(seg.pop())
        self.assertEqual([s.text_en for s in out], [first, second])
        # t_start comes from each speech onset
        self.assertAlmostEqual(out[0].t_start, 0.0)
        self.assertAlmostEqual(out[1].t_start, 1.5)

    def test_flush_commits_remaining_draft(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        seg.update(upd("a whole sentence and a tail that is not", len("a whole sentence")), 2.0)
        seg.pop()
        seg.flush(4.0)
        out = seg.pop()
        self.assertEqual(out[0].text_en, "and a tail that is not")
        self.assertAlmostEqual(out[0].t_end, 4.0)

    def test_partial_rewrite_is_harmless(self):
        # A transducer rewrites its tail; the segmenter must not commit the
        # rewritten part, only the finalized prefix.
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        finalized = "the cat sat on"
        seg.update(upd(f"{finalized} the", len(finalized)), 1.0)
        seg.pop()
        seg.update(upd(f"{finalized} the mat", len(finalized)), 1.5)  # tail rewritten
        self.assertEqual(seg.partial, "the mat")
        self.assertEqual(seg.pop(), [], "a rewrite must not commit anything new")


if __name__ == "__main__":
    unittest.main()
