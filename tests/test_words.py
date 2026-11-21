import unittest

from b_line.asr import _tokens_to_words
from b_line.config import BLineConfig
from b_line.segmenter import StablePrefixSegmenter
from b_line.types import ASRUpdate


class TestTokensToWords(unittest.TestCase):
    def test_groups_on_sentencepiece_marker(self):
        tokens = ["▁WE", "▁WILL", "▁NOW", "▁DISCUSS"]
        times = [0.2, 0.4, 0.6, 0.8]
        self.assertEqual(
            _tokens_to_words(tokens, times),
            [("WE", 0.2), ("WILL", 0.4), ("NOW", 0.6), ("DISCUSS", 0.8)],
        )

    def test_subword_pieces_join(self):
        self.assertEqual(
            _tokens_to_words(["▁SAM", "PLING"], [2.0, 2.2]),
            [("SAMPLING", 2.2)],
        )

    def test_mismatched_lengths_give_none(self):
        self.assertIsNone(_tokens_to_words(["▁A"], []))


class TestWordGapSegmentation(unittest.TestCase):
    def _run(self, words, gap_ms=550, min_chars=25):
        cfg = BLineConfig(sentence_gap_ms=gap_ms, sentence_min_chars=min_chars,
                          split_on_pauses=True)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        text = " ".join(w for w, _ in words)
        seg.update(
            ASRUpdate(text=text, stable_chars=len(text), confidence=1.0, words=words),
            words[-1][1],
        )
        return seg.pop(), text

    def test_split_at_pause(self):
        # first group is 25 chars, so the 1 s pause after AND is a legal cut
        words = [("SIGNAL", 0.2), ("INTO", 0.4), ("SINUSOIDS", 0.6), ("AND", 0.8),
                 ("THE", 1.8), ("SAMPLING", 2.0), ("THEOREM", 2.2)]
        out, _ = self._run(words)
        self.assertEqual([s.text_en for s in out],
                         ["SIGNAL INTO SINUSOIDS AND", "THE SAMPLING THEOREM"])
        self.assertAlmostEqual(out[0].t_end, out[1].t_start)

    def test_short_pause_does_not_split(self):
        # 0.3 s pauses are ordinary phrase-internal gaps, not sentence ends
        words = [("SIGNAL", 0.2), ("INTO", 0.5), ("SINUSOIDS", 0.8), ("AND", 1.1),
                 ("THE", 1.4), ("SAMPLING", 1.7), ("THEOREM", 2.0)]
        out, text = self._run(words)
        self.assertEqual([s.text_en for s in out], [text])

    def test_length_cut_prefers_the_largest_pause(self):
        # every gap is below the gap threshold, so the length cap decides; the
        # cut must land on the widest pause inside the group
        cfg = BLineConfig(sentence_gap_ms=300, max_seg_chars=60,
                          sentence_min_chars=18, split_on_pauses=True)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        A, B, C, D, E = ("A" * 12, "B" * 12, "C" * 12, "D" * 12, "E" * 12)
        words = [(A, 1.00), (B, 1.05), (C, 1.30), (D, 1.35), (E, 1.40)]
        text = " ".join(w for w, _ in words)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                             words=words), 1.40)
        out = seg.pop()
        self.assertEqual([s.text_en for s in out],
                         [f"{A} {B}", f"{C} {D} {E}"])

    def test_zero_max_chars_disables_the_length_cut(self):
        cfg = BLineConfig(sentence_gap_ms=300, max_seg_chars=0,
                          sentence_min_chars=18, split_on_pauses=True)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        words = [(W, 0.1 * i) for i, W in enumerate(["A" * 40] * 4)]
        text = " ".join(w for w, _ in words)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                             words=words), words[-1][1])
        self.assertEqual([s.text_en for s in seg.pop()], [text])

    def test_misaligned_words_fall_back_to_one_segment(self):
        words = [("WE", 0.2), ("WILL", 0.4), ("NOW", 0.6), ("DISCUSS", 0.8),
                 ("THE", 1.8), ("SAMPLING", 2.0), ("THEOREM", 2.2)]
        text = "SOMETHING ELSE ENTIRELY HERE OK"
        cfg = BLineConfig(sentence_gap_ms=300)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                             words=words), 2.2)
        out = seg.pop()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text_en, text)


class TestDefaultCommit(unittest.TestCase):
    """The on-screen draft and the committed caption must have the same words."""

    def test_pause_inside_a_finalized_slice_does_not_split_it(self):
        words = [("SIGNAL", 0.2), ("INTO", 0.4), ("SINUSOIDS", 0.6), ("AND", 0.8),
                 ("THE", 1.8), ("SAMPLING", 2.0), ("THEOREM", 2.2)]
        cfg = BLineConfig(sentence_gap_ms=550, sentence_min_chars=25)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        text = " ".join(w for w, _ in words)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                             words=words), words[-1][1])
        self.assertEqual([s.text_en for s in seg.pop()], [text])


class TestRulePunctuation(unittest.TestCase):
    """The recognizer emits no marks; the segmenter derives them from pauses."""

    def _commit(self, words, **kw):
        cfg = BLineConfig(rule_punctuation=True, sentence_gap_ms=800,
                          sentence_min_chars=18, max_seg_chars=160,
                          split_on_pauses=True, **kw)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        text = " ".join(w for w, _ in words)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                             words=words), words[-1][1])
        return [s.text_en for s in seg.pop()]

    def test_pauses_never_become_commas(self):
        words = [("SIGNAL", 0.2), ("INTO", 0.4), ("SINUSOIDS", 0.6),
                 ("AND", 0.85), ("THE", 1.50), ("SAMPLING", 1.70),
                 ("THEOREM", 1.90)]
        # a 0.65 s pause is not evidence of a comma; no mark is invented
        self.assertEqual(self._commit(words),
                         ["SIGNAL INTO SINUSOIDS AND THE SAMPLING THEOREM."])

    def test_short_pauses_get_no_comma(self):
        words = [("WE", 0.2), ("WILL", 0.35), ("NOW", 0.5), ("DISCUSS", 0.65),
                 ("THE", 0.8), ("THEOREM", 0.95)]
        self.assertEqual(self._commit(words),
                         ["WE WILL NOW DISCUSS THE THEOREM."])

    def test_text_that_already_has_marks_is_left_alone(self):
        words = [("SIGNAL,", 0.2), ("INTO", 0.4), ("SINUSOIDS.", 1.6)]
        self.assertEqual(self._commit(words),
                         ["SIGNAL, INTO SINUSOIDS."])

    def test_no_words_still_gets_a_full_stop(self):
        cfg = BLineConfig(rule_punctuation=True)
        seg = StablePrefixSegmenter(cfg)
        seg.note_speech(True, 0.0, 0.0)
        seg.update(ASRUpdate(text="HELLO THERE", stable_chars=11,
                             confidence=1.0), 1.0)
        self.assertEqual([s.text_en for s in seg.pop()], ["HELLO THERE."])

    def test_a_punctuation_model_suppresses_the_rule(self):
        words = [("SIGNAL", 0.2), ("AND", 0.4), ("THE", 1.2), ("THEOREM", 1.4)]
        self.assertEqual(self._commit(words, punct_model="model.onnx"),
                         ["SIGNAL AND THE THEOREM"])


if __name__ == "__main__":
    unittest.main()
