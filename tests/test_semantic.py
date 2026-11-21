"""Semantic cut policy: confirmation, lookahead, and draft offsets."""
import unittest

from b_line.config import BLineConfig
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline
from b_line.segmenter import StablePrefixSegmenter
from b_line.semantic import SemanticCutter
from b_line.types import ASRUpdate

DRAFT = ("so now we sample the signal and reconstruct it the fourier "
         "transform tells us which frequencies")


class _PunctAfterWord:
    """Punctuate by appending '.' after one chosen word (case-insensitive)."""

    def __init__(self, word, drop_space=False):
        self.word = word.lower()
        self.drop_space = drop_space
        self.calls = 0

    def apply(self, text):
        self.calls += 1
        out = []
        for i, w in enumerate(text.split()):
            if self.drop_space and i == 1:
                out[-1] = out[-1] + w          # glue words 1 and 2 together
            else:
                mark = ("." if w.lower() == self.word and not w.endswith(".")
                        else "")
                out.append(w + mark)
        return " ".join(out)

    restore = apply


def _cfg(**kw):
    base = dict(semantic_cuts=True, semantic_min_chars=10,
                semantic_lookahead_words=4, semantic_confirm=2,
                semantic_step_chars=1, semantic_step_sec=0.0)
    base.update(kw)
    return BLineConfig(**base)


class TestSemanticCutter(unittest.TestCase):
    def test_cut_needs_two_confirmations(self):
        cutter = SemanticCutter(_PunctAfterWord("it"), _cfg())
        self.assertEqual(cutter.feed(DRAFT, 1.0), 0)
        cut = cutter.feed(DRAFT, 1.1)
        self.assertGreater(cut, 0)
        self.assertTrue(DRAFT[:cut].endswith("it"))

    def test_confirm_one_cuts_immediately(self):
        cutter = SemanticCutter(_PunctAfterWord("it"), _cfg(semantic_confirm=1))
        self.assertGreater(cutter.feed(DRAFT, 1.0), 0)

    def test_period_at_the_live_edge_is_ignored(self):
        cutter = SemanticCutter(_PunctAfterWord("frequencies"), _cfg())
        self.assertEqual(cutter.feed(DRAFT, 1.0), 0)
        self.assertEqual(cutter.feed(DRAFT, 1.1), 0)

    def test_lookahead_blocks_a_too_late_cut(self):
        cutter = SemanticCutter(
            _PunctAfterWord("it"), _cfg(semantic_lookahead_words=10))
        self.assertEqual(cutter.feed(DRAFT, 1.0), 0)
        self.assertEqual(cutter.feed(DRAFT, 1.1), 0)

    def test_offset_counts_the_leading_space(self):
        draft = " " + DRAFT
        cutter = SemanticCutter(_PunctAfterWord("it"), _cfg())
        cutter.feed(draft, 1.0)
        cut = cutter.feed(draft, 1.1)
        self.assertGreater(cut, 0)
        self.assertTrue(draft[:cut].endswith("it"))

    def test_word_count_mismatch_gives_no_cut(self):
        cutter = SemanticCutter(
            _PunctAfterWord("it", drop_space=True), _cfg())
        self.assertEqual(cutter.feed(DRAFT, 1.0), 0)
        self.assertEqual(cutter.feed(DRAFT, 1.1), 0)

    def test_model_failure_does_not_crash_the_pipeline(self):
        class Boom:
            def apply(self, text):
                raise RuntimeError("boom")

        cutter = SemanticCutter(Boom(), _cfg())
        self.assertEqual(cutter.feed(DRAFT, 1.0), 0)
        self.assertEqual(cutter.feed(DRAFT, 1.1), 0)

    def test_cadence_waits_for_growth_or_time(self):
        punct = _PunctAfterWord("it")
        cutter = SemanticCutter(
            punct, _cfg(semantic_step_chars=100, semantic_step_sec=10.0))
        cutter.feed(DRAFT, 1.0)
        self.assertEqual(punct.calls, 1)
        cutter.feed(DRAFT, 1.1)               # not enough time or growth
        self.assertEqual(punct.calls, 1)
        cutter.feed(DRAFT, 30.0)              # time has passed
        self.assertEqual(punct.calls, 2)


class TestSegmenterCutSupport(unittest.TestCase):
    def test_end_hint_places_the_boundary(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        text = "one two three"
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=1.0),
                   5.0, end_sec=2.0)
        segs = seg.pop()
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].t_end, 2.0)

    def test_last_word_time_sets_t_end(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        words = [("ONE", 0.4), ("TWO", 1.7)]
        seg.update(ASRUpdate(text="ONE TWO", stable_chars=7, confidence=1.0,
                             words=words), 3.0)
        self.assertAlmostEqual(seg.pop()[0].t_end, 1.7,
                               msg="t_end must come from the last word, not the commit")

    def test_consecutive_captions_never_overlap(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.note_speech(True, 0.0, 0.0)
        seg.update(ASRUpdate(text="ONE TWO", stable_chars=7, confidence=1.0,
                             words=[("ONE", 0.4), ("TWO", 1.7)]), 3.0)
        first = seg.pop()[0]
        # A VAD onset from before the first caption ended must not be reused.
        seg.note_speech(True, 1.2, 1.2)
        seg.update(ASRUpdate(text="ONE TWO THREE", stable_chars=13,
                             confidence=1.0,
                             words=[("ONE", 0.4), ("TWO", 1.7),
                                    ("THREE", 2.6)]), 4.0)
        second = seg.pop()[0]
        self.assertGreaterEqual(second.t_start, first.t_end)
        self.assertAlmostEqual(second.t_end, 2.6)

    def test_semantic_prefix_is_not_replayed_as_draft(self):
        seg = StablePrefixSegmenter(BLineConfig())
        seg.update(ASRUpdate(text="ONE TWO THREE FOUR", stable_chars=7,
                             confidence=1.0), 2.0)
        self.assertEqual(seg.partial, "THREE FOUR")
        self.assertEqual(len(seg.pop()), 1)

        # The recognizer still reports stable_chars=0; that must not undo the
        # cut or commit the same text twice.
        seg.update(ASRUpdate(text="ONE TWO THREE FOUR FIVE", stable_chars=0,
                             confidence=1.0), 3.0)
        self.assertEqual(seg.pop(), [])
        self.assertEqual(seg.partial, "THREE FOUR FIVE")


class _NoEndpointASR:
    def reset(self):
        pass

    def stream(self, samples, start_sec, end_sec):
        return ASRUpdate(text="", stable_chars=0, confidence=1.0)


class _ForceASR(_NoEndpointASR):
    def __init__(self, text="one two"):
        self.text = text
        self.forced = 0

    def force_finalize(self):
        self.forced += 1
        return ASRUpdate(text=self.text, stable_chars=len(self.text),
                         confidence=1.0, finalized=True)


class TestPipelineSemanticCut(unittest.TestCase):
    def test_confirmed_cut_emits_before_the_draft_ends(self):
        cfg = _cfg(semantic_lookahead_words=3)
        pipe = SubtitlePipeline(cfg, _NoEndpointASR(), IdentityMT(),
                                punct=_PunctAfterWord("it"))
        update = ASRUpdate(text=DRAFT, stable_chars=0, confidence=1.0)

        def feed(now):
            # Mirrors SubtitlePipeline._flush_chunk: update, then pop + emit.
            pipe._feed_segmenter(update, now)
            for seg in pipe.segmenter.pop():
                pipe._emit(seg)

        feed(5.0)
        self.assertEqual(pipe.events, [], "one sighting is not a cut")
        self.assertEqual(pipe.segmenter.partial, DRAFT)

        feed(5.2)
        self.assertEqual(len(pipe.events), 1)
        event = pipe.events[0]
        plain = event.text_en.lower().rstrip(".,!?")
        self.assertTrue(plain.startswith("so now we sample"))
        self.assertTrue(plain.endswith("reconstruct it"))
        self.assertLess(event.t_end, 5.2, "t_end must be the cut, not the commit")
        self.assertEqual(pipe.segmenter.partial.lower(),
                         "the fourier transform tells us which frequencies")

    def test_semantic_cut_uses_the_word_time(self):
        cfg = _cfg(semantic_lookahead_words=3)
        pipe = SubtitlePipeline(cfg, _NoEndpointASR(), IdentityMT(),
                                punct=_PunctAfterWord("it"))
        words = [(w, 0.3 * i) for i, w in enumerate(DRAFT.split())]
        update = ASRUpdate(text=DRAFT, stable_chars=0, confidence=1.0,
                           words=words)
        for now in (5.0, 5.2):
            pipe._feed_segmenter(update, now)
            for seg in pipe.segmenter.pop():
                pipe._emit(seg)
        it = [w for w, _ in words].index("it")
        self.assertAlmostEqual(pipe.events[0].t_end, words[it][1])

    def test_forced_commit_backs_off_to_a_pause(self):
        cfg = BLineConfig(max_seg_seconds=10.0, force_pause_ms=400,
                          force_min_chars=5)
        asr = _ForceASR()
        pipe = SubtitlePipeline(cfg, asr, IdentityMT())
        text = "one two three four five six"
        words = [("one", 0.5), ("two", 0.9), ("three", 1.4), ("four", 2.0),
                 ("five", 2.4), ("six", 2.8)]
        pipe._last_update = ASRUpdate(text=text, stable_chars=0,
                                      confidence=1.0, words=words)
        pipe.segmenter.partial = text
        pipe.segmenter._draft_since = 0.0
        pipe._maybe_force_finalize(10.1)

        self.assertEqual(asr.forced, 0, "a pause cut is not a hard finalize")
        self.assertEqual(pipe.pause_cut_count, 1)
        self.assertEqual(len(pipe.events), 1)
        self.assertAlmostEqual(pipe.events[0].t_end, 2.4)
        self.assertEqual(pipe.segmenter.partial, "six")

    def test_no_pause_falls_back_to_the_hard_cut(self):
        cfg = BLineConfig(max_seg_seconds=10.0, force_pause_ms=400)
        asr = _ForceASR(text="one two")
        pipe = SubtitlePipeline(cfg, asr, IdentityMT())
        pipe._last_update = ASRUpdate(
            text="one two", stable_chars=0, confidence=1.0,
            words=[("one", 0.5), ("two", 0.6)])
        pipe.segmenter.partial = "one two"
        pipe.segmenter._draft_since = 0.0
        pipe._maybe_force_finalize(10.1)

        self.assertEqual(asr.forced, 1)
        self.assertEqual(pipe.pause_cut_count, 0)
        self.assertEqual(len(pipe.events), 1)

    def test_semantic_cuts_without_a_model_are_disabled(self):
        cfg = BLineConfig(semantic_cuts=True)
        pipe = SubtitlePipeline(cfg, _NoEndpointASR(), IdentityMT(), punct=None)
        self.assertIsNone(pipe.semantic)


if __name__ == "__main__":
    unittest.main()
