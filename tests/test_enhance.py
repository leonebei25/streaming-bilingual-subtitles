import unittest

import numpy as np

from a_line.enhance import BypassEnhancer, SpectralSubtractionEnhancer


def speech_like(sr=16000, seconds=3.0, seed=1, noise_amp=0.25):
    """Amplitude-modulated two-harmonic bursts separated by silence.

    The default noise level puts the input around 0 dB SNR, where traditional
    spectral subtraction is expected to help. At high SNR it can hurt, which is
    one of the findings the report should show rather than hide.
    """
    n = int(sr * seconds)
    t = np.arange(n) / sr
    env = np.zeros(n, dtype=np.float32)
    for onset in (0.3, 1.0, 1.7, 2.4):
        m = (t >= onset) & (t < onset + 0.4)
        env[m] = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * (t[m] - onset))
    clean = (
        0.35 * np.sin(2 * np.pi * 300.0 * t)
        + 0.20 * np.sin(2 * np.pi * 600.0 * t)
    ).astype(np.float32) * env
    rng = np.random.default_rng(seed)
    noise = noise_amp * rng.standard_normal(n).astype(np.float32)
    return clean, clean + noise


def speech_snr(reference, estimate, delay_samples):
    r = reference[delay_samples:]
    e = estimate[: len(estimate) - delay_samples]
    length = min(len(r), len(e))
    r, e = r[:length], e[:length]
    mask = np.abs(r) > 0.05
    return 10.0 * np.log10(
        np.sum(r[mask] ** 2) / (np.sum((e[mask] - r[mask]) ** 2) + 1e-12)
    )


class TestBypass(unittest.TestCase):
    def test_passthrough_is_exact_and_length_preserving(self):
        x = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        enh = BypassEnhancer()
        out = enh.process(x)
        np.testing.assert_array_equal(out, x)
        self.assertEqual(out.dtype, np.float32)


class TestSpectralSubtraction(unittest.TestCase):
    def test_length_preserved(self):
        clean, noisy = speech_like()
        enh = SpectralSubtractionEnhancer()
        out = enh.process(noisy)
        self.assertEqual(len(out), len(noisy))
        self.assertEqual(out.dtype, np.float32)

    def test_improves_snr_on_speech_like_bursts(self):
        clean, noisy = speech_like()
        enh = SpectralSubtractionEnhancer()
        out = enh.process(noisy)
        delay = int(round(enh.algorithmic_delay_ms * 16))
        before = speech_snr(clean, noisy, 0)
        after = speech_snr(clean, out, delay)
        self.assertGreater(after, before + 3.0,
                           f"SNR before={before:.2f} dB after={after:.2f} dB")

    def test_reset_restores_initial_state(self):
        clean, noisy = speech_like(seconds=1.0)
        enh = SpectralSubtractionEnhancer()
        a = enh.process(noisy)
        enh.reset()
        b = enh.process(noisy)
        np.testing.assert_array_equal(a, b)

    def test_wiener_method_runs(self):
        clean, noisy = speech_like(seconds=1.0)
        enh = SpectralSubtractionEnhancer(method="wiener")
        out = enh.process(noisy)
        self.assertEqual(len(out), len(noisy))
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
