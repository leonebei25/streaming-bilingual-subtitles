"""WAV conversion, resampling and the converter CLI.

Rebuilt from the compiled cache after the source file was lost; the test names
and assertions follow the original bytecode.
"""
import os
import tempfile
import unittest
import wave

import numpy as np

from a_line.audio_io import (
    _cli,
    convert_wav,
    default_output_name,
    is_target_format,
    probe_wav,
    read_wav,
)


def write_pcm_wav(path, samples, sample_rate=16000, channels=1, width=2):
    """Write float32 samples as PCM WAV with 1/2/3/4-byte samples."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1 and channels > 1:
        x = np.repeat(x[:, None], channels, axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    flat = x.reshape(-1)

    if width == 1:
        data = np.clip(np.rint(flat * 127.0 + 128.0), 0, 255).astype(np.uint8).tobytes()
    elif width == 2:
        data = np.clip(np.rint(flat * 32767.0), -32768, 32767).astype("<i2").tobytes()
    elif width == 3:
        v = np.clip(np.rint(flat * 8388607.0), -8388608, 8388607).astype(np.int32)
        b = v.astype("<i4").tobytes()
        data = b"".join(b[i:i + 3] for i in range(0, len(b), 4))
    elif width == 4:
        data = np.clip(
            np.rint(flat * 2147483647.0), -2147483648, 2147483647
        ).astype("<i4").tobytes()
    else:
        raise ValueError(f"unsupported sample width: {width}")

    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sample_rate)
        w.writeframes(data)


def tone(seconds, freq, sample_rate, amp=0.5):
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(path, expected=None):
    x, info = read_wav(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / info.sample_rate)
    return float(freqs[int(np.argmax(spec))])


class TestReadWrite(unittest.TestCase):
    def test_roundtrip_16k_mono(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            x = tone(0.25, 440.0, 16000)
            write_pcm_wav(path, x, sample_rate=16000)
            y, info = read_wav(path)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(len(y), len(x))
        np.testing.assert_allclose(y, x, atol=0.001)

    def test_read_24_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a24.wav")
            x = tone(0.2, 300.0, 16000)
            write_pcm_wav(path, x, sample_rate=16000, width=3)
            y, info = read_wav(path)
        self.assertEqual(info.sample_width, 3)
        np.testing.assert_allclose(y, x, atol=0.0001)


class TestConversion(unittest.TestCase):
    def convert(self, sr, channels, seconds=1.0, freq=440.0, width=2):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        write_pcm_wav(src, tone(seconds, freq, sr), sr, channels=channels, width=width)
        report = convert_wav(src, dst)
        return dst, report

    def test_44100_stereo_to_16000_mono(self):
        dst, rep = self.convert(44100, 2)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.sample_width, 2)
        self.assertAlmostEqual(info.duration_sec, 1.0, delta=0.02)
        self.assertAlmostEqual(dominant_freq(dst), 440.0, delta=10.0)
        self.assertEqual(rep["output"]["path"], dst)
        self.assertEqual(rep["output"]["channels"], 1)

    def test_48000_mono_to_16000(self):
        dst, _ = self.convert(48000, 1)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertAlmostEqual(info.duration_sec, 1.0, delta=0.02)

    def test_8000_mono_to_16000(self):
        dst, _ = self.convert(8000, 1)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertAlmostEqual(info.duration_sec, 1.0, delta=0.02)

    def test_16000_stereo_downmix(self):
        dst, _ = self.convert(16000, 2)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        y, _ = read_wav(dst)
        self.assertEqual(len(y), 16000)

    def test_keep_stereo(self):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        left = tone(1.0, 440.0, 44100)
        right = tone(1.0, 880.0, 44100)
        write_pcm_wav(src, np.stack([left, right], axis=1), 44100, channels=2)
        rep = convert_wav(src, dst, mono=False)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(rep["output"]["channels"], 2)

    def test_24_bit_input(self):
        dst, _ = self.convert(48000, 1, width=3)
        info = probe_wav(dst)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.sample_width, 2)

    def test_same_file_rejected(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "a.wav")
        write_pcm_wav(path, tone(0.1, 440.0, 16000), 16000)
        with self.assertRaises(ValueError):
            convert_wav(path, path)

    def test_is_target_format(self):
        tmp = tempfile.mkdtemp()
        good = os.path.join(tmp, "good.wav")
        bad = os.path.join(tmp, "bad.wav")
        write_pcm_wav(good, tone(0.1, 440.0, 16000), 16000)
        write_pcm_wav(bad, tone(0.1, 440.0, 44100), 44100, channels=2)
        self.assertTrue(is_target_format(good))
        self.assertFalse(is_target_format(bad))


class TestCli(unittest.TestCase):
    def test_default_output_name_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "lecture.wav")
            write_pcm_wav(src, tone(0.2, 440.0, 44100), 44100, channels=2)
            self.assertEqual(
                default_output_name(src), os.path.join(tmp, "lecture_16k.wav")
            )
            rc = _cli([src, "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "lecture_16k.wav")))

    def test_batch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, sr in (("a.wav", 44100), ("b.wav", 48000)):
                write_pcm_wav(os.path.join(tmp, name), tone(0.1, 300.0, sr), sr)
            out = os.path.join(tmp, "out")
            rc = _cli([tmp, "--out-dir", out, "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "a_16k.wav")))
            self.assertTrue(os.path.exists(os.path.join(out, "b_16k.wav")))


if __name__ == "__main__":
    unittest.main()
