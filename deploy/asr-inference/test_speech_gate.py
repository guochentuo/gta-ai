from __future__ import annotations

import os
import tempfile
import unittest
import wave

import app


class SpeechGateTest(unittest.TestCase):
    def test_silence_skips_whisper_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gta-asr-gate-") as directory:
            path = os.path.join(directory, "silence.wav")
            with wave.open(path, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 16000)

            self.assertIsNone(app._model)
            result = app.speech_gate(path)
            self.assertEqual("no_speech", result["status"])
            self.assertEqual("silero_vad", result["gate"])
            self.assertEqual(0, result["speechSec"])
            self.assertIsNone(app._model)


if __name__ == "__main__":
    unittest.main()
