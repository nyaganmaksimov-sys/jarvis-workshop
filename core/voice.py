from __future__ import annotations

import json
import os
import queue
from pathlib import Path
from typing import Callable


class VoiceUnavailable(RuntimeError):
    pass


class JarvisVoice:
    """Offline microphone listener based on Vosk + sounddevice.

    Importing this module does not require audio dependencies. They are loaded
    lazily so the server can run without a microphone attached.
    """

    def __init__(self, wake_word: str | None = None, model_path: str | None = None) -> None:
        self.wake_word = (wake_word or os.getenv("JARVIS_WAKE_WORD", "джарвис")).lower()
        self.model_path = Path(model_path or os.getenv("VOSK_MODEL_PATH", "models/vosk-ru"))
        self._armed = False

    def listen_forever(self, on_command: Callable[[str], None]) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise VoiceUnavailable("Install voice requirements: vosk and sounddevice") from exc

        if not self.model_path.exists():
            raise VoiceUnavailable(f"Vosk model not found: {self.model_path}")

        audio: queue.Queue[bytes] = queue.Queue()
        model = Model(str(self.model_path))
        recognizer = KaldiRecognizer(model, 16000)

        def callback(indata, frames, time, status):  # noqa: ARG001
            audio.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = audio.get()
                if not recognizer.AcceptWaveform(data):
                    continue
                text = json.loads(recognizer.Result()).get("text", "").strip().lower()
                if not text:
                    continue
                self._handle_phrase(text, on_command)

    def _handle_phrase(self, text: str, on_command: Callable[[str], None]) -> None:
        if self.wake_word in text:
            after = text.split(self.wake_word, 1)[1].strip()
            if after:
                on_command(after)
                self._armed = False
            else:
                self._armed = True
            return
        if self._armed:
            on_command(text)
            self._armed = False
