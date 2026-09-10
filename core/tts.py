from __future__ import annotations

import os

import edge_tts


VOICE_PROFILES = {
    "female": os.getenv("JARVIS_TTS_FEMALE_VOICE", "ru-RU-SvetlanaNeural"),
    "male": os.getenv("JARVIS_TTS_MALE_VOICE", "ru-RU-DmitryNeural"),
}


class NeuralTTS:
    def __init__(self) -> None:
        self.rate_female = os.getenv("JARVIS_TTS_FEMALE_RATE", "+0%")
        self.rate_male = os.getenv("JARVIS_TTS_MALE_RATE", "-4%")

    async def synthesize(self, text: str, profile: str = "female") -> bytes:
        profile = profile if profile in VOICE_PROFILES else "female"
        voice = VOICE_PROFILES[profile]
        rate = self.rate_female if profile == "female" else self.rate_male
        communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        chunks: list[bytes] = []
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
        if not chunks:
            raise RuntimeError("TTS_EMPTY_AUDIO")
        return b"".join(chunks)

    @staticmethod
    def profiles() -> dict[str, dict[str, str]]:
        return {
            "female": {"id": "female", "name": "Ксюша", "voice": VOICE_PROFILES["female"]},
            "male": {"id": "male", "name": "Джарвис", "voice": VOICE_PROFILES["male"]},
        }
