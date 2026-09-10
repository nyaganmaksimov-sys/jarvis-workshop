from __future__ import annotations

import os
from typing import TypedDict

import edge_tts


class VoiceProfile(TypedDict):
    id: str
    name: str
    voice: str
    rate: str
    volume: str
    pitch: str
    character: str


def _profile(
    profile_id: str,
    name: str,
    voice_env: str,
    voice_default: str,
    rate_env: str,
    rate_default: str,
    volume_env: str,
    volume_default: str,
    pitch_env: str,
    pitch_default: str,
    character: str,
) -> VoiceProfile:
    return {
        "id": profile_id,
        "name": name,
        "voice": os.getenv(voice_env, voice_default),
        "rate": os.getenv(rate_env, rate_default),
        "volume": os.getenv(volume_env, volume_default),
        "pitch": os.getenv(pitch_env, pitch_default),
        "character": character,
    }


VOICE_PROFILES: dict[str, VoiceProfile] = {
    "female": _profile(
        "female",
        "Ксюша",
        "JARVIS_TTS_FEMALE_VOICE",
        "ru-RU-SvetlanaNeural",
        "JARVIS_TTS_FEMALE_RATE",
        "-2%",
        "JARVIS_TTS_FEMALE_VOLUME",
        "+3%",
        "JARVIS_TTS_FEMALE_PITCH",
        "+1Hz",
        "спокойный, тёплый женский голос",
    ),
    "male": _profile(
        "male",
        "Джарвис",
        "JARVIS_TTS_MALE_VOICE",
        "ru-RU-DmitryNeural",
        "JARVIS_TTS_MALE_RATE",
        "-6%",
        "JARVIS_TTS_MALE_VOLUME",
        "+4%",
        "JARVIS_TTS_MALE_PITCH",
        "-8Hz",
        "спокойный, уверенный мужской голос",
    ),
}


class NeuralTTS:
    async def synthesize(self, text: str, profile: str = "female") -> bytes:
        profile_id = profile if profile in VOICE_PROFILES else "female"
        settings = VOICE_PROFILES[profile_id]
        communicator = edge_tts.Communicate(
            text=text,
            voice=settings["voice"],
            rate=settings["rate"],
            volume=settings["volume"],
            pitch=settings["pitch"],
        )
        chunks: list[bytes] = []
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
        if not chunks:
            raise RuntimeError("TTS_EMPTY_AUDIO")
        return b"".join(chunks)

    @staticmethod
    def profiles() -> dict[str, VoiceProfile]:
        return {key: dict(value) for key, value in VOICE_PROFILES.items()}
