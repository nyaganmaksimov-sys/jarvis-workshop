from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class BrainReply:
    text: str
    intent: str = "chat"
    data: dict[str, Any] | None = None


class JarvisBrain:
    """Local-first reasoning layer.

    Uses Ollama when available. Falls back to deterministic workshop commands so
    basic operation does not depend on an LLM being online.
    """

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    async def answer(self, text: str, context: dict[str, Any] | None = None) -> BrainReply:
        normalized = text.strip().lower()
        if not normalized:
            return BrainReply("Я вас не расслышал.")

        if any(x in normalized for x in ("как дела в цехе", "что в цехе", "статус цеха")):
            return BrainReply("Собираю состояние оборудования и последние тревоги.", "workshop_status")
        if "камер" in normalized and any(x in normalized for x in ("покажи", "открой")):
            return BrainReply("Открываю камеры цеха.", "show_cameras")
        if any(x in normalized for x in ("ошибк", "авари", "тревог")):
            return BrainReply("Проверяю активные тревоги оборудования.", "alerts")

        try:
            return await self._ollama(text, context or {})
        except Exception:
            return BrainReply("Команда принята. Локальная языковая модель сейчас недоступна, но системные команды продолжают работать.")

    async def _ollama(self, text: str, context: dict[str, Any]) -> BrainReply:
        system = (
            "Ты Джарвис — локальный помощник производственного цеха. "
            "Отвечай кратко по-русски. Не выдумывай состояние станков. "
            "Если данных нет — прямо скажи об этом."
        )
        prompt = f"Контекст: {json.dumps(context, ensure_ascii=False)}\nЗапрос: {text}"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "system": system, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            body = r.json()
        return BrainReply(str(body.get("response", "")).strip() or "Нет ответа от модели.")
