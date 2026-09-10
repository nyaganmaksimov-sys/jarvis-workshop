from __future__ import annotations

from typing import Any

from .brain import JarvisBrain
from .memory import JarvisMemory


class JarvisCore:
    def __init__(self) -> None:
        self.brain = JarvisBrain()
        self.memory = JarvisMemory()

    async def handle(self, text: str, workshop_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self.memory.add("user", text)
        history = [item.__dict__ for item in self.memory.recent(10)]
        context = {"workshop": workshop_state or {}, "history": history}
        reply = await self.brain.answer(text, context)
        self.memory.add("assistant", reply.text, {"intent": reply.intent})
        return {"text": reply.text, "intent": reply.intent, "data": reply.data or {}}
