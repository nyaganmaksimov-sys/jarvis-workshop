from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .hub_source import HubSource


class AnnouncementQueue:
    def __init__(self, hub: HubSource, poll_seconds: int = 12) -> None:
        self.hub = hub
        self.poll_seconds = max(5, poll_seconds)
        self.items: deque[dict[str, Any]] = deque(maxlen=200)
        self.seen: set[str] = set()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.items)[:max(1, min(limit, 100))]

    def ack(self, announcement_id: str) -> bool:
        for item in list(self.items):
            if item.get("id") == announcement_id:
                self.items.remove(item)
                return True
        return False

    async def _run(self) -> None:
        while True:
            try:
                snapshot = await self.hub.snapshot()
                self._collect(snapshot)
            except Exception:
                pass
            await asyncio.sleep(self.poll_seconds)

    def _collect(self, snapshot: dict[str, Any]) -> None:
        for msg in (snapshot.get("messages") or {}).get("latest") or []:
            fingerprint = f"chat:{msg.get('created_at')}:{msg.get('message')}"
            if fingerprint in self.seen:
                continue
            self.seen.add(fingerprint)
            text = (msg.get("message") or "Новое сообщение").strip()
            self.items.appendleft({
                "id": fingerprint,
                "type": "message",
                "priority": "normal",
                "text": f"Новое сообщение в HUB. {text[:220]}",
                "created_at": msg.get("created_at") or datetime.now(timezone.utc).isoformat(),
            })

        production = snapshot.get("production") or {}
        high = int(production.get("high_priority") or 0)
        if high:
            fingerprint = f"production-high:{high}:{snapshot.get('captured_at','')[:13]}"
            if fingerprint not in self.seen:
                self.seen.add(fingerprint)
                self.items.appendleft({
                    "id": fingerprint,
                    "type": "production",
                    "priority": "high",
                    "text": f"Внимание. В производстве {high} заданий высокого приоритета.",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
