from __future__ import annotations

from typing import Any

from .brain import JarvisBrain
from .memory import JarvisMemory


class JarvisCore:
    def __init__(self) -> None:
        self.brain = JarvisBrain()
        self.memory = JarvisMemory()

    def _money(self, value: Any) -> str:
        try:
            return f"{float(value or 0):,.0f}".replace(",", " ") + " рублей"
        except Exception:
            return "0 рублей"

    def _render_intent(self, intent: str, hub: dict[str, Any], workshop: dict[str, Any]) -> str | None:
        if intent == "revenue_today":
            r = hub.get("revenue") or {}
            if not hub.get("configured"):
                return "Данные HUB пока не подключены."
            return (
                f"За сегодня чистая выручка {self._money(r.get('net'))}. "
                f"Продаж на {self._money(r.get('gross'))}, возвратов на {self._money(r.get('refunds'))}. "
                f"Продаж: {int(r.get('sales_count') or 0)}."
            )
        if intent == "orders_status":
            o = hub.get("orders") or {}
            if not hub.get("configured"):
                return "Данные HUB пока не подключены."
            return (
                f"Новых заказов {int(o.get('new') or 0)}, в работе {int(o.get('in_progress') or 0)}, "
                f"готовы к выдаче {int(o.get('ready') or 0)}."
            )
        if intent == "unread_messages":
            m = hub.get("messages") or {}
            if not hub.get("configured"):
                return "Данные HUB пока не подключены."
            unread = int(m.get("unread") or 0)
            if unread == 0:
                return "Непрочитанных сообщений нет."
            latest = m.get("latest") or []
            preview = latest[0].get("message") if latest else ""
            return f"Непрочитанных сообщений: {unread}." + (f" Последнее: {preview}" if preview else "")
        if intent == "production_summary":
            p = hub.get("production") or {}
            if not hub.get("configured"):
                return "Данные HUB пока не подключены."
            return (
                f"В очереди {int(p.get('queue') or 0)} заданий, в работе {int(p.get('in_progress') or 0)}, "
                f"высокого приоритета {int(p.get('high_priority') or 0)}."
            )
        if intent == "attention_summary":
            o = hub.get("orders") or {}
            p = hub.get("production") or {}
            m = hub.get("messages") or {}
            r = hub.get("revenue") or {}
            if not hub.get("configured"):
                return "Данные HUB пока не подключены."
            return (
                f"Сводка: выручка {self._money(r.get('net'))}; новых заказов {int(o.get('new') or 0)}; "
                f"в работе {int(o.get('in_progress') or 0)}; непрочитанных сообщений {int(m.get('unread') or 0)}; "
                f"производственных заданий в работе {int(p.get('in_progress') or 0)}."
            )
        if intent == "workshop_status":
            devices = workshop.get("devices") or []
            alerts = workshop.get("alerts") or []
            if not devices:
                return "Оборудование пока не передаёт данные."
            active = sum(1 for x in devices if x.get("status") in {"PRINTING", "ONLINE", "IN_PROGRESS"})
            errors = sum(1 for x in devices if x.get("status") in {"ERROR", "OFFLINE"})
            return f"Устройств видно {len(devices)}. Активных {active}, с ошибкой или офлайн {errors}. Тревог {len(alerts)}."
        return None

    async def handle(
        self,
        text: str,
        workshop_state: dict[str, Any] | None = None,
        hub_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.memory.add("user", text)
        history = [item.__dict__ for item in self.memory.recent(10)]
        context = {"workshop": workshop_state or {}, "hub": hub_state or {}, "history": history}
        reply = await self.brain.answer(text, context)
        rendered = self._render_intent(reply.intent, hub_state or {}, workshop_state or {})
        final_text = rendered or reply.text
        self.memory.add("assistant", final_text, {"intent": reply.intent})
        return {"text": final_text, "intent": reply.intent, "data": reply.data or {}}
