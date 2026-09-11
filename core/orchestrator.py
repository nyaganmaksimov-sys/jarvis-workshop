from __future__ import annotations

from typing import Any

from .brain import JarvisBrain


class JarvisCore:
    def __init__(self) -> None:
        self.brain = JarvisBrain()

    def _money(self, value: Any) -> str:
        try:
            return f"{float(value or 0):,.0f}".replace(",", " ") + " рублей"
        except Exception:
            return "0 рублей"

    def _incident_summary(self, hub: dict[str, Any]) -> str:
        incidents = hub.get("incidents") or {}
        total = int(incidents.get("total") or 0)
        critical = int(incidents.get("critical") or 0)
        warning = int(incidents.get("warning") or 0)
        items = incidents.get("items") or []
        if total <= 0:
            return "Sentinel активных проблем не обнаружил."
        top = items[0] if items else {}
        top_title = str(top.get("title") or "").strip()
        text = f"Sentinel: активных проблем {total}, критических {critical}, предупреждений {warning}."
        if top_title:
            text += f" Самая важная: {top_title}."
        return text

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
                f"производственных заданий в работе {int(p.get('in_progress') or 0)}. "
                f"{self._incident_summary(hub)}"
            )
        if intent == "workshop_status":
            devices = workshop.get("devices") or []
            alerts = workshop.get("alerts") or []
            if not devices:
                incidents_text = self._incident_summary(hub)
                return f"Оборудование пока не передаёт данные. {incidents_text}"
            active = sum(1 for x in devices if x.get("status") in {"PRINTING", "ONLINE", "IN_PROGRESS"})
            errors = sum(1 for x in devices if x.get("status") in {"ERROR", "OFFLINE"})
            return (
                f"Устройств видно {len(devices)}. Активных {active}, с ошибкой или офлайн {errors}. "
                f"Тревог оборудования {len(alerts)}. {self._incident_summary(hub)}"
            )
        return None

    async def handle(
        self,
        text: str,
        workshop_state: dict[str, Any] | None = None,
        hub_state: dict[str, Any] | None = None,
        external_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "workshop": workshop_state or {},
            "hub": hub_state or {},
        }
        if external_context:
            context.update(external_context)
        reply = await self.brain.answer(text, context)
        rendered = self._render_intent(reply.intent, hub_state or {}, workshop_state or {})
        final_text = rendered or reply.text
        return {"text": final_text, "intent": reply.intent, "data": reply.data or {}}
