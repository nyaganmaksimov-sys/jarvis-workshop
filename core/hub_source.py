from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx


class HubSource:
    """Read-only A4Print-HUB operational source via Supabase REST.

    Requires server-side SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
    Only aggregate operational data is exposed to Jarvis responses.
    """

    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    async def _get(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.url}/rest/v1/{table}", params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def snapshot(self) -> dict[str, Any]:
        if not self.configured:
            return {"configured": False, "captured_at": datetime.now(timezone.utc).isoformat()}

        today = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()

        orders = await self._get("orders", {
            "select": "id,status,total,total_amount,business_unit,created_at",
            "order": "created_at.desc",
            "limit": "1000",
        })
        jobs = await self._get("production_jobs", {
            "select": "id,status,title,priority,planned_end,updated_at",
            "order": "updated_at.desc",
            "limit": "1000",
        })
        notifications = await self._get("notifications", {
            "select": "id,type,is_read,title,message,created_at,entity_type,entity_id",
            "is_read": "eq.false",
            "order": "created_at.desc",
            "limit": "100",
        })
        sales = await self._get("pos_sales", {
            "select": "id,total,created_at,status",
            "created_at": f"gte.{today}",
            "limit": "2000",
        })
        returns = await self._get("pos_returns", {
            "select": "id,amount,created_at,status",
            "created_at": f"gte.{today}",
            "limit": "2000",
        })

        gross = sum(float(x.get("total") or 0) for x in sales)
        refunds = sum(float(x.get("amount") or 0) for x in returns)
        chat = [x for x in notifications if x.get("type") == "CHAT_MESSAGE"]

        return {
            "configured": True,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "revenue": {
                "gross": gross,
                "refunds": refunds,
                "net": gross - refunds,
                "sales_count": len(sales),
                "returns_count": len(returns),
            },
            "orders": {
                "total": len(orders),
                "new": sum(1 for x in orders if x.get("status") == "NEW"),
                "in_progress": sum(1 for x in orders if x.get("status") in {"CONFIRMED", "IN_PROGRESS"}),
                "ready": sum(1 for x in orders if x.get("status") == "READY"),
            },
            "production": {
                "total": len(jobs),
                "queue": sum(1 for x in jobs if x.get("status") in {"NEW", "QUEUED"}),
                "in_progress": sum(1 for x in jobs if x.get("status") in {"IN_PROGRESS", "PAUSED"}),
                "done": sum(1 for x in jobs if x.get("status") == "DONE"),
                "high_priority": sum(1 for x in jobs if float(x.get("priority") or 0) >= 70 and x.get("status") not in {"DONE", "CANCELLED"}),
            },
            "messages": {
                "unread": len(chat),
                "latest": [
                    {
                        "title": x.get("title") or "Новое сообщение",
                        "message": (x.get("message") or "")[:240],
                        "created_at": x.get("created_at"),
                    }
                    for x in chat[:10]
                ],
            },
            "attention": [
                {
                    "type": x.get("type"),
                    "title": x.get("title"),
                    "message": (x.get("message") or "")[:240],
                    "created_at": x.get("created_at"),
                }
                for x in notifications[:20]
            ],
        }
