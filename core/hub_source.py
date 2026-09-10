from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx


class HubSource:
    """Read-only operational source from A4Print-HUB.

    Jarvis does not receive the Supabase service role key. Instead it calls a
    protected internal HUB endpoint that returns only aggregated operational data.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("HUB_API_URL", "").rstrip("/")
        self.bridge_key = os.getenv("JARVIS_BRIDGE_KEY", "")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.bridge_key)

    async def snapshot(self) -> dict[str, Any]:
        if not self.configured:
            return {"configured": False, "captured_at": datetime.now(timezone.utc).isoformat()}

        headers = {
            "Authorization": f"Bearer {self.bridge_key}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/internal/jarvis/status",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            return {"configured": False, "captured_at": datetime.now(timezone.utc).isoformat()}
        return data
