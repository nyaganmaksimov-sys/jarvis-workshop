from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class BrainReply:
    text: str
    intent: str = "chat"
    data: dict[str, Any] | None = None


class JarvisBrain:
    """Reasoning layer for Jarvis with cloud LLM, optional web search and Ollama fallback."""

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5.6-sol").strip() or "gpt-5.6-sol"
        self.provider = os.getenv("JARVIS_LLM_PROVIDER", "auto").strip().lower() or "auto"
        self.openai_url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses").strip()
        self.web_search_enabled = os.getenv("JARVIS_WEB_SEARCH_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}

    def status(self) -> dict[str, Any]:
        if self.provider == "openai":
            configured = bool(self.openai_key)
            active = "openai" if configured else "unconfigured"
        elif self.provider == "ollama":
            configured = True
            active = "ollama"
        else:
            configured = bool(self.openai_key) or bool(self.ollama_url)
            active = "openai" if self.openai_key else "ollama"
        return {
            "provider": self.provider,
            "active_provider": active,
            "configured": configured,
            "openai_configured": bool(self.openai_key),
            "openai_model": self.openai_model,
            "web_search_enabled": self.web_search_enabled and bool(self.openai_key),
            "ollama_model": self.ollama_model,
        }

    async def answer(self, text: str, context: dict[str, Any] | None = None) -> BrainReply:
        normalized = text.strip().lower()
        if not normalized:
            return BrainReply("Я вас не расслышал.")

        ctx = context or {}
        task = str(ctx.get("task") or "")
        internet_cfg = ctx.get("internet") if isinstance(ctx.get("internet"), dict) else {}
        wants_web = bool(internet_cfg.get("enabled")) or bool(re.search(
            r"(?:в интернете|в сети|поищи в интернете|найди в интернете|актуальн(?:ая|ые|ую)|последн(?:ие|яя) новости|свеж(?:ая|ие) информац)",
            normalized,
        ))
        force_model = task in {"message_draft", "support_reply", "support_answer", "incident_diagnosis", "internet_research"} or wants_web

        if not force_model:
            if any(x in normalized for x in ("как дела в цехе", "что в цехе", "статус цеха")):
                return BrainReply("Собираю состояние оборудования и последние тревоги.", "workshop_status")
            if "камер" in normalized and any(x in normalized for x in ("покажи", "открой")):
                return BrainReply("Открываю камеры цеха.", "show_cameras")
            if any(x in normalized for x in ("ошибк", "авари", "тревог")):
                return BrainReply("Проверяю активные тревоги оборудования.", "alerts")
            if any(x in normalized for x in ("выручк", "касс", "продаж сегодня", "сколько заработ")):
                return BrainReply("Проверяю сегодняшнюю выручку HUB.", "revenue_today")
            if any(x in normalized for x in ("сколько заказ", "заказы в работе", "статус заказ", "готовы к выдаче")):
                return BrainReply("Проверяю текущие заказы HUB.", "orders_status")
            if any(x in normalized for x in ("сообщени", "непрочитан", "кто написал", "что написали")):
                return BrainReply("Проверяю непрочитанные сообщения HUB.", "unread_messages")
            if any(x in normalized for x in ("производств", "задания в работе", "очередь цеха")):
                return BrainReply("Проверяю производственную очередь HUB.", "production_summary")
            if any(x in normalized for x in ("что требует внимания", "что важного", "что нового", "сводка")):
                return BrainReply("Собираю оперативную сводку HUB.", "attention_summary")

        errors: list[str] = []
        for provider in self._provider_order():
            try:
                if provider == "openai":
                    return await self._openai(text, ctx, wants_web=wants_web)
                if provider == "ollama":
                    return await self._ollama(text, ctx)
            except Exception as exc:
                errors.append(f"{provider}:{type(exc).__name__}")

        return BrainReply(
            "Я понял запрос, но языковая модель сейчас недоступна. Системные команды, поиск, навигация и задачи продолжают работать.",
            data={"llm_errors": errors, "web_search_requested": wants_web},
        )

    def _provider_order(self) -> list[str]:
        if self.provider == "openai":
            return ["openai"] if self.openai_key else []
        if self.provider == "ollama":
            return ["ollama"]
        order: list[str] = []
        if self.openai_key:
            order.append("openai")
        order.append("ollama")
        return order

    def _system_prompt(self, context: dict[str, Any]) -> str:
        personality = context.get("jarvis_personality") or {}
        humor_level = int(personality.get("humor_level") or 0)
        humor = {
            0: "Не шути. Тон спокойный, профессиональный и человеческий.",
            1: "Допускай изредка очень лёгкий сухой юмор в стиле умного помощника, если это уместно.",
            2: "Можно умеренно шутить и использовать короткие остроумные реплики, не мешая делу.",
            3: "Юмор заметный и живой, но ответ всё равно должен оставаться полезным и профессиональным.",
        }.get(max(0, min(humor_level, 3)), "")
        if bool(personality.get("serious_mode")):
            humor = "Никакого юмора: запрос относится к деньгам, кассе, ошибке, безопасности или критическому событию."

        memories = context.get("jarvis_memories") or []
        memory_rule = (
            "Учитывай переданные долговременные предпочтения пользователя, но не выдавай их как системные данные и не придумывай новые."
            if memories else
            "Долговременных пользовательских предпочтений в контексте нет."
        )
        task = str(context.get("task") or "")
        task_rule = ""
        if task in {"support_reply", "support_answer"}:
            task_rule = (
                "Ты помогаешь службе поддержки. Сначала опирайся на контекст HUB и базу знаний. "
                "Внешний веб используй только как дополнение. Не придумывай настройки или факты о конкретной установке HUB. "
                "Если предлагаешь технические шаги, начни с безопасных и обратимых проверок."
            )
        elif task == "incident_diagnosis":
            task_rule = (
                "Ты диагностируешь инцидент HUB. Раздели подтверждённые факты, вероятные причины и безопасные следующие шаги. "
                "Не утверждай, что исправление выполнено. Опасные, удаляющие или денежные действия только предлагай, но не выполняй."
            )
        elif task == "internet_research":
            task_rule = "Используй веб-поиск для актуального ответа и опирайся на найденные источники."

        return (
            "Ты Джарвис — основной интеллектуальный помощник A4Print-HUB и производственного цеха. "
            "Отвечай по-русски, естественно и без канцелярита. Будь инициативным, но не выдумывай факты. "
            "Никогда не утверждай, что выполнил действие, если в переданном контексте нет подтверждения результата. "
            "Данные о клиентах, заказах, деньгах, сотрудниках и производстве используй только из переданного контекста HUB. "
            "Внешние интернет-источники не считай доказательством состояния конкретного HUB. "
            "Если данных недостаточно, прямо скажи, чего не хватает. "
            f"{memory_rule} {humor} {task_rule} "
            "Не шути над клиентами, сотрудниками, ошибками, деньгами, безопасностью или авариями. "
            "Для сообщений выдавай только готовый текст, когда задача контекста помечена как message_draft."
        )

    async def _openai(self, text: str, context: dict[str, Any], wants_web: bool = False) -> BrainReply:
        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY_MISSING")
        system = self._system_prompt(context)
        prompt = f"Контекст приложения: {json.dumps(context, ensure_ascii=False, default=str)}\n\nЗапрос пользователя: {text}"
        payload: dict[str, Any] = {
            "model": self.openai_model,
            "instructions": system,
            "input": prompt,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 1200,
        }
        web_allowed = wants_web and self.web_search_enabled
        if web_allowed:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
            payload["include"] = ["web_search_call.action.sources"]
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.openai_url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        text_out = self._extract_openai_text(body)
        if not text_out:
            raise RuntimeError("OPENAI_EMPTY_RESPONSE")
        sources = self._extract_openai_sources(body) if web_allowed else []
        return BrainReply(text_out, data={"web_search_used": bool(sources), "web_sources": sources})

    @staticmethod
    def _extract_openai_text(body: dict[str, Any]) -> str:
        direct = body.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        chunks: list[str] = []
        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "\n".join(x.strip() for x in chunks if x.strip()).strip()

    @staticmethod
    def _extract_openai_sources(body: dict[str, Any]) -> list[dict[str, str]]:
        seen: set[str] = set()
        sources: list[dict[str, str]] = []

        def add(url: Any, title: Any = "") -> None:
            value = str(url or "").strip()
            if not value or value in seen:
                return
            seen.add(value)
            sources.append({"url": value, "title": str(title or value).strip()[:300]})

        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            for source in action.get("sources") or []:
                if isinstance(source, dict):
                    add(source.get("url"), source.get("title"))
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                for annotation in content.get("annotations") or []:
                    if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
                        add(annotation.get("url"), annotation.get("title"))
        return sources[:20]

    async def _ollama(self, text: str, context: dict[str, Any]) -> BrainReply:
        system = self._system_prompt(context)
        prompt = f"Контекст: {json.dumps(context, ensure_ascii=False, default=str)}\nЗапрос: {text}"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "system": system, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            body = r.json()
        value = str(body.get("response", "")).strip()
        if not value:
            raise RuntimeError("OLLAMA_EMPTY_RESPONSE")
        return BrainReply(value, data={"web_search_used": False, "web_sources": []})
