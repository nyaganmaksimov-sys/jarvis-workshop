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
    """Reasoning layer for Jarvis with OpenAI cloud brain + local Ollama fallback."""

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip() or "gpt-5.6-luna"
        self.provider = os.getenv("JARVIS_LLM_PROVIDER", "auto").strip().lower() or "auto"
        self.openai_url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses").strip()
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower() or "low"
        self.web_search_enabled = self._env_bool("OPENAI_WEB_SEARCH_ENABLED", False)
        self.web_search_mode = os.getenv("OPENAI_WEB_SEARCH_MODE", "auto").strip().lower() or "auto"

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}

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
            "cloud_ready": bool(self.openai_key),
            "openai_configured": bool(self.openai_key),
            "openai_model": self.openai_model,
            "reasoning_effort": self.reasoning_effort,
            "web_search_enabled": self.web_search_enabled,
            "web_search_mode": self.web_search_mode,
            "ollama_model": self.ollama_model,
        }

    async def answer(self, text: str, context: dict[str, Any] | None = None) -> BrainReply:
        normalized = text.strip().lower()
        if not normalized:
            return BrainReply("Я вас не расслышал.")

        ctx = context or {}
        force_model = str(ctx.get("task") or "") == "message_draft"

        if not force_model:
            if any(x in normalized for x in ("как дела в цехе", "что в цехе", "статус цеха")):
                return BrainReply("Собираю состояние оборудования и последние тревоги.", "workshop_status")
            if "камер" in normalized and any(x in normalized for x in ("покажи", "открой")):
                return BrainReply("Открываю камеры цеха.", "show_cameras")
            if any(x in normalized for x in ("ошибк", "авари", "тревог")) and not self._explicit_web_request(normalized):
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
                    return await self._openai(text, ctx)
                if provider == "ollama":
                    return await self._ollama(text, ctx)
            except Exception as exc:
                errors.append(f"{provider}:{type(exc).__name__}")

        return BrainReply(
            "Я понял запрос, но облачный мозг сейчас недоступен. Системные команды, поиск по HUB, навигация и задачи продолжают работать.",
            data={"llm_errors": errors, "cloud_ready": bool(self.openai_key)},
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

    def _system_prompt(self, context: dict[str, Any], web_search: bool = False) -> str:
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
        web_rule = (
            "У тебя доступен веб-поиск. Используй его только когда для ответа нужна актуальная или внешняя информация. "
            "Никогда не отправляй в веб-поиск персональные данные клиентов или сотрудников, телефоны, e-mail, внутренние ID, приватные сообщения, токены, ключи или иные секреты HUB. "
            "Если нужен внешний поиск по внутренней проблеме, обезличь запрос: ищи по модели оборудования, коду ошибки, технологии или общему симптому. "
            "Отделяй факты из HUB от сведений, найденных во внешних источниках."
            if web_search else
            "Веб-поиск в этом ответе не включён. Не утверждай, что проверял интернет."
        )
        return (
            "Ты Джарвис — основной интеллектуальный помощник A4Print-HUB и производственного цеха. "
            "Отвечай по-русски, естественно и без канцелярита. Будь инициативным, но не выдумывай факты. "
            "Никогда не утверждай, что выполнил действие, если в переданном контексте нет подтверждения результата. "
            "Данные о клиентах, заказах, деньгах, сотрудниках и производстве используй только из переданного контекста. "
            "Если данных недостаточно, прямо скажи, чего не хватает. "
            f"{memory_rule} {humor} {web_rule} "
            "Не шути над клиентами, сотрудниками, ошибками, деньгами, безопасностью или авариями. "
            "Для сообщений выдавай только готовый текст, когда задача контекста помечена как message_draft."
        )

    @staticmethod
    def _explicit_web_request(normalized: str) -> bool:
        patterns = (
            "в интернете", "в сети", "поищи в интернете", "найди в интернете", "проверь в интернете",
            "найди инструкцию", "официальная документация", "последние новости", "актуальная информация",
            "свежая информация", "что сейчас", "на сегодня", "последняя версия",
        )
        return any(item in normalized for item in patterns)

    def _should_use_web_search(self, text: str, context: dict[str, Any]) -> bool:
        if not self.web_search_enabled or str(context.get("task") or "") == "message_draft":
            return False
        if bool(context.get("web_search")) or str(context.get("research_mode") or "").lower() in {"web", "internet", "external"}:
            return True
        normalized = text.strip().lower()
        if self._explicit_web_request(normalized):
            return True
        if self.web_search_mode == "always":
            return True
        if self.web_search_mode == "off":
            return False
        freshness = re.search(r"\b(сегодня|сейчас|актуальн\w*|свеж\w*|последн\w*|новост\w*|курс\w*|цена\w*|погода\w*|верси\w*)\b", normalized)
        external = re.search(r"\b(инструкци\w*|документаци\w*|мануал\w*|драйвер\w*|прошивк\w*|ошибк\w*|код\s+ошибки|характеристик\w*)\b", normalized)
        return bool(freshness or external)

    @classmethod
    def _safe_context(cls, value: Any, depth: int = 0) -> Any:
        if depth > 5:
            return "…"
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in list(value.items())[:80]:
                lowered = str(key).lower()
                if any(secret in lowered for secret in ("password", "secret", "token", "api_key", "apikey", "authorization", "service_role")):
                    out[str(key)] = "[REDACTED]"
                    continue
                out[str(key)] = cls._safe_context(item, depth + 1)
            return out
        if isinstance(value, list):
            return [cls._safe_context(item, depth + 1) for item in value[:30]]
        if isinstance(value, tuple):
            return [cls._safe_context(item, depth + 1) for item in value[:30]]
        if isinstance(value, str):
            return value[:6000]
        return value

    async def _openai(self, text: str, context: dict[str, Any]) -> BrainReply:
        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY_MISSING")

        use_web = self._should_use_web_search(text, context)
        safe_context = self._safe_context(context)
        system = self._system_prompt(safe_context, web_search=use_web)
        prompt = f"Контекст приложения: {json.dumps(safe_context, ensure_ascii=False, default=str)}\n\nЗапрос пользователя: {text}"
        payload: dict[str, Any] = {
            "model": self.openai_model,
            "instructions": system,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": 1200,
        }
        if use_web:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.openai_url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        text_out = self._extract_openai_text(body)
        if not text_out:
            raise RuntimeError("OPENAI_EMPTY_RESPONSE")
        sources = self._extract_openai_sources(body)
        return BrainReply(
            text_out,
            data={
                "provider": "openai",
                "model": self.openai_model,
                "web_search_requested": use_web,
                "web_sources": sources,
            },
        )

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
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                for annotation in content.get("annotations") or []:
                    if not isinstance(annotation, dict):
                        continue
                    candidate = annotation.get("url_citation") if isinstance(annotation.get("url_citation"), dict) else annotation
                    url = str(candidate.get("url") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    found.append({"url": url, "title": str(candidate.get("title") or url)[:300]})
                    if len(found) >= 12:
                        return found
        return found

    async def _ollama(self, text: str, context: dict[str, Any]) -> BrainReply:
        system = self._system_prompt(context, web_search=False)
        prompt = f"Контекст: {json.dumps(self._safe_context(context), ensure_ascii=False, default=str)}\nЗапрос: {text}"
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
        return BrainReply(value, data={"provider": "ollama", "model": self.ollama_model})
