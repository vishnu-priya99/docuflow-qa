from __future__ import annotations

import httpx

from app.prompts import answer_prompt, rerank_prompt, router_prompt, sql_prompt
from app.services.llm.base import LLMProvider
from app.services.llm.utils import parse_index_list, strip_code_fence


class OllamaLLMProvider(LLMProvider):
    """Chat provider backed by a local Ollama server (http://localhost:11434
    by default). No API key needed - just `ollama pull <model>` first."""

    def __init__(self, base_url: str, model: str, max_tokens: int, keep_alive: str = "30m") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive
        # Local inference (especially CPU-only) can be slow - give it room.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))

    async def _complete(
        self, *, system: str, user: str, max_tokens: int | None = None, temperature: float | None = None
    ) -> str:
        options: dict[str, object] = {"num_predict": max_tokens or self._max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": options,
                # Without this, Ollama's own default (5 minutes) unloads the
                # model between requests during any natural pause - see
                # Settings.ollama_keep_alive.
                "keep_alive": self._keep_alive,
            },
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message", {}).get("content") or "").strip()

    async def classify_question(
        self, *, question: str, has_documents: bool, has_structured_data: bool
    ) -> str:
        user = router_prompt.build_user_prompt(
            question=question, has_documents=has_documents, has_structured_data=has_structured_data
        )
        raw = await self._complete(system=router_prompt.SYSTEM, user=user, max_tokens=16)
        # Small local models sometimes wrap the single word in punctuation/quotes.
        return raw.strip().strip(".\"'").upper()

    async def generate_sql(self, *, question: str, schema_description: str) -> str:
        user = sql_prompt.build_user_prompt(question=question, schema_description=schema_description)
        # temperature=0: there's one correct query for a given question and
        # schema, not a range of equally-valid creative options - sampling
        # at the default temperature was measured to genuinely flip the
        # output between a correct partial query and a defensive NO_QUERY
        # refusal on the exact same prompt, run back to back.
        raw = await self._complete(system=sql_prompt.SYSTEM, user=user, max_tokens=512, temperature=0.0)
        return strip_code_fence(raw)

    async def generate_answer(self, *, question: str, context: str, question_type: str) -> str:
        user = answer_prompt.build_user_prompt(
            question=question, context=context, question_type=question_type
        )
        return await self._complete(system=answer_prompt.SYSTEM, user=user, max_tokens=512)

    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        if not candidates:
            return []
        user = rerank_prompt.build_user_prompt(question=question, candidates=candidates, top_n=top_n)
        raw = await self._complete(system=rerank_prompt.SYSTEM, user=user, max_tokens=256)
        return parse_index_list(raw, count=len(candidates))
