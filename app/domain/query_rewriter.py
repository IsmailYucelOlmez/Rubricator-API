import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.models.schemas import RewriteResultModel

logger = logging.getLogger(__name__)

TONE_SEARCH_HINTS = {
    "Happy": "uplifting joyful heartwarming",
    "Surprising": "twist unexpected plot",
    "Angry": "revenge justice intense",
    "Suspenseful": "thriller suspense mystery",
    "Sad": "tragedy grief loss emotional",
}

CATEGORY_SEARCH_HINTS = {
    "Fiction": "fiction novel",
    "Nonfiction": "nonfiction",
}

REWRITE_PROMPT = """You prepare Google Books API search queries from a user's book description.

Tasks (single step):
1. Detect the input language.
2. If not English, interpret the meaning in English.
3. Extract 3-6 English search themes (nouns, genres, settings, moods).
4. Build {max_queries} short Google Books "q" strings (max {max_chars} chars each).
   - Use plain keywords, not full sentences.
   - Add genre words when implied (fiction, mystery, biography, etc.).
   - Use subject: only for clear nonfiction themes.
   - Do NOT invent book titles or authors not mentioned by the user.
   - Ignore filler like "a story about", "kitap öner", "benzeri".

User category hint: {category}
User tone hint: {tone}
Tone search hints (optional): {tone_hints}
Category search hints (optional): {category_hints}

Input: {user_query}

Respond with JSON only, no markdown fences."""


def _is_quota_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _extract_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class QueryRewriter:
    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.rewrite_model,
            temperature=0.0,
            timeout=15,
            max_retries=1,
            max_output_tokens=256,
            thinking_budget=0,
            google_api_key=settings.google_api_key,
        )
        self._structured = self._llm.with_structured_output(RewriteResultModel)

    def rewrite(
        self,
        user_query: str,
        category: str = "All",
        tone: str = "All",
    ) -> RewriteResultModel | None:
        user_query = user_query.strip()
        if not user_query:
            return None

        tone_hints = TONE_SEARCH_HINTS.get(tone, "") if tone != "All" else ""
        category_hints = CATEGORY_SEARCH_HINTS.get(category, "") if category != "All" else ""

        prompt = REWRITE_PROMPT.format(
            max_queries=2,
            max_chars=80,
            category=category,
            tone=tone,
            tone_hints=tone_hints or "none",
            category_hints=category_hints or "none",
            user_query=user_query,
        )

        try:
            result = self._structured.invoke([HumanMessage(content=prompt)])
            if isinstance(result, RewriteResultModel):
                return result
            if isinstance(result, dict):
                return RewriteResultModel.model_validate(result)
            return RewriteResultModel.model_validate(result)
        except Exception as structured_error:
            if _is_quota_error(structured_error):
                logger.warning(
                    "Query rewrite skipped (quota/rate limit) for %r: %s",
                    user_query[:80],
                    structured_error,
                )
                return None
            logger.warning("Structured rewrite failed, trying raw JSON parse: %s", structured_error)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            payload = _extract_json_payload(str(content))
            return RewriteResultModel.model_validate(payload)
        except Exception as error:
            logger.warning("Query rewrite failed for %r: %s", user_query[:80], error)
            return None
