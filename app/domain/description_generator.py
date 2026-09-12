import logging

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = """Sen bir Türkçe kitap tanıtım metni yazarısın.

Kitap bilgileri:
Başlık: {title}
Yazar: {author}
ISBN: {isbn}

Görev: Bu kitap için 2-4 cümlelik, akıcı, Türkçe bir tanıtım/arka kapak metni yaz.

Kurallar:
- Spoiler verme: sonu, önemli olay örgüsü dönüşlerini veya karakterlerin akıbetini açıklama.
- Var olan bir tanıtım metnini birebir kopyalama; kendi cümlelerinle özgün bir metin üret.
- Kitabı tanıyorsan gerçek konusuna sadık kal.
- Kitabı tanımıyorsan veya emin değilsen, başlık ve yazardan çıkarılabilecek türde,
  genel ama gerçeğe aykırı olmayan kısa bir tanıtım yaz; uydurma isim, olay veya yer ekleme.
- Yalnızca tanıtım metnini döndür: başlık, etiket, markdown veya açıklama ekleme.
"""


def _is_quota_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


class TrbookDescriptionGenerator:
    """Generates a spoiler-free, non-copied Turkish book description from
    title/author/ISBN alone — used when a user-submitted trbooks entry has
    no description of its own.
    """

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.description_model,
            temperature=settings.description_temperature,
            timeout=20,
            max_retries=1,
            max_output_tokens=settings.description_max_output_tokens,
            google_api_key=settings.google_api_key,
        )

    def generate(self, *, title: str, author: str, isbn: str) -> str:
        title = title.strip()
        author = author.strip()
        isbn = isbn.strip()
        if not title:
            raise ValueError("title is required")
        if not author:
            raise ValueError("author is required")

        prompt = DESCRIPTION_PROMPT.format(
            title=title,
            author=author,
            isbn=isbn or "bilinmiyor",
        )

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
        except Exception as error:
            if _is_quota_error(error):
                raise RuntimeError(
                    "Description generation is rate-limited; try again shortly."
                ) from error
            raise RuntimeError(f"Description generation failed: {error}") from error

        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        text = str(content).strip()
        if not text:
            raise RuntimeError("Description generation returned an empty result")
        return text
