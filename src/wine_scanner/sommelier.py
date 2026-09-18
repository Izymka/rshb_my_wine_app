"""Цифровой сомелье: разговор о найденном вине, заземлённый в его карточке.

Функция после поиска (ТЗ п. 3 задачи, критерий «функция после поиска», 20 баллов): человек
узнал вино — и тут же спрашивает, к чему его подать, при какой температуре, чем оно отличается
от соседнего на полке. Ответ даёт языковая модель, но говорить ей разрешено только о том, что
есть в карточке и в общих знаниях о сорте и регионе; выдумывать рейтинги и цены запрещено
подсказкой явно.

Подключение — те же клиенты, что у VLM-судьи (llm.py), но отдельный флаг: судья работает на
картинке и стоит в критическом пути ответа, сомелье — на тексте и по запросу пользователя.
`WINE_SOMMELIER=1` плюс `WINE_LLM_PROVIDER` (`yandex` — те же ключи, что у судьи, модель
`YANDEX_LLM_MODEL`; `openai` — `WINE_LLM_BASE_URL/MODEL/API_KEY`, при их отсутствии берутся
`WINE_VLM_*`) и `WINE_LLM_FALLBACK`. Без флага ручка отвечает 404, и интерфейс блок не
показывает. Ошибка провайдера — честное «подсказки сейчас недоступны», карточка не страдает.
"""

import os

import httpx

from .llm import ChatError, OpenAICompatibleChat, chat_from_env

DEFAULT_TIMEOUT = 12.0
MAX_HISTORY = 6

SYSTEM_PROMPT = (
    "Ты — сомелье платформы «Своё Вино», помогаешь покупателю у полки магазина. Отвечай по-русски, "
    "коротко (2–5 предложений), дружелюбно и по делу. Опирайся только на карточку вина ниже и на "
    "общие знания о сорте, регионе и стиле. Не выдумывай рейтинги, награды, цены и факты о "
    "конкретной винодельне, которых нет в карточке; если чего-то не знаешь — так и скажи. "
    "Если спрашивают о еде — предложи 2–3 конкретных сочетания и температуру подачи."
)


class Sommelier:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
        llm=None,
    ):
        if llm is None:
            if not base_url or not model:
                raise ValueError("Sommelier: нужен llm или base_url и model")
            llm = OpenAICompatibleChat(
                base_url=base_url, model=model, api_key=api_key, timeout=timeout, client=client
            )
        self.llm = llm
        self.calls = 0
        self.errors = 0

    @classmethod
    def from_env(cls) -> "Sommelier | None":
        if os.environ.get("WINE_SOMMELIER", "0") != "1":
            return None
        return cls(llm=chat_from_env("WINE_LLM", timeout=DEFAULT_TIMEOUT))

    @property
    def provider(self) -> str:
        return getattr(self.llm, "name", "?")

    @staticmethod
    def card_text(card: dict) -> str:
        fields_ = (
            ("Название", "name"),
            ("Винодельня", "winery"),
            ("Категория", "category"),
            ("Цвет", "color"),
            ("Регион", "region"),
            ("Сорта", "grapes"),
            ("Год", "vintage"),
            ("Описание", "description"),
        )
        lines = [f"{label}: {card[key]}" for label, key in fields_ if card.get(key)]
        return "\n".join(lines) if lines else "Карточка без описания."

    def ask(self, card: dict, question: str, history: list[dict] | None = None) -> str:
        """Ответ на вопрос о вине. Поднимает RuntimeError, если провайдер не ответил."""
        self.calls += 1
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nКарточка вина:\n{self.card_text(card)}",
            },
        ]
        for turn in (history or [])[-MAX_HISTORY:]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            content = str(turn.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content[:2000]})
        messages.append({"role": "user", "content": question.strip()[:1000]})
        try:
            content = self.llm.chat(messages, temperature=0.4)
        except ChatError as error:
            self.errors += 1
            raise RuntimeError(f"сомелье: {error}") from error
        if not isinstance(content, str):
            content = str(content)
        return content.strip()
