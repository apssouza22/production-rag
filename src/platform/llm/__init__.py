"""LangChain chat model helpers for OpenAI."""

__all__ = ["make_chat_model", "check_openai_health", "make_default_chat_model"]


def __getattr__(name: str):
    if name == "make_chat_model":
        from src.platform.llm.chat_model import make_chat_model

        return make_chat_model
    if name == "check_openai_health":
        from src.platform.llm.chat_model import check_openai_health

        return check_openai_health
    if name == "make_default_chat_model":
        from src.platform.llm.factory import make_default_chat_model

        return make_default_chat_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
