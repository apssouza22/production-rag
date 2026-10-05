from langchain_openai import ChatOpenAI
from src.config import Settings
from src.platform.llm.factory import make_default_chat_model


def test_make_default_chat_model_uses_agent_model():
    settings = Settings(agent_model="gpt-4o-mini", llm_fallback_models="")
    chat_model = make_default_chat_model(settings)

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == "gpt-4o-mini"
