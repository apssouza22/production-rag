"""LangChain chat-model helpers shared by evaluation suites."""

from typing import Any, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


def make_eval_chat_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
) -> BaseChatModel:
    """Create the chat model used for response generation and LLM judges."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


class RagasChatModel(InstructorBaseRagasLLM):
    """Adapt a LangChain chat model to Ragas' structured-output interface."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model

    def generate(self, prompt: str, response_model: type[StructuredOutput]) -> StructuredOutput:
        structured_model = self.chat_model.with_structured_output(response_model)
        response = structured_model.invoke(prompt)
        return self._validate_response(response, response_model)

    async def agenerate(self, prompt: str, response_model: type[StructuredOutput]) -> StructuredOutput:
        structured_model = self.chat_model.with_structured_output(response_model)
        response = await structured_model.ainvoke(prompt)
        return self._validate_response(response, response_model)

    @staticmethod
    def _validate_response(response: Any, response_model: type[StructuredOutput]) -> StructuredOutput:
        if isinstance(response, response_model):
            return response
        return cast(StructuredOutput, response_model.model_validate(response))
