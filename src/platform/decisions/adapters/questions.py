"""Map :mod:`langchain_core.decisions` questions to OpenAI Decisions questions."""

from langchain_core.decisions import Choice as CoreChoice
from langchain_core.decisions import Noul, Question
from langchain_openai.decisions import Choice as OpenAIChoice
from langchain_openai.decisions import Predicate


def to_openai_questions(questions: dict[str, Question]) -> dict[str, Predicate | OpenAIChoice]:
    """Convert portable questions for :class:`langchain_openai.decisions.OpenAIDecisions`.

    OpenAI names binary questions ``Predicate`` while the portable API uses
    ``Noul``. Categorical questions use ``choices`` instead of ``criteria``.
    """
    converted: dict[str, Predicate | OpenAIChoice] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            instructions = question.instructions if isinstance(question.instructions, str) else str(question.instructions)
            converted[name] = Predicate(instructions=instructions)
            continue
        if isinstance(question, CoreChoice):
            criteria = {str(key): _criteria_value(value) for key, value in question.criteria.items()}
            instructions = question.instructions if isinstance(question.instructions, str) else str(question.instructions)
            converted[name] = OpenAIChoice(instructions=instructions, choices=criteria)
            continue
        raise TypeError(f"Unsupported question type for OpenAI backend: {type(question)!r}")
    return converted


def _criteria_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
