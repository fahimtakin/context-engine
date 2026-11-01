"""LLM-as-judge checks on generated answers: groundedness and answer relevance.

The judge is the same local model that writes the answers, so treat these scores as a
cheap regression signal rather than ground truth.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.nodes import format_context
from app.agents.state import RetrievedDoc

logger = logging.getLogger(__name__)

GROUNDEDNESS_PROMPT = (
    "You check an assistant's answer against the context it was given. "
    "Answer grounded=true if every factual claim in the answer is supported by the context, "
    "or if the answer correctly says the context does not contain the information. "
    "Answer grounded=false if the answer states anything the context does not support."
)

RELEVANCE_PROMPT = (
    "You check whether an assistant's answer addresses the user's question. "
    "Answer relevant=true if it responds to what was asked, even if briefly. "
    "Answer relevant=false if it is off-topic, evasive, or answers a different question."
)


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(description="True if the answer is supported by the context.")


class RelevanceVerdict(BaseModel):
    relevant: bool = Field(description="True if the answer addresses the question.")


def judge_groundedness(llm, answer: str, docs: list[RetrievedDoc]) -> bool | None:
    """None when the judge call fails, so the question is left out of the average."""
    try:
        verdict = llm.with_structured_output(GroundednessVerdict).invoke(
            [
                SystemMessage(content=GROUNDEDNESS_PROMPT),
                HumanMessage(content=f"Context:\n{format_context(docs)}\n\nAnswer:\n{answer}"),
            ]
        )
    except Exception:
        logger.warning("Groundedness judge failed", exc_info=True)
        return None
    return verdict.grounded


def judge_relevance(llm, question: str, answer: str) -> bool | None:
    """None when the judge call fails, so the question is left out of the average."""
    try:
        verdict = llm.with_structured_output(RelevanceVerdict).invoke(
            [
                SystemMessage(content=RELEVANCE_PROMPT),
                HumanMessage(content=f"Question: {question}\n\nAnswer:\n{answer}"),
            ]
        )
    except Exception:
        logger.warning("Relevance judge failed", exc_info=True)
        return None
    return verdict.relevant
