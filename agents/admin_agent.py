"""
Admin agent — the second LangChain agent in the system.

Responsibility:
- Generate user-facing messages for approved / rejected reservations.

The actual approve/reject decision is made by the human admin via the Admin Panel UI.
The human-in-the-loop pattern is enforced by the LangGraph interrupt
in the graph workflow (graph/builder.py).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Prompt: generate user-facing approval/rejection message
# ---------------------------------------------------------------------------

_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Slytherin parking assistant. "
            "Inform the customer of the administrator's decision in a friendly, "
            "professional tone. If approved, include next steps. "
            "If rejected, apologise and suggest contacting customer service.",
        ),
        (
            "human",
            "Reservation details:\n"
            "Name: {name}\n"
            "License plate: {car_number}\n"
            "Period: {start_date} → {end_date}\n\n"
            "Administrator decision: {decision}",
        ),
    ]
)



def format_decision_message(reservation: dict, decision: str) -> str:
    """
    Generate a user-facing message for an approval or rejection.

    decision: "approved" | "rejected"
    """
    name = reservation.get("name", "")
    surname = reservation.get("surname", "")

    chain = _DECISION_PROMPT | _llm()
    return chain.invoke(
        {
            "name": f"{name} {surname}".strip(),
            "car_number": reservation.get("car_number", "—"),
            "start_date": reservation.get("start_date", "—"),
            "end_date": reservation.get("end_date", "—"),
            "decision": decision,
        }
    ).content
