import os
from pathlib import Path

from langchain_community.chat_message_histories import SQLChatMessageHistory


DB_PATH = Path(__file__).parent.parent / "data" / "sessions.db"


def _connection_string() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DB_PATH}"


def get_memory(session_id: str) -> SQLChatMessageHistory:
    """Return a SQLChatMessageHistory instance for the given session."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=_connection_string(),
    )


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history."""
    memory = get_memory(session_id)
    if role == "human":
        from langchain_core.messages import HumanMessage
        memory.add_message(HumanMessage(content=content))
    else:
        from langchain_core.messages import AIMessage
        memory.add_message(AIMessage(content=content))


def get_messages_as_dicts(session_id: str) -> list[dict]:
    """Return messages as plain dicts for inclusion in AgentState."""
    memory = get_memory(session_id)
    return [
        {"role": m.type, "content": m.content}
        for m in memory.messages
    ]
