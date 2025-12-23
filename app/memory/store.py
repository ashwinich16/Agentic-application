from typing import Dict, Any
import uuid

MEMORY_STORE: Dict[str, Dict[str, Any]] = {}

def create_session_id() -> str:
    """
    Creates a unique session ID for each new user session.
    """
    session_id = str(uuid.uuid4())
    MEMORY_STORE[session_id] = {
        "messages": [],
        "extracted_texts": [],
        "summaries": [],
        "last_intent": None,
        "task_history": []
    }
    return session_id

def get_memory(session_id: str) -> Dict[str, Any]:
    """
    Return memory for the given session.
    If not found, initialize a new one.
    """
    if session_id not in MEMORY_STORE:
        MEMORY_STORE[session_id] = {
            "messages": [],
            "extracted_texts": [],
            "summaries": [],
            "last_intent": None,
            "task_history": []
        }
    return MEMORY_STORE[session_id]

def update_memory(session_id: str, key: str, value: Any) -> None:
    """
    Update a specific memory key for a session.
    """
    mem = get_memory(session_id)
    mem[key] = value

def append_memory(session_id: str, key: str, entry: Any) -> None:
    """
    Append an entry to a list in memory.
    """
    mem = get_memory(session_id)
    if key not in mem or not isinstance(mem[key], list):
        mem[key] = []
    mem[key].append(entry)

def clear_memory(session_id: str) -> None:
    """
    Removes memory for a session.
    """
    if session_id in MEMORY_STORE:
        del MEMORY_STORE[session_id]
