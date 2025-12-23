from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse

from app.memory.store import create_session_id, get_memory, append_memory
from app.models.langgraph_flow import compiled_graph

app = FastAPI(
    title="Agentic Application",
    description="Agentic system for text, image, PDF, audio, and YouTube inputs",
)


def _latest_assistant_question(memory: dict) -> Optional[str]:
    """
    If the last assistant message was a follow-up question, return it.
    We store assistant messages in memory["messages"] with role 'assistant' or 'agent'
    depending on your earlier code; we handle both.
    """
    msgs = memory.get("messages", [])
    for m in reversed(msgs):
        role = (m.get("role") or "").lower()
        if role in {"assistant", "agent"}:
            return m.get("text") or m.get("content")
    return None


@app.post("/process")
def process_input(
    text: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    session_id: Optional[str] = Header(None),
):
    if not session_id:
        session_id = create_session_id()

    memory = get_memory(session_id)

    user_text = (text or "").strip()
    upload_files = files or []
    prev_q = _latest_assistant_question(memory)
    combined_text = user_text
    if prev_q and user_text:
        combined_text = f"Previous follow-up question: {prev_q}\nUser answer: {user_text}"
    if user_text:
        append_memory(session_id, "messages", {"role": "user", "content": user_text})

    state_in = {
        "session_id": session_id,
        "user_text": combined_text,
        "files": upload_files,
    }

    try:
        state_out = compiled_graph.invoke(state_in)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Graph execution failed", "detail": str(e)},
        )

    if state_out.get("status") == "follow_up":
        question = state_out.get("question") or state_out.get("final_text") or "Can you clarify?"
        append_memory(session_id, "messages", {"role": "assistant", "content": question})

        return {
            "session_id": session_id,
            "response": {
                "status": "follow_up",
                "question": question,
                "result": question,
            },
        }

    final_text = state_out.get("final_text") or ""
    append_memory(session_id, "messages", {"role": "assistant", "content": final_text})

    return {
        "session_id": session_id,
        "response": {
            "status": "completed",
            "result": final_text,
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
