from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from app.supervisor.intent_classifier import classify_intent
from app.supervisor.followup import get_followup_question

from app.workers.extraction_worker import extract_from_image, extract_from_pdf
from app.workers.youtube_worker import youtube_transcript_task
from app.workers.text_worker import summarize_text, sentiment_analysis, question_answering
from app.workers.code_worker import explain_code

class AgentState(TypedDict):
    session_id: str
    user_text: str
    files: List[Any]

    intent: Optional[str]
    status: str                 # "follow_up" | "completed"
    question: Optional[str]
    result: Optional[Dict[str, Any]]
    final_text: Optional[str]

def node_classify_intent(state: AgentState) -> Dict[str, Any]:
    user_text = state.get("user_text", "")
    intent_data = classify_intent(user_text)
    return {"intent": intent_data["chosen_intent"]}


def node_follow_up(state: AgentState) -> Dict[str, Any]:
    question = get_followup_question(state.get("user_text", ""))
    return {
        "status": "follow_up",
        "question": question,
        "final_text": question,  # text-only guarantee
    }

def node_execute_task(state: AgentState) -> Dict[str, Any]:
    text = state.get("user_text", "")
    files = state.get("files", []) or []
    intent = state.get("intent")

    result: Dict[str, Any] = {}

    if intent == "IMAGE_PDF_EXTRACTION":
        if not files:
            question = get_followup_question(text)
            return {"status": "follow_up", "question": question, "final_text": question}

        f = files[0]
        ext = (f.filename or "").lower().split(".")[-1]
        content = f.file.read()

        if ext in {"jpg", "jpeg", "png"}:
            result = extract_from_image(content)
        elif ext == "pdf":
            result = extract_from_pdf(content)
        else:
            result = {"error": f"Unsupported file type: .{ext}"}

    elif intent == "YOUTUBE_TRANSCRIPT":
        result = youtube_transcript_task(text)

    elif intent == "SUMMARIZATION":
        result = summarize_text(text)

    elif intent == "SENTIMENT_ANALYSIS":
        result = sentiment_analysis(text)

    elif intent == "CODE_EXPLANATION":
        result = explain_code(text)

    elif intent == "CONVERSATIONAL_QA":

        result = question_answering(text, text)

    else:
        question = get_followup_question(text)
        return {"status": "follow_up", "question": question, "final_text": question}

    return {"status": "completed", "result": result}

def node_postprocess(state: AgentState) -> Dict[str, Any]:
    result = state.get("result")
    if result is None:
        return {}

    if isinstance(result, str):
        final_text = result
    elif isinstance(result, dict):
        final_text = "\n".join(f"{k}: {v}" for k, v in result.items())
    else:
        final_text = str(result)

    return {"final_text": final_text}


graph = StateGraph(AgentState)

graph.add_node("classify_intent", node_classify_intent)
graph.add_node("follow_up", node_follow_up)
graph.add_node("execute_task", node_execute_task)
graph.add_node("postprocess", node_postprocess)

graph.add_edge(START, "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    lambda s: s.get("intent") == "UNKNOWN",
    {True: "follow_up", False: "execute_task"},
)

graph.add_edge("execute_task", "postprocess")
graph.add_edge("postprocess", END)


graph.add_edge("follow_up", END)

compiled_graph = graph.compile()
