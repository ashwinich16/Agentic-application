# Agentic Application (FastAPI + LangGraph + Streamlit)

An agentic multimodal application that can handle:
- Text tasks: summarization, sentiment analysis, QA, code explanation
- Image/PDF: OCR extraction (Google Cloud Vision)
- YouTube: transcript fetch + cleanup
- Memory: per-session in-memory store
- LLM: Qwen via NVIDIA NIM (OpenAI-compatible endpoint)

## Project Structure

app/
├── main.py
├── models/
│   └── langgraph_flow.py
├── memory/
│   └── store.py
├── supervisor/
│   ├── intent_classifier.py
│   └── followup.py
├── services/
│   ├── llm_api.py
│   ├── vision_api.py
│   └── youtube_api.py
├── workers/
│   ├── extraction_worker.py
│   ├── youtube_worker.py
│   ├── text_worker.py
│   └── code_worker.py
└── utils/
    └── text_cleaner.py

streamlit.py

---



### Google Vision
Google Vision uses service account credentials
## How to Run

### 1 Create and activate environment
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

## Architecture (High-Level)

User (Streamlit UI)
        |
        v
FastAPI /process  ---------------------+
        |                              |
        v                              |
LangGraph Flow (Supervisor + Workers)  |
        |                              |
        +--> Intent Classifier (NIM/Qwen)
        |
        +--> If UNKNOWN -> Follow-up question
        |
        +--> Else route to correct Worker:
              - extraction_worker (Vision OCR)
              - youtube_worker (Transcript)
              - text_worker (LLM tasks)
              - code_worker (LLM explanation)
        |
        v
Postprocess (always returns TEXT)
        |
        v
Return JSON response + update Memory

---

## LangGraph Flow (Logic)

1) classify_intent
2) if intent == UNKNOWN -> follow_up
3) else -> execute_task
4) postprocess -> END

Memory is stored per session_id (in-memory dict).