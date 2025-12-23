# tests/test_intent_classifier.py

import pytest
from unittest.mock import patch

from app.supervisor.intent_classifier import classify_intent

@pytest.mark.parametrize("user_text,expected_intent", [
    ("Please summarize this text", "SUMMARIZATION"),
    ("What is the sentiment of this?", "SENTIMENT_ANALYSIS"),
    ("Explain this code snippet", "CODE_EXPLANATION"),
    ("Can you fetch the YouTube transcript?", "YOUTUBE_TRANSCRIPT"),
    ("Transcribe this audio file", "AUDIO_TRANSCRIPTION_SUMMARY"),
    ("Hello! How are you?", "CONVERSATIONAL_QA"),
])
@patch("app.supervisor.intent_classifier.client.chat.completions.create")
def test_intent_classifier(mock_create, user_text, expected_intent):
    # Mock LLM output
    mock_response = type("obj", (object,), {
        "choices": [
            type("c", (object,), {"message": {"content": expected_intent}})
        ]
    })
    mock_create.return_value = mock_response

    result = classify_intent(user_text)
    assert result["chosen_intent"] == expected_intent
# tests/test_follow_up_manager.py

from app.supervisor.follow_up_manager import get_followup_question

def test_follow_up_empty():
    question = get_followup_question("")
    assert "provide more details" in question.lower()

def test_follow_up_unclear_text():
    question = get_followup_question("Blah blah")
    assert "clarify your intended task" in question.lower()
# tests/test_extraction_worker.py

import pytest
from app.workers.extraction_worker import extract_from_image

def test_extract_from_image_text(monkeypatch):
    fake_bytes = b"fake image bytes"

    # Mock OCR service
    def fake_ocr(image_bytes):
        return {"text": "Hello world", "confidence": 0.9}

    monkeypatch.setattr("app.services.vision_api.ocr_image_bytes", fake_ocr)

    result = extract_from_image(fake_bytes)
    assert result["text"] == "Hello world"
    assert result["confidence"] == 0.9
# tests/test_extraction_worker.py

from app.workers.extraction_worker import transcribe_audio_file

def test_audio_transcription(monkeypatch, tmp_path):
    fake_file = tmp_path / "audio.wav"
    fake_file.write_bytes(b"fake audio")

    monkeypatch.setattr(
        "app.services.whisper_api.transcribe_audio",
        lambda path: "Transcribed text"
    )

    result = transcribe_audio_file(str(fake_file))
    assert "Transcribed text" in result["text"]
# tests/test_youtube_worker.py

from app.workers.youtube_worker import youtube_transcript_task

def test_youtube_transcript(monkeypatch):
    fake_url = "https://youtu.be/1234"
    monkeypatch.setattr(
        "app.services.youtube_api.fetch_youtube_transcript",
        lambda video_id: "YT Transcript"
    )

    res = youtube_transcript_task(fake_url)
    assert "YT Transcript" in res["text"]
# tests/test_text_worker.py

from app.workers.text_worker import summarize_text, sentiment_analysis

def test_summarize(monkeypatch):
    monkeypatch.setattr(
        "app.models.langchain_chains.summarize_chain.run",
        lambda text: {"summary": "short"}
    )
    result = summarize_text("Some text")
    assert result["summary"] == "short"

def test_sentiment(monkeypatch):
    monkeypatch.setattr(
        "app.models.langchain_chains.sentiment_chain.run",
        lambda text: {"label": "POSITIVE", "confidence": 0.8}
    )
    result = sentiment_analysis("Some text")
    assert result["label"] == "POSITIVE"
# tests/test_code_worker.py

from app.workers.code_worker import explain_code

def test_code_explain(monkeypatch):
    monkeypatch.setattr(
        "app.models.langchain_chains.code_explain_chain.run",
        lambda code: {"explanation": "Explained!"}
    )
    result = explain_code("print('hi')")
    assert "Explained" in result["explanation"]
# tests/test_task_router.py

from app.supervisor.task_router import route_task

def test_unknown_intent(monkeypatch):
    # Force intent classifier to return UNKNOWN
    monkeypatch.setattr(
        "app.supervisor.intent_classifier.classify_intent",
        lambda text: {"chosen_intent": "UNKNOWN"}
    )
    result = route_task({"text": "???", "files": []})
    assert result["follow_up_required"]

def test_summarization_route(monkeypatch):
    monkeypatch.setattr(
        "app.supervisor.intent_classifier.classify_intent",
        lambda text: {"chosen_intent": "SUMMARIZATION"}
    )
    monkeypatch.setattr(
        "app.workers.text_worker.summarize_text",
        lambda text: {"summary": "ok"}
    )
    result = route_task({"text": "This needs a summary", "files": []})
    assert "summary" in result["result"]
# tests/test_api.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200

def test_process_followup(monkeypatch):
    monkeypatch.setattr(
        "app.supervisor.intent_classifier.classify_intent",
        lambda text: {"chosen_intent": "UNKNOWN"}
    )

    res = client.post("/process", data={"text": "?"})
    assert res.json()["status"] == "follow_up"
