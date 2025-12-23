import os
import tempfile
from typing import Dict

from app.services.whisper_api import transcribe_audio_file
from app.services.llm_api import summarize_text_llm


def transcribe_and_summarize(audio_bytes: bytes, filename: str = "audio") -> Dict:
    if not audio_bytes:
        return {"error": "No audio file provided."}
    ext = os.path.splitext(filename)[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio_file(tmp_path)
        if not transcript.strip():
            return {"error": "Transcription produced empty text."}

        summary = summarize_text_llm(transcript) 
        return {"transcript": transcript, **summary}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
