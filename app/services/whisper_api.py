# from openai import OpenAI

# client = OpenAI()

# def transcribe_audio(file_path: str) -> str:
#     try:
#         with open(file_path, "rb") as audio_file:
#             transcription = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_file,
#             )
#         return transcription.text
#     except Exception as e:
#         raise RuntimeError(f"Audio transcription failed: {e}")
# app/services/whisper_api.py

import whisper


_MODEL = whisper.load_model("base")


def transcribe_audio_file(file_path: str) -> str:
    """
    Offline transcription using openai-whisper (local).
    Requires ffmpeg installed and available in PATH.
    """
    try:
        result = _MODEL.transcribe(file_path)
        return (result.get("text") or "").strip()
    except Exception as e:
        raise RuntimeError(f"Audio transcription failed: {e}")

