import re
from typing import Optional


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_youtube_url(text: str) -> Optional[str]:
    """
    Extracts the first YouTube URL from text.
    Supports watch, short, embed, and youtu.be formats.
    """
    pattern = re.compile(
        r"(https?://(?:www\.)?"
        r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)"
        r"|youtu\.be/)"
        r"[^\s]+)"
    )
    match = pattern.search(text)
    return match.group(1) if match else None


def format_ocr_result(text: str, confidence: float) -> dict:
    return {
        "text": (text or "").strip(),
        "confidence": float(confidence or 0.0),
    }
