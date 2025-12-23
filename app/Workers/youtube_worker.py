from typing import Dict
from urllib.parse import urlparse, parse_qs

from app.services.youtube_api import fetch_youtube_transcript
from app.utils.text_cleaner import clean_text, extract_youtube_url


def _extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)

    # youtu.be/<id>
    if parsed.netloc in {"youtu.be"}:
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=<id>
    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]

    # youtube.com/embed/<id> or /shorts/<id>
    if parsed.path.startswith(("/embed/", "/shorts/")):
        return parsed.path.split("/")[2]

    return None


def youtube_transcript_task(text: str) -> Dict:
    url = extract_youtube_url(text)
    if not url:
        return {"error": "No YouTube URL found in input."}

    video_id = _extract_video_id(url)
    if not video_id:
        return {"error": "Could not extract video ID from URL."}

    transcript = fetch_youtube_transcript(video_id)
    cleaned = clean_text(transcript)

    return {"text": cleaned}
