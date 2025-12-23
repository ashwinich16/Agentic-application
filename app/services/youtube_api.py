from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def fetch_youtube_transcript(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)

        return " ".join(snippet.text for snippet in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript not available for this video."
    except Exception as e:
        raise RuntimeError(f"Failed to fetch YouTube transcript: {e}")
