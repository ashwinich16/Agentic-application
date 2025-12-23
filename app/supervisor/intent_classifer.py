import os
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

if not NIM_API_KEY:
    raise RuntimeError("Missing NIM_API_KEY in environment (.env).")

client = OpenAI(
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
)

INTENT_MODEL = os.getenv("INTENT_MODEL", "qwen/qwen-2.5-7b-instruct")

INTENTS = [
    "IMAGE_PDF_EXTRACTION",
    "YOUTUBE_TRANSCRIPT",
    "CONVERSATIONAL_QA",
    "SUMMARIZATION",
    "SENTIMENT_ANALYSIS",
    "CODE_EXPLANATION",
    "AUDIO_TRANSCRIPTION_SUMMARY",
    "UNKNOWN",
]

def build_intent_prompt(user_text: str) -> str:
    intent_list = "\n".join(f"- {intent}" for intent in INTENTS)
    return (
        "You are an intent classifier. Decide the user's intent from the text.\n\n"
        f"User text:\n{user_text}\n\n"
        "Possible intents:\n"
        f"{intent_list}\n\n"
        "Return ONLY the best matching intent name from the above list."
    )

def classify_intent(user_text: str) -> Dict[str, str]:
    prompt = build_intent_prompt(user_text)

    response = client.chat.completions.create(
        model=INTENT_MODEL,
        messages=[
            {"role": "system", "content": "Classify user intent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=30,
        temperature=0.0,
    )

    raw_intent = (response.choices[0].message.content or "").strip().upper()
    chosen_intent = raw_intent if raw_intent in INTENTS else "UNKNOWN"

    return {"chosen_intent": chosen_intent, "raw_output": raw_intent}
