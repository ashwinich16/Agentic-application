# app/workers/code_worker.py

from typing import Dict
from app.utils.text_cleaner import clean_text
from app.services.llm_api import explain_code_llm  # YOU must implement this service


def explain_code(code: str) -> Dict:
    """
    Explain code, detect bugs, and mention time complexity.
    """
    if not code or not code.strip():
        return {"error": "No code provided for explanation."}

    cleaned_code = clean_text(code)

    # API call delegated to service
    return explain_code_llm(cleaned_code)
