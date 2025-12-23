from typing import Dict
from app.utils.text_cleaner import clean_text
from app.services.llm_api import explain_code_llm


def explain_code(code: str) -> Dict:
    if not code or not code.strip():
        return {"error": "No code provided for explanation."}

    cleaned_code = clean_text(code)
    return explain_code_llm(cleaned_code)
