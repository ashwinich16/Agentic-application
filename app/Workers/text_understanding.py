# app/workers/text_worker.py

from typing import Dict
from app.utils.text_cleaner import clean_text
from app.services.llm_api import (
    summarize_text_llm,
    sentiment_analysis_llm,
    question_answering_llm,
)


def summarize_text(text: str) -> Dict:
    """
    1-line summary
    3 bullets
    5-sentence summary
    """
    if not text or not text.strip():
        return {"error": "No text provided for summarization."}

    cleaned_text = clean_text(text)

    return summarize_text_llm(cleaned_text)


def sentiment_analysis(text: str) -> Dict:
    """
    Label + confidence + justification
    """
    if not text or not text.strip():
        return {"error": "No text provided for sentiment analysis."}

    cleaned_text = clean_text(text)

    return sentiment_analysis_llm(cleaned_text)


def question_answering(context: str, question: str) -> Dict:
    """
    Conversational QA
    """
    if not context or not context.strip():
        return {"error": "No context provided."}
    if not question or not question.strip():
        return {"error": "No question provided."}

    cleaned_context = clean_text(context)
    cleaned_question = clean_text(question)

    return question_answering_llm(cleaned_context, cleaned_question)
