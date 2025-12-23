# app/services/llm_api.py
import os
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama3-8b-instruct")

if not NIM_API_KEY:
    raise RuntimeError("Missing NIM_API_KEY in .env")

client = OpenAI(api_key=NIM_API_KEY, base_url=NIM_BASE_URL)


def _chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=NIM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def summarize_text_llm(text: str) -> Dict:
    prompt = f"""Summarize the following text in EXACTLY this format:

1. One-line summary
2. Three bullet points
3. A five-sentence paragraph

Text:
{text}
"""
    return {"summary": _chat_completion("You are an expert summarization assistant.", prompt, max_tokens=400)}


def sentiment_analysis_llm(text: str) -> Dict:
    prompt = f"""Analyze the sentiment of the following text.

Return in this format:
Sentiment: <Positive/Negative/Neutral>
Confidence: <number between 0 and 1>
Justification: <one line>

Text:
{text}
"""
    return {"sentiment": _chat_completion("You are a sentiment analysis expert.", prompt, max_tokens=200)}


def explain_code_llm(code: str) -> Dict:
    prompt = f"""Explain the following code.

Your response MUST include:
1. What the code does
2. Potential bugs or issues
3. Time complexity analysis

Code:

{code}
"""
    return {"explanation": _chat_completion("You are a senior software engineer.", prompt, max_tokens=600)}


def question_answering_llm(context: str, question: str) -> Dict:
    prompt = f"""answer the question like you are a friendly conversational AI"

Context:
{context}

Question:
{question}
"""
    return {"answer": _chat_completion("Answer strictly from the provided context. Do not use outside knowledge.", prompt, max_tokens=300)}
