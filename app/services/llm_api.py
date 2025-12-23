import os
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_BASE_URL = os.getenv("NIM_BASE_URL")
NIM_MODEL = os.getenv("NIM_MODEL")


if not NIM_API_KEY or not NIM_BASE_URL or not NIM_MODEL:
    raise RuntimeError(
        "Missing one or more required env vars: NIM_API_KEY, NIM_BASE_URL, NIM_MODEL"
    )


client = OpenAI(
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
)

def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,) -> str:

    try:
        response = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {type(e).__name__}: {e}"

def summarize_text_llm(text: str) -> Dict:
    prompt = f"""
                Summarize the following text in EXACTLY this format:

                1. One-line summary
                2. Three bullet points
                3. A five-sentence paragraph

                Text:
                {text}
                """
    output = _chat_completion(
        system_prompt="You are an expert summarization assistant.",
        user_prompt=prompt,
        max_tokens=400,
    )
    return {"summary": output}

def sentiment_analysis_llm(text: str) -> Dict:
    prompt = f"""
                Analyze the sentiment of the following text.

                Return in this format:
                Sentiment: <Positive/Negative/Neutral>
                Confidence: <number between 0 and 1>
                Justification: <one line>

                Text:
                {text}
                """
    output = _chat_completion(
        system_prompt="You are a sentiment analysis expert.",
        user_prompt=prompt,
        max_tokens=200,
    )
    return {"sentiment": output}

def explain_code_llm(code: str) -> Dict:
    prompt = f"""
                Explain the following code.

                Your response MUST include:
                1. What the code does
                2. Potential bugs or issues
                3. Time complexity analysis

                Code:
                {code}"""
    output = _chat_completion(
        system_prompt="You are a senior software engineer.",
        user_prompt=prompt,
        max_tokens=600,
    )
    return {"explanation": output}

def question_answering_llm(context: str, question: str) -> Dict:
    prompt = f"""
                Use ONLY the context below to answer the question.
                If the answer is not in the context, say "Not found in the provided context."

                Context:
                {context}

                Question:
                {question}
                """
    output = _chat_completion(
        system_prompt="Answer strictly from the provided context. Do not use outside knowledge.",
        user_prompt=prompt,
        max_tokens=300,
    )
    return {"answer": output}
