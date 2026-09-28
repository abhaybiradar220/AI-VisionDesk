"""
MILESTONE 3 - Gemini LLM connection
------------------------------------
One function: send a prompt, get text back.
Returns None when no API key is set (the app then shows a fallback).
"""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def has_key():
    return bool(os.getenv("GEMINI_API_KEY"))


def ask_llm(prompt):
    if not has_key():
        return None
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


if __name__ == "__main__":
    print(ask_llm("Say hello in 5 words.") or "No GEMINI_API_KEY set")
