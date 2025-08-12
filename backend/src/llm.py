import os, time, requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Practical caps
MAX_CHARS_PER_GRAPH = 8000         # ~2–2.5k tokens input
DEFAULT_MAX_TOKENS = 350           # per-call output cap
MIN_MAX_TOKENS = 180               # fallback if 402/429
MAX_RETRIES = 2                    # 1 retry with backoff + shrink
BACKOFF_SECS = 2.0                 # simple backoff

def clamp_text(s: str, max_chars=MAX_CHARS_PER_GRAPH) -> str:
    if s is None:
        return ""
    return s if len(s) <= max_chars else s[:max_chars]

def call_llm(messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.2):
    """
    Calls OpenRouter with retries on 402/429 (quota/rate).
    On retry, shrinks max_tokens and backs off.
    Returns: {"error": False, "data": <json>} | {"error": True, "warning": "..."}
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "PDF Extractor",
    }

    attempt = 0
    current_max_tokens = max_tokens

    while attempt <= MAX_RETRIES:
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": current_max_tokens,
            "temperature": temperature,
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code in (402, 429):
                # Retry once with a smaller output cap
                if attempt < MAX_RETRIES and current_max_tokens > MIN_MAX_TOKENS:
                    attempt += 1
                    current_max_tokens = max(MIN_MAX_TOKENS, current_max_tokens - 120)
                    time.sleep(BACKOFF_SECS * attempt)
                    continue
                return {"error": True, "warning": f"{r.status_code} from OpenRouter: {r.text}"}

            r.raise_for_status()
            return {"error": False, "data": r.json()}

        except requests.RequestException as e:
            # Retry on transient errors
            if attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(BACKOFF_SECS * attempt)
                continue
            return {"error": True, "warning": f"LLM request failed: {e}"}
