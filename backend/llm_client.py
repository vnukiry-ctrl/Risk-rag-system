import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM PROVIDER - EASY SWAP HERE
# ============================================================
# Current: Groq (free tier, fast, OpenAI-compatible) - used during
# development. Switch LLM_PROVIDER to "anthropic" after deployment
# to move to Claude; every caller in this codebase goes through
# llm_complete() below, so that's the only change needed.

LLM_PROVIDER = "groq"  # Options: "groq" or "anthropic"

if LLM_PROVIDER == "groq":
    from openai import OpenAI

    MODEL = "openai/gpt-oss-120b"
    _client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    def llm_complete(messages: List[Dict], temperature: float = 0, max_tokens: int = 1500) -> str:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

elif LLM_PROVIDER == "anthropic":
    # TODO: TO USE CLAUDE INSTEAD:
    # 1. pip install anthropic
    # 2. Set ANTHROPIC_API_KEY in .env
    # 3. Change LLM_PROVIDER = "anthropic" above
    import anthropic

    MODEL = "claude-sonnet-4-5"
    _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def llm_complete(messages: List[Dict], temperature: float = 0, max_tokens: int = 1500) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        response = _client.messages.create(
            model=MODEL,
            system=system,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text

else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
