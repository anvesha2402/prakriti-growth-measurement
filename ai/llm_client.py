"""
One LLM client for every agent in the project (AI4 copilot, AI5 negotiation, AI6 pacing).

- Every provider below speaks the OpenAI-compatible chat format, so one SDK covers all.
- Providers are tried in order; on a rate-limit (429) or server error the call waits
  briefly and falls through to the next provider.
- Every call is appended to a CSV trace log (provider, model, tokens, latency).
- Answers are cached on disk so the same question never spends quota twice.
- For GRADED runs, pass pinned="<provider>" so every result comes from one model.

Keys are read from environment variables (in Colab: load them from Colab secrets
into os.environ first — see notebooks/00_setup_check.ipynb). Never hard-code keys.
Model names change often: run check_providers() and edit PROVIDERS if one is retired.
"""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

from openai import APIError, APIStatusError, OpenAI, RateLimitError

PROVIDERS = [
    # name,      base_url,                                                   key env var,        model
    ("mistral",  "https://api.mistral.ai/v1",                                "MISTRAL_API_KEY",  "mistral-small-latest"),
    ("cerebras", "https://api.cerebras.ai/v1",                               "CEREBRAS_API_KEY", "gpt-oss-120b"),
    ("groq",     "https://api.groq.com/openai/v1",                           "GROQ_API_KEY",     "openai/gpt-oss-120b"),
    ("gemini",   "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",   "gemini-2.5-flash"),
    ("ollama",   "http://localhost:11434/v1",                                None,               "qwen2.5:7b"),
]

TRACE_PATH = Path(os.environ.get("LLM_TRACE_PATH", "ai/eval/llm_calls.csv"))
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", "ai/eval/cache"))


def _available():
    out = []
    for name, url, key_env, model in PROVIDERS:
        if key_env is None:
            out.append((name, OpenAI(base_url=url, api_key="ollama"), model))
        elif os.environ.get(key_env):
            out.append((name, OpenAI(base_url=url, api_key=os.environ[key_env]), model))
    return out


def _log(row):
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not TRACE_PATH.exists()
    with TRACE_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def chat(messages, tools=None, pinned=None, use_cache=True, temperature=0, run_id=""):
    """Return (message, provider_name). `message` is the OpenAI-style response message."""
    key = hashlib.sha256(json.dumps([messages, tools, pinned], sort_keys=True, default=str).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return cached["message"], cached["provider"] + " (cache)"

    pool = _available()
    if pinned:
        pool = [p for p in pool if p[0] == pinned]
    if not pool:
        raise RuntimeError("No LLM provider available: set an API key or start Ollama.")

    last_err = None
    for name, client, model in pool:
        for attempt in range(2):
            t0 = time.time()
            try:
                kwargs = dict(model=model, messages=messages, temperature=temperature)
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                usage = resp.usage
                _log({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id, "provider": name,
                    "model": model, "prompt_tokens": getattr(usage, "prompt_tokens", ""),
                    "completion_tokens": getattr(usage, "completion_tokens", ""),
                    "latency_s": round(time.time() - t0, 2),
                })
                msg = resp.choices[0].message.model_dump()
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"message": msg, "provider": name}))
                return msg, name
            except RateLimitError as e:
                last_err = e
                time.sleep(20 if attempt == 0 else 0)   # wait out a per-minute window once, then move on
            except (APIStatusError, APIError) as e:
                last_err = e
                break
    raise RuntimeError(f"All providers failed. Last error: {last_err}")


def check_providers():
    """Ping each configured provider with a one-word prompt; print what works."""
    for name, client, model in _available():
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with the word OK."}], max_tokens=5
            )
            print(f"OK    {name:9s} {model}: {r.choices[0].message.content!r}")
        except Exception as e:
            print(f"FAIL  {name:9s} {model}: {type(e).__name__}: {str(e)[:120]}")
