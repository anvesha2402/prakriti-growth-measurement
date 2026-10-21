"""
One LLM client for every agent in the project (AI4 copilot, AI5 negotiation, AI6 pacing).

- Every provider speaks the OpenAI-compatible chat format, so one SDK covers all.
- Free-tier friendly: requests are spaced out (MIN_INTERVAL_S) and rate-limit errors (429) are
  retried with growing waits before moving to the next provider.
- Every call is appended to a CSV log (provider, model, tokens, latency) for the evaluation pack.
- Answers are cached on disk, so re-running a notebook never spends quota twice.
- For GRADED runs pass pinned="groq" so every result comes from one model.
- set_backend(fn) swaps in a stand-in model (used to test the plumbing without any API key).

Keys come from environment variables (in Colab: Colab secrets). Never hard-code keys.
"""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

PROVIDERS = [
    # name,      base_url,                                                   key env var,        model
    ("groq",     "https://api.groq.com/openai/v1",                           "GROQ_API_KEY",     "openai/gpt-oss-120b"),
    ("gemini",   "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",   "gemini-2.5-flash"),
    ("mistral",  "https://api.mistral.ai/v1",                                "MISTRAL_API_KEY",  "mistral-small-latest"),
    ("cerebras", "https://api.cerebras.ai/v1",                               "CEREBRAS_API_KEY", "gpt-oss-120b"),
    ("ollama",   "http://localhost:11434/v1",                                "USE_OLLAMA",       "qwen2.5:7b"),
]
MIN_INTERVAL_S = float(os.environ.get("LLM_MIN_INTERVAL_S", "2.2"))   # 30 requests/minute on Groq's free tier
TRACE_PATH = Path(os.environ.get("LLM_TRACE_PATH", "ai/eval/llm_calls.csv"))
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", "ai/eval/cache"))
_backend = None          # optional stand-in model: fn(messages, tools, temperature) -> message dict
_last_call = [0.0]


def set_backend(fn):
    """Use a stand-in model (for tests). Pass None to go back to real providers."""
    global _backend
    _backend = fn


def _clean(messages):
    """Keep only the fields every provider accepts (drops reasoning/annotations/refusal etc.)."""
    out = []
    for m in messages:
        c = {"role": m["role"], "content": m.get("content") if m.get("content") is not None else ""}
        if m.get("tool_calls"):
            c["tool_calls"] = [{"id": t["id"], "type": "function",
                                "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                               for t in m["tool_calls"]]
        if m["role"] == "tool":
            c["tool_call_id"] = m["tool_call_id"]
        out.append(c)
    return out


def _recover_final_answer(err, tools):
    """gpt-oss sometimes emits its FINAL answer as a call to a tool named 'json' (not in our tool list); Groq then
    answers 400 tool_use_failed and includes what the model wrote. If the call was to an unlisted tool and its
    arguments are valid JSON, treat the arguments as the model's plain-text final answer. Real tool names are never
    recovered this way, and nothing is invented: the content is exactly what the model produced."""
    body = getattr(err, "body", None)
    err_obj = body.get("error", body) if isinstance(body, dict) else {}
    if err_obj.get("code") != "tool_use_failed" or not err_obj.get("failed_generation"):
        return None
    try:
        gen = json.loads(err_obj["failed_generation"])
    except (json.JSONDecodeError, TypeError):
        return None
    listed = {t["function"]["name"] for t in (tools or [])}
    if not isinstance(gen, dict) or gen.get("name") in listed or not isinstance(gen.get("arguments"), dict):
        return None
    return json.dumps(gen["arguments"])


def _available():
    from openai import OpenAI
    out = []
    for name, url, key_env, model in PROVIDERS:
        if os.environ.get(key_env):
            key = "ollama" if name == "ollama" else os.environ[key_env]
            out.append((name, OpenAI(base_url=url, api_key=key, max_retries=0, timeout=90), model))
    return out


def _log(row):
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not TRACE_PATH.exists()
    with TRACE_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def chat(messages, tools=None, pinned=None, use_cache=True, temperature=0, run_id="", max_tokens=1500):
    """Return (message_dict, provider_name, usage_dict)."""
    messages = _clean(messages)
    key = hashlib.sha256(json.dumps([messages, tools, pinned, temperature, run_id if temperature else ""],
                                    sort_keys=True, default=str).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        c = json.loads(cache_file.read_text())
        return c["message"], c["provider"] + " (cache)", c.get("usage", {})

    if _backend is not None:
        msg = _backend(messages, tools, temperature)
        return msg, "stand-in", {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0}

    from openai import APIError, APIStatusError, RateLimitError
    pool = _available()
    if pinned:
        pool = [p for p in pool if p[0] == pinned]
    if not pool:
        raise RuntimeError("No LLM provider available: add GROQ_API_KEY to Colab secrets (or set USE_OLLAMA=1).")

    last_err = None
    for name, client, model in pool:
        wait = 10
        for attempt in range(6):
            gap = time.time() - _last_call[0]
            if gap < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - gap)
            t0 = time.time()
            _last_call[0] = t0
            try:
                kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                u = resp.usage
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0),
                         "latency_s": round(time.time() - t0, 2)}
                _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id, "provider": name, "model": model, **usage})
                m = resp.choices[0].message
                msg = {"role": "assistant", "content": m.content,
                       "tool_calls": [{"id": t.id, "type": "function",
                                       "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                      for t in (m.tool_calls or [])]}
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"message": msg, "provider": name, "usage": usage}))
                return msg, name, usage
            except RateLimitError as e:          # free-tier limit: wait and retry, longer each time
                last_err = e
                time.sleep(wait)
                wait = min(wait * 2, 60)
            except (APIStatusError, APIError) as e:
                last_err = e
                rec = _recover_final_answer(e, tools)
                if rec is not None:              # model "called" a non-existent tool such as `json` to hand in its answer
                    msg = {"role": "assistant", "content": rec, "tool_calls": []}
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": round(time.time() - t0, 2)}
                    _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id, "provider": name, "model": model,
                          **usage, "note": "recovered_json_tool_call"})
                    if use_cache:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps({"message": msg, "provider": name, "usage": usage}))
                    return msg, name, usage
                break
    raise RuntimeError(f"All providers failed. Last error: {str(last_err)[:300]}")


def check_providers():
    """Ping each configured provider with a one-word prompt; print what works."""
    for name, client, model in _available():
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with the word OK."}],
                                               max_tokens=20)
            print(f"OK    {name:9s} {model}: {r.choices[0].message.content!r}")
        except Exception as e:
            print(f"FAIL  {name:9s} {model}: {type(e).__name__}: {str(e)[:160]}")
