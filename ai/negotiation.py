"""
AI5: three AI roles negotiate next year's ₹24 Cr budget; a Python referee enforces the rules.

Roles: Analyst (proposes), Head of Growth (pushes for growth), CFO (decides). Up to 4 rounds.
Referee (code, not AI): total must equal ₹24 Cr (±0.05), every channel within ±50% of today, valid JSON.
Experiment: identical negotiations with two information sets.
- "dashboard": the platform-reported ROAS a typical team sees.
- "evidence": MMM v2 + geo experiment + CLV results from this project.
Every final allocation is then valued with the TRUE response curves (answer key) to see which
information leads to better decisions.
"""

import json
import re

from llm_client import chat

CHANNELS = ["meta", "qc_ads", "search", "trade_promo", "youtube", "influencer"]
BUDGET = 24.0
ROLES = {
    "analyst": "You are the marketing Analyst at Prakriti Foods. Propose the allocation you believe maximises long-term "
               "contribution, using only the information given.",
    "growth": "You are the Head of Growth at Prakriti Foods. You care most about protecting revenue growth and new customers. "
              "Critique the latest proposal and propose your own allocation.",
    "cfo": "You are the CFO of Prakriti Foods. You must hit the ₹24 Cr cap and maximise contribution after marketing. "
           "Weigh both proposals and decide. You may set ACCEPT to true to end the negotiation.",
}
FORMAT = ('Reply with ONLY a JSON object: {"allocation_cr": {"meta": x, "qc_ads": x, "search": x, "trade_promo": x, '
          '"youtube": x, "influencer": x}, "reason": "<one sentence>"ACCEPT}  (x = ₹ Cr per year)')


def referee(alloc, current_cr):
    errs = []
    if not isinstance(alloc, dict) or set(alloc) != set(CHANNELS):
        return [f"allocation must have exactly these keys: {CHANNELS}"]
    try:
        vals = {c: float(alloc[c]) for c in CHANNELS}
    except (TypeError, ValueError):
        return ["all values must be numbers (₹ Cr)"]
    if abs(sum(vals.values()) - BUDGET) > 0.05:
        errs.append(f"total is ₹{sum(vals.values()):.2f} Cr; it must equal ₹{BUDGET} Cr")
    for c in CHANNELS:
        lo, hi = 0.5 * current_cr[c], 1.5 * current_cr[c]
        if not lo - 1e-6 <= vals[c] <= hi + 1e-6:
            errs.append(f"{c} ₹{vals[c]:.2f} Cr is outside its allowed range ₹{lo:.2f}–{hi:.2f} Cr (±50% of today)")
    return errs


def _parse(text):
    s = text or ""
    a, b = s.find("{"), s.rfind("}")
    try:
        return json.loads(s[a:b + 1]) if a >= 0 else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def negotiate(brief, current_cr, run_id, pinned="groq", rounds=4, temperature=0.7):
    rules = (f"Rules enforced by a referee: the total must be exactly ₹{BUDGET} Cr; each channel must stay within ±50% of "
             f"today's spend. Today's spend (₹ Cr/yr): {json.dumps(current_cr)}.")
    transcript, last_valid, calls = [], None, 0
    for r in range(1, rounds + 1):
        for role in ["analyst", "growth", "cfo"]:
            fmt = FORMAT.replace("ACCEPT", ', "accept": true or false' if role == "cfo" else "")
            history = "\n".join(f"[Round {t['round']}] {t['role'].upper()}: {json.dumps(t['proposal'])} — {t['reason']}"
                                + (f" | REFEREE: {'; '.join(t['errors'])}" if t["errors"] else "") for t in transcript[-6:])
            msgs = [{"role": "system", "content": f"{ROLES[role]}\n\n{rules}\n\nInformation available:\n{brief}\n\n{fmt}"},
                    {"role": "user", "content": f"Discussion so far:\n{history or '(none yet)'}\n\nRound {r}: your move."}]
            msg, _, _ = chat(msgs, pinned=pinned, temperature=temperature, run_id=f"{run_id}-{r}-{role}", max_tokens=1200)
            calls += 1
            out = _parse(msg.get("content"))
            alloc = out.get("allocation_cr", {})
            errs = referee(alloc, current_cr)
            transcript.append(dict(round=r, role=role, proposal=alloc, reason=str(out.get("reason", ""))[:200], errors=errs))
            if role == "cfo" and not errs:
                last_valid = {c: float(alloc[c]) for c in CHANNELS}
                if out.get("accept") is True:
                    return dict(final=last_valid, rounds=r, accepted=True, transcript=transcript, calls=calls)
    return dict(final=last_valid, rounds=rounds, accepted=False, transcript=transcript, calls=calls)
