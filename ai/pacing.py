"""
AI6: a weekly budget-pacing agent that may only PROPOSE; a code-based policy decides what happens.

Autonomy policy (enforced in Python, not in the prompt):
  Level 1  AUTO       shift ≤ ₹2 lakh/week, both channels stay within ±50% of plan  -> executed and logged
  Level 2  ESCALATE   shift > ₹2 and ≤ ₹10 lakh/week                                 -> needs CFO approval
  Level 3  BLOCKED    shift > ₹10 lakh, total budget change, bounds breached,
                      unknown channel or malformed proposal                          -> refused
  Kill switch         data feed not OK                                               -> nothing executes ("hold")
Every week writes one audit-log row, whatever the outcome.
"""

import json

from llm_client import chat

CHANNELS = ["meta", "qc_ads", "search", "trade_promo", "youtube", "influencer"]
AUTO_LIMIT, ESCALATE_LIMIT = 2.0, 10.0

SYSTEM = """You are a budget-pacing assistant for Prakriti Foods. Each week you read a pacing report and may propose ONE
budget shift between channels for next week, or hold. You cannot execute anything: a policy engine decides.
Strategy agreed by the CFO: the frozen plan (weekly ₹ lakh) below. Prefer small, evidence-based adjustments.
Reply with ONLY JSON: {"action": "shift" or "hold", "from": "<channel>", "to": "<channel>", "amount_lakh": <number>, "reason": "<one sentence>"}
Channels: meta, qc_ads, search, trade_promo, youtube, influencer."""


def policy(proposal, budget_now, lo, hi, data_ok, approver):
    if not data_ok:
        return "HOLD (kill switch: data feed not OK)", budget_now
    if not isinstance(proposal, dict) or proposal.get("action") not in ("shift", "hold"):
        return "BLOCKED (malformed proposal)", budget_now
    if proposal["action"] == "hold":
        return "NO CHANGE (agent chose hold)", budget_now
    f, t = proposal.get("from"), proposal.get("to")
    try:
        amt = float(proposal.get("amount_lakh", 0))
    except (TypeError, ValueError):
        return "BLOCKED (amount not a number)", budget_now
    if f not in CHANNELS or t not in CHANNELS or f == t or amt <= 0:
        return "BLOCKED (invalid channels or amount)", budget_now
    new = dict(budget_now); new[f] -= amt; new[t] += amt
    if new[f] < lo[f] - 1e-9 or new[t] > hi[t] + 1e-9:
        return "BLOCKED (would breach ±50% bounds)", budget_now
    if amt > ESCALATE_LIMIT:
        return f"BLOCKED (₹{amt:.1f} lakh exceeds ₹{ESCALATE_LIMIT:.0f} lakh limit)", budget_now
    if amt > AUTO_LIMIT:
        ok, why = approver(f, t, amt)
        return (f"ESCALATED -> APPROVED by CFO ({why})", new) if ok else (f"ESCALATED -> REJECTED by CFO ({why})", budget_now)
    return "AUTO-EXECUTED (within Level 1 limit)", new


def run_pacing(weekly_reports, plan_weekly, lo, hi, approver, pinned="groq", run_id="pacing"):
    budget = dict(plan_weekly)
    log = []
    for rep in weekly_reports:
        msgs = [{"role": "system", "content": SYSTEM + f"\nFrozen plan (₹ lakh/week): {json.dumps({c: round(v, 2) for c, v in plan_weekly.items()})}"},
                {"role": "user", "content": json.dumps({**rep, "current_budget_lakh": {c: round(v, 2) for c, v in budget.items()}})}]
        msg, _, _ = chat(msgs, pinned=pinned, run_id=f"{run_id}-w{rep['week']}", max_tokens=800)
        text = msg.get("content") or ""
        a, b = text.find("{"), text.rfind("}")
        try:
            prop = json.loads(text[a:b + 1]) if a >= 0 else None
        except json.JSONDecodeError:
            prop = None
        decision, budget = policy(prop, budget, lo, hi, rep["data_status"] == "ok", approver)
        log.append(dict(week=rep["week"], data_status=rep["data_status"], note=rep.get("note", ""),
                        proposal=json.dumps(prop)[:300], decision=decision, total_after=round(sum(budget.values()), 4),
                        **{f"{c}_after": round(budget[c], 3) for c in CHANNELS}))
    return log
