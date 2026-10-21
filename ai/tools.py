"""
Tools the "Ask Prakriti" copilot may call. Safety lives here, in code, not in the prompt:

- The data is loaded into an in-memory DuckDB database, then external access is switched OFF and
  the configuration is locked. SQL can no longer read any file (e.g. sealed/truth.json) or write anything.
- Only a single SELECT / WITH statement is allowed; write and admin keywords are refused before execution.
- Results are capped at 500 rows.
- Every tool call is counted, so the evaluation can prove zero write queries were executed.
"""

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated" / "full"
MODEL = ROOT / "model"
CHANNELS = ["meta", "qc_ads", "search", "trade_promo", "youtube", "influencer"]
LABEL = dict(meta="Meta", qc_ads="Quick-commerce ads", search="Google Search", trade_promo="Trade promotions",
             youtube="YouTube", influencer="Influencers")
MAX_ROWS = 500
BLOCKED = re.compile(r"\b(insert|update|delete|drop|create|alter|copy|attach|detach|pragma|install|load|export|import|"
                     r"call|set|reset|checkpoint|vacuum|truncate|grant|revoke|replace)\b|read_\w+\s*\(|glob\s*\(", re.I)


class DataStore:
    def __init__(self):
        self.con = duckdb.connect(":memory:")
        self.tables = {}
        for name in ["weekly_geo", "platform_reported", "customers", "orders", "influencers", "cro_sessions"]:
            df = pd.read_csv(DATA / f"{name}.csv")
            self.con.register(f"_{name}", df)
            self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
            self.con.unregister(f"_{name}")
            self.tables[name] = list(df.columns)
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self.stats = {"sql_calls": 0, "sql_blocked": 0, "sql_errors": 0}

    def snapshot(self):
        """Row count and a checksum per table: compare before/after to prove nothing was changed."""
        out = {}
        for t in self.tables:
            n, h = self.con.execute(f"SELECT COUNT(*), SUM(HASH(t)) FROM {t} t").fetchone()
            out[t] = (n, h)
        return out

    def schema(self):
        return {t: cols for t, cols in self.tables.items()}

    def run_sql(self, query: str):
        self.stats["sql_calls"] += 1
        q = query.strip().rstrip(";").strip()
        q_nocomment = re.sub(r"--[^\n]*|/\*.*?\*/", " ", q, flags=re.S)
        if ";" in q_nocomment or not re.match(r"^\s*(select|with)\b", q_nocomment, re.I) or BLOCKED.search(q_nocomment):
            self.stats["sql_blocked"] += 1
            return {"error": "Refused: only a single read-only SELECT query on the listed tables is allowed."}
        try:
            df = self.con.execute(q).fetch_df()
        except Exception as e:
            self.stats["sql_errors"] += 1
            return {"error": f"SQL error: {str(e)[:300]}"}
        truncated = len(df) > MAX_ROWS
        df = df.head(MAX_ROWS)
        return {"rows": json.loads(df.to_json(orient="records", double_precision=6)), "row_count": len(df),
                "truncated": truncated}


def get_channel_roi(channel: str = "all"):
    """MMM v2 (frozen) estimates: ROI, 90% interval, marginal ROI, contribution, spend (weeks 1–96)."""
    t = pd.read_csv(MODEL / "mmm_v2_roi.csv", index_col=0)[["roi", "roi_lo", "roi_hi", "mroi", "contribution_cr", "spend_cr"]]
    t = t.round(4)
    if channel != "all":
        name = LABEL.get(channel, channel)
        if name not in t.index:
            return {"error": f"Unknown channel. Use one of {CHANNELS}."}
        t = t.loc[[name]]
    return {"source": "MMM v2, weeks 1–96, ₹ per ₹1; contribution and spend in ₹ Cr", "rows": t.reset_index().to_dict("records")}


def get_response_curve(channel: str, weekly_spend_lakh: float):
    """MMM v2 median extra weekly revenue (₹ lakh) at a steady weekly spend level."""
    if channel not in CHANNELS:
        return {"error": f"Unknown channel. Use one of {CHANNELS}."}
    d = np.load(MODEL / "mmm_v2_posterior_params.npz", allow_pickle=True)
    i = list(d["channels"]).index(channel)
    u = max(float(weekly_spend_lakh), 0) / d["channel_scale"][i]
    s, k = d["slope"][:, i], d["kappa"][:, i]
    rev = d["beta"][:, i] * float(d["target_scale"]) * u**s / (u**s + k**s)
    return {"channel": channel, "weekly_spend_lakh": weekly_spend_lakh, "extra_weekly_revenue_lakh_median": round(float(np.median(rev)), 4),
            "p5": round(float(np.quantile(rev, .05)), 4), "p95": round(float(np.quantile(rev, .95)), 4)}


def get_reports(name: str):
    """Saved results: budget recommendation, experiment, CLV, validation scorecard."""
    files = {"budget_recommendation": MODEL / "optimiser_recommendation.json", "experiment": MODEL / "experiment_result.json",
             "clv": MODEL / "clv_by_channel.csv", "validation": ROOT / "reports" / "validation_scorecard.csv"}
    if name not in files:
        return {"error": f"Unknown report. Use one of {list(files)}."}
    f = files[name]
    return json.loads(f.read_text()) if f.suffix == ".json" else {"rows": pd.read_csv(f).round(4).to_dict("records")}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only SQL SELECT on the company data (DuckDB dialect). Max 500 rows returned.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_channel_roi",
        "description": "Model estimates per marketing channel (MMM v2): ROI, 90% interval, marginal ROI, contribution, spend.",
        "parameters": {"type": "object", "properties": {"channel": {"type": "string", "enum": CHANNELS + ["all"]}}, "required": ["channel"]}}},
    {"type": "function", "function": {
        "name": "get_response_curve",
        "description": "Model's extra weekly revenue (₹ lakh) for a channel at a steady weekly spend (₹ lakh).",
        "parameters": {"type": "object", "properties": {"channel": {"type": "string", "enum": CHANNELS},
                                                         "weekly_spend_lakh": {"type": "number"}},
                       "required": ["channel", "weekly_spend_lakh"]}}},
    {"type": "function", "function": {
        "name": "get_reports",
        "description": "Saved results: budget_recommendation (₹ Cr by channel), experiment (QC ads geo test), clv (by acquisition channel), validation (scorecard).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": ["budget_recommendation", "experiment", "clv", "validation"]}},
                       "required": ["name"]}}},
]


def make_dispatcher(store: DataStore):
    fns = {"run_sql": lambda a: store.run_sql(a.get("query", "")),
           "get_channel_roi": lambda a: get_channel_roi(a.get("channel", "all")),
           "get_response_curve": lambda a: get_response_curve(a.get("channel", ""), a.get("weekly_spend_lakh", 0)),
           "get_reports": lambda a: get_reports(a.get("name", ""))}

    def dispatch(name, args_json):
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return {"error": "Arguments were not valid JSON."}
        if name not in fns:
            return {"error": f"Unknown tool {name}."}
        return fns[name](args)
    return dispatch
