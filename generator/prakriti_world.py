"""
Prakriti Foods — synthetic world generator
==========================================

Creates two years (104 weeks) of pretend data for a fictional Indian D2C food and
beverage brand, with the TRUE effect of every marketing channel hidden inside it.

Plain-language summary of how one week of sales is made, for each of 8 cities:

    sales = baseline demand  (what would sell with zero ads: trend, season,
                              weather, festivals, price, dark stores, competitors)
          + the true effect of each of 6 ad channels
            (each channel's spend is "remembered" for some weeks  -> adstock
             and gives less and less extra sales as spend rises   -> Hill saturation)
          + random noise

The true numbers (how strong each channel really is) are drawn AT RANDOM from
documented ranges (RANGES below), so nobody knows the exact answer. They are saved to
sealed/truth.json, never printed, and only opened at grading time. The file's SHA-256
fingerprint is published first, which proves later that the answers were not edited.

Two-stage design (like a real company):
  stage 1 -> weeks 1-87 are released. You study them and design a geo experiment.
  stage 2 -> weeks 88-104 are simulated under YOUR experiment design.
Every random number is tied to a fixed (seed, table, week, city) address, so weeks 1-87
come out identical in both stages. The continuity check proves this.

Units: money in ₹ lakh (1 lakh = ₹100,000; 100 lakh = ₹1 Cr) unless a column says _inr.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "1.0.0"

# --------------------------------------------------------------------------------------
# 1. FIXED, PUBLIC SET-UP (the "story" — anyone may read this)
# --------------------------------------------------------------------------------------
N_WEEKS = 104
STAGE1_WEEKS = 87
START = pd.Timestamp("2024-10-07")            # Monday of week 1 (Oct 2024 -> Oct 2026)

# name, market weight, newer market?, mean temp °C, seasonal temp swing °C,
# monsoon cooling °C, dark stores at start, dark stores at end
GEOS = [
    ("Delhi NCR", 0.20, False, 25.5, 9.0, 3.0, 140, 260),
    ("Mumbai",    0.18, False, 27.5, 2.5, 2.0, 120, 230),
    ("Bengaluru", 0.17, False, 24.0, 2.5, 1.5, 110, 220),
    ("Hyderabad", 0.10, False, 27.0, 4.5, 2.5,  60, 130),
    ("Chennai",   0.10, False, 29.0, 3.0, 1.0,  55, 120),
    ("Pune",      0.09, True,  25.5, 3.5, 2.5,  35,  95),
    ("Kolkata",   0.08, True,  27.0, 5.0, 2.0,  30,  85),
    ("Ahmedabad", 0.08, True,  27.5, 6.0, 2.5,  25,  80),
]
GEO_NAMES = [g[0] for g in GEOS]
GEO_W = np.array([g[1] for g in GEOS])
NEWER = np.array([g[2] for g in GEOS])

CHANNELS = ["meta", "search", "youtube", "qc_ads", "influencer", "trade_promo"]
SPEND_SHARE = dict(meta=0.25, qc_ads=0.25, search=0.15, trade_promo=0.13, youtube=0.12, influencer=0.10)

YEAR2_REVENUE_LAKH = 12_000.0     # about ₹120 Cr in weeks 53-104
GROWTH = 0.45                     # about 45% a year (target for total revenue)
BASE_GROWTH = 0.62                # baseline demand grows faster because media effects saturate (tuned)
MARKETING_PCT = 0.28              # marketing about 28% of revenue
WEBSITE_SAMPLE_RATE = 0.125       # customers.csv / orders.csv hold a 1-in-8 random sample
FIRST_ORDER_SHARE = 0.41          # share of website revenue that comes from first orders (tuned)
AOV_FIRST_INR = 600               # used only to turn website revenue into new-customer counts
CM2_RATE = 0.44                   # contribution after product, fulfilment, payment, returns
FESTIVE_WEEKS = {1, 3, 4, 51, 52, 54, 55}   # Navratri / Diwali weeks in Oct 2024 and Sep-Oct 2025

# --------------------------------------------------------------------------------------
# 2. RANGES THE HIDDEN TRUTH IS DRAWN FROM (public; the exact draw is secret)
#    roi            = true incremental revenue per ₹1 of spend, weeks 1-87
#    half_life      = weeks until an ad's leftover effect halves (adstock)
#    k_rel          = spend (relative to the city's usual spend) where the channel gives
#                     half its maximum effect; low k_rel = already saturated
#    s              = curve shape; above 1 = S-shaped
#    platform_bias  = how much the platform over/under-reports its own sales
# --------------------------------------------------------------------------------------
RANGES = {
    "meta":        dict(roi=(1.6, 2.2), half_life=(1.0, 2.0), k_rel=(0.35, 0.60), s=(1.0, 1.5),
                        platform_bias=(2.2, 2.8)),
    "search":      dict(roi=(0.8, 1.2), half_life=(0.3, 0.8), k_rel=(0.8, 1.2), s=(1.0, 1.4),
                        platform_bias=(2.8, 3.6)),
    "youtube":     dict(roi=(2.0, 2.8), half_life=(4.0, 6.0), k_rel=(1.5, 2.5), s=(1.6, 2.4),
                        platform_bias=(0.5, 0.8)),
    "qc_ads":      dict(roi=(1.2, 1.6), half_life=(0.2, 0.6), k_rel=(0.6, 1.0), s=(1.0, 1.3),
                        cannibalisation=(0.50, 0.70), multi_touch=(1.00, 1.15)),
    "influencer":  dict(roi=(1.4, 2.0), half_life=(2.0, 3.0), k_rel=(1.0, 1.6), s=(1.0, 1.5),
                        platform_bias=(1.2, 1.6), newer_geo_mult=(1.3, 1.6)),
    "trade_promo": dict(roi=(1.0, 1.4), half_life=(0.2, 0.6), k_rel=(0.8, 1.2), s=(1.0, 1.2),
                        pull_forward=(0.30, 0.50), platform_bias=(1.10, 1.40)),
}
# How website customers behave, by the channel that first brought them (BG/NBD style).
#   lam  = repeat purchases per week while "alive"     p = chance of stopping after each repeat
#   aov  = typical order value ₹                       disc = usual discount share
#   yield_ = new customers per ₹ of website sales the channel creates (relative to organic)
CUSTOMER_RANGES = {
    "organic":     dict(lam=(0.034, 0.044), p=(0.25, 0.32), aov=(630, 680), disc=(0.04, 0.06), yield_=(1.0, 1.0)),
    "meta":        dict(lam=(0.030, 0.039), p=(0.30, 0.36), aov=(610, 660), disc=(0.07, 0.09), yield_=(0.9, 1.1)),
    "search":      dict(lam=(0.034, 0.044), p=(0.26, 0.32), aov=(640, 690), disc=(0.04, 0.06), yield_=(0.9, 1.1)),
    "youtube":     dict(lam=(0.037, 0.046), p=(0.22, 0.28), aov=(650, 700), disc=(0.04, 0.06), yield_=(0.7, 0.9)),
    "influencer":  dict(lam=(0.044, 0.053), p=(0.17, 0.23), aov=(670, 720), disc=(0.05, 0.07), yield_=(1.1, 1.3)),
    "trade_promo": dict(lam=(0.023, 0.030), p=(0.40, 0.50), aov=(540, 590), disc=(0.16, 0.22), yield_=(1.4, 1.7)),
}
ACQ = list(CUSTOMER_RANGES)
OTHER_RANGES = dict(
    view_through=(1.50, 1.75),     # digital platforms double-count view-through / cross-device sales
    cro_base_cr=(0.026, 0.034),    # checkout conversion rate, 3-step checkout
    cro_lift_abs=(0.0015, 0.0045), # true absolute lift of 1-page checkout
)
INFLUENCER_FIRST_WEEK, N_CREATORS, N_FORTNIGHTS = 35, 150, 26   # fortnights cover weeks 35-86
CRO_FIRST_WEEK, CRO_WEEKS, CRO_SESSIONS = 75, 4, 120_000


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _rng(seed, *address):
    """A random-number stream tied to an address, so the same cell always gets the same draw."""
    return np.random.default_rng([int(seed), *[int(a) for a in address]])


def _geometric_adstock(x, lam):
    """A_t = x_t + lam * A_{t-1}; starts at steady state so week 1 is not artificially low."""
    a = np.empty_like(x, dtype=float)
    prev = x[..., 0] / (1 - lam)
    for t in range(x.shape[-1]):
        prev = x[..., t] + lam * prev
        a[..., t] = prev
    return a


def _hill(u, k, s):
    u = np.maximum(u, 0)
    return u**s / (u**s + k**s)


def calendar():
    t = np.arange(N_WEEKS)
    week_start = START + pd.to_timedelta(7 * t, unit="D")
    doy = (week_start + pd.Timedelta(days=3)).dayofyear.values
    festive = np.array([1 if (i + 1) in FESTIVE_WEEKS else 0 for i in t])
    summer = np.exp(-(((doy - 140) / 35.0) ** 2))      # peaks late May (Apr-Jun)
    monsoon = np.exp(-(((doy - 220) / 30.0) ** 2))     # peaks early Aug (Jul-Sep)
    return week_start, doy, festive, summer, monsoon


# --------------------------------------------------------------------------------------
# 3. DRAW THE HIDDEN TRUTH
# --------------------------------------------------------------------------------------
def draw_truth(seed):
    rng = _rng(seed, 1)
    ch = {}
    for c in CHANNELS:
        d = {k: float(rng.uniform(*v)) for k, v in RANGES[c].items()}
        d["lambda"] = 0.5 ** (1.0 / d["half_life"])
        ch[c] = d
    cust = {a: {k: float(rng.uniform(*v)) for k, v in r.items()} for a, r in CUSTOMER_RANGES.items()}
    other = {k: float(rng.uniform(*v)) for k, v in OTHER_RANGES.items()}
    return ch, cust, other


# --------------------------------------------------------------------------------------
# 4. BUILD THE WORLD
# --------------------------------------------------------------------------------------
def _environment(seed):
    """Weather, festivals, price, dark stores, competitors and baseline demand shape."""
    week_start, doy, festive, summer, monsoon = calendar()
    G, T = len(GEOS), N_WEEKS
    t = np.arange(T)
    r = _rng(seed, 2)
    mean_temp = np.array([g[3] for g in GEOS])[:, None]
    amp = np.array([g[4] for g in GEOS])[:, None]
    cool = np.array([g[5] for g in GEOS])[:, None]
    temp = mean_temp + amp * np.cos(2 * np.pi * (doy - 140) / 365.0) - cool * monsoon + r.normal(0, 1.2, (G, T))

    price = np.where(t >= 79, 1.07, np.where(t >= 39, 1.04, 1.0)) + r.normal(0, 0.005, T)

    d0 = np.array([g[6] for g in GEOS])[:, None]
    d1 = np.array([g[7] for g in GEOS])[:, None]
    dark = np.round(d0 + (d1 - d0) / (1 + np.exp(-(t - 52) / 18.0)))

    comp = 1 + r.normal(0, 0.02, (G, T))
    for g in r.choice(G, size=2, replace=False):       # a rival launches in 2 cities
        w0 = int(r.integers(20, 80))
        comp[g, w0:w0 + 10] += 0.20 * np.exp(-np.arange(min(10, T - w0)) / 6.0)

    season = 1 + 0.010 * (temp - mean_temp) + 0.20 * summer - 0.12 * monsoon + 0.08 * festive
    trend = (1 + BASE_GROWTH) ** ((t - 77.5) / 52.0)
    spend_trend = (1 + GROWTH) ** ((t - 77.5) / 52.0)
    dark_eff = (dark / dark.mean(axis=1, keepdims=True)) ** 0.12

    e = np.zeros((G, T))
    shock = r.normal(0, 0.035, (G, T))
    for k in range(T):
        e[:, k] = (0.3 * e[:, k - 1] if k else 0) + shock[:, k]
    base_unit = GEO_W[:, None] * trend * season * price**-1.0 * dark_eff * (1 - 0.30 * (comp - 1)) * np.exp(e)

    season_nat = (GEO_W[:, None] * (1 + 0.010 * (temp - mean_temp) + 0.20 * summer - 0.12 * monsoon + 0.08 * festive)).sum(0)
    return dict(week_start=week_start, festive=festive, temp=temp, price=price, dark=dark, comp=comp,
                base_unit=base_unit, trend=spend_trend, season_nat=season_nat)


def _spend_plan(seed, env):
    """Planned weekly spend per channel and city (₹ lakh), before any experiment."""
    T = N_WEEKS
    r = _rng(seed, 3)
    demand = env["trend"] * (1 + 0.5 * (env["season_nat"] - 1))

    def ar1(sigma, rho=0.6, eps=None):
        eps = r.normal(0, 1, T) if eps is None else eps
        z = np.zeros(T)
        for k in range(T):
            z[k] = (rho * z[k - 1] if k else 0) + sigma * eps[k]
        return np.exp(z)

    # Meta and YouTube move together in weeks 20-49 (the collinearity trap):
    # inside the window both follow one shared campaign rhythm plus a smaller private wobble,
    # which gives a correlation of about 0.8 whatever the random draw.
    def ar1_log(sigma, rho=0.6):
        eps, z = r.normal(0, 1, T), np.zeros(T)
        for k in range(T):
            z[k] = (rho * z[k - 1] if k else 0) + sigma * eps[k]
        return z
    win = (np.arange(T) >= 19) & (np.arange(T) <= 48)
    z_common, z_meta, z_yt = ar1_log(0.22), ar1_log(0.22), ar1_log(0.22)
    z_m_win, z_y_win = r.normal(0, 0.11, T), r.normal(0, 0.11, T)
    log_meta = np.where(win, z_common + z_m_win, z_meta)
    log_yt = np.where(win, z_common + z_y_win, z_yt)

    raw = {}
    raw["meta"] = demand * np.exp(log_meta)
    yt = demand * np.exp(log_yt)
    flights = np.ones(T)                                   # YouTube runs in campaign bursts
    k = 0
    while k < T:
        on, off = int(r.integers(4, 9)), int(r.integers(2, 5))
        flights[k + on:k + on + off] = 0.12
        k += on + off
    flights[win] = 1.0
    raw["youtube"] = yt * flights
    raw["search"] = demand * env["season_nat"] ** 1.5 * ar1(0.10)   # follows demand (a trap)
    raw["qc_ads"] = demand * (env["dark"].sum(0) / env["dark"].sum(0).mean()) ** 0.5 * ar1(0.15)
    infl = ar1(0.30)
    raw["influencer"] = demand * np.repeat(infl[::2], 2)[:T]       # booked per fortnight
    promo_on = (r.uniform(size=T) < 0.40) | (env["festive"] == 1)
    raw["trade_promo"] = demand * np.where(promo_on, 1.0, 0.15) * ar1(0.10)

    total_y2 = MARKETING_PCT * YEAR2_REVENUE_LAKH
    nat = {c: raw[c] * (SPEND_SHARE[c] * total_y2 / raw[c][52:].sum()) for c in CHANNELS}

    G = len(GEOS)
    dark_share = env["dark"] / env["dark"].sum(0, keepdims=True)
    spend = {}
    for c in CHANNELS:
        base = GEO_W[:, None] * np.ones((G, T))
        if c == "qc_ads":
            base = 0.5 * base + 0.5 * dark_share
        if c == "influencer":
            base = base * np.where(NEWER, 1.15, 1.0)[:, None]
        alloc = base * np.exp(r.normal(0, 0.08, (G, T)))
        alloc /= alloc.sum(0, keepdims=True)
        spend[c] = nat[c][None, :] * alloc
    return spend, promo_on.astype(int)


def _influencer_table(seed):
    """150 creators x 26 fortnights. 8-12% are fake-engagement accounts (flags are secret)."""
    r = _rng(seed, 4)
    n_fake = int(round(N_CREATORS * r.uniform(0.08, 0.12)))
    fake = np.zeros(N_CREATORS, bool)
    fake[r.choice(N_CREATORS, n_fake, replace=False)] = True
    tier = r.choice(["nano", "micro", "macro"], N_CREATORS, p=[0.45, 0.40, 0.15])
    f0 = np.where(tier == "nano", r.lognormal(np.log(15_000), 0.4, N_CREATORS),
          np.where(tier == "micro", r.lognormal(np.log(90_000), 0.4, N_CREATORS),
                   r.lognormal(np.log(600_000), 0.4, N_CREATORS)))
    er_mean = np.where(tier == "nano", 0.05, np.where(tier == "micro", 0.03, 0.015))
    # each fake shows 2-4 of the 4 tell-tale signs, so detection is not trivial
    tells = np.zeros((N_CREATORS, 4), bool)
    for i in np.where(fake)[0]:
        tells[i, r.choice(4, int(r.integers(2, 5)), replace=False)] = True
    rows = []
    followers = f0.copy()
    for f in range(N_FORTNIGHTS):
        growth = r.normal(0.01, 0.005, N_CREATORS)
        jump = tells[:, 0] & (r.uniform(size=N_CREATORS) < 0.25)
        growth = np.where(jump, r.uniform(0.15, 0.40, N_CREATORS), growth)
        followers = followers * (1 + growth)
        er = er_mean * r.lognormal(0, 0.25, N_CREATORS) * np.where(tells[:, 1], r.uniform(2, 3, N_CREATORS), 1)
        likes = followers * er
        c_ratio = np.where(tells[:, 1], r.uniform(0.005, 0.015, N_CREATORS), r.uniform(0.03, 0.06, N_CREATORS))
        generic = np.where(tells[:, 2], r.beta(12, 8, N_CREATORS), r.beta(3, 22, N_CREATORS))
        city = np.where(tells[:, 3], r.beta(4, 12, N_CREATORS), r.beta(11, 9, N_CREATORS))
        for i in range(N_CREATORS):
            rows.append((f"IN{i + 1:03d}", tier[i], f + 1, INFLUENCER_FIRST_WEEK + 2 * f, int(followers[i]),
                         int(likes[i]), int(likes[i] * c_ratio[i]), round(float(growth[i]), 4),
                         round(float(generic[i]), 3), round(float(city[i]), 3)))
    df = pd.DataFrame(rows, columns=["creator_id", "tier", "fortnight", "week_start_num", "followers", "likes",
                                     "comments", "follower_growth", "generic_comment_share", "audience_city_match"])
    weight = df["followers"] ** 0.8
    df["_w"] = weight / weight.groupby(df["fortnight"]).transform("sum")
    fake_ids = sorted(df["creator_id"].unique()[fake].tolist())
    return df, fake_ids


def build_world(seed, design=None, with_customers=True):
    """Return (tables, truth). design = {"channel": "meta", "treated_geos": [...], "weeks": [88, 95]}"""
    G, T = len(GEOS), N_WEEKS
    ch, cust, other = draw_truth(seed)
    env = _environment(seed)
    planned, promo_flag = _spend_plan(seed, env)

    infl_df, fake_ids = _influencer_table(seed)
    fake_spend_share = float(infl_df.loc[infl_df["creator_id"].isin(fake_ids), "_w"].sum() / N_FORTNIGHTS)

    actual = {c: planned[c].copy() for c in CHANNELS}
    if design:
        gi = [GEO_NAMES.index(g) for g in design["treated_geos"]]
        w0, w1 = design["weeks"]
        actual[design["channel"]][np.ix_(gi, np.arange(w0 - 1, w1))] = 0.0

    def media_units(spend):
        """Channel effect shapes, before scaling by beta."""
        out = {}
        for c in CHANNELS:
            p = ch[c]
            x = spend[c] * ((1 - fake_spend_share) if c == "influencer" else 1.0)
            xp = planned[c] * ((1 - fake_spend_share) if c == "influencer" else 1.0)
            norm = _geometric_adstock(xp, p["lambda"]).mean(axis=1, keepdims=True)
            h = _hill(_geometric_adstock(x, p["lambda"]) / norm, p["k_rel"], p["s"])
            mult = np.where(NEWER, p.get("newer_geo_mult", 1.0), 1.0)[:, None]
            gross = GEO_W[:, None] * mult * h
            if c == "trade_promo":
                pf = p["pull_forward"]
                lag1 = np.concatenate([gross[:, :1], gross[:, :-1]], axis=1)
                lag2 = np.concatenate([gross[:, :2], gross[:, :-2]], axis=1)
                out[c] = (gross - pf * (0.6 * lag1 + 0.4 * lag2), gross)
            else:
                out[c] = (gross, gross)
        return out

    units_planned = media_units(planned)
    beta = {}
    for c in CHANNELS:      # scale so that true ROI over weeks 1-87 equals the drawn ROI
        beta[c] = ch[c]["roi"] * planned[c][:, :STAGE1_WEEKS].sum() / units_planned[c][0][:, :STAGE1_WEEKS].sum()
    contrib_planned = {c: beta[c] * units_planned[c][0] for c in CHANNELS}
    media_y2 = sum(contrib_planned[c][:, 52:].sum() for c in CHANNELS)
    L = (YEAR2_REVENUE_LAKH - media_y2) / env["base_unit"][:, 52:].sum()

    units = media_units(actual)
    contrib = {c: beta[c] * units[c][0] for c in CHANNELS}
    gross_contrib = {c: beta[c] * units[c][1] for c in CHANNELS}
    baseline = L * env["base_unit"]
    revenue = baseline + sum(contrib.values())

    # ---- sales-channel split (quick commerce, marketplace, website, modern trade)
    r6 = _rng(seed, 6)
    s_web = r6.uniform(0.20, 0.24, G)[:, None]
    dk = env["dark"]
    s_qc = 0.34 + 0.12 * (dk - dk.min(1, keepdims=True)) / (dk.max(1, keepdims=True) - dk.min(1, keepdims=True))
    s_mp = 0.21
    brand = baseline + sum(contrib[c] for c in ["meta", "search", "youtube", "influencer"])
    rev_qc = s_qc * brand + contrib["qc_ads"] + 0.40 * contrib["trade_promo"]
    rev_mp = s_mp * brand + 0.05 * contrib["trade_promo"]
    rev_web = s_web * brand + 0.15 * contrib["trade_promo"]
    rev_mt = revenue - rev_qc - rev_mp - rev_web

    # ---- reported spend carries small measurement error (as real invoices do)
    r5 = _rng(seed, 5)
    measured = {c: actual[c] * np.exp(r5.normal(0, 0.03, (G, T))) for c in CHANNELS}

    # ---- website new customers, split by acquiring channel
    yields = {a: cust[a]["yield_"] for a in ACQ}
    wts = {"organic": s_web * baseline}
    for c in ["meta", "search", "youtube", "influencer"]:
        wts[c] = s_web * np.maximum(contrib[c], 0) * yields[c]
    wts["trade_promo"] = 0.15 * np.maximum(contrib["trade_promo"], 0) * yields["trade_promo"]
    wsum = sum(wts.values())
    exp_new = FIRST_ORDER_SHARE * rev_web * 1e5 / AOV_FIRST_INR
    true_new_by_channel = {a: float((exp_new * wts[a] / wsum).sum()) for a in ACQ}

    new_full = np.zeros((G, T), int)
    sampled = np.zeros((len(ACQ), G, T), int)
    for t in range(T):
        for g in range(G):
            rr = _rng(seed, 7, t, g)
            n = int(rr.poisson(exp_new[g, t]))
            probs = np.array([wts[a][g, t] for a in ACQ]) / wsum[g, t]
            split = rr.multinomial(n, probs)
            new_full[g, t] = n
            sampled[:, g, t] = rr.binomial(split, WEBSITE_SAMPLE_RATE)

    r8 = _rng(seed, 8)
    orders_full = np.round(rev_web * 1e5 / 640 * np.exp(r8.normal(0, 0.02, (G, T)))).astype(int)

    week_start = env["week_start"]
    rows = []
    for t in range(T):
        for g in range(G):
            row = dict(week=t + 1, week_start=week_start[t].date().isoformat(), geo=GEO_NAMES[g],
                       revenue_lakh=revenue[g, t], rev_quick_commerce_lakh=rev_qc[g, t],
                       rev_marketplace_lakh=rev_mp[g, t], rev_website_lakh=rev_web[g, t],
                       rev_modern_trade_lakh=rev_mt[g, t], orders=orders_full[g, t],
                       new_customers=new_full[g, t])
            for c in CHANNELS:
                row[f"spend_{c}_lakh"] = measured[c][g, t]
            row.update(price_index=env["price"][t], promo_flag=int(promo_flag[t]),
                       dark_store_count=int(env["dark"][g, t]), temp_c=env["temp"][g, t],
                       festive_flag=int(env["festive"][t]), competitor_index=env["comp"][g, t])
            rows.append(row)
    weekly = pd.DataFrame(rows).round(3)

    # ---- platform-reported (biased by design), national channel x week
    vt = other["view_through"]
    prow = []
    for c in CHANNELS:
        p = ch[c]
        if c == "qc_ads":
            rep = contrib[c].sum(0) / (1 - p["cannibalisation"]) * p["multi_touch"] * vt
        elif c == "trade_promo":
            rep = gross_contrib[c].sum(0) * p["platform_bias"]
        else:
            rep = contrib[c].sum(0) * p["platform_bias"] * vt
        for t in range(T):
            noise = float(np.exp(_rng(seed, 10, CHANNELS.index(c), t).normal(0, 0.12)))
            sp = measured[c][:, t].sum()
            rv = max(rep[t], 0) * noise
            prow.append(dict(week=t + 1, week_start=week_start[t].date().isoformat(), channel=c,
                             spend_lakh=round(sp, 3), reported_revenue_lakh=round(rv, 3),
                             reported_roas=round(rv / sp, 3) if sp > 0 else None))
    platform = pd.DataFrame(prow)

    # ---- influencer spend per creator = channel spend in that fortnight x creator weight
    nat_infl = measured["influencer"].sum(0)
    fn_spend = {f + 1: nat_infl[INFLUENCER_FIRST_WEEK - 1 + 2 * f: INFLUENCER_FIRST_WEEK + 1 + 2 * f].sum()
                for f in range(N_FORTNIGHTS)}
    infl = infl_df.copy()
    infl["spend_lakh"] = (infl["_w"] * infl["fortnight"].map(fn_spend)).round(3)
    infl = infl.drop(columns="_w")

    # ---- customers and orders (1-in-8 sample of website customers)
    customers, orders = (_customers_orders(seed, sampled, cust) if with_customers else (None, None))

    # ---- checkout A/B test
    cro = _cro_sessions(seed, other)

    tables = dict(weekly_geo=weekly, platform_reported=platform, influencers=infl,
                  customers=customers, orders=orders, cro_sessions=cro)

    # ---- the hidden truth
    def roi(spend_d, contrib_d, c, w0, w1):
        return float(contrib_d[c][:, w0:w1].sum() / spend_d[c][:, w0:w1].sum())

    mroi = {}
    for c in CHANNELS:                      # marginal ROI at year-2 planned spend (+1%)
        bumped = {k: v.copy() for k, v in planned.items()}
        bumped[c][:, 52:] *= 1.01
        u2 = media_units(bumped)[c][0]
        mroi[c] = float(beta[c] * (u2 - units_planned[c][0])[:, 52:].sum() / (0.01 * planned[c][:, 52:].sum()))

    truth = dict(
        generator_version=GENERATOR_VERSION, seed=int(seed),
        channels={c: dict(**{k: round(v, 6) for k, v in ch[c].items()},
                          beta=round(float(beta[c]), 6),
                          true_roi_weeks_1_87=round(roi(planned, contrib_planned, c, 0, STAGE1_WEEKS), 6),
                          true_roi_weeks_1_104_no_test=round(roi(planned, contrib_planned, c, 0, T), 6),
                          true_mroi_year2=round(mroi[c], 6),
                          true_contribution_weeks_1_87_lakh=round(float(contrib_planned[c][:, :STAGE1_WEEKS].sum()), 3))
                  for c in CHANNELS},
        view_through=round(other["view_through"], 6),
        influencer_fake_creators=fake_ids,
        influencer_fake_spend_share=round(fake_spend_share, 6),
        customers={a: {k: round(v, 6) for k, v in cust[a].items()} for a in ACQ},
        true_new_website_customers_by_channel_weeks_1_104=
            {a: round(v) for a, v in true_new_by_channel.items()},
        cro=dict(base_conversion=round(other["cro_base_cr"], 6), true_abs_lift=round(other["cro_lift_abs"], 6)),
        baseline_share_of_revenue_year2=round(float(baseline[:, 52:].sum() / revenue[:, 52:].sum()), 6),
    )
    truth["true_clv_by_channel"] = _true_clv(seed, cust)
    extras = dict(planned=planned, actual=actual, beta=beta, units_planned=units_planned,
                  media_units=media_units, contrib=contrib, contrib_planned=contrib_planned)
    return tables, truth, extras


def _simulate_customer(rng, first_day, lam_w, p, aov, disc, end_day):
    lam_i = rng.gamma(3.0, lam_w / 3.0) / 7.0               # repeat rate per day
    p_i = rng.beta(8 * p, 8 * (1 - p))
    m_i = aov * rng.gamma(6.0, 1 / 6.0)
    orders = []
    day, n = first_day, 0
    while day < end_day:
        gross = m_i * rng.gamma(12.0, 1 / 12.0)
        d = disc + (0.05 if n == 0 else 0.0)
        cm = CM2_RATE * (1 + rng.normal(0, 0.03))
        orders.append((day, gross, gross * d, gross * (1 - d) * cm))
        n += 1
        if n > 1 and rng.uniform() < p_i:                   # may stop after each repeat
            break
        day += rng.exponential(1 / lam_i)
    return orders


def _customers_orders(seed, sampled, cust):
    G, T = sampled.shape[1], sampled.shape[2]
    end_day = N_WEEKS * 7
    crow, orow = [], []
    cid = 0
    for t in range(T):
        for g in range(G):
            for ai, a in enumerate(ACQ):
                for i in range(sampled[ai, g, t]):
                    rr = _rng(seed, 9, t, g, ai, i)
                    first = t * 7 + rr.uniform(0, 7)
                    cid += 1
                    c_id = f"C{cid:06d}"
                    pr = cust[a]
                    od = _simulate_customer(rr, first, pr["lam"], pr["p"], pr["aov"], pr["disc"], end_day)
                    crow.append((c_id, START + pd.Timedelta(days=int(first)), a, GEO_NAMES[g]))
                    for (day, gross, disc_amt, contrib_amt) in od:
                        orow.append((c_id, START + pd.Timedelta(days=int(day)), round(gross, 2),
                                     round(disc_amt, 2), round(contrib_amt, 2), int(day)))
    customers = pd.DataFrame(crow, columns=["customer_id", "first_order_date", "acquisition_channel", "geo"])
    orders = pd.DataFrame(orow, columns=["customer_id", "date", "gross_value_inr", "discount_inr",
                                         "contribution_inr", "_day"])
    orders = orders.sort_values(["_day", "customer_id"], kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", [f"O{i + 1:07d}" for i in range(len(orders))])
    customers["first_order_date"] = customers["first_order_date"].dt.date.astype(str)
    orders["date"] = orders["date"].dt.date.astype(str)
    return customers, orders


def _true_clv(seed, cust, n=4000, annual_discount=0.12):
    """Expected discounted contribution per new customer over 12 and 24 months, by channel."""
    out = {}
    d_day = (1 + annual_discount) ** (1 / 365) - 1
    for ai, a in enumerate(ACQ):
        rr = _rng(seed, 12, ai)
        v12 = v24 = 0.0
        for _ in range(n):
            for day, _, _, cm in _simulate_customer(rr, 0.0, cust[a]["lam"], cust[a]["p"], cust[a]["aov"],
                                                   cust[a]["disc"], 730):
                pv = cm / (1 + d_day) ** day
                v24 += pv
                if day < 365:
                    v12 += pv
        out[a] = dict(clv_12m_inr=round(v12 / n, 1), clv_24m_inr=round(v24 / n, 1))
    return out


def _cro_sessions(seed, other):
    r = _rng(seed, 11)
    n = CRO_SESSIONS
    day0 = (CRO_FIRST_WEEK - 1) * 7
    days = np.sort(r.integers(0, CRO_WEEKS * 7, n))
    variant = np.where(r.uniform(size=n) < 0.5, "A_three_step", "B_one_page")
    p = other["cro_base_cr"] + np.where(variant == "B_one_page", other["cro_lift_abs"], 0.0)
    device = np.where(r.uniform(size=n) < 0.78, "mobile", "desktop")
    p = p * np.where(device == "mobile", 0.9, 1.35)
    conv = (r.uniform(size=n) < p).astype(int)
    value = np.where(conv == 1, np.round(650 * r.gamma(8, 1 / 8, n), 2), 0.0)
    dates = (START + pd.to_timedelta(day0 + days, unit="D")).date.astype(str)
    return pd.DataFrame(dict(session_id=[f"S{i + 1:06d}" for i in range(n)], date=dates, variant=variant,
                             device=device, geo=r.choice(GEO_NAMES, n, p=GEO_W), converted=conv,
                             order_value_inr=value))


# --------------------------------------------------------------------------------------
# 5. CUT, SAVE, SEAL
# --------------------------------------------------------------------------------------
def cut_stage1(tables):
    """Keep only what a company would know at the end of week 87."""
    last_day = (START + pd.Timedelta(days=STAGE1_WEEKS * 7 - 1)).date().isoformat()
    out = dict(
        weekly_geo=tables["weekly_geo"].query("week <= @STAGE1_WEEKS").reset_index(drop=True),
        platform_reported=tables["platform_reported"].query("week <= @STAGE1_WEEKS").reset_index(drop=True),
        influencers=tables["influencers"],
        cro_sessions=tables["cro_sessions"],
    )
    if tables["customers"] is not None:
        out["customers"] = tables["customers"][tables["customers"]["first_order_date"] <= last_day].reset_index(drop=True)
        out["orders"] = tables["orders"][tables["orders"]["date"] <= last_day].drop(columns="_day").reset_index(drop=True)
    return out


def full_tables(tables):
    out = dict(tables)
    if out.get("orders") is not None:
        out["orders"] = out["orders"].drop(columns="_day")
    return out


def save_tables(tables, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is not None:
            df.to_csv(folder / f"{name}.csv", index=False)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_stage1(out_dir="data/generated/stage1", sealed_dir="sealed", dev_seed=None):
    """Sealed run: dev_seed=None draws a secret random seed that is never shown.
    Practice run: pass dev_seed=<number>; files go to data/dev/ and nothing is sealed."""
    if dev_seed is not None:
        out_dir, sealed_dir = "data/dev/stage1", "data/dev/sealed_practice"
        seed = int(dev_seed)
    else:
        seed = secrets.randbits(63)
    tables, truth, _ = build_world(seed)
    stage1 = cut_stage1(tables)
    save_tables(stage1, out_dir)
    sealed = Path(sealed_dir)
    sealed.mkdir(parents=True, exist_ok=True)
    truth["sealed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (sealed / "truth.json").write_text(json.dumps(truth, indent=1))
    h = sha256_of(sealed / "truth.json")
    (sealed / "HASH.txt").write_text(f"SHA-256 of truth.json: {h}\nSealed at (UTC): {truth['sealed_at_utc']}\n")
    print(f"Stage 1 written to {out_dir}/ ({', '.join(f'{k}: {len(v):,} rows' for k, v in stage1.items())})")
    print(f"Answer key saved to {sealed_dir}/truth.json  -- DO NOT OPEN IT")
    print(f"Fingerprint (SHA-256): {h}")
    return stage1, h


def run_stage2(design_path="preregistration/design.json", truth_path="sealed/truth.json",
               stage1_dir="data/generated/stage1", out_dir="data/generated/full", sealed_dir="sealed"):
    """Simulate weeks 88-104 under the pre-registered experiment. Reads the answer key silently."""
    design = json.loads(Path(design_path).read_text())
    seed = json.loads(Path(truth_path).read_text())["seed"]          # read silently, never printed
    tables, truth, ex = build_world(seed, design=design)
    full = full_tables(tables)

    # continuity check: weeks 1-87 must be identical to what stage 1 released
    s1 = cut_stage1(tables)
    ok = True
    for name, df in s1.items():
        old = pd.read_csv(Path(stage1_dir) / f"{name}.csv")
        new = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))
        same = old.shape == new.shape and np.allclose(old.select_dtypes("number"), new.select_dtypes("number"),
                                                      rtol=0, atol=1e-6) \
            and old.select_dtypes(exclude="number").equals(new.select_dtypes(exclude="number"))
        print(f"continuity {name:18s} {'OK' if same else 'MISMATCH'}")
        ok &= bool(same)
    if not ok:
        raise RuntimeError("Weeks 1-87 changed between stages — do not use this run.")
    save_tables(full, out_dir)

    # experiment truth: what the test really did (sealed separately, opened at grading)
    c, gi = design["channel"], [GEO_NAMES.index(g) for g in design["treated_geos"]]
    w0, w1 = design["weeks"]
    lost = ex["contrib_planned"][c][gi].sum() - ex["contrib"][c][gi].sum()
    saved = ex["planned"][c][gi].sum() - ex["actual"][c][gi].sum()
    exp_truth = dict(design=design,
                     true_revenue_lost_in_treated_geos_lakh=round(float(lost), 3),
                     true_spend_withheld_lakh=round(float(saved), 3),
                     true_test_roi=round(float(lost / saved), 6),
                     true_roi_weeks_1_104_realised={k: round(float(ex["contrib"][k].sum() / ex["actual"][k].sum()), 6)
                                                    for k in CHANNELS})
    p = Path(sealed_dir) / "experiment_truth.json"
    p.write_text(json.dumps(exp_truth, indent=1))
    h = sha256_of(p)
    (Path(sealed_dir) / "HASH_stage2.txt").write_text(f"SHA-256 of experiment_truth.json: {h}\n")
    print(f"Full 104 weeks written to {out_dir}/")
    print(f"Experiment answer key fingerprint (SHA-256): {h}")
    return full, h


# --------------------------------------------------------------------------------------
# 6. CALIBRATION REPORT (uses only released data — reveals nothing secret)
# --------------------------------------------------------------------------------------
def calibration_report(t):
    w, pr, cu, od = t["weekly_geo"], t["platform_reported"], t.get("customers"), t.get("orders")
    spend_cols = [f"spend_{c}_lakh" for c in CHANNELS]
    nat = w.groupby("week")[["revenue_lakh", "rev_quick_commerce_lakh", "rev_website_lakh", *spend_cols]].sum()
    y1, y2 = nat.loc[1:52], nat.loc[53:]
    last52 = nat.iloc[-52:]
    ws = pd.to_datetime(w.groupby("week")["week_start"].first())
    month = (ws + pd.Timedelta(days=3)).dt.month.values
    y = np.log(nat["revenue_lakh"].values)
    tt = np.arange(len(y))
    detr = np.exp(y - np.polyval(np.polyfit(tt, y, 1), tt))      # remove growth before comparing seasons
    summer = detr[(month >= 4) & (month <= 6)].mean() / detr.mean() - 1
    monsoon = detr[(month >= 7) & (month <= 9)].mean() / detr.mean() - 1
    fest = w.groupby("week")["festive_flag"].first().values == 1
    near = ~fest & np.isin(month, [9, 10, 11])
    festive = detr[fest].mean() / detr[near].mean() - 1
    mk = last52[spend_cols].sum().sum()
    rows = [
        ("Revenue, latest 52 weeks", "≈ ₹120 Cr by week 104", f"₹{last52['revenue_lakh'].sum() / 100:.1f} Cr"),
    ]
    if len(nat) >= 104:
        rows.append(("Year-on-year growth", "≈ 45%", f"{y2['revenue_lakh'].sum() / y1['revenue_lakh'].sum() - 1:.0%}"))
    else:
        rows.append(("Growth, weeks 36-87 vs 1-52", "≈ 45% a year", f"{(last52['revenue_lakh'].sum() / y1['revenue_lakh'].sum()) ** (52 / (len(nat) - 52)) - 1:.0%} a year"))
    rows += [
        ("Marketing % of revenue (latest 52 weeks)", "≈ 28%", f"{mk / last52['revenue_lakh'].sum():.1%}"),
        ("Quick-commerce share of revenue", "≈ 45%", f"{nat['rev_quick_commerce_lakh'].sum() / nat['revenue_lakh'].sum():.1%}"),
        ("QC ad spend % of QC sales", "≤ 15%", f"{nat['spend_qc_ads_lakh'].sum() / nat['rev_quick_commerce_lakh'].sum():.1%}"),
        ("Website share of revenue", "≈ 20%", f"{nat['rev_website_lakh'].sum() / nat['revenue_lakh'].sum():.1%}"),
        ("Summer (Apr–Jun) vs average week", "+ (peak)", f"{summer:+.0%}"),
        ("Monsoon (Jul–Sep) vs average week", "− (dip)", f"{monsoon:+.0%}"),
        ("Festive weeks vs nearby weeks", "+8–12%", f"{festive:+.0%}"),
        ("Platform-claimed ÷ actual revenue", "> 1×", f"{pr['reported_revenue_lakh'].sum() / nat['revenue_lakh'].sum():.2f}×"),
    ]
    m = w[(w.week >= 20) & (w.week <= 49)].groupby("week")[["spend_meta_lakh", "spend_youtube_lakh"]].sum()
    rows.append(("Meta–YouTube spend correlation, weeks 20–49", "≈ 0.8", f"{m.corr().iloc[0, 1]:.2f}"))
    if cu is not None:
        od2 = od.copy()
        od2["date"] = pd.to_datetime(od2["date"])
        cu2 = cu.copy()
        cu2["first_order_date"] = pd.to_datetime(cu2["first_order_date"])
        net = (od2["gross_value_inr"] - od2["discount_inr"]).sum()
        rows += [
            ("Website orders in sample (1 in 8)", "70,000–90,000 over 104 weeks", f"{len(od):,}"),
            ("Website customers in sample", "25,000–35,000 over 104 weeks", f"{len(cu):,}"),
            ("Average order value", "₹600–700", f"₹{od['gross_value_inr'].mean():.0f}"),
            ("Contribution per order", "35–45% of order value", f"{od['contribution_inr'].sum() / od['gross_value_inr'].sum():.0%}"),
            ("Sample × 8 vs website revenue", "≈ 1.0", f"{net / WEBSITE_SAMPLE_RATE / (nat['rev_website_lakh'].sum() * 1e5):.2f}"),
        ]
        paid = cu2["acquisition_channel"] != "organic"
        web_share = nat["rev_website_lakh"].sum() / nat["revenue_lakh"].sum()
        paid_spend = nat[[f"spend_{c}_lakh" for c in ["meta", "search", "youtube", "influencer", "trade_promo"]]].sum().sum()
        cac = paid_spend * 1e5 * web_share / (paid.sum() / WEBSITE_SAMPLE_RATE)
        rows.append(("Blended website CAC (website share of paid spend ÷ paid-acquired customers)", "₹800–1,200", f"₹{cac:,.0f}"))
        cutoff = od2["date"].max() - pd.Timedelta(days=91)
        od2 = od2.sort_values(["customer_id", "date"])
        od2["n"] = od2.groupby("customer_id").cumcount()
        first = od2[od2["n"] == 0].set_index("customer_id")["date"]
        second = od2[od2["n"] == 1].set_index("customer_id")["date"]
        elig = cu2[cu2["first_order_date"] <= cutoff].set_index("customer_id")
        rep = ((second.reindex(elig.index) - first.reindex(elig.index)).dt.days <= 90)
        by = rep.groupby(elig["acquisition_channel"]).mean()
        rows.append(("90-day repeat rate by channel", "25–45%",
                     ", ".join(f"{k} {v:.0%}" for k, v in by.sort_values().items())))
    return pd.DataFrame(rows, columns=["Check", "Target (benchmark)", "Generated"])
