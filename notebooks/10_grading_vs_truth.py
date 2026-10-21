# %% [markdown]
# # 10 · Opening the answer key: how right were we? (D4 validation report)
#
# **What this is, simply:** the answers were hidden in `sealed/truth.json` on 21 Oct 2026, before any
# modelling. Every model and decision was frozen before the key was opened. Now we grade them all,
# **including the misses**, and explain each miss with evidence.
#
# **What is graded:**
# 1. Integrity: the key is unchanged, and it regenerates the released data exactly.
# 2. The three channel models (OLS, MMM v1, MMM v2) against the pre-registered pass rules.
# 3. The geo experiment.
# 4. Customer lifetime value.
# 5. The fake-influencer detector (AI3).
# 6. The checkout A/B test.
# 7. The frozen ₹24 Cr budget recommendation: how much it truly earns vs the best possible split.

# %%
import sys, json, hashlib, re
from pathlib import Path
here = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(here / "src")); sys.path.insert(0, str(here / "generator"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.optimize import minimize
import prakriti_world as pw
from common import load, national, FULL, CHANNELS, LABEL, COLOR, INK, INK2, NEUTRAL, style, save, ROOT

style()
truth = json.loads((ROOT / "sealed" / "truth.json").read_text())
exp_truth = json.loads((ROOT / "sealed" / "experiment_truth.json").read_text())

# %% [markdown]
# ## 1. Integrity checks

# %%
for f, hf in [("truth.json", "HASH.txt"), ("experiment_truth.json", "HASH_stage2.txt")]:
    committed = (ROOT / "sealed" / hf).read_text().split(":")[1].split()[0].strip()
    actual = hashlib.sha256((ROOT / "sealed" / f).read_bytes()).hexdigest()
    print(f"{f:24s} fingerprint {'MATCHES' if actual == committed else 'DOES NOT MATCH'} the committed one")
    assert actual == committed

# Rebuild the world from the sealed seed and the pre-registered design; it must equal the released data.
design = json.loads((ROOT / "preregistration" / "design.json").read_text())
tables, truth_rebuilt, ex = pw.build_world(truth["seed"], design=design, with_customers=False)
released = pd.read_csv(FULL / "weekly_geo.csv")
rebuilt = pd.read_csv(pd.io.common.StringIO(tables["weekly_geo"].to_csv(index=False)))
same = np.allclose(released.select_dtypes("number"), rebuilt.select_dtypes("number"), atol=1e-6)
print(f"Rebuilt weekly data identical to the released data: {same}")
assert same
w, p = load(FULL)

# %% [markdown]
# ## 2. True ROI over exactly the weeks each model saw
# Computed from the rebuilt world: true incremental revenue ÷ true spend, for weeks 1–79 (OLS, v1)
# and weeks 1–96 (v2).

# %%
def true_roi(w0, w1):
    return pd.Series({LABEL[c]: ex["contrib"][c][:, w0:w1].sum() / ex["actual"][c][:, w0:w1].sum() for c in CHANNELS})
true79, true96 = true_roi(0, 79), true_roi(0, 96)
true_mroi = pd.Series({LABEL[c]: truth["channels"][c]["true_mroi_year2"] for c in CHANNELS})
pd.DataFrame({"true ROI wk 1–79": true79, "true ROI wk 1–96": true96, "true marginal ROI (yr 2)": true_mroi}).round(3)

# %% [markdown]
# ## 3. Grade OLS, MMM v1 and MMM v2 against the pre-registered rules
# Rules (Measurement Brief D1):
# - True ROI inside the 90% interval for at least 5 of 6 channels.
# - Ranking agrees with the truth: Spearman ρ ≥ 0.8, with the top and bottom channels correct.
# - Holdout error (MAPE) below 10%.

# %%
nat = national(w)
spend_cols = [f"spend_{c}_lakh" for c in CHANNELS]
controls = ["price_index", "promo_flag", "festive_flag", "temp_c", "competitor_index", "dark_store_count"]
d79 = nat.loc[1:87, spend_cols + controls].copy(); d79["t"] = np.arange(len(d79))
Xc = sm.add_constant(d79)
ols = sm.OLS(nat.loc[1:79, "revenue_lakh"], Xc.loc[1:79]).fit()
ols_ci = ols.conf_int(alpha=0.10).loc[spend_cols]
ols_tbl = pd.DataFrame({"roi": ols.params[spend_cols].values, "roi_lo": ols_ci[0].values, "roi_hi": ols_ci[1].values},
                       index=[LABEL[c] for c in CHANNELS])
v1 = pd.read_csv(ROOT / "model" / "mmm_v1_roi.csv", index_col=0)
v2 = pd.read_csv(ROOT / "model" / "mmm_v2_roi.csv", index_col=0)

def holdout_mape(nb):
    for c in json.loads((ROOT / "notebooks" / f"{nb}.ipynb").read_text())["cells"]:
        for o in c.get("outputs", []):
            m = re.search(r"[Hh]oldout MAPE[^:]*: ([\d.]+)%", "".join(o.get("text", "")))
            if m:
                return float(m.group(1)) / 100

def grade(est, tru, mape):
    est = est.reindex(tru.index)
    inside = (tru >= est.roi_lo) & (tru <= est.roi_hi)
    rho = spearmanr(est.roi, tru).correlation
    top_ok, bottom_ok = est.roi.idxmax() == tru.idxmax(), est.roi.idxmin() == tru.idxmin()
    return dict(coverage=f"{inside.sum()} of 6", coverage_pass=inside.sum() >= 5, spearman=round(rho, 2),
                rank_pass=bool(rho >= 0.8 and top_ok and bottom_ok), top_correct=top_ok, bottom_correct=bottom_ok,
                holdout_mape=mape, mape_pass=mape < 0.10, inside=inside)

g = {"OLS (wk 1–79)": grade(ols_tbl, true79, holdout_mape("02A_eda_and_ols_baseline")),
     "MMM v1 (wk 1–79)": grade(v1, true79, holdout_mape("02B_mmm_v1")),
     "MMM v2 (wk 1–96 + experiment)": grade(v2, true96, holdout_mape("02C_mmm_v2"))}
score = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "inside"} for k, v in g.items()}).T
score

# %%
detail = pd.DataFrame({"truth": true96, "v2 estimate": v2.roi, "v2 90% low": v2.roi_lo, "v2 90% high": v2.roi_hi,
                       "inside?": g["MMM v2 (wk 1–96 + experiment)"]["inside"],
                       "v1 estimate": v1.roi, "v1 inside?": g["MMM v1 (wk 1–79)"]["inside"],
                       "OLS estimate": ols_tbl.roi, "OLS inside?": g["OLS (wk 1–79)"]["inside"]})
detail.round(2)

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.2))
order = true96.sort_values().index
for j, n in enumerate(order):
    c = [k for k, v in LABEL.items() if v == n][0]
    ax.plot([v2.loc[n, "roi_lo"], v2.loc[n, "roi_hi"]], [j, j], color=COLOR[c], lw=4, solid_capstyle="round", alpha=0.85)
    ax.plot(v2.loc[n, "roi"], j, "o", color=COLOR[c], ms=8, mec="#fcfcfb", mew=1.5)
    ax.plot(true96[n], j, marker="D", color=INK, ms=7, ls="none")
    ax.text(max(v2.loc[n, "roi_hi"], true96[n]) + 0.1, j, "✓ inside" if detail.loc[n, "inside?"] else "✗ missed",
            va="center", fontsize=8.5, color=INK if detail.loc[n, "inside?"] else "#c0392b")
ax.plot([], [], "D", color=INK, label="True ROI (answer key)")
ax.plot([], [], "o", color=INK2, label="MMM v2 estimate, 90% interval")
ax.set_yticks(range(len(order)), order)
ax.set_xlabel("ROI, ₹ per ₹1 (weeks 1–96)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
cov = g["MMM v2 (wk 1–96 + experiment)"]["coverage"]
ax.set_title(f"MMM v2 vs the answer key: true ROI inside the interval for {cov} channels")
save(fig, "10_v2_vs_truth"); plt.show()

# %% [markdown]
# ## 4. The decision quantity: marginal ROI
# Budget moves depend on the *next* rupee, not the average. Does v2 rank channels' marginal returns correctly?

# %%
mcmp = pd.DataFrame({"v2 marginal ROI": v2.mroi, "true marginal ROI": true_mroi})
rho_m = spearmanr(mcmp.iloc[:, 0], mcmp.iloc[:, 1]).correlation
print(f"Spearman, marginal ROI ranking (v2 vs truth): {rho_m:.2f}")
mcmp.sort_values("true marginal ROI", ascending=False).round(2)

# %% [markdown]
# ## 5. Why the misses? Evidence, not excuses
# **Meta.** The truth makes Meta *near saturation*: today's spend sits far up its curve. A saturated
# channel's weekly contribution barely moves when spend moves. Then a model cannot tell it apart from
# baseline sales, and Meta's spend also tracks the growth trend. Test: how much does each channel's
# *true* weekly contribution vary, and how closely does it follow the trend?

# %%
tot = {c: ex["contrib"][c].sum(0)[:96] for c in CHANNELS}
trend = np.arange(96)
ev = pd.DataFrame({
    "true contribution, week-to-week variation (CV)": {LABEL[c]: np.std(np.diff(v)) / np.mean(v) for c, v in tot.items()},
    "correlation with time trend": {LABEL[c]: np.corrcoef(v, trend)[0, 1] for c, v in tot.items()},
    "true k_rel (lower = more saturated)": {LABEL[c]: truth["channels"][c]["k_rel"] for c in CHANNELS},
})
ev.round(3)

# %% [markdown]
# **Where did the missing credit go?** Compare true and modelled contribution channel by channel.
# Credit the model does not give to media stays in the baseline (intercept + trend).

# %%
units = ex["media_units"](ex["actual"])
credit = pd.DataFrame({"true ₹ Cr (wk 1–96)": {LABEL[c]: ex["contrib"][c][:, :96].sum() / 100 for c in CHANNELS},
                       "MMM v2 ₹ Cr": v2.contribution_cr})
credit["gap ₹ Cr"] = credit["MMM v2 ₹ Cr"] - credit["true ₹ Cr (wk 1–96)"]
credit.loc["All media"] = credit.sum()
print(f"Media truly drove ₹{credit.loc['All media', 'true ₹ Cr (wk 1–96)']:.1f} Cr; v2 credited ₹{credit.loc['All media', 'MMM v2 ₹ Cr']:.1f} Cr. "
      f"The missing ₹{-credit.loc['All media', 'gap ₹ Cr']:.1f} Cr sits in v2's baseline.")
credit.round(1)

# %% [markdown]
# **Trade promotions.** Hypothesis: the model measured the promotion *spike* and missed the pull-forward
# (sales borrowed from the next two weeks). Test: the true *gross* ROI before pull-forward.

# %%
c = "trade_promo"
gross_roi = ex["beta"][c] * units[c][1][:, :96].sum() / ex["actual"][c][:, :96].sum()
print(f"Trade promotions: true GROSS ROI {gross_roi:.2f}, true NET ROI {true96['Trade promotions']:.2f} "
      f"(pull-forward {truth['channels'][c]['pull_forward']:.0%}); v2 estimated {v2.loc['Trade promotions', 'roi']:.2f}.")
print(f"v2's 'week after a promotion' control was too crude: promotions ran in {nat.loc[1:96, 'promo_flag'].mean():.0%} of weeks, "
      "often back to back, so clean post-promotion weeks were rare, and the dip is spread over two weeks.")

# %% [markdown]
# **YouTube.** True half-life is long and the curve S-shaped; v2's half-life and curve can be compared
# directly.

# %%
dd = np.load(ROOT / "model" / "mmm_v2_posterior_params.npz", allow_pickle=True)
hl = np.log(0.5) / np.log(np.clip(dd["alpha"], 1e-6, 0.999999))
hl_tbl = pd.DataFrame({"true half-life": {LABEL[c]: truth["channels"][c]["half_life"] for c in CHANNELS},
                       "v2 median": {LABEL[c]: np.median(hl[:, i]) for i, c in enumerate(CHANNELS)},
                       "v2 5%": {LABEL[c]: np.quantile(hl[:, i], .05) for i, c in enumerate(CHANNELS)},
                       "v2 95%": {LABEL[c]: np.quantile(hl[:, i], .95) for i, c in enumerate(CHANNELS)}})
hl_tbl["inside?"] = (hl_tbl["true half-life"] >= hl_tbl["v2 5%"]) & (hl_tbl["true half-life"] <= hl_tbl["v2 95%"])
hl_tbl.round(2)

# %% [markdown]
# ## 6. The geo experiment vs the truth

# %%
er = json.loads((ROOT / "model" / "experiment_result.json").read_text())
t_roi = exp_truth["true_test_roi"]
print(f"Experiment estimate {er['roi']:.3f} (90% CI {er['roi_ci90'][0]:.3f}–{er['roi_ci90'][1]:.3f}) vs true {t_roi:.3f}: "
      f"{'inside' if er['roi_ci90'][0] <= t_roi <= er['roi_ci90'][1] else 'outside'} the interval, error {er['roi'] / t_roi - 1:+.1%}")
print(f"Revenue lost: estimated ₹{er['revenue_lost_lakh']:.1f} lakh vs true ₹{exp_truth['true_revenue_lost_in_treated_geos_lakh']:.1f} lakh; "
      f"spend withheld: estimated ₹{er['spend_withheld_lakh']:.1f} lakh vs true ₹{exp_truth['true_spend_withheld_lakh']:.1f} lakh")

# %% [markdown]
# ## 7. Customer lifetime value vs the truth

# %%
clv = pd.read_csv(ROOT / "model" / "clv_by_channel.csv", index_col=0)
lab = {"organic": "Organic", **LABEL}
tc = pd.Series({lab[k]: v["clv_24m_inr"] for k, v in truth["true_clv_by_channel"].items()})
clv_g = pd.DataFrame({"true 24m CLV ₹": tc, "estimate ₹": clv["CLV 24m ₹"], "low": clv["CLV 24m low"], "high": clv["CLV 24m high"]}).dropna()
clv_g["error"] = clv_g["estimate ₹"] / clv_g["true 24m CLV ₹"] - 1
clv_g["inside?"] = (clv_g["true 24m CLV ₹"] >= clv_g.low) & (clv_g["true 24m CLV ₹"] <= clv_g.high)
print(f"CLV: mean absolute error {clv_g.error.abs().mean():.1%}; ranking Spearman "
      f"{spearmanr(clv_g['estimate ₹'], clv_g['true 24m CLV ₹']).correlation:.2f}; inside interval {clv_g['inside?'].sum()} of {len(clv_g)}")
clv_g.round(3)

# %% [markdown]
# ## 8. Fake-influencer detector (AI3) vs the sealed fake list

# %%
flags = pd.read_csv(ROOT / "model" / "influencer_flags.csv")
fakes = set(truth["influencer_fake_creators"])
flags["is_fake"] = flags.creator_id.isin(fakes)
rows = {}
for m in ["flag_rule", "flag_if_5", "flag_if_10", "flag_if_15"]:
    sel = flags[m]
    rows[m] = dict(flagged=int(sel.sum()), true_fakes_caught=int((sel & flags.is_fake).sum()),
                   precision=(sel & flags.is_fake).sum() / max(sel.sum(), 1), recall=(sel & flags.is_fake).sum() / len(fakes))
k = len(fakes)
topk = flags.sort_values("anomaly_score", ascending=False).head(k)
rows[f"Isolation Forest, top {k} (precision@k)"] = dict(flagged=k, true_fakes_caught=int(topk.is_fake.sum()),
                                                       precision=topk.is_fake.mean(), recall=topk.is_fake.sum() / len(fakes))
fraud = pd.DataFrame(rows).T
print(f"{len(fakes)} fake creators in the truth ({truth['influencer_fake_spend_share']:.1%} of influencer spend).")
fraud.round(3)

# %%
fs = truth["influencer_fake_spend_share"]
print(f"Influencer ROI on all spend: true {true96['Influencers']:.2f}. On genuine spend only: {true96['Influencers'] / (1 - fs):.2f} "
      f"(+{fs / (1 - fs):.1%}). Removing the flagged creators recovers that waste.")

# %% [markdown]
# ## 9. Checkout A/B test vs the truth

# %%
s = pd.read_csv(FULL / "cro_sessions.csv")
gA, gB = s[s.variant == "A_three_step"].converted, s[s.variant == "B_one_page"].converted
diff = gB.mean() - gA.mean()
se = np.sqrt(gA.var() / len(gA) + gB.var() / len(gB))
t_lift = truth["cro"]["true_abs_lift"]
print(f"Estimated lift {diff * 100:+.2f} pp (95% CI {(diff - 1.96 * se) * 100:+.2f} to {(diff + 1.96 * se) * 100:+.2f}) vs true "
      f"{t_lift * 100:+.2f} pp: {'inside' if diff - 1.96 * se <= t_lift <= diff + 1.96 * se else 'outside'} the interval. "
      "(Conversion differs by device in the data, so the overall true lift is approximate.)")

# %% [markdown]
# ## 10. The frozen ₹24 Cr recommendation, valued with the TRUE response curves
# Each plan is replayed through the real (sealed) generator over a year shaped like year 2. Margins and
# customer-value weights are kept as the optimiser used them; only the curves are swapped for the truth.
# The **oracle** is the best split anyone could have chosen knowing the truth, under the same ±50% limits.

# %%
rec = json.loads((ROOT / "model" / "optimiser_recommendation.json").read_text())
cur = np.array([rec["current_cr"][LABEL[c]] for c in CHANNELS])
plans = {"Today (₹33.5 Cr)": cur, "Pro-rata cut (₹24 Cr)": np.array([rec["pro_rata_cr"][LABEL[c]] for c in CHANNELS]),
         "Frozen recommendation (₹24 Cr)": np.array([rec["optimised_cr"][LABEL[c]] for c in CHANNELS])}
planned = ex["planned"]
beta = ex["beta"]
plan_y2 = {c: planned[c][:, 52:].sum() / 100 for c in CHANNELS}              # ₹ Cr in year 2 (plan)
MARGIN = {c: (0.30 if c == "qc_ads" else 0.37) for c in CHANNELS}
# customer-value weights exactly as the frozen optimiser used them
nbj = json.loads((ROOT / "notebooks" / "04_budget_optimiser.ipynb").read_text())
fv_txt = next("".join(o["data"]["text/plain"]) for c in nbj["cells"] for o in c.get("outputs", [])
              if "future value per ₹ revenue" in "".join(o.get("data", {}).get("text/plain", "")))
FV = {c: float(re.search(rf"{re.escape(LABEL[c])}\s+[\d.]+\s+([\d.]+)", fv_txt).group(1)) for c in CHANNELS}

def true_value(annual_cr):
    """True CLV-adjusted contribution after marketing, ₹ Cr, for an annual allocation by channel."""
    sp = {c: planned[c].copy() for c in CHANNELS}
    for i, c in enumerate(CHANNELS):
        sp[c][:, 52:] *= annual_cr[i] / plan_y2[c]
    units = ex["media_units"](sp)
    rev = np.array([beta[c] * units[c][0][:, 52:].sum() / 100 for c in CHANNELS])     # ₹ Cr incremental revenue
    return float((rev * np.array([MARGIN[c] + FV[c] for c in CHANNELS])).sum() - annual_cr.sum()), rev

vals = {k: true_value(v)[0] for k, v in plans.items()}
lo, hi = cur * 0.5, cur * 1.5
best, best_v = None, -np.inf
rng = np.random.default_rng(3)
starts = [plans["Frozen recommendation (₹24 Cr)"], plans["Pro-rata cut (₹24 Cr)"]] + \
         [lo + rng.dirichlet(np.ones(6)) * (24 - lo.sum()) for _ in range(15)]
for x0 in starts:
    r = minimize(lambda x: -true_value(x)[0], np.clip(x0, lo, hi), method="SLSQP", bounds=list(zip(lo, hi)),
                 constraints=[{"type": "eq", "fun": lambda x: x.sum() - 24}], options={"maxiter": 300})
    if r.success and -r.fun > best_v:
        best, best_v = r.x, -r.fun
vals["Oracle (best possible, ₹24 Cr)"] = best_v
vt = pd.Series(vals).to_frame("true net contribution ₹ Cr/yr")
gap_total = vals["Oracle (best possible, ₹24 Cr)"] - vals["Pro-rata cut (₹24 Cr)"]
captured = (vals["Frozen recommendation (₹24 Cr)"] - vals["Pro-rata cut (₹24 Cr)"]) / gap_total
alloc_cmp = pd.DataFrame({"frozen recommendation ₹ Cr": plans["Frozen recommendation (₹24 Cr)"], "oracle ₹ Cr": best},
                         index=[LABEL[c] for c in CHANNELS])
print(f"Predicted gain over pro-rata (frozen, from MMM v2): ₹{rec['expected_gain_vs_pro_rata_cr']['median']:.1f} Cr "
      f"(90% {rec['expected_gain_vs_pro_rata_cr']['p5']:.1f}–{rec['expected_gain_vs_pro_rata_cr']['p95']:.1f})")
print(f"TRUE gain over pro-rata: ₹{vals['Frozen recommendation (₹24 Cr)'] - vals['Pro-rata cut (₹24 Cr)']:.1f} Cr | "
      f"oracle gain: ₹{gap_total:.1f} Cr | recommendation captures {captured:.0%} of the best possible gain")
display_tbl = pd.concat([vt.round(2), alloc_cmp.round(2)], axis=0)
vt.round(2)

# %%
alloc_cmp.round(2)

# %%
base = vals["Today (₹33.5 Cr)"]
names = [n for n in vals if not n.startswith("Today")]
gains = [vals[n] - base for n in names]
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh(range(len(names)), gains, color=[NEUTRAL, COLOR["meta"], INK2], height=0.55)
for j, gv in enumerate(gains):
    ax.text(gv + 0.15, j, f"+₹{gv:.1f} Cr", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
ax.set_xlim(0, max(gains) * 1.2)
ax.set_xlabel("TRUE gain in contribution vs today's ₹33.5 Cr plan, ₹ Cr/yr (CLV-adjusted)")
ax.grid(axis="y", visible=False)
ax.set_title(f"Graded on the true curves, the frozen plan matches the best possible split")
save(fig, "10_plan_vs_oracle"); plt.show()

# %% [markdown]
# ## 11. Scorecard and so what

# %%
v2g = g["MMM v2 (wk 1–96 + experiment)"]
card = pd.DataFrame([
    ("MMM v2: true ROI inside 90% interval (≥5 of 6)", v2g["coverage"], "PASS" if v2g["coverage_pass"] else "FAIL"),
    ("MMM v2: ranking Spearman ≥0.8, top and bottom correct", f"ρ = {v2g['spearman']}", "PASS" if v2g["rank_pass"] else "FAIL"),
    ("MMM v2: holdout MAPE < 10%", f"{v2g['holdout_mape']:.1%}", "PASS" if v2g["mape_pass"] else "FAIL"),
    ("MMM v2: marginal-ROI ranking (added check, not pre-registered)", f"ρ = {rho_m:.2f}", "PASS" if rho_m >= 0.8 else "FAIL"),
    ("Geo experiment: truth inside 90% CI", f"{er['roi']:.2f} vs {t_roi:.2f}", "PASS" if er['roi_ci90'][0] <= t_roi <= er['roi_ci90'][1] else "FAIL"),
    ("CLV: mean absolute error < 10%", f"{clv_g.error.abs().mean():.1%}", "PASS" if clv_g.error.abs().mean() < 0.10 else "FAIL"),
    ("Fraud detector: precision@k", f"{fraud.iloc[-1].precision:.0%}", "—"),
    ("Checkout test: truth inside 95% CI", f"{diff * 100:+.2f} vs {t_lift * 100:+.2f} pp", "PASS" if diff - 1.96 * se <= t_lift <= diff + 1.96 * se else "FAIL"),
    ("Budget recommendation: share of best possible gain captured", f"{captured:.0%}", "—"),
    ("Budget recommendation: predicted gain vs true gain", f"₹{rec['expected_gain_vs_pro_rata_cr']['median']:.1f} vs ₹{vals['Frozen recommendation (₹24 Cr)'] - vals['Pro-rata cut (₹24 Cr)']:.1f} Cr", "OVERSTATED"),
], columns=["Test", "Result", "Verdict"])
card.to_csv(ROOT / "reports" / "validation_scorecard.csv", index=False)
card

# %%
missed = detail.index[~detail["inside?"]].tolist()
pred = rec["expected_gain_vs_pro_rata_cr"]
true_gain = vals["Frozen recommendation (₹24 Cr)"] - vals["Pro-rata cut (₹24 Cr)"]
print("SO WHAT")
print(f"1. MMM v2 failed its own pre-registered test: true ROI inside the interval for {v2g['coverage']} channels "
      f"(missed: {', '.join(missed)}), ranking ρ = {v2g['spearman']}, marginal-ROI ranking ρ = {rho_m:.2f}. "
      "Reported as is; nothing was re-tuned after opening the key.")
print(f"2. The misses have identifiable causes. Meta is near saturation (true k_rel {truth['channels']['meta']['k_rel']:.2f}), so its "
      f"contribution is almost flat and tracks the trend; v2 put ₹{credit.loc['Meta', 'true ₹ Cr (wk 1–96)'] - credit.loc['Meta', 'MMM v2 ₹ Cr']:.0f} Cr of "
      f"Meta's effect into baseline. Trade promotions: v2 measured the gross spike ({gross_roi:.2f} true gross vs {v2.loc['Trade promotions', 'roi']:.2f} "
      f"estimated) and missed the {truth['channels']['trade_promo']['pull_forward']:.0%} pull-forward.")
print(f"3. What held: the experiment ({er['roi']:.2f} vs true {t_roi:.2f}), CLV (error {clv_g.error.abs().mean():.0%}), the fraud detector "
      f"({fraud.loc['flag_if_10', 'true_fakes_caught']:.0f} of {len(fakes)} fakes at the 10% setting) and the checkout test.")
print(f"4. The budget plan matched the best possible split (captures {captured:.0%} of the achievable gain), but partly because the "
      f"±50% limits forced the same corners: four channels truly have low marginal returns (0.39–0.52) and YouTube high (4.10). "
      f"The predicted gain was overstated: ₹{pred['median']:.1f} Cr predicted (90% {pred['p5']:.1f}–{pred['p95']:.1f}) vs ₹{true_gain:.1f} Cr true.")
print("5. Next experiment: a Meta geo holdout. Meta is the largest budget line and the model's biggest miss, and an experiment is "
      "the reliable fix for a saturated, always-on channel; its result should feed MMM v3.")
