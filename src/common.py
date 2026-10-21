"""Shared helpers for the analysis notebooks: data loading, channel names, chart style."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "data" / "generated" / "stage1"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["meta", "qc_ads", "search", "trade_promo", "youtube", "influencer"]
LABEL = dict(meta="Meta", qc_ads="Quick-commerce ads", search="Google Search", trade_promo="Trade promotions",
             youtube="YouTube", influencer="Influencers")
# fixed colour per channel (validated categorical palette, fixed order, never cycled)
COLOR = dict(meta="#2a78d6", qc_ads="#eb6834", search="#1baf7a", trade_promo="#eda100",
             youtube="#e87ba4", influencer="#008300")
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b7b5ae"


def load(stage_dir=STAGE1):
    w = pd.read_csv(stage_dir / "weekly_geo.csv", parse_dates=["week_start"])
    p = pd.read_csv(stage_dir / "platform_reported.csv", parse_dates=["week_start"])
    return w, p


def national(w):
    spend = [f"spend_{c}_lakh" for c in CHANNELS]
    agg = {c: "sum" for c in ["revenue_lakh", "rev_quick_commerce_lakh", "rev_website_lakh", *spend]}
    agg.update(price_index="first", promo_flag="first", festive_flag="first", week_start="first",
               temp_c="mean", competitor_index="mean", dark_store_count="sum")
    return w.groupby("week").agg(agg)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "lines.linewidth": 2, "figure.dpi": 110,
    })


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    return FIG / f"{name}.png"
