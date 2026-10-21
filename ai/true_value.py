"""Value any annual budget split with the TRUE response curves (answer key), exactly as in notebook 10."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))
import prakriti_world as pw  # noqa: E402

CHANNELS = ["meta", "qc_ads", "search", "trade_promo", "youtube", "influencer"]
LABEL = dict(meta="Meta", qc_ads="Quick-commerce ads", search="Google Search", trade_promo="Trade promotions",
             youtube="YouTube", influencer="Influencers")
MARGIN = {c: (0.30 if c == "qc_ads" else 0.37) for c in CHANNELS}


class TrueValuer:
    def __init__(self):
        truth = json.loads((ROOT / "sealed" / "truth.json").read_text())
        design = json.loads((ROOT / "preregistration" / "design.json").read_text())
        _, _, self.ex = pw.build_world(truth["seed"], design=design, with_customers=False)
        self.plan_y2 = {c: self.ex["planned"][c][:, 52:].sum() / 100 for c in CHANNELS}
        nbj = json.loads((ROOT / "notebooks" / "04_budget_optimiser.ipynb").read_text())
        txt = next("".join(o["data"]["text/plain"]) for c in nbj["cells"] for o in c.get("outputs", [])
                   if "future value per ₹ revenue" in "".join(o.get("data", {}).get("text/plain", "")))
        self.fv = {c: float(re.search(rf"{re.escape(LABEL[c])}\s+[\d.]+\s+([\d.]+)", txt).group(1)) for c in CHANNELS}

    def value(self, alloc_cr):
        """True CLV-adjusted contribution after marketing (₹ Cr/yr) for {channel: ₹ Cr/yr}."""
        sp = {c: self.ex["planned"][c].copy() for c in CHANNELS}
        for c in CHANNELS:
            sp[c][:, 52:] *= alloc_cr[c] / self.plan_y2[c]
        units = self.ex["media_units"](sp)
        rev = np.array([self.ex["beta"][c] * units[c][0][:, 52:].sum() / 100 for c in CHANNELS])
        w = np.array([MARGIN[c] + self.fv[c] for c in CHANNELS])
        return float((rev * w).sum() - sum(alloc_cr.values()))
