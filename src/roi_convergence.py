"""Convergence of the decision quantities (ROI per channel), rebuilt from saved posterior draws."""
import numpy as np
import pandas as pd
import arviz as az


def roi_rhat(npz_path, spend_matrix, channels, labels, chains=4, l_max=24):
    d = np.load(npz_path, allow_pickle=True)
    ts, cs = float(d["target_scale"]), d["channel_scale"]
    rows = []
    for i, c in enumerate(channels):
        a = d["alpha"][:, i][:, None] ** np.arange(l_max)[None, :]
        a = a / a.sum(1, keepdims=True)
        xs = spend_matrix[:, i] / cs[i]
        T = len(xs)
        ad = np.zeros((len(a), T))
        for lag in range(l_max):
            ad[:, lag:] += a[:, [lag]] * xs[None, :T - lag]
        s, k = d["slope"][:, i][:, None], d["kappa"][:, i][:, None]
        contrib = d["beta"][:, i][:, None] * ad**s / (ad**s + k**s) * ts
        roi = contrib.sum(1) / spend_matrix[:, i].sum()
        draws = roi.reshape(chains, -1)                    # samples were stacked chain-major
        rows.append(dict(channel=labels[c], roi_median=np.median(roi),
                         r_hat=float(az.rhat(draws)), ess_bulk=float(az.ess(draws))))
    return pd.DataFrame(rows).set_index("channel")
