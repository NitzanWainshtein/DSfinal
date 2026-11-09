"""Day-block bootstrap intervals for the observed test coverage (report section 3.3).

Reads outputs/predictions.csv only. Errors on the same day are correlated, so whole test days
are resampled with replacement. Writes outputs/coverage_bootstrap.csv.

Usage:  python coverage_bootstrap.py [--draws 5000] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def main(draws: int, seed: int) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(ROOT, "outputs", "predictions.csv"))
    t = p[p.period == "test"].copy()
    t["day"] = t.timestamp.str[:10]
    rng = np.random.default_rng(seed)
    rows = []
    for model, level, lo, hi in [("bayes", 80, "lo80", "hi80"), ("bayes", 95, "lo95", "hi95"),
                                 ("poisson", 80, "lo80", "hi80"), ("poisson", 95, "lo95", "hi95")]:
        d = t[t.model == model].copy()
        d["c"] = ((d.actual >= d[lo]) & (d.actual <= d[hi])).astype(int)   # inclusive endpoints
        g = d.groupby("day")["c"].agg(["sum", "size"])
        s, n, k = g["sum"].to_numpy(), g["size"].to_numpy(), len(g)
        boot = [s[i].sum() / n[i].sum() for i in (rng.integers(0, k, k) for _ in range(draws))]
        q = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(model=model, nominal=level, days=k, hours=int(n.sum()), coverage=s.sum() / n.sum(),
                         ci95_low=q[0], ci95_high=q[1], draws=draws, seed=seed))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "outputs", "coverage_bootstrap.csv"), index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(main(a.draws, a.seed).round(4).to_string(index=False))
