"""Reimherr EPSS analysis for midge wing lengths."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_wing_lengths
from nig import posterior_moments


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    y = load_wing_lengths(os.path.join(ROOT, "data", "raw", "wing_lengths.csv"))
    n = len(y)
    ticks = set(range(1400, 2201, 20)) | set(range(1800, 2001, 5))
    ticks |= set(range(1300, 1381, 20)) | set(range(2220, 2301, 20))
    locations = sorted([v / 1000 for v in ticks] + [float(np.mean(y))])
    B = 1000
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)

    original = y[None, :]
    baseline_reference = posterior_moments(original)
    candidate_m = list(range(-3, 41))  # baseline size n+m, informative size n
    rows = []
    for i, mu0 in enumerate(locations):
        rng = np.random.default_rng(20260312 + i)
        boot = {}
        rejected = {}
        for m in candidate_m:
            count = n + m
            z = rng.choice(y, size=(B, count), replace=True)
            bad = np.ptp(z, axis=1) == 0
            rejects = 0
            # Conditional bootstrap: every accepted sample has a proper Jeffreys posterior.
            while np.any(bad):
                rejects += int(np.sum(bad))
                if rejects > 1000 * B:
                    raise RuntimeError("Bootstrap redraw limit exceeded")
                z[bad] = rng.choice(y, size=(int(np.sum(bad)), count), replace=True)
                bad = np.ptp(z, axis=1) == 0
            boot[count] = z
            rejected[count] = rejects
        fixed = posterior_moments(boot[n], mu0)
        for target in ("mu", "joint_log"):
            dims = (0,) if target == "mu" else (0, 1)
            def risk(parts):
                return float(np.mean(sum((parts[d] - baseline_reference[d][0]) ** 2 + parts[d + 2]
                                         for d in dims)))
            wanted = risk(fixed)
            curve = [risk(posterior_moments(boot[n + m])) for m in candidate_m]
            j = int(np.argmin(np.abs(np.array(curve) - wanted)))
            rows.append(dict(method="reimherr", target=target, mu0=mu0,
                             epss=candidate_m[j], fixed_risk=wanted,
                             candidate_risk=curve[j], boundary=int(j in (0, len(curve)-1)),
                             bootstrap_rejections=sum(rejected.values()), risk_reps=B))
        print("Reimherr", i + 1, "/", len(locations), "mu0", mu0, flush=True)

    out = os.path.join(ROOT, "results", "tables", "reimherr.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for target in ("mu", "joint_log"):
        selected = [r for r in rows if r["target"] == target]
        plt.plot([r["mu0"] for r in selected], [r["epss"] for r in selected], label=target)
    plt.axvline(np.mean(y), color="grey", linestyle="--")
    plt.xlabel("prior location mu0 (mm)")
    plt.ylabel("EPSS (observations)")
    plt.legend()
    plt.savefig(os.path.join(ROOT, "results", "figures", "reimherr.png"), dpi=150)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
