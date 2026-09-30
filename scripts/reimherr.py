import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, polygamma


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "raw", "wing_lengths.csv"), newline="") as f:
    y = np.array([float(row["wing_length_mm"]) for row in csv.DictReader(f)])
n = len(y)
assert n == 9 and np.all(np.isfinite(y))
ticks = set(range(1400, 2201, 20)) | set(range(1800, 2001, 5))
ticks |= set(range(1300, 1381, 20)) | set(range(2220, 2301, 20))
locations = sorted([v / 1000 for v in ticks] + [float(np.mean(y))])
B = 1000
os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)


def stuff(z, mu0=None):
    # Rows are bootstrap datasets. The baseline is p(mu,sigma2) proportional to 1/sigma2.
    count = z.shape[1]
    avg = z.mean(axis=1)
    sse = np.sum((z - avg[:, None]) ** 2, axis=1)
    if mu0 is None:
        lam, center, a, b = count, avg, (count - 1) / 2, sse / 2
    else:
        lam = count + 1
        center = (count * avg + mu0) / lam
        a = .5 + count / 2
        b = .005 + .5 * (sse + count / lam * (avg - mu0) ** 2)
    if np.any(b <= 0) or a <= 1:
        raise ValueError("Posterior risk moments unavailable")
    first = center / .1
    second = np.log(b) - digamma(a) - np.log(.01)
    var1 = b / ((a - 1) * lam * .1 ** 2)
    var2 = np.full_like(first, polygamma(1, a))
    return first, second, var1, var2


original = y[None, :]
baseline_reference = stuff(original)
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
    fixed = stuff(boot[n], mu0)
    for target in ("mu", "joint_log"):
        dims = (0,) if target == "mu" else (0, 1)
        def risk(parts):
            return float(np.mean(sum((parts[d] - baseline_reference[d][0]) ** 2 + parts[d + 2]
                                     for d in dims)))
        wanted = risk(fixed)
        curve = [risk(stuff(boot[n + m])) for m in candidate_m]
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
print(out)
