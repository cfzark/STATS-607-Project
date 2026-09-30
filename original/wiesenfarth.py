import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma


HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "wing_lengths.csv"), newline="") as f:
    y = np.array([float(row["wing_length_mm"]) for row in csv.DictReader(f)])
n = len(y)
assert n == 9 and np.all(np.isfinite(y))
ticks = set(range(1400, 2201, 20)) | set(range(1800, 2001, 5))
ticks |= set(range(1300, 1381, 20)) | set(range(2220, 2301, 20))
locations = sorted([v / 1000 for v in ticks] + [float(np.mean(y))])
B = 1000
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)


def posterior_means(z, mu0=None):
    count = z.shape[1]
    avg = z.mean(axis=1)
    sse = np.sum((z - avg[:, None]) ** 2, axis=1)
    if mu0 is None:
        center = avg
        a = (count - 1) / 2
        b = sse / 2
    else:
        center = (count * avg + mu0) / (count + 1)
        a = .5 + count / 2
        b = .005 + .5 * (sse + count / (count + 1) * (avg - mu0) ** 2)
    if np.any(b <= 0) or a <= 1:
        raise ValueError("Posterior target mean unavailable")
    return center / .1, np.log(b) - digamma(a) - np.log(.01)


ref = posterior_means(y[None, :])
plugin_mu = float(np.mean(y))
plugin_sigma2 = float(np.sum((y - plugin_mu) ** 2) / (n - 3))
candidate_m = list(range(-40, 6))  # informative size n-m, baseline size n
rows = []
for i, mu0 in enumerate(locations):
    rng = np.random.default_rng(20260312 + i)
    simulated = {}
    for m in candidate_m:
        count = n - m
        simulated[count] = rng.normal(plugin_mu, np.sqrt(plugin_sigma2), size=(B, count))
    fixed = posterior_means(simulated[n])
    for target in ("mu", "joint_log"):
        dims = (0,) if target == "mu" else (0, 1)
        def risk(parts):
            return float(np.mean(sum((parts[d] - ref[d][0]) ** 2 for d in dims)))
        wanted = risk(fixed)
        curve = [risk(posterior_means(simulated[n - m], mu0)) for m in candidate_m]
        j = int(np.argmin(np.abs(np.array(curve) - wanted)))
        rows.append(dict(method="wiesenfarth", target=target, mu0=mu0,
                         epss=candidate_m[j], fixed_risk=wanted,
                         candidate_risk=curve[j], boundary=int(j in (0, len(curve)-1)),
                         risk_reps=B))
    print("Wiesenfarth", i + 1, "/", len(locations), "mu0", mu0, flush=True)

out = os.path.join(HERE, "output", "wiesenfarth.csv")
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
plt.savefig(os.path.join(HERE, "output", "wiesenfarth.png"), dpi=150)
print(out)
