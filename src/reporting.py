"""Write result tables, figures, and the scientific choices behind them."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from matching import MatchingRule
from config import prior_locations


def write_results(rows, y, config, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    for folder in ("tables", "figures", "metadata"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    stem = config.output_stem
    table = output_dir / "tables" / f"{stem}.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.figure()
    for target in config.targets:
        selected = [row for row in rows if row["target"] == target]
        plt.plot([row["mu0"] for row in selected], [row["epss"] for row in selected], label=target)
    plt.axvline(np.mean(y), color="grey", linestyle="--")
    plt.xlabel("prior location mu0 (mm)")
    plt.ylabel("EPSS (observations)")
    plt.legend()
    plt.savefig(output_dir / "figures" / f"{stem}.png", dpi=150)
    plt.close()
    metadata = {
        "config": asdict(config),
        "model": "Normal likelihood with NIG informative prior",
        "informative_prior": {"lambda": 1, "shape": .5, "scale": .005, "mu0": "prior_locations"},
        "baseline_prior": "p(mu,sigma2) proportional to 1/sigma2",
        "target_coordinates": ["mu/0.1", "log(sigma2/0.01)"],
        "reference": "baseline posterior mean in target coordinates",
        "likelihood_plugin": "baseline E[mu|y], E[sigma2|y] (not MLE)",
        "posterior_update": "fit each complete pseudo-dataset; do not append observed data",
        "matching": "baseline_n_plus_m_vs_informative_n" if config.method == "reimherr" else "informative_n_minus_m_vs_baseline_n",
        "candidate_m": MatchingRule(config.method).grid,
        "prior_locations": prior_locations(y),
        "observations_mm": np.asarray(y).tolist(),
        "aggregation": "mean of per-dataset losses",
        "selection": "minimum absolute risk gap; first grid entry wins ties",
        "sampling_policy": "conditional_on_baseline_posterior_proper_at_every_size" if config.sampling == "bootstrap" else "unconditional_normal_likelihood",
        "bootstrap_rejection_count": "per prior location, repeated across target rows; do not sum targets",
        "coupling": "shared by size within each prior location, across both sides and targets",
        "seed_rule": "seed + original prior grid index; candidate sizes drawn in grid order",
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    (output_dir / "metadata" / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return table
