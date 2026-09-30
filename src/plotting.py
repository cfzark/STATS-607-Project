"""Regenerate figures exclusively from validated CSV results."""

import csv
from pathlib import Path

import numpy as np

from validation import validate_result


def plot_result(output_dir, y, config):
    validate_result(output_dir, y, config)
    root = Path(output_dir)
    with (root / "tables" / f"{config.output_stem}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.8))
    for target in config.targets:
        selected = sorted((row for row in rows if row["target"] == target), key=lambda row: float(row["mu0"]))
        plt.plot([float(row["mu0"]) for row in selected],
                 [float(row["epss"]) for row in selected], label=target)
    plt.axvline(np.mean(y), color="grey", linestyle="--")
    plt.xlabel("prior location mu0 (mm)")
    plt.ylabel("EPSS (observations)")
    plt.legend()
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure = folder / f"{config.output_stem}.png"
    plt.savefig(figure, dpi=150)
    plt.close()
    return figure
