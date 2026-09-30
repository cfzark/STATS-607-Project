"""Command-line choices for the analysis runner."""

import argparse
from pathlib import Path

from analysis import run_analysis
from config import AnalysisConfig
from data import load_wing_lengths
from reporting import write_results


def main(project_root, argv=None):
    parser = argparse.ArgumentParser(description="Midge EPSS risk matching")
    parser.add_argument("--method", required=True, choices=("reimherr", "wiesenfarth"))
    parser.add_argument("--sampling", choices=("bootstrap", "likelihood"))
    parser.add_argument("--loss", choices=("posterior_mse", "squared_mean"))
    parser.add_argument("--target", choices=("both", "mu", "joint_log"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path(project_root) / "results",
                        help="Output root (explicit relative paths are relative to the working directory)")
    args = parser.parse_args(argv)
    method = args.method
    targets = ("mu", "joint_log") if args.target == "both" else (args.target,)
    config = AnalysisConfig(method, sampling=args.sampling, loss=args.loss, targets=targets)
    y = load_wing_lengths(Path(project_root) / "data/raw/wing_lengths.csv")
    print(f"{config.method}: sampling={config.sampling}, loss={config.loss}, targets={','.join(targets)}", flush=True)
    def progress(done, total, mu0):
        print(method.capitalize(), done, "/", total, "mu0", mu0, flush=True)
    rows = run_analysis(y, config, progress)
    print(write_results(rows, y, config, args.output_dir))
