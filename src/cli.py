"""Command-line choices and the serial reproduce/plot/validate workflow."""

import argparse
from pathlib import Path

from analysis import run_analysis
from config import AnalysisConfig, all_configs
from data import load_wing_lengths
from plotting import plot_result
from reporting import write_results
from validation import validate_all, validate_result


def main(project_root, argv=None):
    parser = argparse.ArgumentParser(description="Midge EPSS risk matching")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--method", choices=("reimherr", "wiesenfarth"))
    selection.add_argument("--all", action="store_true", help="Run all 8 method/sampling/loss combinations with both targets")
    parser.add_argument("--sampling", choices=("bootstrap", "likelihood"))
    parser.add_argument("--loss", choices=("posterior_mse", "squared_mean"))
    parser.add_argument("--target", choices=("both", "mu", "joint_log"), default="both")
    parser.add_argument("--stage", choices=("reproduce", "plot", "validate"), default="reproduce",
                        help="Reproduce everything, replot saved CSVs, or only validate saved outputs")
    parser.add_argument("--output-dir", type=Path, default=Path(project_root) / "results",
                        help="Output root (explicit relative paths are relative to the working directory)")
    args = parser.parse_args(argv)
    if args.all and (args.sampling is not None or args.loss is not None or args.target != "both"):
        parser.error("--all requires all samplers, losses, and both targets; do not combine it with configuration filters")
    targets = ("mu", "joint_log") if args.target == "both" else (args.target,)
    configs = all_configs() if args.all else [AnalysisConfig(args.method, args.sampling, args.loss, targets)]
    try:
        # A previous completion record must not survive a failed rerun or validation.
        (args.output_dir / "validation.json").unlink(missing_ok=True)
        y = load_wing_lengths(Path(project_root) / "data/raw/wing_lengths.csv")
        for config in configs:
            print(f"{config.method}: sampling={config.sampling}, loss={config.loss}, targets={','.join(config.targets)}, stage={args.stage}", flush=True)
            if args.stage == "reproduce":
                def progress(done, total, mu0):
                    print(config.method.capitalize(), done, "/", total, "mu0", mu0, flush=True)
                rows = run_analysis(y, config, progress)
                print(write_results(rows, y, config, args.output_dir))
            if args.stage in ("reproduce", "plot"):
                print(plot_result(args.output_dir, y, config))
            check = validate_result(args.output_dir, y, config, require_figure=True)
            print(f"Validated {check['configuration']}: {check['rows']} rows", flush=True)
        if args.all:
            report = validate_all(args.output_dir, y)
            print(f"Validation passed: {report['combinations']} combinations, {report['rows']} rows; {args.output_dir / 'validation.json'}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Workflow failed: {error}\n")
