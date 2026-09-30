"""Command-line entry point for configurable risk matching."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli import main as run_cli


def main(argv=None):
    run_cli(ROOT, argv)


if __name__ == "__main__":
    main()
