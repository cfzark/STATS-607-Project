"""Validate persisted scientific results before plotting or declaring completion."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import all_configs, prior_locations
from matching import MatchingRule
from reporting import result_metadata


def validate_result(output_dir, y, config, require_figure=False):
    root = Path(output_dir)
    stem = config.output_stem
    table = root / "tables" / f"{stem}.csv"
    metadata_file = root / "metadata" / f"{stem}.json"
    with table.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = ["method", "target", "mu0", "epss", "fixed_risk", "candidate_risk", "boundary"]
        if config.method == "reimherr" or config.sampling == "bootstrap":
            columns.append("bootstrap_rejections")
        columns.append("risk_reps")
        if reader.fieldnames != columns:
            raise ValueError(f"{stem}: unexpected CSV columns")
        rows = list(reader)
    expected = {(float(mu0), target) for mu0 in prior_locations(y) for target in config.targets}
    if len(rows) != len(expected):
        raise ValueError(f"{stem}: expected {len(expected)} rows, found {len(rows)}")
    grid = MatchingRule(config.method).grid
    seen = set()
    for row in rows:
        if None in row or row["method"] != config.method:
            raise ValueError(f"{stem}: malformed row or mismatched method")
        numeric = {key: float(row[key]) for key in columns if key not in ("method", "target")}
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError(f"{stem}: nonfinite numeric value")
        key = (numeric["mu0"], row["target"])
        if key in seen:
            raise ValueError(f"{stem}: duplicate prior-target key {key}")
        seen.add(key)
        if numeric["epss"] not in grid or numeric["boundary"] != int(numeric["epss"] in (grid[0], grid[-1])):
            raise ValueError(f"{stem}: invalid EPSS or boundary flag")
        if numeric["fixed_risk"] < 0 or numeric["candidate_risk"] < 0:
            raise ValueError(f"{stem}: negative risk")
        if numeric["risk_reps"] != config.risk_reps:
            raise ValueError(f"{stem}: risk_reps does not match configuration")
        if "bootstrap_rejections" in numeric:
            count = numeric["bootstrap_rejections"]
            if count < 0 or count != int(count):
                raise ValueError(f"{stem}: invalid rejection count")
    if seen != expected:
        raise ValueError(f"{stem}: missing or unexpected prior-target keys")
    metadata = json.loads(metadata_file.read_text())
    if metadata != result_metadata(y, config, table):
        raise ValueError(f"{stem}: metadata, configuration, or table hash mismatch")
    files = [table, metadata_file]
    if require_figure:
        figure = root / "figures" / f"{stem}.png"
        with Image.open(figure) as image:
            if image.format != "PNG" or image.size != (960, 720):
                raise ValueError(f"{stem}: invalid figure format or dimensions")
            image.verify()
        files.append(figure)
    return {"configuration": stem, "rows": len(rows), "files": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def validate_all(output_dir, y):
    """Write a success report only after all eight complete outputs pass."""
    root = Path(output_dir)
    report = root / "validation.json"
    report.unlink(missing_ok=True)
    checks = [validate_result(root, y, config, require_figure=True) for config in all_configs()]
    payload = {"status": "passed", "combinations": len(checks),
               "rows": sum(check["rows"] for check in checks), "checks": checks}
    report.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
