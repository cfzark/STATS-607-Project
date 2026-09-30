"""Read and validate the nine observed midge wing lengths."""

import csv

import numpy as np


def load_wing_lengths(path):
    """Require one named column and nine finite, positive lengths in mm."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["wing_length_mm"]:
            raise ValueError("Expected the single CSV column wing_length_mm")
        values = []
        for row in reader:
            if None in row:
                raise ValueError("Each data row must contain exactly one wing length")
            try:
                values.append(float(row["wing_length_mm"]))
            except (TypeError, ValueError) as error:
                raise ValueError("Wing lengths must be numeric") from error
    y = np.asarray(values, dtype=float)
    if len(y) != 9:
        raise ValueError("Expected exactly nine wing lengths")
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Wing lengths must be finite and positive")
    return y
