from pathlib import Path

import numpy as np
import pytest

from data import load_wing_lengths


def test_published_observations():
    path = Path(__file__).resolve().parents[1] / "data/raw/wing_lengths.csv"
    np.testing.assert_array_equal(
        load_wing_lengths(path),
        [1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08],
    )


@pytest.mark.parametrize("contents, message", [
    ("length\n" + "1.8\n" * 9, "single CSV column"),
    ("wing_length_mm\n" + "1.8\n" * 8, "exactly nine"),
    ("wing_length_mm\n" + "1.8\n" * 8 + "nan\n", "finite and positive"),
    ("wing_length_mm\n" + "1.8\n" * 8 + "inf\n", "finite and positive"),
    ("wing_length_mm\n" + "1.8\n" * 8 + "0\n", "finite and positive"),
    ("wing_length_mm\n" + "1.8\n" * 8 + "-1\n", "finite and positive"),
    ("wing_length_mm\n" + "1.8\n" * 8 + "bad\n", "numeric"),
    ("wing_length_mm\n1.8,2.0\n", "exactly one"),
])
def test_reject_invalid_data(tmp_path, contents, message):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        load_wing_lengths(path)
