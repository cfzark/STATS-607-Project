"""Per-dataset target losses followed by mean aggregation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskLoss:
    name: str

    def __post_init__(self):
        if self.name not in ("posterior_mse", "squared_mean"):
            raise ValueError(f"Unknown loss: {self.name}")

    @property
    def include_variance(self):
        return self.name == "posterior_mse"

    def per_dataset(self, parts, reference, target):
        if self.include_variance:
            return sum((parts[d] - reference[d][0]) ** 2 + parts[d + 2]
                       for d in target.dimensions)
        return sum((parts[d] - reference[d][0]) ** 2 for d in target.dimensions)

    def risk(self, parts, reference, target):
        value = float(np.mean(self.per_dataset(parts, reference, target)))
        if not np.isfinite(value):
            raise ValueError("Risk must be finite")
        return value
