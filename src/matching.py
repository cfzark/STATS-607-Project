"""Sample-size update rules and discrete risk matching."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchingRule:
    name: str

    def __post_init__(self):
        if self.name not in ("reimherr", "wiesenfarth"):
            raise ValueError(f"Unknown matching method: {self.name}")

    @property
    def grid(self):
        return tuple(range(-3, 41)) if self.name == "reimherr" else tuple(range(-40, 6))

    def fixed_prior(self, mu0):
        return mu0 if self.name == "reimherr" else None

    def candidate_prior(self, mu0):
        return None if self.name == "reimherr" else mu0

    def candidate_size(self, n, m):
        return n + m if self.name == "reimherr" else n - m


def select_match(grid, fixed_risk, candidate_risks):
    """Minimize absolute risk gap; the first grid entry wins exact ties."""
    values = np.asarray(candidate_risks)
    if values.shape != (len(grid),) or not len(grid):
        raise ValueError("Candidate risks must match a nonempty grid")
    if not np.all(np.isfinite(values)) or not np.isfinite(fixed_risk):
        raise ValueError("Cannot select from nonfinite risks")
    index = int(np.argmin(np.abs(values - fixed_risk)))
    return index
