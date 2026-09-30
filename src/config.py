"""Validated analysis choices and the fixed real-data prior grid."""

from dataclasses import dataclass

import numpy as np

from losses import RiskLoss
from matching import MatchingRule
from targets import TARGETS


DEFAULTS = {
    "reimherr": ("bootstrap", "posterior_mse"),
    "wiesenfarth": ("likelihood", "squared_mean"),
}


@dataclass(frozen=True)
class AnalysisConfig:
    method: str
    sampling: str | None = None
    loss: str | None = None
    targets: tuple[str, ...] = ("mu", "joint_log")
    risk_reps: int = 1000
    seed: int = 20260312

    def __post_init__(self):
        MatchingRule(self.method)
        sampling, loss = DEFAULTS[self.method]
        if self.sampling is None:
            object.__setattr__(self, "sampling", sampling)
        if self.loss is None:
            object.__setattr__(self, "loss", loss)
        if self.sampling not in ("bootstrap", "likelihood"):
            raise ValueError(f"Unknown sampling strategy: {self.sampling}")
        RiskLoss(self.loss)
        if not self.targets or len(set(self.targets)) != len(self.targets) or any(t not in TARGETS for t in self.targets):
            raise ValueError("Choose unique targets from mu and joint_log")
        if not isinstance(self.risk_reps, int) or self.risk_reps < 1:
            raise ValueError("risk_reps must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @property
    def output_stem(self):
        if (self.sampling, self.loss) == DEFAULTS[self.method] and self.targets == ("mu", "joint_log"):
            return self.method
        target_suffix = "" if self.targets == ("mu", "joint_log") else "__" + "_".join(self.targets)
        return f"{self.method}__{self.sampling}__{self.loss}{target_suffix}"


def prior_locations(y):
    ticks = set(range(1400, 2201, 20)) | set(range(1800, 2001, 5))
    ticks |= set(range(1300, 1381, 20)) | set(range(2220, 2301, 20))
    return sorted([v / 1000 for v in ticks] + [float(np.mean(y))])
