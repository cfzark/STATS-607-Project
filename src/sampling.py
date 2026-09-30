"""Pseudo-data laws, independent of risk matching direction and loss."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapSampler:
    """Bootstrap conditional on a proper baseline posterior at every size.

    Constant rows are redrawn and counted, including informative-side draws.
    This same explicit conditional law applies to both matching methods.
    """
    data: np.ndarray

    def draw(self, count, repetitions, rng):
        if count < 2 or np.ptp(self.data) == 0:
            raise ValueError("A proper baseline bootstrap posterior is unattainable")
        z = rng.choice(self.data, size=(repetitions, count), replace=True)
        bad = np.ptp(z, axis=1) == 0
        rejected = 0
        while np.any(bad):
            rejected += int(np.sum(bad))
            if rejected > 1000 * repetitions:
                raise RuntimeError("Bootstrap redraw limit exceeded")
            z[bad] = rng.choice(self.data, size=(int(np.sum(bad)), count), replace=True)
            bad = np.ptp(z, axis=1) == 0
        return z, rejected


@dataclass(frozen=True)
class LikelihoodSampler:
    """Normal draws at fixed baseline posterior-mean natural parameters."""
    mean: float
    variance: float

    def draw(self, count, repetitions, rng):
        return rng.normal(self.mean, np.sqrt(self.variance), size=(repetitions, count)), 0


def make_sampler(name, model, data):
    if name == "bootstrap":
        return BootstrapSampler(np.asarray(data))
    if name == "likelihood":
        return LikelihoodSampler(*model.likelihood_parameters(data))
    raise ValueError(f"Unknown sampling strategy: {name}")
