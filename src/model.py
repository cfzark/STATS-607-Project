"""Shared NIG posterior calculations for batches of equal-size datasets."""

import numpy as np
from scipy.special import digamma, polygamma
from dataclasses import dataclass


@dataclass(frozen=True)
class NIGModel:
    """Normal likelihood, fixed NIG strength, and independence-Jeffreys baseline.

    The informative prior is specified by mu0; None means the baseline.
    Updating means fitting each complete pseudo-dataset, not appending it to y.
    """

    def update(self, datasets, mu0=None, *, include_variance=True):
        if include_variance:
            return posterior_moments(datasets, mu0)
        return posterior_means(datasets, mu0)

    def reference(self, y):
        return posterior_means(np.asarray(y)[None, :])

    def likelihood_parameters(self, y):
        # Baseline E[mu|y] and E[sigma^2|y], not the MLE variance.
        y = np.asarray(y)
        self.reference(y)
        mean = float(np.mean(y))
        variance = float(np.sum((y - mean) ** 2) / (len(y) - 3))
        return mean, variance


def posterior_parameters(z, mu0=None):
    """Return (lambda, center, shape, scale), one dataset per row.

    mu0=None selects the independence-Jeffreys baseline 1/sigma^2.
    Otherwise the prior has lambda=1, shape=0.5, and scale=0.005.
    Both analyses retain the requirement shape > 1 for finite risk moments.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or 0 in z.shape or not np.all(np.isfinite(z)):
        raise ValueError("Datasets must be a nonempty finite two-dimensional array")
    if mu0 is not None and (np.ndim(mu0) != 0 or not np.isfinite(mu0)):
        raise ValueError("Prior location must be a finite scalar")
    count = z.shape[1]
    avg = z.mean(axis=1)
    sse = np.sum((z - avg[:, None]) ** 2, axis=1)
    if mu0 is None:
        lam, center, a, b = count, avg, (count - 1) / 2, sse / 2
    else:
        lam = count + 1
        center = (count * avg + mu0) / lam
        a = .5 + count / 2
        b = .005 + .5 * (sse + count / lam * (avg - mu0) ** 2)
    if np.any(b <= 0) or a <= 1:
        raise ValueError("Posterior risk moments unavailable")
    return lam, center, a, b


def _target_means(center, a, b):
    return center / .1, np.log(b) - digamma(a) - np.log(.01)


def posterior_means(z, mu0=None):
    """Return E[mu/0.1] and E[log(sigma^2/0.01)] for each dataset."""
    _, center, a, b = posterior_parameters(z, mu0)
    return _target_means(center, a, b)


def posterior_moments(z, mu0=None):
    """Return the two target means followed by their marginal variances."""
    lam, center, a, b = posterior_parameters(z, mu0)
    first, second = _target_means(center, a, b)
    var1 = b / ((a - 1) * lam * .1 ** 2)
    var2 = np.full_like(first, polygamma(1, a))
    return first, second, var1, var2
