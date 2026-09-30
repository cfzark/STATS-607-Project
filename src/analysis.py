"""Compose model updates, sampling, losses, and matching for each prior."""

import numpy as np

from config import prior_locations
from losses import RiskLoss
from matching import MatchingRule, select_match
from model import NIGModel
from sampling import make_sampler
from targets import TARGETS


def analyze_location(y, mu0, grid_index, config):
    """One independent task; seed uses the original full-grid index."""
    model = NIGModel()
    reference = model.reference(y)
    rule, loss = MatchingRule(config.method), RiskLoss(config.loss)
    sampler = make_sampler(config.sampling, model, y)
    rng = np.random.default_rng(config.seed + grid_index)
    n = len(y)
    datasets, rejected = {}, {}
    # Preserve candidate-grid draw order and reuse size n for both sides/targets.
    for m in rule.grid:
        count = rule.candidate_size(n, m)
        if count <= 3:
            raise ValueError("The current risk engine requires sample sizes greater than three")
        datasets[count], rejected[count] = sampler.draw(count, config.risk_reps, rng)
    fixed = model.update(datasets[n], rule.fixed_prior(mu0), include_variance=loss.include_variance)
    candidates = [model.update(datasets[rule.candidate_size(n, m)], rule.candidate_prior(mu0),
                               include_variance=loss.include_variance) for m in rule.grid]
    rows = []
    for name in config.targets:
        target = TARGETS[name]
        wanted = loss.risk(fixed, reference, target)
        curve = [loss.risk(parts, reference, target) for parts in candidates]
        j = select_match(rule.grid, wanted, curve)
        row = dict(method=config.method, target=name, mu0=mu0,
                   epss=rule.grid[j], fixed_risk=wanted, candidate_risk=curve[j],
                   boundary=int(j in (0, len(curve) - 1)))
        if config.method == "reimherr" or config.sampling == "bootstrap":
            row["bootstrap_rejections"] = sum(rejected.values())
        row["risk_reps"] = config.risk_reps
        rows.append(row)
    return rows


def run_analysis(y, config, progress=None):
    rows = []
    locations = prior_locations(y)
    for index, mu0 in enumerate(locations):
        rows.extend(analyze_location(y, mu0, index, config))
        if progress is not None:
            progress(index + 1, len(locations), mu0)
    return rows
