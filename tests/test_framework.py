"""Scientific contracts of the independently selectable risk components."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from analysis import analyze_location
from config import AnalysisConfig, prior_locations
from data import load_wing_lengths
from losses import RiskLoss
from matching import MatchingRule, select_match
from model import NIGModel
from sampling import BootstrapSampler, LikelihoodSampler
from targets import TARGETS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def y():
    return load_wing_lengths(ROOT / "data/raw/wing_lengths.csv")


@pytest.mark.parametrize("method", ["reimherr", "wiesenfarth"])
def test_default_risks_match_preserved_initial_results(y, method):
    with (ROOT / "original/output" / f"{method}.csv").open() as stream:
        expected = list(csv.DictReader(stream))
    grid = prior_locations(y)
    # Both endpoints and the observed-mean location, using original full-grid seeds.
    for index in (0, grid.index(float(y.mean())), len(grid) - 1):
        actual = analyze_location(y, grid[index], index, AnalysisConfig(method))
        for row, saved in zip(actual, expected[2 * index:2 * index + 2]):
            assert set(row) == set(saved)
            for key, value in row.items():
                if isinstance(value, str):
                    assert value == saved[key]
                else:
                    assert value == pytest.approx(float(saved[key]), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("method", ["reimherr", "wiesenfarth"])
@pytest.mark.parametrize("sampling", ["bootstrap", "likelihood"])
@pytest.mark.parametrize("loss", ["posterior_mse", "squared_mean"])
def test_all_eight_combinations_and_target_invariance(y, method, sampling, loss):
    config = AnalysisConfig(method, sampling, loss, risk_reps=16)
    rows = analyze_location(y, 1.9, 0, config)
    assert len(rows) == 2
    for row in rows:
        assert row["epss"] in MatchingRule(method).grid
        assert row["fixed_risk"] >= 0 and np.isfinite(row["candidate_risk"])
        single = analyze_location(y, 1.9, 0, replace(config, targets=(row["target"],)))
        assert single == [row]
    assert rows == analyze_location(y, 1.9, 0, config)


def test_two_loss_definitions():
    parts = (np.array([1., 3.]), np.array([2., 4.]),
             np.array([.5, 1.5]), np.array([2., 3.]))
    reference = (np.array([2.]), np.array([1.]))
    assert RiskLoss("squared_mean").risk(parts, reference, TARGETS["mu"]) == 1
    assert RiskLoss("posterior_mse").risk(parts, reference, TARGETS["mu"]) == 2
    assert RiskLoss("squared_mean").risk(parts, reference, TARGETS["joint_log"]) == 6
    assert RiskLoss("posterior_mse").risk(parts, reference, TARGETS["joint_log"]) == 9.5


def test_matching_directions_and_ties():
    r, w = MatchingRule("reimherr"), MatchingRule("wiesenfarth")
    assert (r.fixed_prior(1.9), r.candidate_prior(1.9), r.candidate_size(9, 2)) == (1.9, None, 11)
    assert (w.fixed_prior(1.9), w.candidate_prior(1.9), w.candidate_size(9, 2)) == (None, 1.9, 7)
    assert select_match([-1, 0, 1], 2., [1., 3., 8.]) == 0
    with pytest.raises(ValueError, match="nonfinite"):
        select_match([0], 2., [np.nan])


def test_conditional_bootstrap_counts_redraws():
    sampler = BootstrapSampler(np.array([1., 2.]))
    actual, rejected = sampler.draw(4, 100, np.random.default_rng(7))
    assert rejected > 0
    assert np.all(np.ptp(actual, axis=1) > 0)
    assert set(actual.ravel()) == {1., 2.}
    repeated, count = sampler.draw(4, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(actual, repeated)
    assert count == rejected
    with pytest.raises(ValueError, match="unattainable"):
        BootstrapSampler(np.ones(9)).draw(4, 100, np.random.default_rng(7))


def test_likelihood_uses_baseline_posterior_variance(y):
    mean, variance = NIGModel().likelihood_parameters(y)
    assert mean == pytest.approx(16.24 / 9)
    assert variance == pytest.approx((1519 / 22500) / 3)
    assert variance != pytest.approx(float(np.var(y)))
    actual, rejected = LikelihoodSampler(mean, variance).draw(4, 10, np.random.default_rng(7))
    expected = np.random.default_rng(7).normal(mean, np.sqrt(variance), size=(10, 4))
    np.testing.assert_array_equal(actual, expected)
    assert rejected == 0


@pytest.mark.parametrize("kwargs", [
    {"method": "unknown"}, {"sampling": "unknown"}, {"loss": "unknown"},
    {"targets": ("bad",)}, {"targets": ("mu", "mu")}, {"risk_reps": 0}, {"seed": -1},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**({"method": "reimherr"} | kwargs))
