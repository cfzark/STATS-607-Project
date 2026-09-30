import numpy as np
import pytest
from scipy.stats import invgamma, t

from model import posterior_means, posterior_moments, posterior_parameters


Y = np.array([[1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08]])


def test_hoff_informative_posterior():
    # Independently calculated conjugate posterior for the book prior mu0=1.9.
    lam, center, shape, scale = posterior_parameters(Y, 1.9)
    assert lam == 10
    assert shape == 5
    np.testing.assert_allclose(center, [1.814], rtol=0, atol=1e-14)
    np.testing.assert_allclose(scale, [.07662], rtol=0, atol=1e-14)


@pytest.mark.parametrize("mu0", [None, 1.9])
def test_target_moments_against_distribution_integrals(mu0):
    # Reference parameters are calculated independently of the function under test.
    # sum(y)=406/25 and sum(y^2)=36799/1250 give baseline scale=1519/22500.
    if mu0 is None:
        lam, center, shape, scale = 9, 16.24 / 9, 4, 1519 / 22500
    else:
        lam, center, shape, scale = 10, 1.814, 5, .07662
    mu_dist = t(df=2 * shape, loc=center, scale=np.sqrt(scale / (shape * lam)))
    variance_dist = invgamma(a=shape, scale=scale)
    log_mean = variance_dist.expect(lambda x: np.log(x / .01))
    log_var = variance_dist.expect(lambda x: (np.log(x / .01) - log_mean) ** 2)
    expected = [mu_dist.mean() / .1, log_mean, mu_dist.var() / .1**2, log_var]
    actual = np.asarray(posterior_moments(Y, mu0))[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(posterior_means(Y, mu0))[:, 0], expected[:2], rtol=1e-9)


def test_batch_rows_are_independent():
    batch = np.concatenate([Y, Y + .2])
    actual = np.asarray(posterior_moments(batch, 1.9))
    expected = np.concatenate([np.asarray(posterior_moments(row[None, :], 1.9)) for row in batch], axis=1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("z, mu0", [
    (np.ones((1, 9)), None),  # Zero baseline scale: improper posterior.
    (np.array([[1., 2., 3.]]), None),  # Infinite baseline risk moments.
    (np.array([[1., np.nan, 2., 3.]]), 1.9),
    (np.array([1., 2., 3., 4.]), 1.9),
    (np.empty((0, 9)), 1.9),
    (Y, np.inf),
])
def test_reject_unavailable_moments(z, mu0):
    with pytest.raises(ValueError):
        posterior_moments(z, mu0)
