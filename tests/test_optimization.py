import numpy as np
from src.optimization import minimum_variance_portfolio,minimum_variance_kkt,minimum_variance_long_only,minimum_variance_long_only_active_set,minimum_variance_target_return
def test_weights_sum_to_one():
    covariance=np.array([[0.04,0.01,0.005],[0.01,0.09,0.002],[0.005,0.002,0.025]])
    weights=minimum_variance_portfolio(covariance)
    assert np.isclose(weights.sum(),1.0)
def test_kkt_matches_closed_form():
    covariance = np.array([[0.04, 0.01, 0.005],[0.01, 0.09, 0.002],[0.005, 0.002, 0.025]])
    w1 = minimum_variance_portfolio(covariance)
    w2, _ = minimum_variance_kkt(covariance)
    assert np.allclose(w1, w2)
def test_long_only_weights():
    covariance = np.array([[0.04, 0.01, 0.005],[0.01, 0.09, 0.002],[0.005, 0.002, 0.025]])
    weights = minimum_variance_long_only(covariance)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-10)
def test_long_only_active_set():
    covariance = np.array([[0.04, 0.01, 0.005],[0.01, 0.09, 0.002],[0.005, 0.002, 0.025]])
    weights = minimum_variance_long_only_active_set(covariance)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-10)
def test_target_return_portfolio():
    covariance=np.array([[0.04, 0.01, 0.005],[0.01, 0.09, 0.002],[0.005, 0.002, 0.025]])
    expected_returns = np.array([0.10,0.15,0.08])
    target_return=0.12
    weights=minimum_variance_target_return(covariance,expected_returns,target_return)
    assert np.isclose(weights.sum(),1.0)
    assert np.isclose(weights@expected_returns,target_return)
    assert np.all(weights>=-1e-10)