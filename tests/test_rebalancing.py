
import numpy as np
from src.optimization import rebalance_portfolio
def test_rebalance_respects_constraints():
    covariance = np.array([[0.04,0.01,0.005],[0.01,0.09,0.002],[0.005,0.002,0.025],])
    current = np.array([0.50,0.30,0.20])
    weights = rebalance_portfolio(covariance,current,max_weight=0.60,max_weight_change=0.10,turnover_penalty=0.01,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-7)
    assert np.all(weights >= -1e-8)
    assert np.all(weights <= 0.60 + 1e-7)
    assert np.all(np.abs(weights - current) <= 0.10 + 1e-7)
def test_rebalance_rejects_invalid_current_weights():
    covariance = np.eye(3)
    try:
        rebalance_portfolio(covariance, [0.5, 0.5, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid current weights should be rejected.")
