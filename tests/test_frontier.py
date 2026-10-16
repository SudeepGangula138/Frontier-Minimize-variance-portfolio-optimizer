import numpy as np
from src.frontier import generate_efficient_frontier
def test_frontier_returns_correct_shape():
    covariance = np.array([[0.04, 0.01, 0.005],[0.01, 0.09, 0.002],[0.005, 0.002, 0.025]])
    expected_returns = np.array([0.10,0.15,0.08])
    frontier = generate_efficient_frontier(covariance,expected_returns,num_points=10)
    assert len(frontier["returns"]) > 0
    assert len(frontier["returns"]) == \
           len(frontier["volatilities"])
    assert len(frontier["returns"]) == \
           len(frontier["weights"])