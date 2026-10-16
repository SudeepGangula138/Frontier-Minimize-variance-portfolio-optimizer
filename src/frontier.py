import numpy as np
from src.optimization import minimum_variance_target_return
from src.portfolio import (portfolio_return,portfolio_volatility)
def generate_efficient_frontier(covariance_matrix,expected_returns,num_points=50):
    covariance_matrix=np.asarray(covariance_matrix,dtype=float)
    expected_returns=np.asarray(expected_returns,dtype=float)
    min_return=np.min(expected_returns)
    max_return=np.max(expected_returns)
    target_returns=np.linspace(min_return,max_return,num_points)
    frontier_returns=[]
    frontier_volatilities=[]
    frontier_weights=[]
    for target_return in target_returns:
        try:
            weights=minimum_variance_target_return(covariance_matrix,expected_returns,target_return)
        except RuntimeError:
            continue
        actual_return=portfolio_return(weights,expected_returns)
        volatility=portfolio_volatility(weights,covariance_matrix)
        frontier_returns.append(actual_return)
        frontier_volatilities.append(volatility)
        frontier_weights.append(weights)
    return {"returns":np.array(frontier_returns),"volatilities":np.array(frontier_volatilities),"weights":np.array(frontier_weights)}
def find_global_minimum_variance_portfolio(covariance_matrix):
    from src.optimization import (minimum_variance_long_only_active_set)
    weights = minimum_variance_long_only_active_set(covariance_matrix)
    volatility = portfolio_volatility(weights,covariance_matrix)
    return {"weights": weights,"volatility": volatility}

