import numpy as np
def portfolio_return(weights,expected_returns):
    # Returns R_p = w^T μ
    weights=np.asarray(weights,dtype=float)
    expected_returns=np.asarray(expected_returns,dtype=float)
    return weights@expected_returns

def portfolio_variance(weights,covariance_matrix):
    # Variance  σ²_p = w^T Σ w
    weights=np.asarray(weights,dtype=float)
    covariance_matrix=np.asarray(covariance_matrix,dtype=float)
    return weights@covariance_matrix@weights

def portfolio_volatility(weights,covariance_matrix):
    # Volatility  σ_p = sqrt(w^T Σ w)
    variance=portfolio_variance(weights,covariance_matrix)
    return np.sqrt(variance)