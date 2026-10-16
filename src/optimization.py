import numpy as np
import itertools
from scipy.optimize import minimize
def _solve_equality_min_variance(covariance_matrix):
    sigma=np.asarray(covariance_matrix,dtype=float)
    n=sigma.shape[0]
    ones=np.ones(n)
    KKT=np.block([[sigma,-ones.reshape(-1,1)],[ones.reshape(1,-1),np.zeros((1,1))]])
    rhs=np.concatenate([np.zeros(n),[1.0]])
    solution=np.linalg.solve(KKT,rhs)
    return solution[:n]
def minimum_variance_portfolio(covariance_matrix):
    sigma=np.asarray(covariance_matrix,dtype=float)
    n=sigma.shape[0]
    ones=np.ones(n)
    x=np.linalg.solve(sigma,ones)
    weights=x/(ones@x)
    return weights
def minimum_variance_kkt(covariance_matrix):
    Sigma = np.asarray(covariance_matrix, dtype=float)
    n = Sigma.shape[0]
    ones = np.ones(n)
    KKT = np.block([[Sigma,-ones.reshape(-1, 1)],[ones.reshape(1, -1),np.zeros((1, 1))]])
    rhs = np.concatenate([np.zeros(n),[1.0]])
    solution = np.linalg.solve(KKT, rhs)
    weights = solution[:n]
    lagrange_multiplier = solution[n]
    return weights, lagrange_multiplier
def minimum_variance_long_only(covariance_matrix):
    sigma=np.asarray(covariance_matrix,dtype=float)
    n=sigma.shape[0]
    initial_weights=np.ones(n)/n
    def objective(weights):
        return 0.5*weights@sigma@weights
    constraints=[{"type":"eq","fun":lambda weights:np.sum(weights)-1}]
    bounds=[(0.0,None)]*n
    result=minimize(objective,initial_weights,method="SLSQP",bounds=bounds,constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed : {result.message}")
    return result.x
def minimum_variance_long_only_active_set(covariance_matrix):
    Sigma = np.asarray(covariance_matrix, dtype=float)
    n = Sigma.shape[0]
    best_weights = None
    best_variance = np.inf
    for size in range(1, n + 1):
        for active_set in itertools.combinations(range(n), size):
            active_set = list(active_set)
            Sigma_active = Sigma[
                np.ix_(active_set, active_set)
            ]
            try:
                active_weights = _solve_equality_min_variance(Sigma_active)
            except np.linalg.LinAlgError:
                continue
            if np.any(active_weights < -1e-10):
                continue
            weights = np.zeros(n)
            weights[active_set] = active_weights
            variance = weights @ Sigma @ weights
            if variance < best_variance:
                best_variance = variance
                best_weights = weights
    if best_weights is None:
        raise RuntimeError("No feasible long-only portfolio was found.")
    best_weights[np.abs(best_weights) < 1e-10] = 0.0
    return best_weights
def _solve_target_return_equality(covariance_matrix,expected_returns,target_return):
    Sigma=np.asarray(covariance_matrix,dtype=float)
    mu=np.asarray(expected_returns,dtype=float)
    n=len(mu)
    ones=np.ones(n)
    KKT=np.block([[Sigma,-ones.reshape(-1,1),-mu.reshape(-1,1)],[ones.reshape(1,-1),np.zeros((1,2))],[mu.reshape(1,-1),np.zeros((1,2))]])
    rhs=np.concatenate([np.zeros(n),[1.0],[target_return]])
    solution=np.linalg.solve(KKT,rhs)
    return solution[:n]
def minimum_variance_target_return(covariance_matrix,expected_returns,target_return):
    Sigma=np.asarray(covariance_matrix,dtype=float)
    mu=np.asarray(expected_returns,dtype=float)
    n=len(mu)
    min_return=np.min(mu)
    max_return=np.max(mu)
    tolerance=1e-10
    if target_return<min_return-tolerance:
        raise ValueError("Target return is below the minimum feasible return")
    if target_return>max_return+tolerance:
        raise ValueError("Target return is above the maximum feasible return")
    best_weights=None
    best_variance=np.inf
    for size in range(1,n+1):
        for active_set in itertools.combinations(range(n),size):
            active_set=list(active_set)
            if(len(active_set)<2):
                continue
            Sigma_active=Sigma[np.ix_(active_set,active_set)]
            mu_active=mu[active_set]
            try:
                active_weights=_solve_target_return_equality(Sigma_active,mu_active,target_return)
            except np.linalg.LinAlgError:
                continue
            if(np.any(active_weights<-1e-10)):
                continue
            weights=np.zeros(n)
            weights[active_set]=active_weights
            if not np.isclose(weights.sum(),1.0):
                continue
            if not np.isclose(weights@mu,target_return,atol=1e-8):
                continue
            variance=weights@Sigma@weights
            if variance<best_variance:
                best_variance=variance
                best_weights=weights
    if best_weights is None:
        raise RuntimeError("No feasible portfolio achieves the target return")
    best_weights[np.abs(best_weights)<1e-10]=0.0
    return best_weights
def rebalance_portfolio(covariance_matrix,current_weights,max_weight=None,max_weight_change=None,min_weight=0.0,turnover_penalty=0.0,):
    Sigma = np.asarray(covariance_matrix, dtype=float)
    current = np.asarray(current_weights, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    n = Sigma.shape[0]
    if current.shape != (n,):
        raise ValueError("Current weights must match the asset count.")
    if not np.all(np.isfinite(Sigma)) or not np.all(np.isfinite(current)):
        raise ValueError("Inputs must contain only finite values.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")
    if np.linalg.eigvalsh(Sigma).min() < -1e-9:
        raise ValueError("Covariance matrix must be positive semidefinite.")
    if np.any(current < 0) or not np.isclose(current.sum(), 1.0):
        raise ValueError("Current weights must be long-only and sum to one.")
    if turnover_penalty < 0:
        raise ValueError("Turnover penalty cannot be negative.")
    lower = np.broadcast_to(np.asarray(min_weight, dtype=float), (n,)).copy()
    upper = (
        np.ones(n)
        if max_weight is None
        else np.broadcast_to(np.asarray(max_weight, dtype=float), (n,)).copy())
    if max_weight_change is not None:
        change = np.broadcast_to(
            np.asarray(max_weight_change, dtype=float), (n,)
        )
        if np.any(change < 0):
            raise ValueError("Maximum weight changes cannot be negative.")
        lower = np.maximum(lower, current - change)
        upper = np.minimum(upper, current + change)
    if np.any(lower < 0) or np.any(lower > upper):
        raise ValueError("Invalid or contradictory weight bounds.")
    if lower.sum() > 1 + 1e-10 or upper.sum() < 1 - 1e-10:
        raise ValueError("Weight bounds cannot produce a fully invested portfolio.")
    def objective(weights):
        difference = weights - current
        variance = weights @ Sigma @ weights
        penalty = turnover_penalty * (difference @ difference)
        return 0.5 * (variance + penalty)
    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],options={"maxiter": 2000, "ftol": 1e-12},)
    if not result.success:
        raise RuntimeError(f"Rebalancing failed: {result.message}")
    weights = result.x
    if not np.isclose(weights.sum(), 1.0, atol=1e-7):
        raise RuntimeError("Solver returned weights that do not sum to one.")
    if np.any(weights < lower - 1e-7) or np.any(weights > upper + 1e-7):
        raise RuntimeError("Solver returned weights outside the bounds.")
    return weights
