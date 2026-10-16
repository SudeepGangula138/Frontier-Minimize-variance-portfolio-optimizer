import numpy as np
def calculate_rebalance_trades(current_weights,new_weights,portfolio_value,):
    current = np.asarray(current_weights, dtype=float)
    new = np.asarray(new_weights, dtype=float)
    if current.shape != new.shape:
        raise ValueError("Current and new weights must have matching shapes.")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(new)):
        raise ValueError("Weights must be finite.")
    if np.any(current < 0) or np.any(new < 0):
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(current.sum(), 1.0) or not np.isclose(new.sum(), 1.0):
        raise ValueError("Both sets of weights must sum to one.")
    if not np.isfinite(portfolio_value) or portfolio_value < 0:
        raise ValueError("Portfolio value must be finite and non-negative.")
    current_values = current * portfolio_value
    target_values = new * portfolio_value
    trade_values = target_values - current_values
    return {"current_values": current_values,"target_values": target_values,"trade_values": trade_values,"buys_value": float(np.maximum(trade_values, 0).sum()),"sells_value": float(-np.minimum(trade_values, 0).sum()),"one_way_turnover": float(np.abs(new - current).sum() / 2),}
