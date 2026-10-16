import numpy as np
from src.rebalancing import calculate_rebalance_trades
def test_calculate_rebalance_trades():
    result = calculate_rebalance_trades(
        [0.50, 0.30, 0.20],
        [0.40, 0.35, 0.25],
        100000,
    )
    assert np.allclose(result["current_values"], [50000, 30000, 20000])
    assert np.allclose(result["target_values"], [40000, 35000, 25000])
    assert np.allclose(result["trade_values"], [-10000, 5000, 5000])
    assert np.isclose(result["buys_value"], 10000)
    assert np.isclose(result["sells_value"], 10000)
    assert np.isclose(result["one_way_turnover"], 0.10)