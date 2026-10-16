import matplotlib.pyplot as plt
import numpy as np
from src.data import (download_prices,calculate_returns,annualized_returns,annualized_covariance)
from src.frontier import (generate_efficient_frontier,find_global_minimum_variance_portfolio)
from src.portfolio import portfolio_return
tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN"]
prices = download_prices(tickers,"2021-01-01","2026-01-01")
returns = calculate_returns(prices)
expected_returns = annualized_returns(returns)
covariance = annualized_covariance(returns)
frontier = generate_efficient_frontier(covariance.values,expected_returns.values,num_points=50)
gmvp = find_global_minimum_variance_portfolio(covariance.values)
gmvp_return = portfolio_return(gmvp["weights"],expected_returns.values)
plt.figure(figsize=(10, 6))
plt.plot(frontier["volatilities"],frontier["returns"],label="Efficient Frontier")
plt.scatter(gmvp["volatility"],gmvp_return,marker="*",s=200,label="Global Minimum-Variance Portfolio")
plt.xlabel("Annualised Volatility")
plt.ylabel("Expected Annual Return")
plt.title("Efficient Frontier")
plt.legend()
plt.grid(True)
plt.savefig("results/efficient_frontier.png", dpi=150, bbox_inches="tight")
plt.show()