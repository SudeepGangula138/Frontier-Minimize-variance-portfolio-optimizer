# Effective Portfolio Optimizer

A Python implementation of mean-variance portfolio optimization, built from the underlying mathematics rather than around a black-box solver. The core problems are derived from their KKT conditions, solved directly, and checked with unit tests at each step.

The project downloads price data from Yahoo Finance and can:

- find the minimum-variance portfolio, with and without short selling
- find the lowest-risk portfolio for a given target return
- plot the efficient frontier
- rebalance an existing portfolio under constraints (maximum weight, maximum change per asset, turnover penalty) and list the trades required

## The math

Returns are daily simple returns, annualized using 252 trading days. With expected returns $\mu$ and covariance matrix $\Sigma$, a portfolio with weights $w$ has

$$R_p = w^\top \mu, \qquad \sigma_p^2 = w^\top \Sigma w$$

### Minimum variance (shorts allowed)

The problem is to minimize $\tfrac12 w^\top \Sigma w$ subject to $\mathbf{1}^\top w = 1$. Setting the gradient of the Lagrangian to zero gives $\Sigma w = \lambda \mathbf{1}$, and substituting into the constraint gives

$$w^* = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}$$

This is implemented in two ways: through the closed-form expression, and by solving the KKT system directly:

```math
\begin{bmatrix} \Sigma & -\mathbf{1} \\ \mathbf{1}^\top & 0 \end{bmatrix}
\begin{bmatrix} w \\ \lambda \end{bmatrix}
=
\begin{bmatrix} \mathbf{0} \\ 1 \end{bmatrix}
```

A test confirms that both give the same weights. Neither version forms $\Sigma^{-1}$ explicitly; both use `np.linalg.solve`.

Note that $\mu$ does not appear in the solution. The minimum-variance portfolio depends only on risk, which is why its expected return is often modest.

### Long-only

Adding $w_i \ge 0$ breaks the closed-form solution. Clipping negative weights to zero and renormalizing does not give the true optimum.

The long-only solver relies on a property of the optimum: the assets with non-zero weight solve the ordinary equality-constrained problem among themselves. The solver therefore enumerates every subset of assets, solves the KKT system for each, discards any candidate with a negative weight, and keeps the one with the lowest variance. Because the problem is convex, this always finds the true optimum.

The cost is $2^n - 1$ linear solves for $n$ assets, which is fine for 5 to 10 assets and impractical beyond about 20. A SciPy (SLSQP) version is included as an independent reference.

### Target return

The same approach, with an additional constraint $\mu^\top w = R^*$:

```math
\begin{bmatrix} \Sigma & -\mathbf{1} & -\mu \\ \mathbf{1}^\top & 0 & 0 \\ \mu^\top & 0 & 0 \end{bmatrix}
\begin{bmatrix} w \\ \lambda \\ \gamma \end{bmatrix}
=
\begin{bmatrix} \mathbf{0} \\ 1 \\ R^* \end{bmatrix}
```

Without short selling, the portfolio return is a weighted average of the asset returns, so any target outside $[\min_i \mu_i, \max_i \mu_i]$ is rejected immediately. Subsets that make the KKT matrix singular (a single asset, or two assets with equal $\mu$) are skipped.

### Efficient frontier

The target return is swept from the lowest to the highest asset return, the problem above is solved for each target, and volatility is plotted against return. The minimum-variance portfolio is marked with a star. Only the part of the curve above it is efficient.

![Efficient frontier](results/efficient_frontier.png)

### Rebalancing

Given current holdings $w_0$, the rebalancer minimizes

$$\tfrac12\left( w^\top \Sigma w + \kappa \lVert w - w_0 \rVert^2 \right)$$

where $\kappa$ penalizes excessive trading. Individual weights can be capped, and the change in each weight can be limited. This step uses SciPy, but inputs are validated beforehand (symmetric, positive semidefinite covariance; bounds that can sum to 100%) and the result is checked against every constraint afterwards. A separate function converts the new weights into buy and sell amounts for a given portfolio value.

## Getting started

```bash
pip install -r requirements.txt
```

From the project root:

```bash
python -m notebooks.efficient_frontier   # frontier for AAPL, MSFT, NVDA, GOOGL, AMZN
pytest                                   # run the test suite
```

Example:

```python
from src.data import download_prices, calculate_returns, annualized_covariance
from src.optimization import minimum_variance_long_only_active_set

prices = download_prices(["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"], "2021-01-01", "2026-01-01")
returns = calculate_returns(prices)
Sigma = annualized_covariance(returns)

w = minimum_variance_long_only_active_set(Sigma.values)
print(dict(zip(Sigma.columns, w.round(4))))
```

yfinance returns columns in alphabetical order, so weights should be labelled with `Sigma.columns` rather than the original ticker list.

## Project structure

- `src/data.py`: price download, returns, mean and covariance
- `src/portfolio.py`: portfolio return, variance and volatility
- `src/optimization.py`: all solvers
- `src/frontier.py`: efficient frontier
- `src/rebalancing.py`: trade amounts from current to new weights
- `tests/`: 12 tests covering the formulas, agreement between solvers, constraint satisfaction and invalid inputs

## Limitations

- **Historical mean returns are a poor estimate of future returns**, and mean-variance optimization is known to amplify that error. With 2021–2025 data, the high-return end of the frontier concentrates heavily in NVDA, largely because of its strong run over that period. No out-of-sample testing has been done yet.
- The subset-enumeration solver does not scale beyond roughly 20 assets.
- With many assets and limited history, the sample covariance matrix becomes close to singular.
- Rebalancing still depends on SciPy rather than an in-house solver.

## Roadmap

- Interior-point solver (Newton's method with a log barrier) so the long-only problem scales to larger universes
- Ledoit-Wolf covariance shrinkage
- Walk-forward backtest against an equal-weight portfolio, mainly because I want to know whether the "optimal" portfolio actually beats it
- A test that checks the full KKT conditions, not only feasibility
- Maximum Sharpe ratio portfolio

---

Sudeep Gangula, B.Tech Mathematics and Computing, IIIT Bhopal
