import yfinance as yf
import pandas as pd
def download_prices(tickers,start,end):
    data=yf.download(tickers,start=start,end=end,auto_adjust=True,progress=False)
    prices=data["Close"]
    return prices
def calculate_returns(prices):
    returns=prices.pct_change()
    return returns.dropna()
def annualized_returns(returns):
    return returns.mean()*252
def annualized_covariance(returns):
    return returns.cov()*252
