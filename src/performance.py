"""Performance metrics (returns, volatility, CAGR, Sharpe ratio, drawdown).

Functions operate on tidy (long-format) DataFrames with a 'Ticker' column,
so each metric is computed independently per ticker via groupby.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices):
    """Add a 'Return' column: simple daily percent return per ticker.

    The first row for each ticker has no prior day to compare against, so
    its Return is NaN.
    """
    df = prices.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    return df


def cumulative_returns(returns):
    """Add a 'CumReturn' column: growth of $1 invested at the start, per
    ticker (e.g. CumReturn of 0.5 means +50% since the first day).
    """
    df = returns.copy()
    df["CumReturn"] = (
        df.groupby("Ticker")["Return"]
        .apply(lambda r: (1 + r.fillna(0)).cumprod() - 1)
        .reset_index(level=0, drop=True)
    )
    return df


def annualized_volatility(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Annualized volatility (std of daily returns scaled by sqrt(N)) per
    ticker, as a Series indexed by Ticker.
    """
    daily_std = returns.groupby("Ticker")["Return"].std()
    return daily_std * np.sqrt(periods_per_year)


def rolling_annualized_volatility(returns, window=21, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Add a 'RollingVol' column: rolling N-day annualized volatility per
    ticker, useful for visualizing how risk changes over time.
    """
    df = returns.copy()
    df["RollingVol"] = (
        df.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(periods_per_year)
    )
    return df


def summary_stats(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Build a per-ticker summary table: mean daily return, daily volatility,
    annualized return (mean daily return * N), and annualized volatility.

    Note: annualized return here is a simple scaling approximation, not
    CAGR (compound annual growth rate) -- CAGR is computed separately once
    we're comparing full backtest equity curves.
    """
    grouped = returns.groupby("Ticker")["Return"]
    stats = grouped.agg(mean_daily_return="mean", daily_volatility="std")
    stats["annualized_return"] = stats["mean_daily_return"] * periods_per_year
    stats["annualized_volatility"] = stats["daily_volatility"] * np.sqrt(periods_per_year)
    return stats
