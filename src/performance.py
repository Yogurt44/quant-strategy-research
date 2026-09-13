"""Performance metrics (returns, volatility, CAGR, Sharpe ratio, drawdown).

Functions operate on tidy (long-format) DataFrames with a 'Ticker' column,
so each metric is computed independently per ticker via groupby.
"""

import numpy as np
import pandas as pd

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


def cagr(cum_return_series, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Compound annual growth rate implied by a cumulative-return series.

    Years elapsed is inferred from the number of rows (one per trading
    day) divided by periods_per_year, so this stays consistent with how
    the rest of this module annualizes (252 trading days/year), rather
    than using calendar days.
    """
    final_wealth = 1 + cum_return_series.iloc[-1]
    years = len(cum_return_series) / periods_per_year
    return final_wealth ** (1 / years) - 1


def sharpe_ratio(return_series, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Annualized Sharpe ratio: mean excess return over its volatility.

    risk_free_rate is an ANNUAL rate, converted to a per-period rate by
    dividing by periods_per_year. Defaults to 0% -- a common simplification
    for a first pass, but a real limitation worth remembering: a nonzero
    risk-free rate would lower every Sharpe ratio here, especially in
    higher-rate periods.
    """
    period_rf = risk_free_rate / periods_per_year
    excess = return_series.dropna() - period_rf
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def performance_summary(backtest_df, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Build a per-ticker, per-series (Strategy vs. Benchmark) performance
    table: CAGR, annualized volatility, Sharpe ratio, and max drawdown.

    Expects backtest_df to already have StrategyReturn/StrategyCumReturn/
    StrategyDrawdown and Return/BenchmarkCumReturn/BenchmarkDrawdown
    columns (see src/backtest.py: run_backtest, add_drawdowns).
    """
    series_specs = [
        ("Strategy", "StrategyReturn", "StrategyCumReturn", "StrategyDrawdown"),
        ("Benchmark", "Return", "BenchmarkCumReturn", "BenchmarkDrawdown"),
    ]

    rows = []
    for ticker, group in backtest_df.groupby("Ticker"):
        for label, return_col, cum_col, dd_col in series_specs:
            returns = group[return_col]
            rows.append(
                {
                    "Ticker": ticker,
                    "Series": label,
                    "CAGR": cagr(group[cum_col], periods_per_year),
                    "AnnualizedVolatility": returns.dropna().std() * np.sqrt(periods_per_year),
                    "SharpeRatio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
                    "MaxDrawdown": group[dd_col].min(),
                }
            )

    return pd.DataFrame(rows).set_index(["Ticker", "Series"])
