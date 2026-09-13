"""Strategy backtesting: apply Position to realized returns to build an
equity curve, and compare against a buy-and-hold benchmark.
"""


def run_backtest(signals_df):
    """Add 'StrategyReturn', 'StrategyCumReturn', and 'BenchmarkCumReturn'
    columns per ticker.

    StrategyReturn on day t is Position(t) * Return(t). Position(t) was
    already decided using information available at day t-1's close (see
    signals.generate_signals), so multiplying it by day t's return
    correctly reflects money earned on a bet placed before day t's return
    was known -- no lookahead bias.

    BenchmarkCumReturn is the buy-and-hold equivalent: always fully
    invested (equivalent to Position == 1 for the whole history).
    """
    df = signals_df.copy()
    df["StrategyReturn"] = df["Position"] * df["Return"]

    df["StrategyCumReturn"] = (
        df.groupby("Ticker")["StrategyReturn"]
        .apply(lambda r: (1 + r.fillna(0)).cumprod() - 1)
        .reset_index(level=0, drop=True)
    )
    df["BenchmarkCumReturn"] = (
        df.groupby("Ticker")["Return"]
        .apply(lambda r: (1 + r.fillna(0)).cumprod() - 1)
        .reset_index(level=0, drop=True)
    )
    return df


def drawdown(cum_return_series):
    """Drawdown series (a fraction <= 0): how far below its running peak
    the equity curve implied by a cumulative-return series currently sits.
    """
    wealth = 1 + cum_return_series
    running_max = wealth.cummax()
    return wealth / running_max - 1


def add_drawdowns(backtest_df):
    """Add 'StrategyDrawdown' and 'BenchmarkDrawdown' columns per ticker."""
    df = backtest_df.copy()
    df["StrategyDrawdown"] = (
        df.groupby("Ticker")["StrategyCumReturn"]
        .apply(drawdown)
        .reset_index(level=0, drop=True)
    )
    df["BenchmarkDrawdown"] = (
        df.groupby("Ticker")["BenchmarkCumReturn"]
        .apply(drawdown)
        .reset_index(level=0, drop=True)
    )
    return df
