"""Moving-average trend-following signal construction.

Strategy: long when the short (50-day) simple moving average is above the
long (200-day) simple moving average, otherwise flat (cash).
"""

SHORT_WINDOW = 50
LONG_WINDOW = 200


def add_moving_averages(prices, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    """Add SMA_<short> and SMA_<long> simple moving average columns of
    Close, computed independently per ticker.
    """
    df = prices.sort_values(["Ticker", "Date"]).copy()
    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    df[short_col] = (
        df.groupby("Ticker")["Close"]
        .rolling(short_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[long_col] = (
        df.groupby("Ticker")["Close"]
        .rolling(long_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def generate_signals(prices_with_ma, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    """Add 'Signal' and 'Position' columns.

    'Signal' (1 = long, 0 = cash) reflects the moving-average state as of
    that day's close: 1 when SMA_short > SMA_long. It's NaN during the
    long_window warm-up period, before both averages exist.

    'Position' is 'Signal' shifted forward one day -- what we actually
    trade tomorrow, based on today's already-known close. This avoids
    lookahead bias: without the shift, "today's position" would depend on
    "today's closing price", which isn't knowable until the market has
    already closed for the day. Position is filled with 0 (no position)
    during the warm-up period, since we don't yet have a signal to act on.
    """
    df = prices_with_ma.copy()
    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    df["Signal"] = (df[short_col] > df[long_col]).astype(float)
    df.loc[df[long_col].isna(), "Signal"] = float("nan")

    df["Position"] = df.groupby("Ticker")["Signal"].shift(1)
    df["Position"] = df["Position"].fillna(0)

    return df


def count_trades(signals):
    """Count position changes (0->1 or 1->0) per ticker. A 'trade' is any
    day where Position differs from the previous day's Position.
    """
    df = signals.copy()
    changed = df.groupby("Ticker")["Position"].diff().fillna(0) != 0
    return changed.groupby(df["Ticker"]).sum().rename("NumTrades")
