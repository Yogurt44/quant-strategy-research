"""Market data download and cleaning utilities.

Downloads daily OHLCV price data via yfinance and reshapes it into a tidy
(long-format) DataFrame that's easy to filter, group, and join on downstream
(one row per Date/Ticker combination, rather than one column per ticker).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_prices(tickers=None, start="2010-01-01", end=None):
    """Download daily adjusted OHLCV data for the given tickers.

    Uses auto_adjust=True, so Open/High/Low/Close are already adjusted for
    stock splits and dividends (this matters for return calculations later).

    Returns a tidy DataFrame with columns:
    ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    frames = []
    for ticker in tickers:
        df = raw[ticker].copy()
        df["Ticker"] = ticker
        frames.append(df)

    tidy = pd.concat(frames)
    tidy = tidy.reset_index().rename(columns={"index": "Date"})
    tidy = tidy[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    tidy = tidy.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return tidy


def save_prices(df, filename="prices.csv"):
    """Cache a prices DataFrame to data/<filename> so we don't re-download."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_prices(filename="prices.csv"):
    """Load a previously cached prices DataFrame from data/<filename>."""
    path = DATA_DIR / filename
    return pd.read_csv(path, parse_dates=["Date"])
