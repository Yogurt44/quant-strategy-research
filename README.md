# Quant Strategy Research

A learning project exploring core quantitative finance concepts through hands-on
implementation, built alongside study from QuantStart and Quantcademy material.

## Purpose

This is **not** an attempt to build a profitable trading system or to claim that
markets can be predicted. The goal is to practice the research workflow used in
quantitative finance — data acquisition, return/risk calculation, signal
construction, backtesting, and performance evaluation — while learning both the
underlying quant concepts and the software engineering practices used to
implement them correctly and reproducibly.

## Research Question

> Does a simple moving-average trend-following strategy outperform buy-and-hold
> on a selected universe of equities on a risk-adjusted basis?

## Initial Scope (Project 1)

**Universe:** SPY, QQQ, AAPL, MSFT, NVDA

**Strategy:**
- 50-day and 200-day simple moving averages
- Long the asset when the 50-day MA is above the 200-day MA
- Otherwise, hold cash

**Benchmark:** Buy & hold on the same asset

**Performance metrics:**
- Cumulative return
- CAGR (compound annual growth rate)
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Number of trades

**Visuals:**
- Price with moving averages overlaid
- Strategy equity curve
- Drawdown chart
- Strategy vs. benchmark comparison

## Methodology (Planned)

The project follows the standard quantitative research pipeline, built up
incrementally step by step rather than all at once:

1. Download historical market data
2. Explore and clean the data
3. Calculate daily and cumulative returns
4. Calculate basic risk statistics
5. Create moving-average trading signals
6. Backtest the strategy
7. Compare the strategy against buy-and-hold
8. Calculate performance metrics (CAGR, volatility, Sharpe ratio, max drawdown)
9. Analyze results and limitations
10. Document the project so the research is reproducible

## Planned Future Extensions

Once the initial strategy and pipeline are working end to end, later phases will
explore:

- **Phase 2:** Transaction costs, alternate parameter choices, a broader set of
  stocks, out-of-sample testing, walk-forward testing
- **Phase 3:** Volatility targeting, position sizing, momentum factors, regime
  analysis, statistical significance testing

## Project Status

This project is under active, incremental development. Structure and
documentation are being built first; the actual data pipeline, signal
generation, and backtesting logic have **not** been implemented yet.

## Repository Structure

```
quant-strategy-research/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/            # Exploratory analysis, one notebook per pipeline stage
├── src/                  # Reusable Python modules backing the notebooks
├── tests/                # Unit tests for src/ modules
└── figures/              # Saved plots/exported visuals
```

## Tools

- Python
- pandas, NumPy
- matplotlib
- yfinance
- Jupyter
- pytest
- Git/GitHub

## Disclaimer

This project is for educational purposes only. Nothing here constitutes
financial advice, and no results from this project should be interpreted as a
recommendation to trade any security.
