"""
Shared pytest fixtures for the fred_econ test suite.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _monthly(name: str, start: str, periods: int, base: float) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="MS")
    return pd.Series([base + i * 0.1 for i in range(periods)], index=idx, name=name)


def _quarterly(name: str, start: str, periods: int, base: float) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="QS")
    return pd.Series([base + i * 10 for i in range(periods)], index=idx, name=name)


@pytest.fixture()
def raw_series() -> dict[str, pd.Series]:
    """Raw dict mimicking the output of fetch_raw (2 years of data)."""
    return {
        "gdp": _quarterly("gdp", "2020-01-01", 8, 21_000.0),
        "unemployment_rate": _monthly("unemployment_rate", "2020-01-01", 24, 4.0),
        "cpi": _monthly("cpi", "2020-01-01", 24, 260.0),
    }


@pytest.fixture()
def clean_df(raw_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Cleaned DataFrame produced by clean_and_align."""
    from fred_econ.cleaner import clean_and_align
    return clean_and_align(raw_series)
