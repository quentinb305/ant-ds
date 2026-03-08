"""Tests for fred_econ.cleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from fred_econ.cleaner import clean_and_align


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(start: str = "2020-01-01", months: int = 24, quarters: int = 8):
    def monthly(name, base):
        idx = pd.date_range(start, periods=months, freq="MS")
        return pd.Series([base + i * 0.1 for i in range(months)], index=idx, name=name)

    def quarterly(name, base):
        idx = pd.date_range(start, periods=quarters, freq="QS")
        return pd.Series([base + i * 10 for i in range(quarters)], index=idx, name=name)

    return {
        "gdp": quarterly("gdp", 21_000.0),
        "unemployment_rate": monthly("unemployment_rate", 4.0),
        "cpi": monthly("cpi", 260.0),
    }


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_returns_dataframe(self):
        assert isinstance(clean_and_align(_make_raw()), pd.DataFrame)

    def test_has_expected_columns(self):
        df = clean_and_align(_make_raw())
        assert set(df.columns) == {"gdp", "unemployment_rate", "cpi"}

    def test_index_is_monthly_start(self):
        df = clean_and_align(_make_raw())
        assert pd.infer_freq(df.index) == "MS"

    def test_index_name_is_date(self):
        df = clean_and_align(_make_raw())
        assert df.index.name == "date"

    def test_index_is_datetimeindex(self):
        df = clean_and_align(_make_raw())
        assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Forward-fill behaviour
# ---------------------------------------------------------------------------

class TestForwardFill:
    def test_gdp_present_in_intra_quarter_months(self):
        """Quarter-start GDP must be forward-filled into Feb and Mar."""
        df = clean_and_align(_make_raw(start="2020-01-01", months=6, quarters=2))
        assert not pd.isna(df.loc["2020-02-01", "gdp"])
        assert not pd.isna(df.loc["2020-03-01", "gdp"])

    def test_q2_gdp_forward_filled(self):
        # quarters=3 ensures the Jul quarter exists so ffill reaches May/Jun.
        df = clean_and_align(_make_raw(start="2020-01-01", months=9, quarters=3))
        assert not pd.isna(df.loc["2020-05-01", "gdp"])
        assert not pd.isna(df.loc["2020-06-01", "gdp"])


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

class TestDataQuality:
    def test_no_all_nan_rows(self):
        df = clean_and_align(_make_raw())
        assert not df.isna().all(axis=1).any()

    def test_gdp_rounded_to_3dp(self):
        df = clean_and_align(_make_raw())
        assert all(round(v, 3) == v for v in df["gdp"].dropna())

    def test_unemployment_rounded_to_2dp(self):
        df = clean_and_align(_make_raw())
        assert all(round(v, 2) == v for v in df["unemployment_rate"].dropna())

    def test_cpi_rounded_to_3dp(self):
        df = clean_and_align(_make_raw())
        assert all(round(v, 3) == v for v in df["cpi"].dropna())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_observation_does_not_raise(self):
        raw = {
            "gdp": pd.Series(
                [25_000.0],
                index=pd.date_range("2023-01-01", periods=1, freq="QS"),
                name="gdp",
            ),
            "unemployment_rate": pd.Series(
                [3.5],
                index=pd.date_range("2023-01-01", periods=1, freq="MS"),
                name="unemployment_rate",
            ),
            "cpi": pd.Series(
                [300.0],
                index=pd.date_range("2023-01-01", periods=1, freq="MS"),
                name="cpi",
            ),
        }
        df = clean_and_align(raw)
        assert len(df) >= 1

    def test_does_not_mutate_input(self):
        raw = _make_raw()
        original_index = raw["gdp"].index.copy()
        clean_and_align(raw)
        assert raw["gdp"].index.equals(original_index)

    def test_all_nan_rows_dropped(self):
        """A row where every value is NaN must be removed."""
        raw = _make_raw(months=3, quarters=1)
        # Inject all-NaN row by using non-overlapping date ranges
        raw["unemployment_rate"] = pd.Series(
            [4.0],
            index=pd.date_range("2021-01-01", periods=1, freq="MS"),
            name="unemployment_rate",
        )
        raw["cpi"] = pd.Series(
            [260.0],
            index=pd.date_range("2021-01-01", periods=1, freq="MS"),
            name="cpi",
        )
        df = clean_and_align(raw)
        assert not df.isna().all(axis=1).any()
