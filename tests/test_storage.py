"""Tests for fred_econ.storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fred_econ.storage import load, save


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    idx.name = "date"
    return pd.DataFrame(
        {
            "gdp": [21_000.0 + i * 10 for i in range(24)],
            "unemployment_rate": [4.0 + i * 0.05 for i in range(24)],
            "cpi": [260.0 + i * 0.2 for i in range(24)],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        assert path.exists()

    def test_accepts_string_path(self, sample_df, tmp_path):
        path = str(tmp_path / "out.csv")
        save(sample_df, path)
        assert Path(path).exists()

    def test_creates_parent_dirs(self, sample_df, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.csv"
        save(sample_df, path)
        assert path.exists()

    def test_csv_row_count_matches(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(sample_df)

    def test_overwrites_existing_file(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        save(sample_df.iloc[:5], path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 5


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_returns_dataframe(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_datetimeindex(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_name_is_date(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert result.index.name == "date"

    def test_original_columns_present(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert {"gdp", "unemployment_rate", "cpi"}.issubset(result.columns)

    def test_gdp_growth_yoy_column_added(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert "gdp_growth_yoy" in result.columns

    def test_cpi_yoy_column_added(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert "cpi_yoy" in result.columns

    def test_gdp_values_preserved(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        pd.testing.assert_series_equal(
            sample_df["gdp"], result["gdp"], check_freq=False
        )

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load(tmp_path / "does_not_exist.csv")

    def test_roundtrip_row_count(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        save(sample_df, path)
        result = load(path)
        assert len(result) == len(sample_df)
