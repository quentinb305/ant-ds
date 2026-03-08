"""
Tests for fetch_fred_data.py.

Run with:
  pip install pytest pandas fredapi python-dotenv
  pytest test_fetch_fred_data.py -v
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fetch_fred_data import clean_and_align, fetch_raw, main, save


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_series(name: str, start: str, periods: int, base: float) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="MS")
    return pd.Series([base + i * 0.1 for i in range(periods)], index=idx, name=name)


def _quarterly_series(name: str, start: str, periods: int, base: float) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="QS")
    return pd.Series([base + i * 10 for i in range(periods)], index=idx, name=name)


def _make_raw(start="2020-01-01", months=24, quarters=8):
    return {
        "gdp": _quarterly_series("gdp", start, quarters, 21000.0),
        "unemployment_rate": _monthly_series("unemployment_rate", start, months, 4.0),
        "cpi": _monthly_series("cpi", start, months, 260.0),
    }


# ---------------------------------------------------------------------------
# clean_and_align
# ---------------------------------------------------------------------------

class TestCleanAndAlign:

    def test_returns_dataframe(self):
        df = clean_and_align(_make_raw())
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = clean_and_align(_make_raw())
        assert set(df.columns) == {"gdp", "unemployment_rate", "cpi"}

    def test_index_is_monthly_start(self):
        df = clean_and_align(_make_raw())
        freq = pd.infer_freq(df.index)
        assert freq == "MS"

    def test_index_name_is_date(self):
        df = clean_and_align(_make_raw())
        assert df.index.name == "date"

    def test_gdp_forward_filled_into_intra_quarter_months(self):
        """Quarter-start GDP values must be present in Feb and Mar too."""
        raw = _make_raw(start="2020-01-01", months=6, quarters=2)
        df = clean_and_align(raw)
        # Jan GDP forward-fills into Feb and Mar
        assert not pd.isna(df.loc["2020-02-01", "gdp"])
        assert not pd.isna(df.loc["2020-03-01", "gdp"])

    def test_no_all_nan_rows(self):
        df = clean_and_align(_make_raw())
        assert not df.isna().all(axis=1).any()

    def test_gdp_rounded_to_3_decimal_places(self):
        df = clean_and_align(_make_raw())
        for val in df["gdp"].dropna():
            assert round(val, 3) == val

    def test_unemployment_rounded_to_2_decimal_places(self):
        df = clean_and_align(_make_raw())
        for val in df["unemployment_rate"].dropna():
            assert round(val, 2) == val

    def test_cpi_rounded_to_3_decimal_places(self):
        df = clean_and_align(_make_raw())
        for val in df["cpi"].dropna():
            assert round(val, 3) == val

    def test_index_is_datetime(self):
        df = clean_and_align(_make_raw())
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_single_row_does_not_raise(self):
        raw = {
            "gdp": _quarterly_series("gdp", "2023-01-01", 1, 25000.0),
            "unemployment_rate": _monthly_series("unemployment_rate", "2023-01-01", 1, 3.5),
            "cpi": _monthly_series("cpi", "2023-01-01", 1, 300.0),
        }
        df = clean_and_align(raw)
        assert len(df) >= 1

    def test_does_not_mutate_input(self):
        raw = _make_raw()
        original_gdp_index = raw["gdp"].index.copy()
        clean_and_align(raw)
        assert raw["gdp"].index.equals(original_gdp_index)


# ---------------------------------------------------------------------------
# fetch_raw
# ---------------------------------------------------------------------------

class TestFetchRaw:

    def _make_fred_mock(self):
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda series_id, **_: pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2023-01-01", periods=3, freq="MS"),
            name=series_id,
        )
        return mock_fred

    def test_returns_dict_with_correct_keys(self):
        series_map = {"gdp": "GDP", "unemployment_rate": "UNRATE", "cpi": "CPIAUCSL"}
        result = fetch_raw(self._make_fred_mock(), series_map, "2023-01-01", "2023-03-01")
        assert set(result.keys()) == {"gdp", "unemployment_rate", "cpi"}

    def test_calls_get_series_for_each_entry(self):
        mock_fred = self._make_fred_mock()
        series_map = {"gdp": "GDP", "unemployment_rate": "UNRATE", "cpi": "CPIAUCSL"}
        fetch_raw(mock_fred, series_map, "2023-01-01", "2023-03-01")
        assert mock_fred.get_series.call_count == 3

    def test_passes_dates_to_get_series(self):
        mock_fred = self._make_fred_mock()
        series_map = {"gdp": "GDP"}
        fetch_raw(mock_fred, series_map, "2015-01-01", "2025-01-01")
        _, kwargs = mock_fred.get_series.call_args
        assert kwargs["observation_start"] == "2015-01-01"
        assert kwargs["observation_end"] == "2025-01-01"

    def test_each_value_is_series(self):
        result = fetch_raw(
            self._make_fred_mock(),
            {"gdp": "GDP"},
            "2023-01-01",
            "2023-03-01",
        )
        assert isinstance(result["gdp"], pd.Series)

    def test_empty_series_map_returns_empty_dict(self):
        result = fetch_raw(MagicMock(), {}, "2023-01-01", "2023-03-01")
        assert result == {}


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSave:

    def test_creates_csv_file(self):
        df = clean_and_align(_make_raw())
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save(df, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_csv_is_readable_back(self):
        df = clean_and_align(_make_raw())
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save(df, path)
            loaded = pd.read_csv(path, index_col="date", parse_dates=True)
            assert list(loaded.columns) == ["gdp", "unemployment_rate", "cpi"]
            assert len(loaded) == len(df)
        finally:
            os.unlink(path)

    def test_csv_values_match_dataframe(self):
        df = clean_and_align(_make_raw())
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save(df, path)
            loaded = pd.read_csv(path, index_col="date", parse_dates=True)
            pd.testing.assert_frame_equal(df, loaded, check_freq=False)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# main (integration, API mocked)
# ---------------------------------------------------------------------------

class TestMain:

    def _patch_main(self, tmp_path):
        raw = _make_raw(start="2016-01-01", months=120, quarters=40)

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.side_effect = lambda sid, **_: {
            "GDP": raw["gdp"],
            "UNRATE": raw["unemployment_rate"],
            "CPIAUCSL": raw["cpi"],
        }[sid]

        return patch.multiple(
            "fetch_fred_data",
            Fred=MagicMock(return_value=mock_fred_instance),
            OUTPUT_FILE=str(tmp_path / "out.csv"),
        )

    def test_main_returns_dataframe(self, tmp_path):
        with self._patch_main(tmp_path):
            with patch.dict(os.environ, {"FRED_API_KEY": "fake_key"}):
                df = main()
        assert isinstance(df, pd.DataFrame)

    def test_main_dataframe_has_three_columns(self, tmp_path):
        with self._patch_main(tmp_path):
            with patch.dict(os.environ, {"FRED_API_KEY": "fake_key"}):
                df = main()
        assert set(df.columns) == {"gdp", "unemployment_rate", "cpi"}

    def test_main_creates_csv(self, tmp_path):
        with self._patch_main(tmp_path):
            with patch.dict(os.environ, {"FRED_API_KEY": "fake_key"}):
                main()
        assert (tmp_path / "out.csv").exists()

    def test_main_raises_without_api_key(self, tmp_path):
        with self._patch_main(tmp_path):
            with patch("fetch_fred_data.load_dotenv"):           # skip .env file
                with patch.dict(os.environ, {}, clear=True):
                    with pytest.raises(ValueError, match="FRED_API_KEY"):
                        main()

    def test_main_raises_with_placeholder_key(self, tmp_path):
        with self._patch_main(tmp_path):
            with patch.dict(os.environ, {"FRED_API_KEY": "your_api_key_here"}):
                with pytest.raises(ValueError, match="FRED_API_KEY"):
                    main()
