"""Tests for fred_econ.fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from fred_econ.fetcher import fetch_raw


def _make_fred_mock() -> MagicMock:
    mock = MagicMock()
    mock.get_series.side_effect = lambda sid, **_: pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2023-01-01", periods=3, freq="MS"),
        name=sid,
    )
    return mock


SERIES_MAP = {"gdp": "GDP", "unemployment_rate": "UNRATE", "cpi": "CPIAUCSL"}


class TestFetchRaw:
    def test_returns_dict_with_correct_keys(self):
        result = fetch_raw(_make_fred_mock(), SERIES_MAP, "2023-01-01", "2023-03-01")
        assert set(result.keys()) == {"gdp", "unemployment_rate", "cpi"}

    def test_each_value_is_series(self):
        result = fetch_raw(_make_fred_mock(), SERIES_MAP, "2023-01-01", "2023-03-01")
        for s in result.values():
            assert isinstance(s, pd.Series)

    def test_calls_get_series_once_per_entry(self):
        mock = _make_fred_mock()
        fetch_raw(mock, SERIES_MAP, "2023-01-01", "2023-03-01")
        assert mock.get_series.call_count == 3

    def test_passes_correct_series_ids(self):
        mock = _make_fred_mock()
        fetch_raw(mock, SERIES_MAP, "2023-01-01", "2023-03-01")
        called_ids = {call.args[0] for call in mock.get_series.call_args_list}
        assert called_ids == {"GDP", "UNRATE", "CPIAUCSL"}

    def test_passes_start_date_kwarg(self):
        mock = _make_fred_mock()
        fetch_raw(mock, {"gdp": "GDP"}, "2015-01-01", "2025-01-01")
        _, kwargs = mock.get_series.call_args
        assert kwargs["observation_start"] == "2015-01-01"

    def test_passes_end_date_kwarg(self):
        mock = _make_fred_mock()
        fetch_raw(mock, {"gdp": "GDP"}, "2015-01-01", "2025-01-01")
        _, kwargs = mock.get_series.call_args
        assert kwargs["observation_end"] == "2025-01-01"

    def test_empty_series_map_returns_empty_dict(self):
        assert fetch_raw(MagicMock(), {}, "2023-01-01", "2023-12-01") == {}

    def test_single_series(self):
        result = fetch_raw(_make_fred_mock(), {"gdp": "GDP"}, "2023-01-01", "2023-03-01")
        assert list(result.keys()) == ["gdp"]

    def test_fred_exception_propagates(self):
        mock = MagicMock()
        mock.get_series.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            fetch_raw(mock, {"gdp": "GDP"}, "2023-01-01", "2023-03-01")
