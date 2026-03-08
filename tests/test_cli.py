"""Tests for fred_econ.cli."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from fred_econ.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods, freq="MS")
    idx.name = "date"
    return pd.DataFrame(
        {
            "gdp": [21_000.0 + i * 10 for i in range(periods)],
            "unemployment_rate": [4.0 + i * 0.05 for i in range(periods)],
            "cpi": [260.0 + i * 0.2 for i in range(periods)],
        },
        index=idx,
    )


def _mock_fred_for_fetch(df: pd.DataFrame):
    """Return a patch context that wires up Fred + fetch_raw + clean_and_align."""
    mock_fred_instance = MagicMock()
    # fetch_raw will call fred.get_series; we bypass the whole chain here
    return patch.multiple(
        "fred_econ.cli",
        fetch_raw=MagicMock(return_value={
            "gdp": df["gdp"],
            "unemployment_rate": df["unemployment_rate"],
            "cpi": df["cpi"],
        }),
        clean_and_align=MagicMock(return_value=df),
        save=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Group-level
# ---------------------------------------------------------------------------

class TestCLIGroup:
    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_subcommands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "fetch" in result.output
        assert "serve" in result.output

    def test_unknown_command_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# fetch sub-command
# ---------------------------------------------------------------------------

class TestFetchCommand:
    def _run_fetch(self, df, extra_args=(), env=None):
        runner = CliRunner()
        default_env = {"FRED_API_KEY": "fake_key"}
        if env:
            default_env.update(env)
        with _mock_fred_for_fetch(df):
            with patch("fred_econ.cli.load_dotenv"):
                with patch("fredapi.Fred", MagicMock()):
                    return runner.invoke(
                        main, ["fetch", *extra_args], env=default_env, catch_exceptions=False
                    )

    def test_exits_zero_on_success(self):
        result = self._run_fetch(_sample_df())
        assert result.exit_code == 0

    def test_output_contains_saved_message(self):
        result = self._run_fetch(_sample_df())
        assert "Saved" in result.output

    def test_custom_output_path_passed_through(self, tmp_path):
        out = str(tmp_path / "custom.csv")
        df = _sample_df()
        runner = CliRunner()
        # Pre-create save mock so we can inspect it after the context exits.
        mock_save = MagicMock()
        with patch.multiple(
            "fred_econ.cli",
            fetch_raw=MagicMock(return_value={k: df[k] for k in df.columns}),
            clean_and_align=MagicMock(return_value=df),
            save=mock_save,
        ):
            with patch("fred_econ.cli.load_dotenv"):
                with patch("fredapi.Fred", MagicMock()):
                    runner.invoke(
                        main,
                        ["fetch", "--output", out],
                        env={"FRED_API_KEY": "fake_key"},
                        catch_exceptions=False,
                    )
        mock_save.assert_called_once_with(df, out)

    def test_missing_api_key_exits_nonzero(self):
        runner = CliRunner()
        with patch("fred_econ.cli.load_dotenv"):
            result = runner.invoke(
                main, ["fetch"], env={"FRED_API_KEY": ""}, catch_exceptions=False
            )
        assert result.exit_code != 0

    def test_placeholder_api_key_exits_nonzero(self):
        runner = CliRunner()
        with patch("fred_econ.cli.load_dotenv"):
            result = runner.invoke(
                main,
                ["fetch"],
                env={"FRED_API_KEY": "your_api_key_here"},
                catch_exceptions=False,
            )
        assert result.exit_code != 0

    def test_fetch_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--help"])
        assert result.exit_code == 0

    def test_years_option_accepted(self):
        result = self._run_fetch(_sample_df(), extra_args=["--years", "5"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# serve sub-command
# ---------------------------------------------------------------------------

class TestServeCommand:
    def test_serve_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0

    def test_serve_help_mentions_port(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "port" in result.output.lower()

    def test_serve_loads_csv_and_calls_run(self, tmp_path):
        """serve must load the CSV and start app.run()."""
        df = _sample_df()
        # Write a real CSV so load() can read it
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path)

        mock_app = MagicMock()
        with patch("fred_econ.dashboard.create_app", return_value=mock_app) as mock_create:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["serve", "--csv", str(csv_path), "--port", "9999"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        mock_app.run.assert_called_once_with(debug=False, port=9999)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_daterange_last_n_years_end_is_today(self):
        from datetime import date
        from fred_econ.config import DateRange
        dr = DateRange.last_n_years(5)
        assert dr.end == date.today()

    def test_daterange_raises_if_start_after_end(self):
        from datetime import date
        from fred_econ.config import DateRange
        with pytest.raises(ValueError, match="before"):
            DateRange(start=date(2025, 1, 1), end=date(2020, 1, 1))

    def test_daterange_frozen(self):
        from datetime import date
        from fred_econ.config import DateRange
        dr = DateRange(start=date(2020, 1, 1), end=date(2025, 1, 1))
        with pytest.raises(Exception):
            dr.start = date(2021, 1, 1)  # type: ignore[misc]
