"""
Plotly Dash dashboard for FRED economic indicators.

Entry point
-----------
Call :func:`create_app` with a loaded DataFrame (from :func:`~fred_econ.storage.load`)
to obtain a configured :class:`dash.Dash` instance, then call ``app.run()``.

Panels
------
* Four-panel time-series chart: GDP level, GDP YoY growth (bar), unemployment
  rate (area), and CPI with YoY inflation on a secondary axis.
* Correlation scatter of GDP growth vs unemployment rate with an OLS
  regression line and summary statistics annotation.

All panels include NBER recession shading defined in
:data:`~fred_econ.config.RECESSIONS`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import dash
from dash import Input, Output, dcc, html

from fred_econ.config import RECESSIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private figure builders
# ---------------------------------------------------------------------------

def _add_recession_bands(fig: go.Figure, row: int) -> None:
    """Shade NBER recession periods on *row* of *fig* (in-place)."""
    for start, end in RECESSIONS:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(150,150,150,0.25)",
            layer="below",
            line_width=0,
            row=row,
            col=1,
            annotation_text="Recession",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="grey",
        )


def _build_timeseries(dff: pd.DataFrame) -> go.Figure:
    """Return a four-panel time-series figure for the given slice *dff*.

    Panels
    ------
    1. GDP level (billions USD)
    2. GDP YoY growth rate (bar, green/red)
    3. Unemployment rate (area chart)
    4. CPI index (left axis) + inflation YoY % (right axis)

    Parameters
    ----------
    dff:
        Filtered DataFrame (output of :func:`~fred_econ.storage.load`).
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "GDP (Billions USD)",
            "GDP Growth YoY (%)",
            "Unemployment Rate (%)",
            "CPI & Inflation YoY (%)",
        ),
        specs=[[{"secondary_y": False}]] * 3 + [[{"secondary_y": True}]],
    )

    # Row 1 – GDP level
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["gdp"],
            name="GDP",
            line=dict(color="#2563eb", width=2),
            hovertemplate="%{x|%b %Y}<br>GDP: $%{y:,.1f}B<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Row 2 – GDP YoY growth (bar, green = positive, red = negative)
    growth = dff["gdp_growth_yoy"]
    fig.add_trace(
        go.Bar(
            x=dff.index,
            y=growth,
            name="GDP Growth YoY",
            marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in growth.fillna(0)],
            hovertemplate="%{x|%b %Y}<br>Growth: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1, row=2, col=1)

    # Row 3 – Unemployment rate (area)
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["unemployment_rate"],
            name="Unemployment",
            line=dict(color="#d97706", width=2),
            fill="tozeroy",
            fillcolor="rgba(217,119,6,0.1)",
            hovertemplate="%{x|%b %Y}<br>Unemployment: %{y:.1f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Row 4 – CPI level (left) + inflation YoY (right)
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["cpi"],
            name="CPI",
            line=dict(color="#7c3aed", width=2),
            hovertemplate="%{x|%b %Y}<br>CPI: %{y:.2f}<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["cpi_yoy"],
            name="Inflation YoY",
            line=dict(color="#be185d", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>Inflation: %{y:.2f}%<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    for row in range(1, 5):
        _add_recession_bands(fig, row=row)

    fig.update_layout(
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=40, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Billions USD", row=1, col=1)
    fig.update_yaxes(title_text="% YoY", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="Index", row=4, col=1)
    fig.update_yaxes(title_text="% YoY", row=4, col=1, secondary_y=True)
    return fig


def _build_correlation(dff: pd.DataFrame) -> go.Figure:
    """Return a scatter figure of GDP growth vs unemployment with OLS fit.

    The annotation box shows slope, intercept, Pearson r, R², p-value, and n.
    A negative slope confirms Okun's Law over the selected period.

    Parameters
    ----------
    dff:
        Filtered DataFrame (output of :func:`~fred_econ.storage.load`).
    """
    clean = dff[["gdp_growth_yoy", "unemployment_rate"]].dropna()

    if len(clean) < 5:
        logger.warning("Too few observations (%d) for correlation plot", len(clean))
        return go.Figure().add_annotation(
            text="Not enough data for correlation analysis",
            showarrow=False,
            font=dict(size=14),
        )

    x = clean["unemployment_rate"].values
    y = clean["gdp_growth_yoy"].values
    slope, intercept, r, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=["#16a34a" if v >= 0 else "#dc2626" for v in y],
                size=7,
                opacity=0.75,
                line=dict(color="white", width=0.5),
            ),
            text=clean.index.strftime("%b %Y"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Unemployment: %{x:.1f}%<br>"
                "GDP Growth: %{y:.2f}%<extra></extra>"
            ),
            name="Monthly obs.",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            line=dict(color="#1e40af", width=2, dash="dash"),
            name=f"OLS  r={r:.3f}  p={p_value:.4f}",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        align="left",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=12),
        text=(
            "<b>OLS Regression</b><br>"
            f"Slope: {slope:.3f}<br>"
            f"Intercept: {intercept:.3f}<br>"
            f"r = {r:.3f}<br>"
            f"R\u00b2 = {r**2:.3f}<br>"
            f"p-value = {p_value:.4f}<br>"
            f"n = {len(clean)}"
        ),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig.update_layout(
        title="GDP Growth YoY (%) vs Unemployment Rate (%)",
        xaxis_title="Unemployment Rate (%)",
        yaxis_title="GDP Growth YoY (%)",
        template="plotly_white",
        height=460,
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_app(df: pd.DataFrame) -> dash.Dash:
    """Build and return the configured :class:`dash.Dash` application.

    The *df* is captured by closure inside the update callback so the app
    is fully self-contained and easy to test without a running server.

    Parameters
    ----------
    df:
        Loaded DataFrame as returned by :func:`~fred_econ.storage.load`.
        Must contain columns ``gdp``, ``unemployment_rate``, ``cpi``,
        ``gdp_growth_yoy``, and ``cpi_yoy``.

    Returns
    -------
    dash.Dash
        Configured application ready to be started with ``app.run()``.
    """
    all_dates = df.index.tolist()
    year_marks = {
        i: str(d.year)
        for i, d in enumerate(all_dates)
        if d.month == 1
    }

    app = dash.Dash(__name__, title="FRED Economic Dashboard")

    app.layout = html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1(
                        "US Economic Indicators",
                        style={"margin": "0", "fontSize": "1.6rem"},
                    ),
                    html.P(
                        "GDP \u00b7 Unemployment Rate \u00b7 CPI  |  Source: FRED (St. Louis Fed)",
                        style={
                            "margin": "4px 0 0",
                            "color": "#64748b",
                            "fontSize": "0.9rem",
                        },
                    ),
                ],
                style={
                    "padding": "20px 32px 16px",
                    "borderBottom": "1px solid #e2e8f0",
                    "background": "#f8fafc",
                },
            ),
            # Date range slider
            html.Div(
                [
                    html.Label(
                        "Date range",
                        style={
                            "fontWeight": "600",
                            "fontSize": "0.85rem",
                            "color": "#475569",
                            "marginBottom": "6px",
                            "display": "block",
                        },
                    ),
                    dcc.RangeSlider(
                        id="date-slider",
                        min=0,
                        max=len(all_dates) - 1,
                        step=1,
                        value=[0, len(all_dates) - 1],
                        marks=year_marks,
                        allowCross=False,
                    ),
                ],
                style={"padding": "20px 48px 8px"},
            ),
            html.Div(
                id="date-display",
                style={
                    "textAlign": "center",
                    "color": "#94a3b8",
                    "fontSize": "0.82rem",
                    "marginBottom": "4px",
                },
            ),
            # Time-series panel
            html.Div(
                [dcc.Graph(id="timeseries-chart", config={"displayModeBar": True})],
                style={"padding": "0 24px"},
            ),
            # Correlation panel
            html.Div(
                [
                    html.H2(
                        "Correlation Analysis: GDP Growth vs Unemployment",
                        style={
                            "fontSize": "1rem",
                            "fontWeight": "600",
                            "color": "#334155",
                            "margin": "0 0 4px",
                        },
                    ),
                    html.P(
                        "A negative slope (Okun's Law) is expected: higher unemployment "
                        "accompanies lower or negative GDP growth.",
                        style={
                            "fontSize": "0.83rem",
                            "color": "#64748b",
                            "margin": "0 0 12px",
                        },
                    ),
                    dcc.Graph(
                        id="correlation-chart", config={"displayModeBar": True}
                    ),
                ],
                style={
                    "padding": "16px 32px 32px",
                    "borderTop": "1px solid #e2e8f0",
                    "marginTop": "8px",
                },
            ),
        ],
        style={
            "fontFamily": "'Inter', sans-serif",
            "maxWidth": "1200px",
            "margin": "0 auto",
            "background": "#fff",
        },
    )

    @app.callback(
        Output("timeseries-chart", "figure"),
        Output("correlation-chart", "figure"),
        Output("date-display", "children"),
        Input("date-slider", "value"),
    )
    def update(slider_range: list[int]) -> tuple[go.Figure, go.Figure, str]:
        """Re-render both figures whenever the date slider moves."""
        lo, hi = slider_range
        dff = df.iloc[lo : hi + 1]
        label = (
            f"{all_dates[lo].strftime('%b %Y')}  -  {all_dates[hi].strftime('%b %Y')}"
            f"  ({hi - lo + 1} months)"
        )
        logger.debug("Rendering slice [%d:%d] (%s)", lo, hi, label)
        return _build_timeseries(dff), _build_correlation(dff), label

    return app
