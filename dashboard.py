"""
Interactive Plotly Dash dashboard for FRED economic indicators.

Features:
  - GDP, Unemployment Rate, and CPI time series with recession shading
  - GDP YoY growth rate
  - Correlation scatter (GDP growth vs Unemployment) with OLS regression
  - Date range slider to zoom into any sub-period

Run:
  python dashboard.py
  Then open http://127.0.0.1:8050 in your browser.

Requirements:
  pip install dash plotly pandas scipy
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import dash
from dash import Input, Output, dcc, html

# ---------------------------------------------------------------------------
# NBER recession periods (within the 10-year window)
# ---------------------------------------------------------------------------

RECESSIONS = [
    ("2020-02-01", "2020-04-30"),  # COVID-19
]

# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data(path: str = "fred_economic_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df["gdp_growth_yoy"] = df["gdp"].pct_change(12) * 100   # % YoY
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100           # inflation %
    return df


DF = load_data()
DATE_MIN = DF.index.min()
DATE_MAX = DF.index.max()
ALL_DATES = DF.index.tolist()

# Map slider marks to actual dates (one tick per year)
YEAR_MARKS = {
    i: str(d.year)
    for i, d in enumerate(ALL_DATES)
    if d.month == 1
}

# ---------------------------------------------------------------------------
# Helper: add recession bands to a figure
# ---------------------------------------------------------------------------

def add_recession_bands(fig: go.Figure, row: int, xref: str = "x") -> None:
    for start, end in RECESSIONS:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(150,150,150,0.25)",
            layer="below", line_width=0,
            row=row, col=1,
            annotation_text="Recession",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="grey",
        )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_timeseries(dff: pd.DataFrame) -> go.Figure:
    """Four-panel time-series figure: GDP, GDP growth, Unemployment, CPI/Inflation."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "GDP (Billions USD)",
            "GDP Growth YoY (%)",
            "Unemployment Rate (%)",
            "CPI & Inflation YoY (%)",
        ),
        specs=[[{"secondary_y": False}]] * 3 + [[{"secondary_y": True}]],
    )

    # --- Row 1: GDP level ---
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff["gdp"],
        name="GDP", line=dict(color="#2563eb", width=2),
        hovertemplate="%{x|%b %Y}<br>GDP: $%{y:,.1f}B<extra></extra>",
    ), row=1, col=1)

    # --- Row 2: GDP YoY growth ---
    colors_growth = [
        "#16a34a" if v >= 0 else "#dc2626"
        for v in dff["gdp_growth_yoy"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=dff.index, y=dff["gdp_growth_yoy"],
        name="GDP Growth YoY", marker_color=colors_growth,
        hovertemplate="%{x|%b %Y}<br>Growth: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1, row=2, col=1)

    # --- Row 3: Unemployment ---
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff["unemployment_rate"],
        name="Unemployment", line=dict(color="#d97706", width=2),
        fill="tozeroy", fillcolor="rgba(217,119,6,0.1)",
        hovertemplate="%{x|%b %Y}<br>Unemployment: %{y:.1f}%<extra></extra>",
    ), row=3, col=1)

    # --- Row 4: CPI level (left axis) + Inflation YoY (right axis) ---
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff["cpi"],
        name="CPI", line=dict(color="#7c3aed", width=2),
        hovertemplate="%{x|%b %Y}<br>CPI: %{y:.2f}<extra></extra>",
    ), row=4, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=dff.index, y=dff["cpi_yoy"],
        name="Inflation YoY", line=dict(color="#be185d", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Inflation: %{y:.2f}%<extra></extra>",
    ), row=4, col=1, secondary_y=True)

    # Recession shading on all rows
    for row in range(1, 5):
        add_recession_bands(fig, row=row)

    fig.update_layout(
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=40, b=40),
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Billions USD", row=1, col=1)
    fig.update_yaxes(title_text="% YoY", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="Index", row=4, col=1)
    fig.update_yaxes(title_text="% YoY", row=4, col=1, secondary_y=True)
    return fig


def build_correlation(dff: pd.DataFrame) -> go.Figure:
    """Scatter: GDP growth YoY vs Unemployment rate, with OLS regression."""
    clean = dff[["gdp_growth_yoy", "unemployment_rate"]].dropna()

    if len(clean) < 5:
        return go.Figure().add_annotation(text="Not enough data", showarrow=False)

    x = clean["unemployment_rate"].values
    y = clean["gdp_growth_yoy"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    # Color points by GDP growth (green = positive, red = negative)
    point_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in y]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color=point_colors, size=7, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        text=clean.index.strftime("%b %Y"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Unemployment: %{x:.1f}%<br>"
            "GDP Growth: %{y:.2f}%<extra></extra>"
        ),
        name="Monthly obs.",
    ))

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#1e40af", width=2, dash="dash"),
        name=f"OLS  r={r:.3f}  p={p:.4f}",
    ))

    stats_text = (
        f"<b>OLS Regression</b><br>"
        f"Slope: {slope:.3f}<br>"
        f"Intercept: {intercept:.3f}<br>"
        f"r = {r:.3f}<br>"
        f"R² = {r**2:.3f}<br>"
        f"p-value = {p:.4f}<br>"
        f"n = {len(clean)}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=stats_text,
        align="left", showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=12),
    )

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
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="FRED Economic Dashboard")

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("US Economic Indicators", style={"margin": "0", "fontSize": "1.6rem"}),
        html.P(
            "GDP · Unemployment Rate · CPI  |  Source: FRED (St. Louis Fed)",
            style={"margin": "4px 0 0", "color": "#64748b", "fontSize": "0.9rem"},
        ),
    ], style={
        "padding": "20px 32px 16px",
        "borderBottom": "1px solid #e2e8f0",
        "background": "#f8fafc",
    }),

    # Date range slider
    html.Div([
        html.Label("Date range", style={"fontWeight": "600", "fontSize": "0.85rem",
                                         "color": "#475569", "marginBottom": "6px",
                                         "display": "block"}),
        dcc.RangeSlider(
            id="date-slider",
            min=0,
            max=len(ALL_DATES) - 1,
            step=1,
            value=[0, len(ALL_DATES) - 1],
            marks=YEAR_MARKS,
            tooltip={"placement": "bottom",
                     "transform": "indexToDate"},
            allowCross=False,
        ),
    ], style={"padding": "20px 48px 8px"}),

    # Date display
    html.Div(id="date-display", style={
        "textAlign": "center", "color": "#94a3b8",
        "fontSize": "0.82rem", "marginBottom": "4px",
    }),

    # Time-series panel
    html.Div([
        dcc.Graph(id="timeseries-chart", config={"displayModeBar": True}),
    ], style={"padding": "0 24px"}),

    # Correlation panel
    html.Div([
        html.H2("Correlation Analysis: GDP Growth vs Unemployment",
                style={"fontSize": "1rem", "fontWeight": "600",
                       "color": "#334155", "margin": "0 0 4px"}),
        html.P(
            "A negative correlation (Okun's Law) is expected: higher unemployment "
            "accompanies lower or negative GDP growth.",
            style={"fontSize": "0.83rem", "color": "#64748b", "margin": "0 0 12px"},
        ),
        dcc.Graph(id="correlation-chart", config={"displayModeBar": True}),
    ], style={
        "padding": "16px 32px 32px",
        "borderTop": "1px solid #e2e8f0",
        "marginTop": "8px",
    }),

], style={"fontFamily": "'Inter', sans-serif", "maxWidth": "1200px",
          "margin": "0 auto", "background": "#fff"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("timeseries-chart", "figure"),
    Output("correlation-chart", "figure"),
    Output("date-display", "children"),
    Input("date-slider", "value"),
)
def update(slider_range):
    lo, hi = slider_range
    dff = DF.iloc[lo : hi + 1]
    label = (
        f"{ALL_DATES[lo].strftime('%b %Y')}  –  {ALL_DATES[hi].strftime('%b %Y')}"
        f"  ({hi - lo + 1} months)"
    )
    return build_timeseries(dff), build_correlation(dff), label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
