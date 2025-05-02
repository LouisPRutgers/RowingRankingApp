# streamlit_app.py
# ────────────────────────────────────────────────────────────────────
"""
Rowing Race Ranker – web front‑end
Show interactive Rank / Percentile / Rating trend lines for
any subset of schools, using the results CSV produced by the
desktop GUI.  Built for Streamlit Community Cloud.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----- repo‑local modules ------------------------------------------------------
# `rank_math.py` must expose two helpers:
#   timeline(dframe)  -> (dates, rank_rel, pct, rating)
#   ivy_colors()      -> dict{school: hex}
from rank_math import timeline, ivy_colors


# =======================================================================
# 1 · CONFIG
# =======================================================================
CSV_PATH = Path("data/rowing_races.csv")

st.set_page_config(
    page_title="Rowing Race Ranker",
    page_icon="🚣",
    layout="wide",
)

# =======================================================================
# 2 · DATA LOAD + PRE‑CALC
# =======================================================================
if not CSV_PATH.exists():
    st.error(f"CSV not found at {CSV_PATH}.  Push data and redeploy.")
    st.stop()

df = pd.read_csv(CSV_PATH)
if df.empty:
    st.warning("No races yet – add some via the desktop app.")
    st.stop()

dates, rank_rel, pct, rating = timeline(df)
teams_all = sorted(rank_rel)              # every team ever seen
ivy_color = ivy_colors()                  # hard‑coded palette
default_sel = [t for t in teams_all if t in ivy_color]

# =======================================================================
# 3 · SIDEBAR – CONTROLS
# =======================================================================
st.sidebar.header("Filters")
chosen = st.sidebar.multiselect(
    "Schools on chart",
    options=teams_all,
    default=default_sel,
)

mode = st.sidebar.radio(
    "Metric to plot",
    ["Rank", "Percentile", "Rating"],
)

# =======================================================================
# 4 · MAIN – TITLE + CHART
# =======================================================================
st.title("NCAA Women's Collegiate Rowing Ranker")
st.caption(
    f"Data last updated: {datetime.now():%Y‑%m‑%d %H:%M} ET &nbsp;•&nbsp; "
    "CSV path: `data/rowing_races.csv`"
)

metric_map = {
    "Rank":       rank_rel,
    "Percentile": pct,
    "Rating":     rating,
}
invert_y = mode == "Rank"
y_label = {
    "Rank":       "Rank (1 = best)",
    "Percentile": "Percentile (100 = best)",
    "Rating":     "Massey rating (higher = better)",
}[mode]

fig = go.Figure()

# Get all selected series with their latest available value
plottables = []
for team in chosen:
    series = metric_map[mode][team]
    if not any(pd.notna(series)):
        continue
    # get latest non-None value
    latest_val = next((v for v in reversed(series) if v is not None), None)
    if latest_val is not None:
        plottables.append((latest_val, team, series))

# Sort: lower is better for rank, higher is better otherwise
plottables.sort(key=lambda x: x[0], reverse=(mode != "Rank"))

# Add traces in sorted order → controls hover legend order
for _, team, series in plottables:
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series,
            mode="lines+markers",
            name=team,
            line=dict(color=ivy_color.get(team)),
        )
    )


fig.update_layout(
    xaxis_title="Date",
    yaxis_title=y_label,
    yaxis_autorange="reversed" if invert_y else True,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(font=dict(size=10)),
)

st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# 5 · FOOTER – DOWNLOAD BUTTON
# =======================================================================
with open(CSV_PATH, "rb") as fh:
    st.download_button(
        label="⬇️ Download raw CSV",
        data=fh.read(),
        file_name="rowing_races.csv",
        mime="text/csv",
    )
