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
    ["Rank", "Percentile", "Rating"],index=2
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


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# Pre-group races for quick access
df_by_date_race = df.groupby(["date", "race_id"])

for _, team, series in plottables:
    hover_labels = []

    for date_str in [d.strftime("%Y-%m-%d") for d in dates]:
        races_today = df[(df["date"] == date_str)]
        races_with_team = races_today[races_today["school"] == team]

        if races_with_team.empty:
            hover_labels.append(f"{team}<br>Did not race on this date")
            continue

        all_race_texts = []
        for i, (_, row) in enumerate(races_with_team.iterrows(), start=1):
            race_id = row["race_id"]
            race_df = df_by_date_race.get_group((date_str, race_id)).sort_values("position")

            result_lines = [
                f"{int(r['position'])}{ordinal_suffix(int(r['position']))} — {r['school']}"
                for _, r in race_df.iterrows()
            ]
            race_text = f"Race {i}:<br>" + "<br>".join(result_lines)
            all_race_texts.append(race_text)

        hover_text = f"{team}<br>" + "<br><br>".join(all_race_texts)
        hover_labels.append(hover_text)

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series,
            mode="lines+markers",
            name=team,
            text=hover_labels,
            hoverinfo="text+name",
            line=dict(color=ivy_color.get(team)),
        )
    )


fig.update_layout(
    xaxis_title="Date",
    yaxis_title=y_label,
    yaxis_autorange="reversed" if invert_y else True,
    hovermode="closest",
    template="plotly_white",
    legend=dict(font=dict(size=10)),
)

st.plotly_chart(fig, use_container_width=True)
# =======================================================================
# 4.5 · EXPLANATION – METHOD DESCRIPTION
# =======================================================================
with st.expander("ℹ️  What do Rank, Percentile, and Rating mean?"):
    st.markdown("""
**• Rank**  
Each team's position relative to all others seen so far.  
Rank 1 means the top-performing team based on all past races.

**• Percentile**  
Translates a team's rank into a 0–100 scale.  
100 = best, 50 = middle of the pack, 0 = lowest.  
Example: a team in the 90th percentile is outperforming 90% of teams.

**• Massey Rating**  
A score based on **who you raced, who you beat, and by how much**.  
The rating system builds equations from each matchup (e.g., *Team A beat Team B by 5 seconds*)  
and solves them to find the most consistent set of scores across all teams.

- A higher rating means stronger performance across multiple races.
- Ratings are centered around 0 — average teams score near 0, strong teams rise above.

All calculations use only the results **up to each date** in the season.
""")

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
