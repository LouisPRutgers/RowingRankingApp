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
from pytz import timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io
from datetime import datetime, timedelta
import numpy as np   

from rank_math import timeline, school_colors, rolling_rating


######### CONSTANTS #############
crca_top25_schools  = [
    "Stanford","Texas","Washington - UW","Tennessee","Yale University","Princeton University",
    "Rutgers","Brown University","California","Michigan","Syracuse","Virginia - UVA",
    "University of Pennsylvania","Harvard University","UCF","Indiana","Ohio","Duke",
    "Columbia University","Oregon State - OSU","Dartmouth College","Clemson",
    "USC", "North Carolina Chapel Hill - UNC","Oklahoma"
]

ivy_schools = [
    "Brown University", "Columbia University", "Cornell University",
    "Dartmouth College", "Harvard University",
    "University of Pennsylvania", "Princeton University", "Yale University"
]

ACC_SCHOOLS = [
    "Boston College", "California", "Clemson", "Duke", "Louisville",
    "Miami (FL)", "North Carolina", "Notre Dame",
    "Southern Methodist University (SMU)", "Stanford", "Syracuse"
]

BIG_TEN_SCHOOLS = [
    "Indiana", "Iowa", "Michigan", "Michigan State", "Minnesota",
    "Ohio State", "Rutgers", "UCLA", "USC", "Washington", "Wisconsin"
]

SEC_SCHOOLS = [
    "Alabama", "Oklahoma", "Tennessee", "Texas"
]

WCC_SCHOOLS = [
    "Creighton", "Gonzaga", "Loyola Marymount", "Oregon State",
    "Portland", "Saint Mary's", "San Diego", "Santa Clara", "Washington State"
]

A10_SCHOOLS = [
    "Dayton", "Duquesne", "Fordham", "George Mason",
    "George Washington", "La Salle", "Rhode Island",
    "Saint Joseph's", "Umass"
]

PATRIOT_SCHOOLS = [
    "Boston University", "Bucknell", "Colgate", "Holy Cross",
    "Lehigh", "Loyola (MD)", "MIT", "Navy"
]

CAA_SCHOOLS = [
    "Delaware", "Drexel", "Eastern Michigan", "Monmouth",
    "Northeastern", "UC San Diego", "UConn", "Villanova"
]

MAAC_SCHOOLS = [
    "Canisius", "Drake", "Fairfield", "Iona", "Jacksonville",
    "Manhattan", "Marist", "Robert Morris", "Sacred Heart", "Stetson"
]

BIG_12_SCHOOLS = [
    "Kansas", "Kansas State", "Old Dominion", "Tulsa", "UCF", "West Virginia"
]

INDEPENDENT_SCHOOLS = [
    "Temple", "Georgetown University", "Sacramento State University",
    "Seattle University", "Bryant University"
]

# Config
CSV_PATH = Path("data/rowing_races.csv")
st.set_page_config(page_title="Rowing Race Ranker", page_icon="🚣", layout="wide")

# Load CSV
if not CSV_PATH.exists():
    st.error(f"CSV not found at {CSV_PATH}. Push data and redeploy.")
    st.stop()

df = pd.read_csv(CSV_PATH)
if df.empty or "Boat Class" not in df.columns:
    st.warning("CSV missing or malformed. Must include 'Boat Class'.")
    st.stop()

priority_order = [
    "1st Varsity 8+",
    "2nd Varsity 8+",
    "1st Varsity 4+",
    "2nd Varsity 4+",
    "3rd Varsity 8+"
]

st.sidebar.header("Filters")
#Boat-class selector (priority order respected)
available_classes = sorted(df["Boat Class"].unique())
sorted_boats = ([b for b in priority_order if b in available_classes] +
                sorted(set(available_classes) - set(priority_order)))
default_boat = "1st Varsity 8+" if "1st Varsity 8+" in sorted_boats else sorted_boats[0]

boat_class = st.sidebar.selectbox(
    "Boat Class", sorted_boats, index=sorted_boats.index(default_boat)
)

df_filtered = df.query("`Boat Class` == @boat_class")
if df_filtered.empty:
    st.warning(f"No races found for {boat_class}")
    st.stop()

# League selector
LEAGUES = [
    "CRCA Top25", "Ivy League", "Southeastern Conference (SEC)",
    "Atlantic Coast Conference (ACC)", "Big Ten Conference",
    "Big 12 Conference", "West Coast Conference (WCC)",
    "Atlantic 10 Conference (A-10)", "Patriot League",
    "Coastal Athletic Association (CAA)",
    "Metro Atlantic Athletic Conference (MAAC)",
    "Independents & Other Programs", "All",
]
selected_league = st.sidebar.selectbox("League", LEAGUES)

# Teams present *after* Boat-class filter
teams_all = sorted(df_filtered["school"].unique())

# League → schools lookup (dynamic “All” entry)
school_mapping = {
    "CRCA Top25": crca_top25_schools,
    "Ivy League": ivy_schools,
    "Southeastern Conference (SEC)": SEC_SCHOOLS,
    "Atlantic Coast Conference (ACC)": ACC_SCHOOLS,
    "Big Ten Conference": BIG_TEN_SCHOOLS,
    "Big 12 Conference": BIG_12_SCHOOLS,
    "West Coast Conference (WCC)": WCC_SCHOOLS,
    "Atlantic 10 Conference (A-10)": A10_SCHOOLS,
    "Patriot League": PATRIOT_SCHOOLS,
    "Coastal Athletic Association (CAA)": CAA_SCHOOLS,
    "Metro Atlantic Athletic Conference (MAAC)": MAAC_SCHOOLS,
    "Independents & Other Programs": INDEPENDENT_SCHOOLS,
    "All": teams_all,
}

preselected = (
    [s for s in school_mapping[selected_league] if s in teams_all]
    if selected_league != "All" else teams_all
)

# Session-state helpers ----------------------------------------------------
def _ensure_state():
    st.session_state.setdefault("chosen_schools", preselected)
    st.session_state.setdefault("modifications", {})          # {school: "added"/"removed"}
    st.session_state.setdefault("previous_league", selected_league)

_ensure_state()

# ── 5️⃣  Handle league changes & keep per-run selection alive ──────────────────
if "previous_league" not in st.session_state:
    st.session_state.previous_league = selected_league
if "previous_boat" not in st.session_state:
    st.session_state.previous_boat = boat_class
if "schools_on_chart" not in st.session_state:
    st.session_state.schools_on_chart = []
if "modifications" not in st.session_state:
    st.session_state.modifications = {}
if "chosen_schools" not in st.session_state:
    st.session_state.chosen_schools = []
if "init_done" not in st.session_state:
    st.session_state.init_done = False

if not st.session_state.init_done:                    # ← changed condition
    first_list = (
        teams_all
        if selected_league == "All"
        else [s for s in school_mapping[selected_league] if s in teams_all]
    )
    st.session_state["schools_on_chart"] = first_list
    st.session_state.chosen_schools = first_list.copy()
    st.session_state.init_done = True                 # ← mark as done

# -------- When the user picks a *different* LEAGUE ----------------------------
if selected_league != st.session_state.previous_league:
    # fresh full list for that league (restricted to teams in current boat-class data)
    if selected_league == "All":
        fresh_list = teams_all
    else:
        league_list = school_mapping[selected_league]
        fresh_list = [s for s in league_list if s in teams_all]

    # wipe any old add/remove history and push the fresh list into state + widget
    st.session_state.modifications.clear()
    st.session_state.chosen_schools = fresh_list
    st.session_state.schools_on_chart = fresh_list.copy()

    # remember which league is now active
    st.session_state.previous_league = selected_league

# ── Handle a change in the Boat-class dropdown ──────────────────────
if boat_class != st.session_state.previous_boat:
    # show only the schools (from the user’s current selection) that
    # actually race in this boat class
    visible = [s for s in st.session_state.chosen_schools if s in teams_all]

    # push that list into the widget so it never starts empty
    st.session_state.schools_on_chart = visible

    # remember this boat class
    st.session_state.previous_boat = boat_class
# ────────────────────────────────────────────────────────────────────

# ── 6️⃣  School multiselect (widget reads/writes `schools_on_chart`) ───────────
chosen_schools = st.sidebar.multiselect(
    "Schools on chart",
    options=teams_all,
    key="schools_on_chart",        # <- widget’s single source of truth
)

# ---- Track what the user just did -------------------------------------------
# additions
for s in chosen_schools:
    if st.session_state.modifications.get(s) == "removed":
        st.session_state.modifications[s] = "added"
    elif s not in st.session_state.modifications:
        st.session_state.modifications[s] = "added"

# removals
for s in set(st.session_state.chosen_schools) - set(chosen_schools):
    st.session_state.modifications[s] = "removed"

# stash latest list for downstream plotting etc.
st.session_state.chosen_schools = chosen_schools
chosen = st.session_state.chosen_schools

# 🎨 Color dictionary
school_color = school_colors()




# Metric selection
mode = st.sidebar.radio("Metric to plot", ["Rank", "Percentile", "Rating"], index=2)

# Rolling toggle + settings
st.sidebar.markdown("### Rolling")
use_rolling = st.sidebar.toggle("Apply rolling window", value=False)

if use_rolling:
    days_window = st.sidebar.number_input("Rolling window (days)", 1, 180, value=30)
    dropoff = st.sidebar.selectbox("Drop-off", ["Sudden Decay", "Linear Decay", "Exponential Decay"], index=1)
    decay_rate = st.sidebar.slider("Decay rate (r)", 0.01, 1.0, 0.1, step=0.01) if dropoff == "Exponential Decay" else None

# Apply timeline logic after rolling toggle + settings
if use_rolling:
    dates, rolling = rolling_rating(df_filtered, window_days=days_window, dropoff=dropoff, decay_rate=decay_rate)
    rating = rolling
    pct = {t: [] for t in teams_all}
    rank_rel = {t: [] for t in teams_all}
    for i in range(len(dates)):
        scores_today = {t: rating[t][i] for t in teams_all if rating[t][i] is not None}
        ranked = sorted(scores_today.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        ranks_today = {team: j + 1 for j, (team, _) in enumerate(ranked)}
        for t in teams_all:
            score = rating[t][i]
            rank_rel[t].append(ranks_today.get(t))
            pct[t].append(100 * (n - ranks_today[t] + 1) / n if t in ranks_today else None)
else:
    dates, rank_rel, pct, rating = timeline(df_filtered)

#Show weight Overlay
if use_rolling:
    show_overlay = st.sidebar.toggle("Show weighting overlay", value=True)
else:
    show_overlay = False

# ── helper that builds green-background shapes ───────────────────────
def _weight_shapes() -> list[dict]:
    if not show_overlay:
        return []                 # overlay disabled

    latest = dates[-1]            # we shade w.r.t. *today's* rating
    g = (0, 255, 0)               # pure green
    max_a = 0.18                  # peak opacity

    shapes = []
    if not use_rolling:           # no window → solid green
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=min(dates), x1=max(dates),
                           y0=0, y1=1,
                           fillcolor=f"rgba({g[0]},{g[1]},{g[2]},{max_a})",
                           line_width=0))
        return shapes

    start = latest - timedelta(days=days_window)

    if dropoff == "Sudden Decay":
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=start, x1=latest, y0=0, y1=1,
                           fillcolor=f"rgba({g[0]},{g[1]},{g[2]},{max_a})",
                           line_width=0))
        return shapes

    # Linear or Exponential → fade in ~50 slices
    slices = 50
    step = days_window / slices
    for i in range(slices):
        seg_end   = latest - timedelta(days=i * step)
        seg_start = latest - timedelta(days=(i + 1) * step)
        age_mid   = (latest - (seg_start + (seg_end - seg_start) / 2)).days
        if dropoff == "Linear Decay":
            w = max(0.0, 1 - age_mid / days_window)
        else:                       # Exponential Decay
            w = np.exp(-decay_rate * age_mid)
        alpha = max_a * w
        if alpha < 0.01:            # skip if almost invisible
            continue
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=seg_start, x1=seg_end, y0=0, y1=1,
                           fillcolor=f"rgba({g[0]},{g[1]},{g[2]},{alpha:.3f})",
                           line_width=0))
    return shapes


# Chart logic
st.title(f"NCAA Women's Collegiate Rowing Ranker – {boat_class}")
now_et = datetime.now(timezone("US/Eastern"))
st.caption(f"Data last updated: {now_et:%B %d, %Y at %I:%M %p} ET • CSV path: `data/rowing_races.csv`")

metric_map = {"Rank": rank_rel, "Percentile": pct, "Rating": rating}
invert_y = (mode == "Rank")
y_label = f"{mode} (Rolling, {days_window}d, drop-off: {dropoff})" if use_rolling else {
    "Rank": "Rank (1 = best)",
    "Percentile": "Percentile (100 = best)",
    "Rating": "Massey rating (higher = better)",
}[mode]

def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

# Sort chosen schools by most recent value of selected metric
latest_values = {}
for team in chosen:
    series = metric_map[mode][team]
    latest_val = next((v for v in reversed(series) if v is not None), None)
    if latest_val is not None:
        latest_values[team] = latest_val

# Sorting logic: lower Rank = better, higher Percentile/Rating = better
reverse = (mode != "Rank")
chosen_sorted = sorted(latest_values, key=latest_values.get, reverse=reverse)


fig = go.Figure()
for team in chosen_sorted:
    series = metric_map[mode][team]
    if not any(pd.notna(series)):
        continue
    latest_val = next((v for v in reversed(series) if v is not None), None)
    if latest_val is None:
        continue
    hover_labels = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        races_today = df_filtered[df_filtered["date"] == date_str]
        races_with_team = races_today[races_today["school"] == team]
        val = series[dates.index(d)]
        label_val = f"{mode}: {int(round(val))}" if mode == "Rank" and val is not None else f"{mode}: {val:.2f}" if val is not None else f"{mode}: N/A"
        if races_with_team.empty:
            hover_labels.append(f"{team} ({d.strftime('%m/%d/%y')})<br>{label_val}<br>No race on this date")
            continue
        all_race_texts = []
        for i, (_, row) in enumerate(races_with_team.iterrows(), start=1):
            race_id = row["race_id"]
            race_df = df_filtered[df_filtered["race_id"] == race_id].sort_values("position")
            first_time = race_df.iloc[0]["time"]
            result_lines = []
            for _, r in race_df.iterrows():
                pos, secs = int(r["position"]), r["time"]
                margin_str = f", +{int(secs - first_time)}s" if secs > first_time else ""
                result_lines.append(f"{pos}{ordinal_suffix(pos)} — {r['school']} ({int(secs)//60}:{int(secs)%60:02d}{margin_str})")
            all_race_texts.append(f"Race {i}:<br>" + "<br>".join(result_lines))
        hover_labels.append(f"{team} ({d.strftime('%m/%d/%y')})<br>{label_val}<br>" + "<br>".join(all_race_texts))
        
    marker_symbols = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        if not df_filtered[(df_filtered["date"] == date_str) & (df_filtered["school"] == team)].empty:
            marker_symbols.append("circle")
        else:
            marker_symbols.append("line-ew")
  
    fig.add_trace(go.Scatter(
        x=dates,
        y=series,
        mode="lines+markers",
        name=team,
        text=hover_labels,
        hoverinfo="text+name",
        line=dict(color=school_color.get(team)),
        marker=dict(size=6, color=school_color.get(team), symbol=marker_symbols)
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title=y_label,
    yaxis_autorange="reversed" if invert_y else True,
    hovermode="closest",
    template="plotly_white",
    legend=dict(font=dict(size=10)),
    shapes=_weight_shapes(),
)
st.plotly_chart(fig, use_container_width=True)
with st.expander("🏆 What if the NCAA were to happen today?", expanded=False):
    st.markdown("Below is the predicted team standings if the NCAA Championship happened today, based on the latest ratings!")

    def calculate_points(sorted_teams, base, step=3):
        """
        Assigns both:
        - Official display points (e.g., 66, 63, ...)
        - Hidden placement scores used for sorting (same as display, but even 0 gets negative scores)
        """
        points = {}
        for i, team in enumerate(sorted_teams):
            pts = base - i * step  # Displayed NCAA points
            points[team] = pts
        return points

    boat_classes = {
        "1st Varsity 8+": {"points_start": 66, "step": 3, "col": "1st Varsity 8+ Points"},
        "2nd Varsity 8+": {"points_start": 44, "step": 2, "col": "2nd Varsity 8+ Points"},
        "1st Varsity 4+": {"points_start": 22, "step": 1, "col": "1st Varsity 4+ Points"},
    }

    team_points = {}

    # For each boat class, calculate the predicted points using the rating data
    for boat, config in boat_classes.items():
        df_boat = df[df["Boat Class"] == boat]
        if df_boat.empty:
            continue

        # Use the previously computed rank_rel for the selected rating method (Rating or Rolling Rating)
        latest_ranks = {
            team: next((r for r in reversed(rank_rel[team]) if r is not None), None)
            for team in rank_rel  # We are using rank_rel computed earlier, not from timeline(df_boat)
        }

        # Sort teams by latest rank or rating (based on the latest ranks from rank_rel)
        sorted_teams = [t for t, _ in sorted(
            latest_ranks.items(), key=lambda x: x[1] if x[1] is not None else float("inf")
        )]

        # Assign official points and hidden placement scores
        display_pts = calculate_points(
            sorted_teams, config["points_start"], config["step"]
        )

        for team in sorted_teams:
            if team not in team_points:
                team_points[team] = {}
            team_points[team][config["col"]] = display_pts[team]


    # Build the results table
    results_table = []
    for team in team_points:
        # Display points for UI
        p1 = team_points[team].get("1st Varsity 8+ Points", 0)
        p2 = team_points[team].get("2nd Varsity 8+ Points", 0)
        p4 = team_points[team].get("1st Varsity 4+ Points", 0)
        #If there's a positive score, disregard the negatives in the sum
        if(p1>0 or p2>0 or p4>0):
            total_points = max(p1,0) + max(p2,0) + max(p4,0)
        else:
            total_points = p1 + p2 + p4

        # Only include teams with at least one positive score 
        results_table.append({
            "School Name": team,
            "1st Varsity 8+ Points": p1,
            "2nd Varsity 8+ Points": p2,
            "1st Varsity 4+ Points": p4,
            "Overall Team points": total_points,
            "_1V8 Points for Tie": p1  # Needed for tie-breaking per NCAA rules
        })

    results_df = pd.DataFrame(results_table)

    # Break ties using official 1V8+ points (also descending)
    results_df = results_df.sort_values(
        by=["Overall Team points", "_1V8 Points for Tie"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Assign placement numbers
    results_df["Placement"] = range(1, len(results_df) + 1)

    # Remove internal scoring columns before displaying
    results_df.drop(columns=["_1V8 Points for Tie"], inplace=True)

    # Move 'Placement' column to the front
    cols = ["Placement"] + [col for col in results_df.columns if col != "Placement"]
    results_df = results_df[cols]

    #replace non zero values with zero
    results_df = results_df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' and x.str.isdigit().all() else x)
    results_df = results_df.where(results_df.apply(lambda x: x >= 0 if pd.api.types.is_numeric_dtype(x) else True), 0)
    # Show table in UI
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Convert results_df to CSV in memory
    csv = results_df.to_csv(index=False)
    csv_bytes = csv.encode("utf-8")
    buffer = io.BytesIO(csv_bytes)
    st.download_button(
        label="⬇️ Download Prediction Table",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="ncaa_prediction_table.csv",
        mime="text/csv"
    )


# Explanation
with st.expander("ℹ️  Methods. How are Rank, Percentile, and Rating calculated?"):
    st.markdown("""
### 📊 Data Source & Processing
All race data is manually collected from **Row2k.com Results**, and includes:
- **Date**
- **Boat Class**
- **Team Name (School)**
- **Position (Finish Order)**
- **Elapsed Time (in seconds)**

Not all NCAA Schools were included in this analysis. If a school was in a race but was not included in this analysis, they were simply skipped 
over in the race results.
The general inclusion criteria was to include all regattas of schools included in the analysis from march 21st onwards. 
That being said, it is possible I missed regattas of schools that are in this list. 

Each race is processed into **pairwise comparisons** (e.g., Team A beat Team B by 5 seconds), and all methods below rely on that foundation.

---

### 🥇 Rank
For each date, a **Massey Rating** is calculated for all teams based on races **up to that point**.  
Teams are then **ranked from best to worst** by rating.

- Rank 1 = top team so far  
- If a team has not yet raced, Rank is **empty**

---

### 📈 Percentile
Percentile is a **rescaled version of rank**, expressed as a score from 0 to 100:
- 100 = top-ranked team  
- 50 = middle of the pack  
- 0 = lowest-ranked team

This helps compare teams more intuitively, especially when field size changes.

---

### 🧠 Massey Rating
This is a continuous numeric score that reflects **how dominant a team has been**, factoring in:
- Who they raced
- Who they beat
- By how much

Massey ratings are **centered around 0**, with top teams rising above and underperformers falling below.

---

### 🔁 Rolling Ratings (New!)
Optionally, you can compute a team’s rating using only **recent races** from a moving window of days.

You can also choose how older results “fade out”:

- **Sudden**: All races within the window have equal weight; older ones drop off immediately.
- **Linear**: Races lose influence gradually as they age within the window.
- **Exponential Decay**: Older races decline quickly in influence, controlled by a **decay rate**.

Rolling ratings are helpful for tracking **momentum** and avoiding outdated performance bias.

---
""")

with st.expander("ℹ️  List of Schools Included in the Results."):
    st.markdown("""
There are many schools in the NCAA and it was not feasable to include them all.
The following schools were included for the purposes of this analysis:

**Schools:**""")
    
    schools = sorted([
        "Brown University", "Columbia University", "Cornell University", "Dartmouth College",
        "Harvard University", "University of Pennsylvania", "Princeton University", "Yale University",
        "Michigan", "Rutgers", "Tennessee", "Washington - UW", "Northeastern",
        "Boston University - BU", "Colgate", "Stanford", "Syracuse", "Ohio", "Gonzaga", "Texas",
        "Duke", "California", "Indiana", "Jacksonville", "Virginia - UVA", "SMU",
        "Rhode Island (RIU)", "Iowa", "Oregon State - OSU", "UCF", "Louisville", "Notre Dame",
        "Duquesne", "Clemson", "Portland", "George Washington", "Oklahoma",
        "University of Minnesota", "Michigan State", "Boston College",
        "North Carolina Chapel Hill - UNC", "Kansas State", "Miami", "Washington State - WSU",
        "USC", "UCLA", "UCSD", "Wisconsin"
    ])
    
    cols = st.columns(3)
    col_blocks = ["", "", ""]

    for i, school in enumerate(schools):
        col_blocks[i % 3] += f"- {school}\n"

    for col, text in zip(cols, col_blocks):
        col.markdown(text)


with st.expander("ℹ️  Errors? Contact me!"):
    st.markdown("""
**Methods:**  
The database for these rankings sources data from Row2k.com Results.  
Data is acquired manually – so mistakes can happen!  
If you notice something wrong, or an important race is missing, let me know!

**Contact:**  
collegiate.rowing.rankings@gmail.com
""")

# Download
with open(CSV_PATH, "rb") as fh:
    st.download_button("⬇️ Download raw CSV", fh.read(), file_name="rowing_races.csv", mime="text/csv")



