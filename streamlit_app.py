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

from rank_math import timeline, school_colors, rolling_rating

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

# Boat class dropdown
priority_order = [
    "1st Varsity 8+",
    "2nd Varsity 8+",
    "1st Varsity 4+",
    "2nd Varsity 4+",
    "3rd Varsity 8+"
]
available_classes = sorted(df["Boat Class"].unique())
sorted_boats = [b for b in priority_order if b in available_classes] + \
               sorted([b for b in available_classes if b not in priority_order])
default_boat = "1st Varsity 8+" if "1st Varsity 8+" in sorted_boats else sorted_boats[0]

st.sidebar.header("Filters")
boat_class = st.sidebar.selectbox("Boat Class", options=sorted_boats, index=sorted_boats.index(default_boat))
df_filtered = df[df["Boat Class"] == boat_class]
if df_filtered.empty:
    st.warning(f"No races found for {boat_class}")
    st.stop()

preselected_schools = st.sidebar.selectbox("League", ["CRCA Top25", "Ivy League", "All"])

# All teams (for filtering and plotting)
teams_all = sorted(df_filtered["school"].unique())

# School filter
school_color = school_colors()
crca_top25_schools  = [
    "Stanford",
    "Texas",
    "Washington - UW",
    "Tennessee",
    "Yale University",
    "Princeton University",
    "Rutgers",
    "Brown University",
    "California",
    "Michigan",
    "Syracuse",
    "Virginia - UVA",
    "University of Pennsylvania",
    "Harvard University",
    "UCF",
    "Indiana",
    "Ohio",
    "Duke",
    "Columbia University",  
    "Oregon State - OSU",
    "Dartmouth College",
    "Clemson",
    "USC", 
    "North Carolina Chapel Hill - UNC",
    "Oklahoma"
]

ivy_schools = [
    "Brown University", "Columbia University", "Cornell University",
    "Dartmouth College", "Harvard University",
    "University of Pennsylvania", "Princeton University", "Yale University"
]

# Map league selections to corresponding school lists
school_mapping = {
    "Ivy League": ivy_schools,
    "CRCA Top25": crca_top25_schools,
    "All": teams_all
}

# Get the list of schools based on the selected league
preselected_schoolsed_list = school_mapping.get(preselected_schools, [])

# Filter the schools to include only those present in 'teams_all'
# This ensures that we don't include schools that are not part of the dataset
preselected_schoolsed_list = [school for school in preselected_schoolsed_list if school in teams_all] if preselected_schools != "All" else teams_all

# Initialize session state for chosen schools if it hasn't been initialized
# This is the list of schools that the user will see and can modify
if "chosen_schools" not in st.session_state:
    st.session_state.chosen_schools = preselected_schoolsed_list

# Initialize session state for modifications if it hasn't been initialized
# The 'modifications' dictionary keeps track of schools the user has added or removed
if "modifications" not in st.session_state:
    st.session_state.modifications = {}

# Initialize 'previous_league' to keep track of the last selected league
# This allows us to compare if the user switches between leagues
if "previous_league" not in st.session_state:
    st.session_state.previous_league = preselected_schools

# Check if the league has changed
if preselected_schools != st.session_state.previous_league:
    # If the league has changed, reset the modifications
    # Update the list of schools based on the selected league and preserve user modifications
    if preselected_schools != "All":
        # Reset to the list of schools for the selected league but exclude schools marked as "removed"
        st.session_state.chosen_schools = [school for school in preselected_schoolsed_list if school not in st.session_state.modifications.get("removed", [])]
    else:
        # If "All" schools are selected, allow the user to modify all schools
        st.session_state.chosen_schools = preselected_schoolsed_list
    
    # Update the 'previous_league' session state to the current league
    st.session_state.previous_league = preselected_schools  # Store the current league

# Handle school modifications (additions and removals)
chosen_schools = st.sidebar.multiselect(
    "Schools on chart", options=teams_all, default=[school for school in st.session_state.chosen_schools if school in teams_all]
)

# Track schools the user has added
for school in chosen_schools:
    if school not in st.session_state.modifications or st.session_state.modifications[school] == "removed":
        st.session_state.modifications[school] = "added"

# Track schools the user has removed
removed_schools = [school for school in teams_all if school not in chosen_schools and school in st.session_state.chosen_schools]
for school in removed_schools:
    st.session_state.modifications[school] = "removed"

# Update the session state with the latest list of chosen schools
st.session_state.chosen_schools = chosen_schools

# Store the final list of chosen schools in the session state (after modification tracking)
chosen = st.session_state.chosen_schools




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
    fig.add_trace(go.Scatter(
        x=dates,
        y=series,
        mode="lines+markers",
        name=team,
        text=hover_labels,
        hoverinfo="text+name",
        line=dict(color=school_color.get(team)),
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title=y_label,
    yaxis_autorange="reversed" if invert_y else True,
    hovermode="closest",
    template="plotly_white",
    legend=dict(font=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("🏆 What if the NCAA were to happen today?"):
    st.markdown("Below is the predicted team standings if the NCAA Championship happened today, based on the latest ratings!")

    def assign_points_with_hidden_ranking(sorted_teams, base, step=3):
        """
        Assigns both:
        - Official display points (e.g., 66, 63, ...)
        - Hidden placement scores used for sorting (same as display, but even 0 gets negative scores)
        """
        display_points = {}
        hidden_scores = {}
        for i, team in enumerate(sorted_teams):
            pts = max(base - i * step, 0)  # Displayed NCAA points
            display_points[team] = pts
            # Hidden placement score: zero scores are treated as negative for fair sorting
            hidden_scores[team] = pts if pts > 0 else -step * (len(sorted_teams) - i)
        return display_points, hidden_scores

    boat_classes = {
        "1st Varsity 8+": {"points_start": 66, "step": 3, "col": "1st Varsity 8+ Points"},
        "2nd Varsity 8+": {"points_start": 44, "step": 2, "col": "2st Varsity 8+ Points"},
        "1st Varsity 4+": {"points_start": 22, "step": 1, "col": "1st Varsity 4+ Points"},
    }

    # Collect points (shown) and hidden placement scores (used for ranking)
    team_display_points = {}
    team_hidden_scores = {}

    for boat, config in boat_classes.items():
        df_boat = df[df["Boat Class"] == boat]
        if df_boat.empty:
            continue

        # Get each team's most recent relative rank
        dates_b, rank_rel_b, _, _ = timeline(df_boat)
        latest_ranks = {
            team: next((r for r in reversed(rank_rel_b[team]) if r is not None), None)
            for team in rank_rel_b
        }

        # Sort teams by latest rank (lowest rank = best)
        sorted_teams = [t for t, _ in sorted(
            latest_ranks.items(), key=lambda x: x[1] if x[1] is not None else float("inf")
        )]

        # Assign official points and hidden placement scores
        display_pts, hidden_pts = assign_points_with_hidden_ranking(
            sorted_teams, config["points_start"], config["step"]
        )

        for team in sorted_teams:
            if team not in team_display_points:
                team_display_points[team] = {}
                team_hidden_scores[team] = {}
            team_display_points[team][config["col"]] = display_pts[team]
            team_hidden_scores[team][config["col"]] = hidden_pts[team]

    # Build the results table
    results_table = []
    for team in team_display_points:
        # Display points for UI
        p1 = team_display_points[team].get("1st Varsity 8+ Points", 0)
        p2 = team_display_points[team].get("2st Varsity 8+ Points", 0)
        p4 = team_display_points[team].get("1st Varsity 4+ Points", 0)
        total_display = p1 + p2 + p4

        # Hidden score logic — only apply negative fallback if all events are 0
        has_positive = any(x > 0 for x in [p1, p2, p4])
        if has_positive:
            total_hidden = p1 + p2 + p4
        else:
            h1 = team_hidden_scores[team].get("1st Varsity 8+ Points", 0)
            h2 = team_hidden_scores[team].get("2st Varsity 8+ Points", 0)
            h4 = team_hidden_scores[team].get("1st Varsity 4+ Points", 0)
            total_hidden = h1 + h2 + h4


        results_table.append({
            "School Name": team,
            "1st Varsity 8+ Points": p1,
            "2st Varsity 8+ Points": p2,
            "1st Varsity 4+ Points": p4,
            "Overall Team points": total_display,
            "_Total Placement Score": total_hidden,
            "_1V8 Points for Tie": p1  # Needed for tie-breaking per NCAA rules
        })

    results_df = pd.DataFrame(results_table)

    # Sort using hidden score (descending = higher score is better)
    # Break ties using official 1V8+ points (also descending)
    results_df = results_df.sort_values(
        by=["_Total Placement Score", "_1V8 Points for Tie"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Assign placement numbers
    results_df["Placement"] = range(1, len(results_df) + 1)

    # Remove internal scoring columns before displaying
    results_df.drop(columns=["_Total Placement Score", "_1V8 Points for Tie"], inplace=True)

    # Move 'Placement' column to the front
    cols = ["Placement"] + [col for col in results_df.columns if col != "Placement"]
    results_df = results_df[cols]

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



