# rank_math.py
# ────────────────────────────────────────────────────────────────────
"""
Pure logic helpers shared by the desktop GUI and Streamlit front‑end.

Exposes two public functions:

    timeline(df)  -> dates, rank_rel, pct, rating
    ivy_colors()  -> {school: hex}

`df` must be a pandas DataFrame with columns:
    race_id (int / str)    – unique per race
    date    (YYYY‑MM‑DD)   – same for all rows of a race
    position (1, 2, 3…)    – place within a race (smaller = faster)
    school   (str)         – team name
    time     (float)       – seconds elapsed (lower = faster)
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────
IVY_SCHOOLS = [
    "Brown University", "Columbia University", "Cornell University",
    "Dartmouth College", "Harvard University",
    "University of Pennsylvania", "Princeton University", "Yale University",
]

SCHOOL_COLORS = {
    "Alabama": "#9E1B32",
    "Boston College": "#98002E",
    "Boston University - BU": "#CC0000",
    "Brown University": "#4E3629",
    "Bryant University": "#FFC72C",
    "Bucknell": "#FF5F00",
    "California": "#003262",
    "Canisius": "#00205B",
    "Clemson": "#F66733",
    "Colgate": "#821019",
    "Columbia University": "#9BDDFF",
    "Cornell University": "#B31B1B",
    "Creighton": "#005CA9",
    "Dartmouth College": "#00693E",
    "Dayton": "#C8102E",
    "Delaware": "#00539B",
    "Drake": "#004B87",
    "Drexel": "#07294D",
    "Duke": "#0736A4",
    "Duquesne": "#041E42",
    "Eastern Michigan": "#00694E",
    "Fairfield": "#E31B23",
    "Fordham": "#7C0A02",
    "George Mason": "#006633",
    "George Washington": "#002855",
    "Georgetown University": "#112E51",
    "Harvard University": "#A41034",
    "Holy Cross": "#582C83",
    "Indiana": "#990000",
    "Iona": "#7B3F00",
    "Iowa": "#FFCD00",
    "Jacksonville": "#007A33",
    "Kansas": "#0051BA",
    "Kansas State": "#512888",
    "La Salle": "#00205B",
    "Lehigh": "#6B4C3B",
    "Louisville": "#AD0000",
    "Loyola (MD)": "#00543C",
    "Loyola Marymount": "#00205C",
    "MIT": "#A31F34",
    "Manhattan": "#00703C",
    "Marist": "#C8102E",
    "Miami": "#F47321",
    "Michigan": "#00274C",
    "Michigan State": "#18453B",
    "Monmouth": "#00205B",
    "Navy": "#00205B",
    "North Carolina": "#7BAFD4",
    "Northeastern": "#C8102E",
    "Notre Dame": "#0C2340",
    "Ohio": "#666666",
    "Oklahoma": "#841617",
    "Old Dominion": "#00205B",
    "Oregon State - OSU": "#DC4405",
    "Portland": "#5E2A84",
    "Princeton University": "#E77500",
    "Rhode Island (RIU)": "#75B2DD",
    "Robert Morris": "#00205B",
    "Rutgers": "#CC0033",
    "Sacramento State University": "#043927",
    "Sacred Heart": "#C8102E",
    "Saint Joseph's": "#9E1B32",
    "Saint Mary's": "#A6192E",
    "San Diego": "#00205C",
    "Santa Clara": "#862633",
    "Seattle University": "#C8102E",
    "Southern Methodist University (SMU)": "#CC0000",
    "Stanford": "#8C1515",
    "Stetson": "#006747",
    "Syracuse": "#F76900",
    "Temple": "#9E1B32",
    "Tennessee": "#FF8200",
    "Texas": "#BF5700",
    "Tulsa": "#002D72",
    "UCF": "#BA9B37",
    "UCLA": "#2774AE",
    "UCSD": "#00629B",
    "UConn": "#000E2F",
    "Umass": "#881C1C",
    "University of Minnesota": "#7A0019",
    "University of Pennsylvania": "#011F5B",
    "University of Wisconsin": "#C5050C",
    "Villanova": "#003366",
    "Virginia - UVA": "#232D4B",
    "Washington - UW": "#4B2E83",
    "Washington State - WSU": "#981E32",
    "West Virginia": "#002855",
    "Yale University": "#00356B"
}



# ── public helper ─────────────────────────────────────────────────────
def school_colors() -> Dict[str, str]:
    """Hard‑coded palette used by both GUIs."""
    return SCHOOL_COLORS.copy()


# ── internal math helpers ─────────────────────────────────────────────
def _pairwise(rows: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Convert each race into pairwise comparisons.
    Return a list of dicts with keys:
        school_a, school_b, winner, margin
    """
    pairs: List[Dict[str, str]] = []
    for race_id, boats in rows.groupby("race_id"):
        boats_sorted = boats.sort_values("position")
        boats_list = list(boats_sorted.itertuples(index=False))

        for i, a in enumerate(boats_list):
            for b in boats_list[i + 1:]:
                margin = b.time - a.time
                pairs.append(
                    {
                        "school_a": a.school,
                        "school_b": b.school,
                        "winner":   a.school,
                        "margin":   f"{margin:.3f}",
                    }
                )
    return pairs


def _ratings(pairs: List[Dict[str, str]]) -> Dict[str, float]:
    """Solve the Massey rating system via least squares."""
    if not pairs:
        return {}

    teams = list({s for r in pairs for s in (r["school_a"], r["school_b"])})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    M = np.zeros((n, n))
    b = np.zeros(n)

    for r in pairs:
        i, j = idx[r["school_a"]], idx[r["school_b"]]
        margin = float(r["margin"])

        # Massey matrix
        M[i, i] += 1
        M[j, j] += 1
        M[i, j] -= 1
        M[j, i] -= 1

        # point‑differential vector
        b[i] += margin
        b[j] -= margin

    # Anchor the ratings (replace last equation)
    M[-1, :] = 1
    b[-1] = 0

    sol = np.linalg.lstsq(M, b, rcond=None)[0]
    return {team: float(sol[idx[team]]) for team in teams}


def _pairwise_subset(rows_subset: pd.DataFrame) -> List[Dict[str, str]]:
    """Same as _pairwise but accepts a pre‑filtered DataFrame."""
    return _pairwise(rows_subset)



# ── public timeline builder ───────────────────────────────────────────
def timeline(
    df: pd.DataFrame,
) -> Tuple[List[datetime], Dict[str, list], Dict[str, list], Dict[str, list]]:
    """
    Compute the evolution of (rank, percentile, rating) for each team
    as the season unfolds.

    Returns
    -------
    dates         : list[datetime]
    rank_rel      : dict[team → list[int|None]]
    pct           : dict[team → list[float|None]]
    rating_abs    : dict[team → list[float|None]]
    """
    if df.empty:
        return [], {}, {}, {}

    # --- prepare chronological date list ------------------------------
    df_dates = (
        df[["race_id", "date"]]
        .drop_duplicates()
        .assign(date_obj=lambda x: pd.to_datetime(x["date"]))
    )
    dates_sorted = (
        df_dates.sort_values("date_obj")["date"].unique().tolist()
    )
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates_sorted]

    # --- containers ----------------------------------------------------
    teams_all = sorted(df["school"].unique())
    rank_rel   = {t: [] for t in teams_all}
    pct        = {t: [] for t in teams_all}
    rating_abs = {t: [] for t in teams_all}

    # --- progressive accumulation -------------------------------------
    seen_race_ids: List[int] = []
    for d in dates_sorted:
        # races up to and including day d
        seen_race_ids.extend(
            df_dates.loc[df_dates["date"] == d, "race_id"].tolist()
        )
        subset = df[df["race_id"].isin(seen_race_ids)]

        rating_now = _ratings(_pairwise_subset(subset))
        ranked = sorted(rating_now.items(), key=lambda kv: kv[1], reverse=True)
        ranks_now = {team: i + 1 for i, (team, _) in enumerate(ranked)}
        n = len(ranked)

        for team in teams_all:
            # Rank (1 = best).  None before team races.
            rank_rel[team].append(ranks_now.get(team))

            # Percentile (100 = best)
            if team in ranks_now:
                r = ranks_now[team]
                pct_val = 100 * (n - r + 1) / n      # 1/10 →100, 5/10 →60
                pct[team].append(pct_val)
            else:
                pct[team].append(None)

            # Absolute rating
            rating_abs[team].append(rating_now.get(team))

    return date_objs, rank_rel, pct, rating_abs

def rolling_rating(
    df: pd.DataFrame,
    window_days: int,
    dropoff: str = "Sudden Decay",
    decay_rate: float | None = None
) -> Tuple[List[datetime], Dict[str, List[float]]]:
    """
    For each date in the season, compute a Massey rating using only
    races within the last `window_days`, with optional drop-off.
    dropoff ∈ {"Sudden Decay", "Linear Decay", "Exponential Decay"}.
    If dropoff=="Exponential Decay", decay_rate ∈ (0,1) controls slope.
    """
    from datetime import timedelta

    # 1) build your sorted list of unique dates
    df_dates = (
        df[["race_id", "date"]].drop_duplicates()
        .assign(date_obj=lambda x: pd.to_datetime(x["date"]))
    )
    dates = sorted(df_dates["date"].unique(),
                   key=lambda d: datetime.strptime(d, "%Y-%m-%d"))
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    # 2) prepare containers
    teams_all = sorted(df["school"].unique())
    rolling_map = {t: [] for t in teams_all}

    # 3) for each date d, pick races in (d - window_days, d]
    for d_str in dates:
        d_obj = datetime.strptime(d_str, "%Y-%m-%d")
        window_start = d_obj - timedelta(days=window_days)

        # slice subset
        sub = df[
            (pd.to_datetime(df["date"]) > window_start) &
            (pd.to_datetime(df["date"]) <= d_obj)
        ]

        # now build weighted pairwise list
        pairs_weighted = []
        for race_id, boats in sub.groupby("race_id"):
            race_date = pd.to_datetime(boats["date"].iloc[0])
            age_days = (d_obj - race_date.to_pydatetime()).days

            # compute weight
            if dropoff == "Sudden Decay":
                w = 1.0
            elif dropoff == "Linear Decay":
                w = max(0.0, 1 - age_days / window_days)
            else:  # Exponential Decay
                w = float(np.exp(-decay_rate * age_days))

            # for each pair in that race, carry weight into its dict
            boats_sorted = boats.sort_values("position")
            rows = list(boats_sorted.itertuples(index=False))
            for i, a in enumerate(rows):
                for b in rows[i+1:]:
                    margin = b.time - a.time
                    pairs_weighted.append({
                        "school_a": a.school,
                        "school_b": b.school,
                        "winner":   a.school,
                        "margin":   f"{margin:.3f}",
                        "weight":   w,
                    })

        # solve weighted Massey
        rating_now = _ratings_weighted(pairs_weighted)
        # append each team’s rating or None
        for t in teams_all:
            rolling_map[t].append(rating_now.get(t))

    return date_objs, rolling_map


def _ratings_weighted(pairs: list[dict]) -> dict[str, float]:
    """Solve Massey with per-pair weights."""
    # very similar to _ratings, but scale each equation by weight
    teams = list({s for r in pairs for s in (r["school_a"], r["school_b"])})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    M = np.zeros((n, n))
    b = np.zeros(n)

    for r in pairs:
        w = float(r["weight"])
        margin = float(r["margin"])
        i, j = idx[r["school_a"]], idx[r["school_b"]]

        # weighted Massey entries
        M[i, i] += w
        M[j, j] += w
        M[i, j] -= w
        M[j, i] -= w

        b[i] += margin * w
        b[j] -= margin * w

    # anchor
    M[-1, :] = 1
    b[-1] = 0

    sol = np.linalg.lstsq(M, b, rcond=None)[0]
    return {team: float(sol[idx[team]]) for team in teams}



# ── dunder exports ----------------------------------------------------
__all__ = ["timeline", "school_colors", "rolling_rating"]
