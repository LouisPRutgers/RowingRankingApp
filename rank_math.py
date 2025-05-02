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

_IVY_COLORS = {
    "Brown University":           "#4E3629",
    "Columbia University":        "#9BDDFF",
    "Cornell University":         "#B31B1B",
    "Dartmouth College":          "#00693E",
    "Harvard University":         "#A41034",
    "University of Pennsylvania": "#011F5B",
    "Princeton University":       "#E77500",
    "Yale University":            "#00356B",
}


# ── public helper ─────────────────────────────────────────────────────
def ivy_colors() -> Dict[str, str]:
    """Hard‑coded palette used by both GUIs."""
    return _IVY_COLORS.copy()


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
        for i, a in boats_sorted.iterrows():
            for _, b in boats_sorted.iloc[i + 1 :].iterrows():
                margin = b["time"] - a["time"]
                pairs.append(
                    {
                        "school_a": a["school"],
                        "school_b": b["school"],
                        "winner":   a["school"],      # first is winner
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


# ── dunder exports ----------------------------------------------------
__all__ = ["timeline", "ivy_colors"]
