# src/custom_data.py
import json
from typing import Iterable, Tuple
import pandas as pd
import requests

POS_KEEP = {"QB","RB","WR","TE"}  # train only skill spots

def _espn_week_pool(season: int, week: int, scoring_id: int = 4) -> pd.DataFrame:
    url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
           f"seasons/{season}/segments/0/leaguedefaults/{scoring_id}"
           f"?scoringPeriodId={week}&view=kona_player_info")
    xff = {"players": {"limit": 2000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    r = requests.get(url, headers={"Accept":"application/json","X-Fantasy-Filter": json.dumps(xff)}, timeout=30)
    r.raise_for_status()
    players = r.json().get("players", [])
    rows = []
    for p in players:
        info = p.get("player", p)
        pid = info.get("id")
        name = info.get("fullName") or info.get("name") or ""
        pos_id = info.get("defaultPositionId", 0)
        pos_map = {1:"QB",2:"RB",3:"WR",4:"TE",5:"K",16:"D/ST"}
        pos = pos_map.get(pos_id, "")
        team_map = {
            0:"FA",1:"ATL",2:"BUF",3:"CHI",4:"CIN",5:"CLE",6:"DAL",7:"DEN",8:"DET",
            9:"GB",10:"TEN",11:"IND",12:"KC",13:"LV",14:"LAR",15:"MIA",16:"MIN",17:"NE",
            18:"NO",19:"NYG",20:"NYJ",21:"PHI",22:"ARI",23:"PIT",24:"LAC",25:"SF",26:"SEA",
            27:"TB",28:"WSH",29:"CAR",30:"JAX",33:"BAL",34:"HOU"
        }
        team = team_map.get(info.get("proTeamId", 0), "FA")

        proj_week = None
        actual_week = None
        for s in info.get("stats", []):
            if s.get("seasonId") != season: 
                continue
            # ESPN projection for THIS week (statSourceId 1)
            if s.get("statSourceId") == 1 and s.get("scoringPeriodId") == week:
                at = s.get("appliedTotal")
                if isinstance(at, (int, float)):
                    proj_week = float(at)
            # ESPN actuals for THIS week (statSourceId 0)
            if s.get("statSourceId") == 0 and s.get("scoringPeriodId") == week:
                at = s.get("appliedTotal")
                if at is None and isinstance(s.get("appliedStats"), dict):
                    at = sum(v for v in s["appliedStats"].values() if isinstance(v,(int,float)))
                if isinstance(at, (int, float)):
                    actual_week = float(at)

        rows.append({
            "season": season, "week": week,
            "player_id": pd.to_numeric(pid, errors="coerce"),
            "player_name": name, "pos": pos, "pro_team": team,
            "espn_proj": proj_week, "actual": actual_week
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = df["player_id"].astype("Int64")
    return df

def build_2024_table(weeks: Iterable[int] = range(1, 19)) -> pd.DataFrame:
    """One row per (player, week) with features + label=actual."""
    frames = []
    for w in weeks:
        frames.append(_espn_week_pool(2024, w))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pos"].isin(POS_KEEP)].copy()

    # Rolling features within the same season per player (no leakage)
    df = df.sort_values(["player_id","week"])
    df["r3"] = df.groupby("player_id")["actual"].shift(1).rolling(3, min_periods=1).mean()
    df["r5"] = df.groupby("player_id")["actual"].shift(1).rolling(5, min_periods=1).mean()

    # Opponent-vs-position (use team vs def is hard with this endpoint; use public proxy:
    # compute rolling league-average by position in trailing 4 weeks as a context prior)
    df["pos_r4_league"] = (
        df.groupby(["pos","week"])["actual"].transform("mean")
          .rolling(4, min_periods=1).mean()
    )

    # Feature matrix and label
    # We’ll predict `actual` using simple signals available everywhere.
    feat_cols = ["espn_proj","r3","r5","pos_r4_league"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df, feat_cols
