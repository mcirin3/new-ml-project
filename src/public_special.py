# src/public_special.py
import pandas as pd
from nfl_data_py import import_weekly_data

DST_LABELS = {"DST","D/ST","DEF"}

def _ensure_fppr(df: pd.DataFrame) -> pd.Series:
    if "fantasy_points_ppr" in df.columns:
        return df["fantasy_points_ppr"].fillna(0.0)
    # conservative fallback
    return pd.Series(0.0, index=df.index)

def kicker_rolling_projection(season: int, week: int, kicker_ids: list[str], window: int = 5):
    """
    Returns a dataframe with columns: player_id, week, pred_points for kickers.
    Uses rolling average of fantasy_points_ppr over last `window` weeks (current season fallback to last season if needed).
    """
    if not kicker_ids:
        return pd.DataFrame(columns=["player_id","week","pred_points"])

    # try current season history up to week-1
    w = import_weekly_data([season])
    cur = w[w["player_id"].isin(kicker_ids)].copy()
    cur["fantasy_ppr"] = _ensure_fppr(cur)
    cur = cur[cur["week"] < week]

    if cur.empty:
        # fallback to last season
        w = import_weekly_data([season-1])
        cur = w[w["player_id"].isin(kicker_ids)].copy()
        cur["fantasy_ppr"] = _ensure_fppr(cur)

    if cur.empty:
        return pd.DataFrame({"player_id": kicker_ids, "week": week, "pred_points": [6.0]*len(kicker_ids)})

    cur.sort_values(["player_id","week"], inplace=True)
    cur["rolling"] = cur.groupby("player_id")["fantasy_ppr"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    last = cur.groupby("player_id")["rolling"].last().reset_index().rename(columns={"rolling":"pred_points"})
    last["week"] = week
    return last[["player_id","week","pred_points"]]

def dst_rolling_projection(season: int, week: int, team_codes: list[str], window: int = 5):
    """
    Team defenses aren’t always in player weekly tables. We approximate by aggregating
    fantasy_points_ppr for 'DEF' entries if present; else fallback to 6.0 pts baseline.
    team_codes: team abbreviations like 'MIN','GB', etc. (use your D/ST team from roster)
    """
    if not team_codes:
        return pd.DataFrame(columns=["team","week","pred_points"])

    w = import_weekly_data([season])
    if "position" in w.columns and "team" in w.columns:
        df = w[(w["position"].isin({"DEF"})) & (w["team"].isin(team_codes))].copy()
        df["fantasy_ppr"] = _ensure_fppr(df)
        df = df[df["week"] < week]
        if df.empty:
            w2 = import_weekly_data([season-1])
            df = w2[(w2["position"].isin({"DEF"})) & (w2["team"].isin(team_codes))].copy()
            df["fantasy_ppr"] = _ensure_fppr(df)
        if not df.empty:
            df.sort_values(["team","week"], inplace=True)
            df["rolling"] = df.groupby("team")["fantasy_ppr"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            last = df.groupby("team")["rolling"].last().reset_index().rename(columns={"rolling":"pred_points"})
            last["week"] = week
            return last[["team","week","pred_points"]]

    # hard fallback
    return pd.DataFrame({"team": team_codes, "week": week, "pred_points": [6.0]*len(team_codes)})
