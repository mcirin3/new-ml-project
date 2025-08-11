import os, pandas as pd
from tqdm import tqdm
from .espn_client import get_league

def pull_week(league, week):
    rows = []
    for m in (league.box_scores(week=week) or []):
        for p in (m.home_lineup + m.away_lineup):
            rows.append(dict(
                season=league.year,
                week=week,
                player_id=p.playerId,
                player_name=p.name,
                pos=p.position,
                pro_team=p.proTeam,
                actual_points=getattr(p, "points", None),
                proj_points=getattr(p, "projected_points", None),
                opponent_team_id=(m.away_team.team_id if p in m.home_lineup else m.home_team.team_id),
                is_starter=(p.slot_position not in ("BE", "IR")),
            ))
    return pd.DataFrame(rows)

def build_season(season: int) -> pd.DataFrame:
    os.makedirs("data/raw", exist_ok=True)
    league = get_league(season)
    cur = getattr(league, "scoringPeriodId", 1) or 1
    # If we're building a *past* season, just pull weeks 1..18
    max_week = 18 if season < getattr(league, "year", season) or season < 2025 else cur
    dfs, used = [], []
    for wk in tqdm(range(1, max_week + 1)):
        dfw = pull_week(league, wk)
        if not dfw.empty:
            dfs.append(dfw); used.append(wk)
    if not dfs:
        return pd.DataFrame(columns=[
            "season","week","player_id","player_name","pos","pro_team",
            "actual_points","proj_points","opponent_team_id","is_starter"
        ])
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(f"data/raw/season_{season}.csv", index=False)
    return df

def build_current_season():
    # Convenience for current season pulls (used by predict to get this week’s pool)
    from .config import settings
    return build_season(settings.season)
