# tools/savant_scrape.py
import time
from pathlib import Path
import requests
import pandas as pd

BASE_URL = "https://nflsavant.com/stats.php?year={year}&week={week}&position=all"
OUT_DIR = Path("data/raw/savant")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ff-ml-lineup/1.0)"
}

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _ensure_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "player": "player_name", "name": "player_name",
        "team": "team", "pos": "position", "position_": "position",
        "opp": "opponent", "opponent_": "opponent",
        "passing_yards": "pass_yds", "pass_yards": "pass_yds",
        "passing_tds": "pass_td", "pass_tds": "pass_td",
        "interceptions": "pass_int",
        "rushing_yards": "rush_yds", "rush_yards": "rush_yds",
        "rushing_tds": "rush_td", "rush_tds": "rush_td",
        "receptions": "rec",
        "receiving_yards": "rec_yds", "rec_yards": "rec_yds",
        "receiving_tds": "rec_td", "rec_tds": "rec_td",
        "fumbles_lost": "fumbles_lost", "fumbles": "fumbles",
        "targets": "targets",
        "passing_attempts": "pass_att", "pass_attempts": "pass_att",
        "rushing_attempts": "rush_att", "rush_attempts": "rush_att",
        "kick_return_tds": "kick_ret_td", "punt_return_tds": "punt_ret_td",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    numeric_cols = [
        "pass_yds","pass_td","pass_int",
        "rush_yds","rush_td",
        "rec","rec_yds","rec_td",
        "fumbles_lost","kick_ret_td","punt_ret_td",
        "targets","pass_att","rush_att"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "player_name" not in df.columns:
        cand = next((c for c in ["player","name","player_"] if c in df.columns), None)
        df["player_name"] = df[cand].astype(str) if cand else ""
    if "team" not in df.columns:
        df["team"] = ""
    if "position" not in df.columns:
        df["position"] = ""
    return df

def _compute_ppr(df: pd.DataFrame) -> pd.DataFrame:
    df["fantasy_ppr"] = (
        df["pass_yds"]/25.0 + df["pass_td"]*4.0 - df["pass_int"]*2.0 +
        df["rush_yds"]/10.0 + df["rush_td"]*6.0 +
        df["rec"]*1.0 + df["rec_yds"]/10.0 + df["rec_td"]*6.0 -
        df["fumbles_lost"]*2.0 + df["kick_ret_td"]*6.0 + df["punt_ret_td"]*6.0
    ).fillna(0.0)
    return df

def _choose_stats_table(dfs):
    """Pick the table that looks like player-week stats."""
    want = {"player", "name"}  # at least one of these must be present pre-clean
    for df in dfs:
        cols = set(c.strip().lower() for c in df.columns)
        if ("player" in cols or "name" in cols) and ("team" in cols or "pos" in cols or "position" in cols):
            return df
    return None

def fetch_nfl_savant_weekly(year: int, week: int, timeout: int = 20) -> pd.DataFrame:
    url = BASE_URL.format(year=year, week=week)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()

    # Try lxml first, then bs4/html5lib as fallback to avoid SoupStrainer weirdness
    dfs = []
    for flavor in (["lxml"], ["bs4", "html5lib"]):
        try:
            dfs = pd.read_html(r.text, flavor=flavor)
            if dfs: break
        except Exception:
            dfs = []
    if not dfs:
        raise ValueError(f"No valid stats table for {year} W{week} (no tables parsed)")

    tbl = _choose_stats_table(dfs)
    if tbl is None:
        raise ValueError(f"No valid stats table for {year} W{week} (no matching columns)")

    df = _clean_columns(tbl)
    df = _ensure_basic_columns(df)
    df["season"] = year
    df["week"] = week
    df = _compute_ppr(df)

    keep_cols = [
        "season","week","player_name","team","position",
        "pass_att","pass_yds","pass_td","pass_int",
        "rush_att","rush_yds","rush_td",
        "targets","rec","rec_yds","rec_td",
        "fumbles_lost","kick_ret_td","punt_ret_td","fantasy_ppr"
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()

def scrape_season(year: int, weeks=range(1,19), sleep_sec: float = 1.0) -> pd.DataFrame:
    frames = []
    for w in weeks:
        out_week = OUT_DIR / f"{year}_week{w}.csv"
        if out_week.exists():
            try:
                dfw = pd.read_csv(out_week)
                frames.append(dfw)
                print(f"[cache] loaded {out_week}")
                continue
            except Exception:
                pass

        try:
            print(f"Fetching {year} week {w} ...")
            dfw = fetch_nfl_savant_weekly(year, w)
            dfw.to_csv(out_week, index=False)
            frames.append(dfw)
            time.sleep(sleep_sec)
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else "?"
            print(f"  -> HTTP {sc} for {year} W{w}; skipping")
            continue
        except ValueError as e:
            print(f"  -> {e}; skipping this week")
            continue
        except Exception as e:
            print(f"  -> Unexpected error {year} W{w}: {e}; skipping")
            continue

    if not frames:
        print(f"WARNING: No data scraped for {year}.")
        return pd.DataFrame()

    season_df = pd.concat(frames, ignore_index=True)
    out_season = OUT_DIR / f"season_{year}.csv"
    season_df.to_csv(out_season, index=False)
    print(f"Saved season CSV → {out_season}  ({len(season_df)} rows)")
    return season_df

if __name__ == "__main__":
    # Scrape full 2024 regular season (weeks 1–18)
    scrape_season(2024, weeks=range(1, 19), sleep_sec=1.0)
