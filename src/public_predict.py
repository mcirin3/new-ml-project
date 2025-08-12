# src/public_predict.py
from typing import List, Tuple, Dict
import json
import pandas as pd
import requests
from rapidfuzz import process, fuzz

from .espn_client import get_team
from .roster import get_current_roster  # used as an additional name source

POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
TEAM_MAP = {
    0:"FA",1:"ATL",2:"BUF",3:"CHI",4:"CIN",5:"CLE",6:"DAL",7:"DEN",8:"DET",
    9:"GB",10:"TEN",11:"IND",12:"KC",13:"LV",14:"LAR",15:"MIA",16:"MIN",17:"NE",
    18:"NO",19:"NYG",20:"NYJ",21:"PHI",22:"ARI",23:"PIT",24:"LAC",25:"SF",26:"SEA",
    27:"TB",28:"WSH",29:"CAR",30:"JAX",33:"BAL",34:"HOU"
}

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    return (s.lower()
            .replace(" jr.", "").replace(" sr.", "")
            .replace(".", "").replace("'", "")
            .replace("-", " ").replace("  ", " ").strip())

def _espn_players(season: int, week: int, scoring_id: int = 4) -> pd.DataFrame:
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/segments/0/leaguedefaults/{scoring_id}"
        f"?scoringPeriodId={week}&view=kona_player_info"
    )
    xff = {"players": {"limit": 2000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    headers = {"Accept": "application/json", "X-Fantasy-Filter": json.dumps(xff)}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    players = data.get("players", data)

    rows = []
    for p in players:
        info = p.get("player", p)
        pid = info.get("id")
        full = info.get("fullName") or info.get("name") or ""
        pos_id = info.get("defaultPositionId", 0)
        pos = POS_MAP.get(pos_id, "")
        team = TEAM_MAP.get(info.get("proTeamId", 0), "FA")

        proj_week = 0.0
        for s in info.get("stats", []):
            if s.get("seasonId") == season and s.get("statSourceId") == 1:
                at = s.get("appliedTotal")
                if isinstance(at, (int, float)):
                    proj_week = float(at)

        rows.append({
            "player_id": pd.to_numeric(pid, errors="coerce"),
            "player_name": full,
            "position": pos,
            "pro_team": team,
            "proj_week": proj_week,
            "name_key": _norm(full),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = df["player_id"].astype("Int64")
    return df

def _extract_team_roster() -> List[Tuple[int | None, str, str]]:
    """
    Return list of (player_id, name, position) for YOUR team using espn_api,
    with broad attribute fallbacks to handle library differences.
    """
    team = get_team()
    out: List[Tuple[int | None, str, str]] = []

    for p in getattr(team, "roster", []):
        # id fallbacks
        pid = getattr(p, "player_id", None)
        if pid is None: pid = getattr(p, "id", None)
        if pid is None: pid = getattr(p, "playerId", None)

        # name fallbacks
        name = getattr(p, "name", None)
        if not name: name = getattr(p, "playerName", None)
        if not name: name = getattr(p, "fullName", None)
        if not name: name = str(pid) if pid is not None else ""

        # position fallbacks
        pos = getattr(p, "position", None)
        if not pos:
            # sometimes the slot position is available
            pos = getattr(p, "eligibleSlots", None)
            if isinstance(pos, list) and pos:
                # crude best guess: first skill-ish slot
                for code in pos:
                    # QB/RB/WR/TE/K/DST are primary
                    if code in [0,2,4,6,17,16]:  # league slot codes vary; harmless if wrong
                        break
                pos = str(code)
        # ensure text pos
        pos = str(pos).upper()
        # normalize DST variants
        if pos in {"DEF", "DST"}:
            pos = "D/ST"

        try:
            pid_num = int(pid) if pid is not None else None
        except Exception:
            pid_num = None

        out.append((pid_num, name, pos))
    return out

def _fuzzy_ids_from_names(names: List[Tuple[str, str]], pool: pd.DataFrame) -> List[int]:
    """Match roster names to pool by normalized name; ignore position if necessary."""
    if pool.empty:
        return []
    pool = pool.copy()
    pool["name_key"] = pool["name_key"].fillna("").astype(str)
    cand = pool["name_key"].tolist()
    got: List[int] = []
    for nm, pos in names:
        target = _norm(nm)
        if not target:
            continue
        m = process.extractOne(target, cand, scorer=fuzz.WRatio)
        if m and m[1] >= 80:
            got.append(int(pool.iloc[m[2]]["player_id"]))
    return list(dict.fromkeys(got))  # unique preserve order-ish

def predict_public(season: int, week: int) -> pd.DataFrame:
    pool = _espn_players(season=season, week=week, scoring_id=4)

    # 1) Get your roster (IDs + names)
    roster_items = _extract_team_roster()
    # split by pos groups we care about
    skill_ids, k_ids, dst_codes = [], [], []
    names_for_fuzzy: List[Tuple[str, str]] = []

    for pid, name, pos in roster_items:
        if pos in {"QB","RB","WR","TE"}:
            if pid is not None:
                skill_ids.append(pid)
            names_for_fuzzy.append((name, pos))
        elif pos == "K":
            if pid is not None:
                k_ids.append(pid)
            names_for_fuzzy.append((name, pos))
        elif pos in {"D/ST","DST","DEF"}:
            # use team code if we can infer it later from name (fallback in pool by D/ST position + team)
            # we will not rely on pid for DST
            pass
        else:
            # unknown pos? still try to include via name match
            names_for_fuzzy.append((name, pos))

    # 2) If we missed any IDs, backfill by name against the pool
    if pool.empty:
        return pd.DataFrame(columns=["player_id","player_name","pos","pro_team","pred_points","floor","ceiling"])

    if not skill_ids:
        back = _fuzzy_ids_from_names([(n,p) for (n,p) in names_for_fuzzy if p in {"QB","RB","WR","TE"}], pool)
        skill_ids = back or skill_ids
    if not k_ids:
        back = _fuzzy_ids_from_names([(n,"K") for (n,p) in names_for_fuzzy if p == "K"], pool)
        k_ids = back or k_ids

    blocks = []

    # 3) Skill + K by IDs
    need_ids = list(dict.fromkeys([*skill_ids, *k_ids]))
    if need_ids:
        sel = pool[pool["player_id"].isin(need_ids)].copy()
        if not sel.empty:
            block = pd.DataFrame({
                "player_id": sel["player_id"].astype("Int64"),
                "player_name": sel["player_name"],
                "pos": sel["position"],
                "pro_team": sel["pro_team"],
                "pred_points": sel["proj_week"].fillna(0.0),
            })
            blocks.append(block)

    # 4) D/ST — choose the D/ST that matches your team roster’s defense name (by team code in pool)
    # If your roster has a D/ST player object, its team abbrev usually appears in the name.
    # Simpler: just include ALL D/ST that are owned by you? The team library doesn’t expose that cleanly.
    # So we include *any* D/ST that matches your team’s rostered defense name by fuzzy match.
    # Extra safety: if your team doesn’t have a DST in roster_items, skip.
    dst_names = [nm for (_pid, nm, pos) in roster_items if pos in {"D/ST","DST","DEF"}]
    if dst_names:
        dst_pool = pool[pool["position"] == "D/ST"].copy()
        dst_pool["name_key"] = dst_pool["name_key"].fillna("")
        cand = dst_pool["name_key"].tolist()
        for nm in dst_names:
            m = process.extractOne(_norm(nm), cand, scorer=fuzz.WRatio)
            if m and m[1] >= 75:
                row = dst_pool.iloc[m[2]]
                blocks.append(pd.DataFrame({
                    "player_id": [row["player_id"]],
                    "player_name": [row["player_name"]],
                    "pos": ["D/ST"],
                    "pro_team": [row["pro_team"]],
                    "pred_points": [row["proj_week"] if pd.notnull(row["proj_week"]) else 0.0],
                }))

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
        columns=["player_id","player_name","pos","pro_team","pred_points"]
    )
    if out.empty:
        # keep downstream happy
        out["floor"] = pd.Series(dtype=float)
        out["ceiling"] = pd.Series(dtype=float)
        return out

    out = out.drop_duplicates(subset=["player_id","pos"])
    out["floor"] = (out["pred_points"] * 0.80).clip(lower=0)
    out["ceiling"] = out["pred_points"] * 1.25
    return out
