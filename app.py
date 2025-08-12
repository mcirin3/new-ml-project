# app.py
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.public_predict import predict_public
from src.custom_model import predict_custom
from src.optimize import optimize_lineup
from src.explain import reason_row
from src.espn_client import get_current_scoring_period, get_league
from src.roster import get_opponent_for_week, get_opponent_roster, get_my_roster_ids_and_names
from src.config import settings

# ---------- constants / helpers ----------
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
TEAM = {
    0:"FA",1:"ATL",2:"BUF",3:"CHI",4:"CIN",5:"CLE",6:"DAL",7:"DEN",8:"DET",
    9:"GB",10:"TEN",11:"IND",12:"KC",13:"LV",14:"LAR",15:"MIA",16:"MIN",17:"NE",
    18:"NO",19:"NYG",20:"NYJ",21:"PHI",22:"ARI",23:"PIT",24:"LAC",25:"SF",26:"SEA",
    27:"TB",28:"WSH",29:"CAR",30:"JAX",33:"BAL",34:"HOU"
}

def normalize_name_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.lower()
         .str.replace(r"\b(jr|sr)\.?$", "", regex=True)
         .str.replace(r"[^\w\s]", "", regex=True)
         .str.replace("-", " ")
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

@st.cache_data
def load_2024_totals_csv(path: str = "data/raw/espn/espn_players_2024_scoring4.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # expected: player_id, player_name, team, position, season, fantasy_total
    df = df.rename(columns={"team": "pro_team"})
    df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce").astype("Int64")
    df = df[["player_id", "player_name", "pro_team", "position", "fantasy_total"]]
    df = df.rename(columns={"fantasy_total": "total_2024"})
    return df

@st.cache_data
def fetch_ytd_totals(season: int, week: int, scoring_id: int = 4) -> pd.DataFrame:
    base = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{season}/segments/0/leaguedefaults/{scoring_id}"
        f"?scoringPeriodId={week}&view=kona_player_info"
    )
    xff = {"players": {"limit": 2000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    headers = {"Accept": "application/json", "X-Fantasy-Filter": json.dumps(xff)}
    r = requests.get(base, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    players = data.get("players", data)

    def extract_ytd(pobj) -> float:
        best = None
        for s in pobj.get("stats", []):
            if s.get("seasonId") != season or s.get("statSourceId") != 0:
                continue
            total = s.get("appliedTotal")
            if total is None and isinstance(s.get("appliedStats"), dict):
                total = sum(v for v in s["appliedStats"].values() if isinstance(v, (int, float)))
            if total is None:
                continue
            split = s.get("statSplitTypeId")
            score = (2 if split == 2 else 1, float(total))
            if best is None or score > best[0]:
                best = (score, total)
        return 0.0 if best is None else best[1]

    rows = []
    for p in players:
        info = p.get("player", p)
        rows.append({
            "player_id": info.get("id"),
            "player_name": info.get("fullName") or info.get("name") or "",
            "ytd_total": extract_ytd(info),
            "position_id": info.get("defaultPositionId", 0),
        })
    df = pd.DataFrame(rows)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["pos_from_api"] = df["position_id"].map(POS).fillna("")
    return df[["player_id", "player_name", "ytd_total", "pos_from_api"]]

@st.cache_data
def fetch_player_pool(season: int, week: int, scoring_id: int = 4) -> pd.DataFrame:
    """ESPN public pool for week: player_id, player_name, pos, pro_team, pred_points."""
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
        pid = pd.to_numeric(info.get("id"), errors="coerce")
        full = info.get("fullName") or info.get("name") or ""
        pos = POS.get(info.get("defaultPositionId", 0), "")
        team = TEAM.get(info.get("proTeamId", 0), "FA")
        proj_week = 0.0
        for s in info.get("stats", []):
            if s.get("seasonId") == season and s.get("statSourceId") == 1:
                at = s.get("appliedTotal")
                if isinstance(at, (int, float)):
                    proj_week = float(at)
        rows.append({
            "player_id": pid,
            "player_name": full,
            "pos": pos,
            "pro_team": team,
            "pred_points": proj_week,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = df["player_id"].astype("Int64")
    return df

def ensure_proj_column(df: pd.DataFrame) -> pd.DataFrame:
    """explain.py expects 'proj_points'; create it from pred_points if missing."""
    if "proj_points" not in df.columns and "pred_points" in df.columns:
        df["proj_points"] = df["pred_points"]
    return df


# ---------- UI ----------
st.set_page_config(page_title="FF ML Lineup", layout="wide")
st.title("Fantasy Football ML Lineup (Public Data + ESPN Roster)")

source = st.radio("Projection source", ["ESPN (public)", "Custom ML"], horizontal=True)

default_week = 1
try:
    default_week = max(1, int(get_current_scoring_period()))
except Exception:
    pass

week = st.number_input("Week", min_value=1, max_value=18, value=default_week, step=1)

# Optional: clear week-scoped caches when the week changes
if "last_week" not in st.session_state:
    st.session_state.last_week = week
if week != st.session_state.last_week:
    st.cache_data.clear()
    st.session_state.last_week = week

if st.button("Run"):
    # 1) Predictions for your roster (choose source)
    if source == "ESPN (public)":
        preds = predict_public(season=settings.season, week=week)   # already only your roster
    else:
        preds = predict_custom(season=settings.season, week=week)   # returns pool
        # ↓ Filter to your roster (by IDs first, then name)
        my_ids, my_names = get_my_roster_ids_and_names(week)

        def _norm_series(s: pd.Series) -> pd.Series:
            return (s.astype(str).str.lower()
                    .str.replace(r"\b(jr|sr)\.?\b", "", regex=True)
                    .str.replace(r"[^\w\s]", "", regex=True)
                    .str.replace("-", " ")
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip())

        preds["name_key"] = _norm_series(preds["player_name"])
        preds = preds[
            preds["player_id"].isin(pd.Series(list(my_ids), dtype="Int64")) |
            preds["name_key"].isin(my_names)
        ].copy()

    # 2) Priors (2024 totals + YTD totals)
    tot24 = load_2024_totals_csv()
    ytd = fetch_ytd_totals(settings.season, week, scoring_id=4)

    # Normalize for name-based merge (IDs may differ)
    if "player_name" not in preds.columns:
        st.error("Predictions missing 'player_name' column.")
        st.stop()

    preds["name_key"] = normalize_name_series(preds["player_name"])
    if not tot24.empty and "player_name" in tot24.columns:
        tot24["name_key"] = normalize_name_series(tot24["player_name"])
    if not ytd.empty and "player_name" in ytd.columns:
        ytd["name_key"] = normalize_name_series(ytd["player_name"])

    # Merge by normalized name keys (preserve preds' player_name)
    if not tot24.empty:
        preds = preds.merge(
            tot24.drop(columns=[c for c in ["player_name"] if c in tot24.columns]),
            on="name_key", how="left"
        )
    if not ytd.empty:
        preds = preds.merge(
            ytd.drop(columns=[c for c in ["player_name"] if c in ytd.columns]),
            on="name_key", how="left"
        )

    # Derived priors
    if "total_2024" in preds.columns:
        preds["prior_pg_2024"] = (preds["total_2024"] / 17.0).round(2)
    else:
        preds["prior_pg_2024"] = None
    preds["ytd_total"] = preds.get("ytd_total", pd.Series([0]*len(preds))).fillna(0).round(2)

    # Uncertainty fallback
    if "uncert" not in preds.columns and {"ceiling", "floor"} <= set(preds.columns):
        preds["uncert"] = (preds["ceiling"] - preds["floor"]) / 2.0
    preds["uncert"] = preds.get("uncert", pd.Series([3.0]*len(preds)))  # fallback

    preds = ensure_proj_column(preds)

    # 3) Optimize your lineup
    starters, bench = optimize_lineup(preds)

    # 4) Opponent: week-accurate via HTTP, render pretty team name only
    opp_info = get_opponent_for_week(week)
    opp_starters = pd.DataFrame()
    opp_bench = pd.DataFrame()

    # Resolve opponent team name (only team name or location+nickname)
    opp_name = "Opponent not found"
    if opp_info:
        opp_id, fallback_name = opp_info
        lg = get_league()  # fresh league object
        opp_team_obj = next((t for t in lg.teams if int(getattr(t, "team_id", -1)) == int(opp_id)), None)
        if opp_team_obj:
            tn = getattr(opp_team_obj, "team_name", None)
            loc = getattr(opp_team_obj, "location", None)
            nick = getattr(opp_team_obj, "nickname", None)
            pretty = tn or (" ".join(x for x in [loc, nick] if x) if (loc or nick) else fallback_name)
            opp_name = pretty or fallback_name
        else:
            opp_name = fallback_name

    st.caption(f"Week {week}: opponent → {opp_name}")

    if opp_info:
        opp_roster = get_opponent_roster(week)

        # Build opponent names
        opp_names = []
        for p in opp_roster:
            nm = getattr(p, "name", getattr(p, "playerName", getattr(p, "fullName", "")))
            if nm:
                opp_names.append(nm)

        # Pull ESPN public pool and select opponent players by name
        pool = fetch_player_pool(settings.season, week, scoring_id=4)
        if not pool.empty and opp_names:
            pool["name_key"] = normalize_name_series(pool["player_name"])
            sel = pool[pool["name_key"].isin(normalize_name_series(pd.Series(opp_names)))]
            opp_preds = sel.copy()

            # uncertainty + compat column
            opp_preds["uncert"] = (opp_preds["pred_points"] * 0.2).fillna(3.0)
            opp_preds["floor"] = (opp_preds["pred_points"] * 0.8).clip(lower=0)
            opp_preds["ceiling"] = (opp_preds["pred_points"] * 1.25)
            opp_preds = ensure_proj_column(opp_preds)

            # Optimize opponent lineup
            opp_starters, opp_bench = optimize_lineup(opp_preds)

    # 5) UI — Your roster summary (with priors)
    with st.expander("📊 Your players — 2024 totals + YTD", expanded=True):
        cols = ["player_name", "pos", "pro_team", "total_2024", "ytd_total"]
        have = [c for c in cols if c in preds.columns]
        tbl = preds[have].copy()
        if "pos" in tbl.columns:
            tbl["pos_sort"] = tbl["pos"].map({"QB":0,"RB":1,"WR":2,"TE":3,"K":4,"D/ST":5}).fillna(9)
            sort_cols = ["pos_sort"]
        else:
            sort_cols = []
        if "total_2024" in tbl.columns:
            sort_cols.append("total_2024")
        if sort_cols:
            tbl = tbl.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True)
        if "pos_sort" in tbl.columns:
            tbl = tbl.drop(columns=["pos_sort"])
        st.dataframe(tbl.reset_index(drop=True))
        if "total_2024" in preds.columns and preds["total_2024"].isna().any():
            missing = int(preds["total_2024"].isna().sum())
            st.warning(f"{missing} player(s) missing 2024 totals (no ESPN match). "
                       "Confirm data/raw/espn/espn_players_2024_scoring4.csv exists.")

    # 6) UI — Your optimized lineup
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Your Optimal Starters (risk-adjusted)")
        cols = [c for c in ["player_name","pos","pred_points","floor","ceiling","prior_pg_2024","ytd_total"] if c in starters.columns]
        st.dataframe(starters[cols])
        st.metric("Your Starter Total (expected)", f"{starters['pred_points'].sum():.2f}")

    with c2:
        st.subheader("Your Bench")
        cols = [c for c in ["player_name","pos","pred_points","floor","ceiling","prior_pg_2024","ytd_total"] if c in bench.columns]
        st.dataframe(bench[cols])

    # 7) UI — Head-to-Head (opponent)
    st.subheader(f"🏈 Head-to-Head — Week {week}: {opp_name}")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**Opponent Starters**")
        if not opp_starters.empty:
            cols = [c for c in ["player_name","pos","pred_points","floor","ceiling"] if c in opp_starters.columns]
            st.dataframe(opp_starters[cols])
            st.metric("Opponent Starter Total (expected)", f"{opp_starters['pred_points'].sum():.2f}")
        else:
            st.info("No opponent data available for this week (or opponent not found).")

    with oc2:
        st.markdown("**Opponent Bench**")
        if not opp_bench.empty:
            cols = [c for c in ["player_name","pos","pred_points"] if c in opp_bench.columns]
            st.dataframe(opp_bench[cols])
        else:
            st.write("—")

    # 8) Optional explanation strings for your starters
    try:
        starters_disp = starters.copy()
        starters_disp["why"] = starters_disp.apply(reason_row, axis=1)
        with st.expander("🧠 Why these starters?"):
            cols = [c for c in ["player_name","pos","pred_points","floor","ceiling","why"] if c in starters_disp.columns]
            st.dataframe(starters_disp[cols])
    except Exception:
        pass

    st.caption(
        "Notes: Predictions use ESPN public projections (PPR) or your Custom ML (if selected). "
        "Head-to-head pulls opponent roster and optimizes their lineup with the same rules."
    )
