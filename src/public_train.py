# src/public_train.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib
from lightgbm import LGBMRegressor
from rapidfuzz import process, fuzz
from nfl_data_py import import_weekly_data

MODEL_OUT = Path("data/models/public_points_skill.pkl")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

SKILL_POS = {"QB", "RB", "WR", "TE"}     # we ML-train these
SPECIAL_POS = {"K"}                      # handled separately
DST_LABELS = {"DST", "D/ST", "DEF"}      # handled separately (team defenses)

def _normalize_name(s: str) -> str:
    return (
        s.replace(" Jr.", "").replace(" Sr.", "")
         .replace(".", "").replace("'", "")
         .replace("-", " ").strip().lower()
    )

def _ensure_fppr(df: pd.DataFrame) -> pd.Series:
    # Use built-in fantasy points if present, else a conservative PPR calc
    if "fantasy_points_ppr" in df.columns:
        return df["fantasy_points_ppr"].fillna(0.0)
    # fallback (covers skill + K partially)
    return (
        df.get("pass_yds", 0)/25
        + df.get("pass_td", 0)*4 - df.get("pass_int", 0)*2
        + df.get("rush_yds", 0)/10 + df.get("rush_td", 0)*6
        + df.get("rec", 0)*1 + df.get("rec_yds", 0)/10 + df.get("rec_td", 0)*6
        + df.get("fumbles_lost", 0)*-2
        + df.get("kick_ret_td", 0)*6 + df.get("punt_ret_td", 0)*6
    ).fillna(0.0)

def fuzzy_map_names(roster_names, weekly_pool):
    """
    Map (name,pos) -> player_id using WEEKLY data (has player_id reliably).
    """
    pool = weekly_pool[['player_id','player_display_name','position']].dropna().copy()
    pool["norm_name"] = pool["player_display_name"].map(_normalize_name)
    pool = pool.drop_duplicates(subset=["player_id"])

    mapped = {}
    names_by_pos = {pos: pool[pool["position"]==pos] for pos in pool["position"].unique()}
    all_norm = pool["norm_name"].tolist()
    idx_by_norm = dict(zip(pool["norm_name"], pool.index))

    for raw_name, pos in roster_names:
        # skip DST here (handled by team method)
        if pos in DST_LABELS:
            continue
        target = _normalize_name(raw_name)

        # prefer same-position candidates
        sub = names_by_pos.get(pos, pd.DataFrame())
        if not sub.empty:
            cand = sub["norm_name"].tolist()
            m = process.extractOne(target, cand, scorer=fuzz.WRatio)
            if m:
                norm, score, i = m
                pid = sub.iloc[i]["player_id"]
                mapped[raw_name] = {"player_id": pid, "pos": pos, "matched": sub.iloc[i]["player_display_name"], "score": score}
                continue

        # fallback: any position
        m = process.extractOne(target, all_norm, scorer=fuzz.WRatio)
        if m:
            norm, score, _ = m
            r = pool.iloc[idx_by_norm[norm]]
            mapped[raw_name] = {"player_id": r["player_id"], "pos": pos, "matched": r["player_display_name"], "score": score}

    return mapped

def _resolve_cols(df):
    team_col = next((c for c in ["recent_team","team","posteam"] if c in df.columns), None)
    opp_col  = next((c for c in ["opponent_team","opp_team","defteam"] if c in df.columns), None)
    pos_col  = "position" if "position" in df.columns else "pos"
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    return team_col, opp_col, pos_col, name_col

def build_skill_history(roster_names, seasons=(2020, 2024)):
    start, end = seasons
    weekly = import_weekly_data(range(start, end+1))
    team_col, opp_col, pos_col, name_col = _resolve_cols(weekly)

    needed = {"player_id", pos_col, "season", "week"}
    missing = needed - set(weekly.columns)
    if missing:
        raise RuntimeError(f"weekly missing columns: {missing}")

    # Keep only skill positions for ML
    weekly = weekly[weekly[pos_col].isin(SKILL_POS)].copy()

    # mapping uses weekly as pool
    mapping = fuzzy_map_names(roster_names, weekly.rename(columns={pos_col:"position", name_col:"player_display_name"}))
    ids = [v["player_id"] for v in mapping.values() if v["pos"] in SKILL_POS]
    df = weekly[weekly["player_id"].isin(ids)].copy()
    if df.empty:
        return df, mapping

    df["fantasy_ppr"] = _ensure_fppr(df)
    df.sort_values(["player_id","season","week"], inplace=True)

    # rolling features
    df["r3_mean"] = df.groupby("player_id")["fantasy_ppr"].shift(1).rolling(3, min_periods=1).mean()
    df["r5_mean"] = df.groupby("player_id")["fantasy_ppr"].shift(1).rolling(5, min_periods=1).mean()

    # opponent vs. position (last 4) — robust, name-safe (no duplicate 'season' on reset)
    if opp_col is None:
        opp_col = "defteam" if "defteam" in df.columns else None
    if opp_col:
        # Build per-(def,pos,season) week series and roll over week axis
        # 1) average PPR allowed by defense vs position per (season, week)
        base = (
            df.groupby([opp_col, pos_col, "season", "week"])["fantasy_ppr"]
              .mean()
              .rename("fantasy_ppr_allowed")
        ).reset_index()

        # 2) for each (def,pos,season), compute 4-week rolling mean over week
        base = base.sort_values([opp_col, pos_col, "season", "week"])
        base["opp_pos_r4"] = (
            base.groupby([opp_col, pos_col, "season"])["fantasy_ppr_allowed"]
                .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )

        # 3) keep only needed cols and merge back
        tmp = base[[opp_col, pos_col, "season", "week", "opp_pos_r4"]]
        df = df.merge(tmp, on=[opp_col, pos_col, "season", "week"], how="left")
    else:
        df["opp_pos_r4"] = 0.0

    # opportunity (okay if NaN for some)
    df["rush_att_r3"] = df.groupby("player_id")["rush_att"].shift(1).rolling(3, min_periods=1).mean() if "rush_att" in df.columns else 0.0
    df["targets_r3"]  = df.groupby("player_id")["targets"].shift(1).rolling(3, min_periods=1).mean() if "targets"  in df.columns else 0.0
    df["pass_att_r3"] = df.groupby("player_id")["pass_att"].shift(1).rolling(3, min_periods=1).mean() if "pass_att" in df.columns else 0.0

    # normalize to expected names downstream
    df.rename(columns={pos_col:"position"}, inplace=True)
    df["is_qb"] = (df["position"]=="QB").astype(int)
    df["is_rb"] = (df["position"]=="RB").astype(int)
    df["is_wr"] = (df["position"]=="WR").astype(int)
    df["is_te"] = (df["position"]=="TE").astype(int)

    return df, mapping

def train_public(roster_names, seasons=(2020, 2024)):
    df, mapping = build_skill_history(roster_names, seasons)
    feats = ["r3_mean","r5_mean","opp_pos_r4","rush_att_r3","targets_r3","pass_att_r3","is_qb","is_rb","is_wr","is_te"]
    use = df.dropna(subset=["fantasy_ppr"] + feats)
    if use.empty:
        raise RuntimeError("No public history for your current skill players. Check name mapping.")

    n_estimators = 200 if len(use) < 50 else 800
    model = LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05)
    model.fit(use[feats], use["fantasy_ppr"])

    joblib.dump((model, feats, mapping), MODEL_OUT)
    print(f"Saved skill model ({len(use)} rows) → {MODEL_OUT}")
    return MODEL_OUT
