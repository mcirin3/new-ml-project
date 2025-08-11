# src/public_predict.py
from pathlib import Path
import pandas as pd
import joblib

from nfl_data_py import import_weekly_data
from rapidfuzz import process, fuzz

from .roster import get_current_roster
from .public_train import MODEL_OUT, _ensure_fppr, SKILL_POS, train_public
from .public_special import kicker_rolling_projection, dst_rolling_projection

from urllib.error import HTTPError
from nfl_data_py import import_weekly_data

MAX_PUBLIC_YEAR = 2024  # bump this once nflfastR publishes 2025 weekly

def import_weekly_safe(years):
    """Import weekly data, skipping years that 404 (not published yet)."""
    ok_years = []
    for y in years:
        if y <= MAX_PUBLIC_YEAR:
            ok_years.append(y)
    if not ok_years:
        raise RuntimeError(f"No public weekly data available for years={years}. Try <= {MAX_PUBLIC_YEAR}.")
    try:
        return import_weekly_data(ok_years)
    except HTTPError as e:
        # Extremely defensive; if a year inside ok_years still 404s, drop it one-by-one
        agg = []
        for y in ok_years:
            try:
                agg.append(import_weekly_data([y]))
            except HTTPError:
                pass
        if not agg:
            raise
        return pd.concat(agg, ignore_index=True)
def _resolve_cols(df):
    """Return (team_col, opp_col, pos_col, name_col) that exist in this dataframe."""
    team_col = next((c for c in ["recent_team", "team", "posteam"] if c in df.columns), None)
    opp_col  = next((c for c in ["opponent_team", "opp_team", "defteam"] if c in df.columns), None)
    pos_col  = "position" if "position" in df.columns else ("pos" if "pos" in df.columns else None)
    name_col = "player_display_name" if "player_display_name" in df.columns else (
               "player_name" if "player_name" in df.columns else None)
    return team_col, opp_col, pos_col, name_col


def _build_skill_feature_rows(weekly: pd.DataFrame, player_ids, week: int) -> pd.DataFrame:
    """Assemble rolling features for the target week for the given player_ids."""
    team_col, opp_col, pos_col, name_col = _resolve_cols(weekly)

    df = weekly[weekly["player_id"].isin(player_ids)].copy()
    if df.empty:
        return pd.DataFrame(columns=["player_id","player_display_name","position","week"])

    # Normalize required columns
    if pos_col and pos_col != "position":
        df.rename(columns={pos_col: "position"}, inplace=True)
    if name_col and name_col != "player_display_name":
        df.rename(columns={name_col: "player_display_name"}, inplace=True)

    df["fantasy_ppr"] = _ensure_fppr(df)
    df.sort_values(["player_id", "week"], inplace=True)

    # rolling recent form (no leakage)
    df["r3_mean"] = df.groupby("player_id")["fantasy_ppr"].shift(1).rolling(3, min_periods=1).mean()
    df["r5_mean"] = df.groupby("player_id")["fantasy_ppr"].shift(1).rolling(5, min_periods=1).mean()

    # opponent vs position last 4 weeks
    if opp_col is None:
        opp_col = "defteam" if "defteam" in df.columns else None

    if opp_col and "position" in df.columns:
        tmp = (
            df.groupby([opp_col, "position", "week"])["fantasy_ppr"].mean()
              .groupby(level=[0, 1]).rolling(4, min_periods=1).mean()
              .reset_index().rename(columns={"fantasy_ppr": "opp_pos_r4"})
        )
        df = df.merge(tmp, left_on=[opp_col, "position", "week"],
                          right_on=[opp_col, "position", "week"], how="left")
    else:
        df["opp_pos_r4"] = 0.0

    # opportunity (ok if absent)
    if "rush_att" in df.columns:
        df["rush_att_r3"] = df.groupby("player_id")["rush_att"].shift(1).rolling(3, min_periods=1).mean()
    else:
        df["rush_att_r3"] = 0.0
    if "targets" in df.columns:
        df["targets_r3"]  = df.groupby("player_id")["targets"].shift(1).rolling(3, min_periods=1).mean()
    else:
        df["targets_r3"] = 0.0
    if "pass_att" in df.columns:
        df["pass_att_r3"] = df.groupby("player_id")["pass_att"].shift(1).rolling(3, min_periods=1).mean()
    else:
        df["pass_att_r3"] = 0.0

    # take the target week rows
    feat_week = df[df["week"] == week].copy()
    return feat_week.fillna(0)


def predict_public(season: int, week: int) -> pd.DataFrame:
    """
    Predict your current roster’s points for (season, week) using:
      - ML model trained on public data for skill positions (QB/RB/WR/TE)
      - Rolling baselines for K and D/ST
    Returns columns: player_id, player_name, pos, pred_points, floor, ceiling
    """
    roster = get_current_roster()

    # Ensure skill model exists; train on first run (2020-2024 history of your players)
    if not Path(MODEL_OUT).exists():
        train_public(roster["skill"], seasons=(2020, 2024))

    model, feats, mapping = joblib.load(MODEL_OUT)
    skill_ids = [v["player_id"] for v in mapping.values()]

    # Current season weekly data (schema-flexible)
    weekly = import_weekly_safe([season])

    # ==== Skill positions (ML) ====
    skill_feat = _build_skill_feature_rows(weekly, skill_ids, week)
    skill_preds = pd.DataFrame(columns=["player_id","player_name","pos","pred_points"])
    if not skill_feat.empty:
        # Only keep skill positions we trained on
        skill_feat = skill_feat[skill_feat["position"].isin(SKILL_POS)]
        if not skill_feat.empty:
            preds = model.predict(skill_feat[feats])
            names = skill_feat.get("player_display_name", skill_feat.get("player_name", skill_feat["player_id"]))
            skill_preds = pd.DataFrame({
                "player_id": skill_feat["player_id"].values,
                "player_name": names.values,
                "pos": skill_feat["position"].values,
                "pred_points": preds,
            })

    # ==== Kickers (rolling baseline) ====
    k_names = [n for (n, p) in roster["kickers"]]
    k_block = pd.DataFrame(columns=["player_id","player_name","pos","pred_points"])
    if k_names:
        # Build name->id mapping from weekly (position K)
        team_col, opp_col, pos_col, name_col = _resolve_cols(weekly)
        kpool = weekly[weekly.get(pos_col or "position", "position") == "K"] if pos_col else weekly[weekly["position"]=="K"]
        if not kpool.empty:
            kp = kpool[["player_id", name_col or "player_display_name"]].dropna().drop_duplicates("player_id")
            disp_col = name_col or "player_display_name"
            names_list = kp[disp_col].tolist()
            kid_map = {}
            for n in k_names:
                m = process.extractOne(n, names_list, scorer=fuzz.WRatio)
                if m:
                    matched, _, _ = m
                    pid = kp.loc[kp[disp_col] == matched, "player_id"].iloc[0]
                    kid_map[n] = pid
            k_ids = list(kid_map.values())
            if k_ids:
                kdf = kicker_rolling_projection(season, week, k_ids)
                if not kdf.empty:
                    kmerge = kdf.merge(kp, on="player_id", how="left")
                    kmerge.rename(columns={disp_col: "player_name"}, inplace=True)
                    kmerge["pos"] = "K"
                    k_block = kmerge[["player_id","player_name","pos","pred_points"]]

    # ==== D/ST (rolling baseline by team code) ====
    dst_codes = roster["dst"]  # e.g., ["MIN"]
    dst_block = pd.DataFrame(columns=["player_id","player_name","pos","pred_points"])
    if dst_codes:
        ddf = dst_rolling_projection(season, week, dst_codes)
        if not ddf.empty:
            ddf = ddf.rename(columns={"team": "player_name"})
            ddf["pos"] = "D/ST"
            ddf["player_id"] = ddf["player_name"]
            dst_block = ddf[["player_id","player_name","pos","pred_points"]]

    # ==== Combine and add bands ====
    out = pd.concat([skill_preds, k_block, dst_block], ignore_index=True)
    out.drop_duplicates(subset=["player_id","pos"], inplace=True)

    # Simple uncertainty band
    out["floor"] = (out["pred_points"] * 0.8).clip(lower=0)
    out["ceiling"] = out["pred_points"] * 1.25

    return out
