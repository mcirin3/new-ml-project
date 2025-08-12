# src/custom_model.py
from pathlib import Path
from typing import Tuple, Iterable
import json
import numpy as np
import pandas as pd
import requests
import joblib

from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "custom_blend_2024.pkl"

POS_MAP = {1:"QB",2:"RB",3:"WR",4:"TE",5:"K",16:"D/ST"}
TEAM_MAP = {
    0:"FA",1:"ATL",2:"BUF",3:"CHI",4:"CIN",5:"CLE",6:"DAL",7:"DEN",8:"DET",
    9:"GB",10:"TEN",11:"IND",12:"KC",13:"LV",14:"LAR",15:"MIA",16:"MIN",17:"NE",
    18:"NO",19:"NYG",20:"NYJ",21:"PHI",22:"ARI",23:"PIT",24:"LAC",25:"SF",26:"SEA",
    27:"TB",28:"WSH",29:"CAR",30:"JAX",33:"BAL",34:"HOU"
}
POS_KEEP = {"QB","RB","WR","TE"}  # train/predict only skill spots

def _espn_pool(season: int, week: int, scoring_id: int = 4) -> pd.DataFrame:
    """Public ESPN player pool for a scoring period with proj + actual for that week."""
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
        pid = pd.to_numeric(info.get("id"), errors="coerce")
        name = info.get("fullName") or info.get("name") or ""
        pos = POS_MAP.get(info.get("defaultPositionId", 0), "")
        team = TEAM_MAP.get(info.get("proTeamId", 0), "FA")
        proj = None
        actual = None
        for s in info.get("stats", []):
            if s.get("seasonId") != season:
                continue
            if s.get("scoringPeriodId") != week:
                continue
            at = s.get("appliedTotal")
            if at is None and isinstance(s.get("appliedStats"), dict):
                at = sum(v for v in s["appliedStats"].values() if isinstance(v,(int,float)))
            if s.get("statSourceId") == 1 and isinstance(at,(int,float)):  # projection
                proj = float(at)
            if s.get("statSourceId") == 0 and isinstance(at,(int,float)):  # actual
                actual = float(at)
        rows.append({
            "season": season, "week": week,
            "player_id": pid, "player_name": name, "pos": pos, "pro_team": team,
            "espn_proj": proj, "actual": actual
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = df["player_id"].astype("Int64")
    return df

def _rolling_actuals_same_season(season: int, weeks: Iterable[int]) -> pd.DataFrame:
    """Collect actuals for prior weeks in the same season to compute r3/r5."""
    frames = []
    for w in weeks:
        frames.append(_espn_pool(season, w))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df

def build_2024_table(weeks: Iterable[int] = range(1,19)) -> Tuple[pd.DataFrame, list]:
    """Training table for 2024 only (label = actual)."""
    df = _rolling_actuals_same_season(2024, weeks)
    df = df[df["pos"].isin(POS_KEEP)].copy()
    df = df.sort_values(["player_id","week"])
    df["r3"] = df.groupby("player_id")["actual"].shift(1).rolling(3, min_periods=1).mean()
    df["r5"] = df.groupby("player_id")["actual"].shift(1).rolling(5, min_periods=1).mean()
    # League context prior: trailing league avg by position (approximate)
    df["pos_r4_league"] = (
        df.groupby(["pos","week"])["actual"].transform("mean")
          .rolling(4, min_periods=1).mean()
    )
    feat_cols = ["espn_proj","r3","r5","pos_r4_league"]
    for c in feat_cols + ["actual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feat_cols

def train_custom() -> Tuple[Path, dict]:
    df, feats = build_2024_table()
    df = df.dropna(subset=["actual"]).copy()

    gkf = GroupKFold(n_splits=5)
    groups = df["player_id"].fillna(-1)

    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    enet = ElasticNet(alpha=0.15, l1_ratio=0.15, max_iter=5000, random_state=42)

    oof_lgb = np.zeros(len(df))
    oof_en  = np.zeros(len(df))

    for tr, va in gkf.split(df, df["actual"], groups):
        Xtr, Xva = df.iloc[tr][feats], df.iloc[va][feats]
        ytr, yva = df.iloc[tr]["actual"], df.iloc[va]["actual"]

        lgb.fit(Xtr, ytr)
        enet.fit(Xtr.fillna(0), ytr)

        oof_lgb[va] = lgb.predict(Xva)
        oof_en[va]  = enet.predict(Xva.fillna(0))

    # choose blend weight by OOF MAE
    w_grid = np.linspace(0.0, 1.0, 21)
    maes = [mean_absolute_error(df["actual"], w*oof_lgb + (1-w)*oof_en) for w in w_grid]
    best_w = float(w_grid[int(np.argmin(maes))])

    lgb.fit(df[feats], df["actual"])
    enet.fit(df[feats].fillna(0), df["actual"])

    joblib.dump({"lgb": lgb, "enet": enet, "feats": feats, "weight": best_w}, MODEL_PATH)
    return MODEL_PATH, {
        "oof_mae_lgb": float(mean_absolute_error(df["actual"], oof_lgb)),
        "oof_mae_en": float(mean_absolute_error(df["actual"], oof_en)),
        "best_weight": best_w,
        "best_oof_mae": float(min(maes)),
        "n_rows": int(len(df)),
    }

def _features_for_week(season: int, week: int) -> Tuple[pd.DataFrame, list]:
    """
    Build features for arbitrary (season, week) WITHOUT any league calls.
    Uses ESPN public pool and actuals from prior weeks in the same season for rolling stats.
    """
    # current week pool (projections + this-week actuals if already played)
    now = _espn_pool(season, week)
    if now.empty:
        return now, ["espn_proj","r3","r5","pos_r4_league"]

    # prior weeks for rolling
    prior_weeks = [w for w in range(1, week) if w >= 1]
    hist = _rolling_actuals_same_season(season, prior_weeks) if prior_weeks else pd.DataFrame(columns=now.columns)

    # compute rolling on combined (hist + now) but only keep rows for 'now' week
    comb = pd.concat([hist, now], ignore_index=True)
    comb = comb.sort_values(["player_id","week"])

    comb["r3"] = comb.groupby("player_id")["actual"].shift(1).rolling(3, min_periods=1).mean()
    comb["r5"] = comb.groupby("player_id")["actual"].shift(1).rolling(5, min_periods=1).mean()
    comb["pos_r4_league"] = (
        comb.groupby(["pos","week"])["actual"].transform("mean")
            .rolling(4, min_periods=1).mean()
    )

    feats = ["espn_proj","r3","r5","pos_r4_league"]
    for c in feats:
        comb[c] = pd.to_numeric(comb[c], errors="coerce")

    return comb[comb["week"] == week].copy(), feats

def predict_custom(season: int, week: int) -> pd.DataFrame:
    """
    Predict for requested (season, week) using the 2024-trained blend.
    No league/leagueId dependency. Works for 2024+ as long as ESPN public pool is available.
    """
    obj = joblib.load(MODEL_PATH)
    lgb, enet, feats_trained, w = obj["lgb"], obj["enet"], obj["feats"], obj["weight"]

    feat_df, feats_now = _features_for_week(season, week)
    if feat_df.empty:
        return pd.DataFrame(columns=["player_id","player_name","pos","pro_team","pred_points","floor","ceiling"])

    # Align feature names (if future seasons miss some rolling stats early in the year)
    for f in feats_trained:
        if f not in feat_df.columns:
            feat_df[f] = np.nan

    X = feat_df[feats_trained]
    p1 = lgb.predict(X)
    p2 = enet.predict(X.fillna(0))
    pred = w*p1 + (1-w)*p2

    out = feat_df[["player_id","player_name","pos","pro_team"]].copy()
    out["pred_points"] = pred
    out["floor"] = (out["pred_points"] * 0.80).clip(lower=0)
    out["ceiling"] = out["pred_points"] * 1.25
    return out
