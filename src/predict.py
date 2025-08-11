import joblib
from pathlib import Path
from .config import settings
from .build_dataset import build_current_season
from .features import make_predict
from .espn_client import get_current_scoring_period
from .train import model_path_for, train_and_save

def _ensure_trained_model() -> Path:
    train_season = settings.season - 1
    mp = model_path_for(train_season)
    if not mp.exists():
        print(f"[predict] Missing model for {train_season}. Training now…")
        train_and_save(train_season)
    return mp

def predict_week(target_week=None):
    # Ensure we have last season’s model
    mp = _ensure_trained_model()
    model, feats = joblib.load(mp)

    # Build feature rows from *current* season and score this week
    df = build_current_season()
    target_week = target_week or get_current_scoring_period()
    pred_df = make_predict(df, target_week)
    pred_df = pred_df.fillna(0)  # early-season NaNs
    pred_df['pred_points'] = model.predict(pred_df[feats])

    # simple floor/ceiling
    pred_df['uncert'] = (pred_df['pred_points'] - pred_df.get('actual_points_r3_med', 0)).abs()
    pred_df['floor'] = (pred_df['pred_points'] - 0.8 * pred_df['uncert']).clip(lower=0)
    pred_df['ceiling'] = pred_df['pred_points'] + 1.2 * pred_df['uncert']
    Path("data/features").mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(f"data/features/week_{target_week}_preds.csv", index=False)
    return pred_df
