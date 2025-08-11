import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from .config import settings
from .build_dataset import build_season
from .features import make_train
from .baseline import BaselineModel  # you already created this

def model_path_for(season: int) -> Path:
    p = Path(f"data/models/lgbm_points_{season}.pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def train_and_save(train_season: int | None = None):
    train_season = train_season or (settings.season - 1)  # last full season
    df = build_season(train_season)
    X, y, feats = make_train(df)

    mp = model_path_for(train_season)

    if X.shape[0] < 50:
        print(f"[train] Not enough samples for {train_season} (n={X.shape[0]}). Saving BaselineModel.")
        joblib.dump((BaselineModel(feats), feats), mp)
        return mp

    model = LGBMRegressor(
        n_estimators=800, learning_rate=0.03,
        max_depth=-1, subsample=0.9, colsample_bytree=0.9
    )
    n_splits = min(5, max(2, X.shape[0] // 100))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, val in cv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[val])
        rmse = np.sqrt(np.mean((preds - y.iloc[val])**2))
        scores.append(rmse)
    print(f"[train] {train_season} CV RMSE: {np.mean(scores):.2f} (splits={n_splits})")
    model.fit(X, y)
    joblib.dump((model, feats), mp)
    return mp

if __name__ == "__main__":
    train_and_save()
