# src/baseline.py
import numpy as np

class BaselineModel:
    """Fallback model that returns ESPN projection (or rolling means)."""
    def __init__(self, feats): 
        self.feats = feats

    def predict(self, X):
        proj = X.get("proj_points", None)
        if proj is not None:
            return proj.fillna(0).to_numpy(dtype=float)
        cols = [c for c in X.columns if c.endswith("_r3_mean") or c.endswith("_r5_mean")]
        if cols:
            return X[cols].mean(axis=1).fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(X), dtype=float)
