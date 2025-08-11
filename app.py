import streamlit as st
import pandas as pd
from src.public_predict import predict_public   # ⬅️ use public-data predictor
from src.optimize import optimize_lineup
from src.explain import reason_row
from src.espn_client import get_current_scoring_period
from src.config import settings

st.set_page_config(page_title="FF ML Lineup", layout="wide")
st.title("Fantasy Football ML Lineup (Public Data + ESPN Roster)")

# Default to ESPN’s current scoring period if available, else week 1
default_week = 1
try:
    default_week = max(1, int(get_current_scoring_period()))
except Exception:
    pass

week = st.number_input("Week", min_value=1, max_value=18, value=default_week, step=1)

if st.button("Run"):
    # Auto-trains a public model for your current roster (if missing), then predicts this week
    preds = predict_public(season=settings.season, week=week)  # settings.season should be 2025
    # Optimizer expects an 'uncert' column; derive from band if not present
    if 'uncert' not in preds.columns:
        preds['uncert'] = (preds['ceiling'] - preds['floor']) / 2.0

    # If you want to filter to just your roster, preds already are just your players
    starters, bench = optimize_lineup(preds)

    starters = starters.copy()
    bench = bench.copy()
    starters['why'] = starters.apply(reason_row, axis=1)
    bench['why'] = bench.apply(reason_row, axis=1)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Optimal Starters (risk-adjusted)")
        st.dataframe(starters[['player_name','pos','pred_points','floor','ceiling','why']])
        st.metric("Starter total (expected)", f"{starters['pred_points'].sum():.2f}")
    with c2:
        st.subheader("Bench")
        st.dataframe(bench[['player_name','pos','pred_points','floor','ceiling','why']])

    st.caption("K and D/ST use rolling public baselines; skill positions use a model trained on your players’ public history.")
