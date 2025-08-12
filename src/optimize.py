# src/optimize.py
import pulp
import pandas as pd

# Your league’s lineup
SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,   # RB/WR/TE eligible
    "D/ST": 1,
    "K": 1,
}

FLEX_ELIGIBLE = {"RB", "WR", "TE"}

def _eligible_slots(pos: str):
    if pos == "QB": return {"QB"}
    if pos == "RB": return {"RB", "FLEX"}
    if pos == "WR": return {"WR", "FLEX"}
    if pos == "TE": return {"TE", "FLEX"}
    if pos in {"D/ST", "DST", "DEF"}: return {"D/ST"}
    if pos == "K": return {"K"}
    return set()  # unknown positions -> not eligible

def optimize_lineup(pred_df: pd.DataFrame):
    df = pred_df.copy()

    # safety fills
    df["pred_points"] = pd.to_numeric(df.get("pred_points", 0), errors="coerce").fillna(0.0)
    df["uncert"] = pd.to_numeric(df.get("uncert", 3.0), errors="coerce").fillna(3.0)
    df["pos"] = df["pos"].replace({"DST": "D/ST", "DEF": "D/ST"})

    # build eligibility map
    elig_map = {i: _eligible_slots(pos) for i, pos in df["pos"].items()}

    # prune players with no eligible slots
    valid_idx = [i for i, slots in elig_map.items() if slots]
    df = df.loc[valid_idx].copy()
    elig_map = {i: elig_map[i] for i in valid_idx}

    # decision vars: y[i, s] = 1 if player i is assigned to slot s
    prob = pulp.LpProblem("FF_Lineup", pulp.LpMaximize)
    y = {
        (i, s): pulp.LpVariable(f"y_{i}_{s}", lowBound=0, upBound=1, cat="Binary")
        for i, slots in elig_map.items() for s in slots
    }

    # objective: risk-adjusted projection
    adj = {i: (df.at[i, "pred_points"] - 0.15 * df.at[i, "uncert"]) for i in df.index}
    prob += pulp.lpSum(adj[i] * y[i, s] for (i, s) in y.keys())

    # slot fill constraints (exactly count for each slot)
    for s, cnt in SLOTS.items():
        eligible_players = [i for i in df.index if s in elig_map[i]]
        prob += pulp.lpSum(y[i, s] for i in eligible_players) == cnt, f"fill_{s}"

    # each player at most one slot
    for i in df.index:
        prob += pulp.lpSum(y[i, s] for s in elig_map[i]) <= 1, f"one_slot_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # extract starters with the slot they occupy
    assigned = []
    for (i, s), var in y.items():
        if var.value() == 1:
            row = df.loc[i].copy()
            row["slot"] = s
            assigned.append(row)

    starters = pd.DataFrame(assigned)
    # nice ordering of starters
    order = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "D/ST", "K"]
    starters["slot_order"] = starters["slot"].apply(lambda x: order.index(x) if x in order else 999)
    starters = starters.sort_values(["slot_order", "pred_points"], ascending=[True, False])
    starters = starters.drop(columns=["slot_order"])

    # bench = everyone not assigned
    started_idx = set(starters.index)
    bench = pred_df.loc[~pred_df.index.isin(started_idx)].copy()
    bench = bench.sort_values("pred_points", ascending=False)

    return starters, bench
