import pulp
import pandas as pd

# Adjust these to your league’s lineup settings
SLOTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,   # RB/WR/TE
}
BENCH_SLOTS = 6

def optimize_lineup(pred_df: pd.DataFrame):
    # only your team’s roster for the week (pred_df should already be filtered if needed)
    players = pred_df.copy()
    players['eligible'] = players['pos'].apply(lambda p: ['QB'] if p=='QB' else (['RB','FLEX'] if p=='RB'
                                            else (['WR','FLEX'] if p=='WR' else (['TE','FLEX'] if p=='TE' else []))))
    # Decision vars: x_i = 1 if started
    prob = pulp.LpProblem("Lineup", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f'x_{i}', lowBound=0, upBound=1, cat='Binary') for i in players.index}

    # Objective: expected points – small penalty for uncertainty (risk-adjusted)
    obj = pulp.lpSum( (players.loc[i,'pred_points'] - 0.15*players.loc[i,'uncert']) * x[i] for i in players.index )
    prob += obj

    # Slot constraints
    for slot, count in SLOTS.items():
        if slot == 'FLEX':
            eligible_idx = [i for i in players.index if 'FLEX' in players.loc[i,'eligible']]
        else:
            eligible_idx = [i for i in players.index if slot in players.loc[i,'eligible']]
        prob += pulp.lpSum(x[i] for i in eligible_idx) == count

    # Each player either started (1) or benched (0), no more than roster size
    # (If you want to force exactly starters + bench, add a bench constraint too.)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    players['start'] = [int(x[i].value()) for i in players.index]
    starters = players[players.start==1].sort_values('pred_points', ascending=False)
    bench = players[players.start==0].sort_values('pred_points', ascending=False)
    return starters, bench
