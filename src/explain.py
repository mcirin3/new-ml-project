# src/explain.py
def reason_row(r):
    # accept either 'proj_points' or 'pred_points'
    proj = r.get("proj_points", r.get("pred_points", None))
    bits = []
    if proj is not None and r.get("pred_points") is not None and r["pred_points"] > proj + 1:
        bits.append("model > ESPN proj")
    if r.get("opp_pos_pts_r4", 0) > 10:
        bits.append("good matchup vs position")
    if r.get("actual_points_r3_mean", 0) > 10:
        bits.append("recent usage")
    if r.get("uncert", 0) > 4:
        bits.append("boom/bust")
    return ", ".join(bits) or "balanced"
