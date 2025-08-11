def reason_row(r):
    bits = []
    if r['proj_points'] is not None and r['pred_points'] > r['proj_points'] + 1:
        bits.append("model > ESPN proj")
    if r.get('opp_pos_pts_r4', 0) > 10:
        bits.append("good matchup vs position")
    if r.get('actual_points_r3_mean', 0) > 10:
        bits.append("recent usage")
    if r['uncert'] > 4:
        bits.append("boom/bust")
    return ", ".join(bits) or "balanced"
