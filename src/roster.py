# src/roster.py
from .espn_client import get_league

def get_current_roster():
    """
    Returns dict with:
      skill: list[(name, pos)]
      kickers: list[(name, 'K')]
      dst: list[team_code]
    """
    lg = get_league()
    my_team = next((t for t in lg.teams if t.owners), lg.teams[0])
    skill, kickers, dst = [], [], []

    for p in my_team.roster:
        slot = getattr(p, "slot_position", None) or getattr(p, "lineupSlot", None)
        if hasattr(slot, "name"): slot = slot.name
        # include bench too if you want training volume: remove this if for full roster
        # if slot in ("BE","IR"): continue

        pos = p.position
        if pos in {"QB","RB","WR","TE"}:
            skill.append((p.name, pos))
        elif pos == "K":
            kickers.append((p.name, "K"))
        elif pos in {"DST","D/ST","DEF"}:
            # espn_api Player for D/ST usually has name like "Vikings D/ST" and proTeam like "MIN"
            team = getattr(p, "proTeam", None) or getattr(p, "pro_team", None)
            if isinstance(team, str):
                dst.append(team)
    return {"skill": skill, "kickers": kickers, "dst": list(dict.fromkeys(dst))}
