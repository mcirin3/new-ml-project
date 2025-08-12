# tools/espn_players_2024.py (patched)
import json, requests, pandas as pd
from pathlib import Path

SEASON = 2024
SCORING_ID = 4   # 1=Standard, 3=Half-PPR, 4=PPR
BASE = (
  "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
  f"seasons/{SEASON}/segments/0/leaguedefaults/{SCORING_ID}"
  "?scoringPeriodId=0&view=kona_player_info"
)

OUT = Path("data/raw/espn"); OUT.mkdir(parents=True, exist_ok=True)
POS = {1:"QB",2:"RB",3:"WR",4:"TE",5:"K",16:"DST"}
TEAM = {0:"FA",1:"ATL",2:"BUF",3:"CHI",4:"CIN",5:"CLE",6:"DAL",7:"DEN",8:"DET",
        9:"GB",10:"TEN",11:"IND",12:"KC",13:"LV",14:"LAR",15:"MIA",16:"MIN",17:"NE",
        18:"NO",19:"NYG",20:"NYJ",21:"PHI",22:"ARI",23:"PIT",24:"LAC",25:"SF",26:"SEA",
        27:"TB",28:"WSH",29:"CAR",30:"JAX",33:"BAL",34:"HOU"}

def fetch_all():
    xff = {"players": {"limit": 2000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    headers = {"Accept":"application/json", "X-Fantasy-Filter": json.dumps(xff)}
    r = requests.get(BASE, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_season_total(player_obj, season):
    best = None
    for s in player_obj.get("stats", []):
        if s.get("seasonId") != season or s.get("statSourceId") != 0:
            continue
        total = s.get("appliedTotal")
        if total is None and isinstance(s.get("appliedStats"), dict):
            total = sum(v for v in s["appliedStats"].values() if isinstance(v, (int, float)))
        if total is None:
            continue
        split = s.get("statSplitTypeId")
        score = (2 if split == 2 else 1, float(total))  # prefer season-to-date split
        if best is None or score > best[0]:
            best = (score, total)
    return 0.0 if best is None else best[1]

def main():
    data = fetch_all()
    players = data.get("players", data)
    rows = []
    for p in players:
        info = p.get("player", p)
        rows.append({
            "player_id": info.get("id"),
            "player_name": info.get("fullName") or info.get("name"),
            "team": TEAM.get(info.get("proTeamId", 0), "FA"),
            "position": POS.get(info.get("defaultPositionId", 0), ""),
            "season": SEASON,
            "fantasy_total": extract_season_total(info, SEASON),
        })
    df = pd.DataFrame(rows).sort_values("fantasy_total", ascending=False)
    out = OUT / f"espn_players_{SEASON}_scoring{SCORING_ID}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} players → {out}")

if __name__ == "__main__":
    main()
