# src/roster.py
from __future__ import annotations

from .espn_client import get_league, get_team
from .config import settings
import re
from typing import Optional, Tuple, List
import requests
import pandas as pd

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _box_scores_for_week(lg, week: int):
    """
    Always fetch box scores for the requested matchup period.
    Tries multiple call signatures to support different espn_api versions,
    then hard-filters by the week if the objects expose a matchupPeriod attribute.
    """
    boxes = []

    # Preferred: explicit kw (many recent versions)
    try:
        b = lg.box_scores(matchupPeriodId=week)
        if b: boxes.extend(b)
    except TypeError:
        pass
    except Exception:
        pass

    # Fallback: positional arg
    if not boxes:
        try:
            b = lg.box_scores(week)
            if b: boxes.extend(b)
        except Exception:
            pass

    # Last-resort: get all and filter manually if attribute exists
    if not boxes:
        try:
            b = lg.box_scores()
            if b: boxes.extend(b)
        except Exception:
            pass

    # Strict filter by week if the objects expose a period attribute
    if boxes and hasattr(boxes[0], "matchupPeriod"):
        boxes = [x for x in boxes if getattr(x, "matchupPeriod", None) == week]
    if boxes and hasattr(boxes[0], "matchupPeriodId"):
        boxes = [x for x in boxes if getattr(x, "matchupPeriodId", None) == week]

    return boxes or []

def get_week_matchup(week: int):
    """
    Return the BoxScore for YOUR team in the given week (no caching).
    """
    lg = get_league()  # new League each call (avoid stale internal caches)
    boxes = _box_scores_for_week(lg, int(week))
    for b in boxes:
        home = getattr(b, "home_team", None)
        away = getattr(b, "away_team", None)
        if not home or not away:
            continue
        if getattr(home, "team_id", None) == settings.team_id or getattr(away, "team_id", None) == settings.team_id:
            return b
    return None

def get_opponent_roster(week: int):
    box = get_week_matchup(week)
    if not box:
        return []
    home = getattr(box, "home_team", None)
    away = getattr(box, "away_team", None)
    if not home or not away:
        return []
    my_is_home = getattr(home, "team_id", None) == settings.team_id
    opp_team = away if my_is_home else home
    return getattr(opp_team, "roster", []) or []
def get_current_roster():
    lg = get_league()
    team = get_team(lg, settings.team_id)

    # Build the structure your predictor expects
    # Adjust these depending on how your Team/Player objects look in your version of espn_api
    skill = []
    kickers = []
    dst = []

    for p in team.roster:
        name = getattr(p, "name", getattr(p, "playerName", ""))
        pos  = getattr(p, "position", "")
        if pos in {"QB", "RB", "WR", "TE"}:
            skill.append((name, pos))
        elif pos == "K":
            kickers.append((name, "K"))
        elif pos in {"D/ST", "DST", "DEF"}:
            # use the NFL code from the player's pro team if available
            # espn_api team defenses may present differently; fall back to team abbreviation
            nfl = getattr(p, "proTeam", getattr(p, "pro_team", getattr(p, "proTeamAbbrev", "")))
            label = nfl or getattr(team, "team_abbrev", "")
            if label:
                dst.append(label)

    # if your league has a "watch list" bench or IR, you may need to include those too
    return {"skill": skill, "kickers": kickers, "dst": dst}

def _fetch_matchups_http(week: int) -> dict:
    """
    Fetch league scoreboard for a specific scoring period (week).
    Tries the official fantasy host first, then lm-api fallback.
    Sends cookies + UA + referer to avoid 302->403.
    """
    season = int(settings.season)
    league_id = int(settings.league_id)

    headers = {
        "Accept": "application/json",
        "User-Agent": UA,
        "Referer": f"https://fantasy.espn.com/football/league?leagueId={league_id}",
        "Accept-Language": "en-US,en;q=0.9",
    }
    cookies = {
        "espn_s2": settings.espn_s2,
        "SWID": settings.swid,
    }

    urls = [
        # Primary (league-scoped)
        f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
        f"/segments/0/leagues/{league_id}"
        f"?scoringPeriodId={int(week)}&view=mMatchupScore",
        # Sometimes this also works for league views
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
        f"/segments/0/leagues/{league_id}"
        f"?scoringPeriodId={int(week)}&view=mMatchupScore",
    ]

    with requests.Session() as s:
        s.headers.update(headers)
        s.cookies.update(cookies)

        last_err = None
        for url in urls:
            try:
                # block silent 302 to /fantasy/ which then 403s; we want the API JSON
                r = s.get(url, timeout=30, allow_redirects=False)
                # follow a single redirect manually if it stays on the same host
                if r.is_redirect:
                    loc = r.headers.get("Location", "")
                    if loc.startswith("http") and "espn.com" in loc:
                        r = s.get(loc, timeout=30, allow_redirects=False)

                r.raise_for_status()
                # ensure JSON
                if "application/json" not in r.headers.get("Content-Type", ""):
                    # try to parse anyway
                    try:
                        return r.json()
                    except Exception:
                        raise requests.HTTPError(f"Unexpected content-type for {url}: {r.headers.get('Content-Type')}")
                return r.json()
            except requests.HTTPError as e:
                last_err = e
                continue
            except requests.RequestException as e:
                last_err = e
                continue

    if last_err:
        raise last_err
    raise RuntimeError("Unexpected failure calling ESPN matchup API")

def get_opponent_for_week(week: int) -> Optional[Tuple[int, str]]:
    """
    Returns (opponent_team_id, opponent_team_name) for your team in the given week.
    Filters schedule strictly by matchupPeriodId == week (with fallback logic).
    """
    data = _fetch_matchups_http(week)
    schedule = data.get("schedule", [])
    teams = {t.get("id"): f"{(t.get('location') or '').strip()} {(t.get('nickname') or '').strip()}".strip()
             for t in data.get("teams", [])}
    my_id = int(settings.team_id)

    # 1) Primary filter: matchupPeriodId == requested week
    target_mp = int(week)
    candidates = [m for m in schedule if int(m.get("matchupPeriodId", -1)) == target_mp]

    # 2) Fallback: if nothing matched (some rare leagues), keep all but prefer rows
    # that have scoring data for this scoring period.
    if not candidates:
        def has_week_scoring(m):
            for side in ("home", "away"):
                s = (m.get(side) or {}).get("rosterForCurrentScoringPeriod") or {}
                if int(s.get("scoringPeriodId", -1)) == int(week):
                    return True
            return False
        prefer = [m for m in schedule if has_week_scoring(m)]
        candidates = prefer or schedule

    # 3) From candidates, choose the one that involves my team
    for m in candidates:
        home = (m.get("home") or {})
        away = (m.get("away") or {})
        h_id = home.get("teamId")
        a_id = away.get("teamId")
        if h_id is None or a_id is None:
            continue
        if h_id == my_id:
            opp_id = a_id
        elif a_id == my_id:
            opp_id = h_id
        else:
            continue
        opp_name = teams.get(opp_id, f"Team {opp_id}")
        return int(opp_id), opp_name

    return None
def get_opponent_roster(week: int):
    """
    Given (league_id, season) + cookies, resolve opponent teamId via HTTP,
    then return that espn_api Team's roster (list of Player objects).
    """
    opp = get_opponent_for_week(week)
    if not opp:
        return []
    opp_id, _ = opp
    lg = get_league()
    team = next((t for t in lg.teams if int(getattr(t, "team_id", -1)) == int(opp_id)), None)
    return getattr(team, "roster", []) if team else []

    my_is_home = getattr(home, "team_id", None) == settings.team_id
    opp_team = away if my_is_home else home
    # espn_api Team exposes .roster
    return getattr(opp_team, "roster", []) or []
def _norm_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\b(jr|sr)\.?\b", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_my_roster_ids_and_names(week: int | None = None):
    """
    Return (ids, norm_names) for the current team roster.
    Uses espn_api league; week arg isn’t required by espn_api for roster(),
    but we keep it for future-proofing.
    """
    lg = get_league()
    team = next(t for t in lg.teams if t.team_id == settings.team_id)
    ids, names = set(), set()
    for p in team.roster:
        pid = getattr(p, "playerId", getattr(p, "id", None))
        if pid is not None:
            try:
                ids.add(int(pid))
            except Exception:
                pass
        nm = getattr(p, "name", getattr(p, "playerName", getattr(p, "fullName", "")))
        if nm:
            names.add(_norm_name(nm))
    return ids, names