# src/espn_client.py
from espn_api.football import League
from .config import settings

def get_league(year: int | None = None) -> League:
    """Create an ESPN League object (no team_id here)."""
    return League(
        league_id=settings.league_id,
        year=int(year or settings.season),
        espn_s2=settings.espn_s2,
        swid=settings.swid,
    )

def get_team(league: League | None = None, team_id: int | None = None):
    """Return the Team object for settings.team_id (or provided team_id)."""
    lg = league or get_league()
    tid = int(team_id or settings.team_id)
    for t in lg.teams:
        # espn_api uses .team_id on Team objects
        if getattr(t, "team_id", None) == tid:
            return t
    raise ValueError(f"Team ID {tid} not found in league {settings.league_id} for season {lg.year}")

def get_current_scoring_period() -> int:
    """Return ESPN's current scoring period (week)."""
    lg = get_league()
    # Some versions expose .current_week; others have .scoringPeriodId
    return getattr(lg, "current_week", getattr(lg, "scoringPeriodId", 1))
