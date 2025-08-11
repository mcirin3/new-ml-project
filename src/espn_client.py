from espn_api.football import League
from .config import settings

def get_league(year: int | None = None, league_id: int | None = None):
    return League(
        league_id=league_id or settings.league_id,
        year=year or settings.season,
        espn_s2=settings.espn_s2,
        swid=settings.swid
    )

def get_current_scoring_period(league=None):
    league = league or get_league(settings.season, settings.league_id)
    return getattr(league, "scoringPeriodId", 1) or 1
