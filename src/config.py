from pydantic import BaseSettings

class Settings(BaseSettings):
    espn_s2: str
    swid: str
    league_id: int
    season: int = 2025

    class Config:
        env_prefix = ''
        env_file = '.env'
        fields = {
            'espn_s2': {'env': 'ESPN_S2'},
            'swid': {'env': 'ESPN_SWID'},
            'league_id': {'env': 'LEAGUE_ID'},
            'season': {'env': 'SEASON'},
        }

settings = Settings()
