# src/config.py
# Works with Pydantic v2 (pydantic-settings) and v1 fallback.

try:
    # ---- Pydantic v2 path ----
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        # load from .env in repo root
        model_config = SettingsConfigDict(env_file='.env', env_prefix='', extra='ignore')

        espn_s2: str = Field(default='', alias='ESPN_S2')
        swid: str = Field(default='', alias='ESPN_SWID')
        league_id: int = Field(alias='LEAGUE_ID')
        season: int = Field(default=2025, alias='SEASON')
        team_id: int = Field(default=1, alias='TEAM_ID')

    settings = Settings()

except ImportError:
    # ---- Pydantic v1 path ----
    from pydantic import BaseSettings, Field

    class Settings(BaseSettings):
        espn_s2: str = Field('', env='ESPN_S2')
        swid: str = Field('', env='ESPN_SWID')
        league_id: int = Field(..., env='LEAGUE_ID')
        season: int = Field(2025, env='SEASON')
        team_id: int = Field(1, env='TEAM_ID')

        class Config:
            env_file = '.env'
            env_prefix = ''
            case_sensitive = False

    settings = Settings()
