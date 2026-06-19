from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DB connection: only DB_PASSWORD is required; the rest default to the local dev
    # container. Kept optional so models (which only need embed_dim) import without it.
    # If we need just the embed_dim etc. and don't want to do anything with the database
    # no need for password
    db_user: str = "postgres"
    db_password: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ragdb"
    # Optional full override; if set, takes precedence over the DB_* parts above.
    database_url: str | None = None

    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_device: str = "cpu"
    embed_batch_size: int = 32

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.db_password is None:
            raise ValueError("DB_PASSWORD is required to build the database URL")
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
