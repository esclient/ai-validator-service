import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    host: str = Field(validation_alias="HOST")
    port: int = Field(validation_alias="PORT")
    database_url: str = Field(validation_alias="DATABASE_URL")

    log_level: str = Field(validation_alias="LOG_LEVEL")
    log_format: str = Field(validation_alias="LOG_FORMAT")
    log_datefmt: str = Field(validation_alias="LOG_DATEFMT")

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            datefmt=self.log_datefmt,
        )
