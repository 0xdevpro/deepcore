from typing import Optional

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ['.env', '../.env', '../../.env', '../../../.env']
        env_file_encoding = 'utf-8'
        extra = 'ignore'

    APP_NAME: str = "deepcore"
    OTEL_ENABLED: bool = True
    OTEL_TRACE_UPLOAD_ENABLED: bool = False
    OTEL_TRACE_UPLOAD_URL: str = "http://127.0.0.1:4318/v1/traces"

    API_KEYS: Optional[list[str]] = []
    TWITTER_TOKEN: Optional[str] = ""

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 1
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    MODEL_NAME: str = "gpt-4o-2024-11-20"
    MCP_MODEL_NAME: str = "grok-3-mini-beta"
    MODEL_TEMPERATURE:float = 0.1

    COIN_HOST: str = ""
    COIN_HOST_V2: str = ""
    COIN_API_KEY: str = ""
    COIN_API_KEY_V2: str = ""
    API_KEY: str = ""
    AI_SEARCH_HOST: str = ""
    AI_SEARCH_KEY: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = None
    REDIS_SSL: bool = False
    REDIS_DB: int = 0
    REDIS_PREFIX: str = "default"
    LOG_LEVEL: str = "INFO"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "mydatabase"

    # AWS S3 Configuration
    STORAGE_TYPE: str = "s3"  # Options: "database", "s3"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-southeast-2"
    AWS_S3_BUCKET: str = ""
    AWS_S3_PREFIX: str = "uploads/"
    AWS_S3_ENDPOINT_URL: Optional[str] = None  # For custom S3-compatible storage
    AWS_S3_URL_EXPIRATION: int = 3600  # Presigned URL expiration time (seconds), default 1 hour

    OPENAPI_FITTER_FIELDS: list[str] = []

    JWT_SECRET: str = ""
    JWT_EXPIRATION_TIME: int = 1  # default one day
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12 * 30 # default 30 days
    REFRESH_TOKEN_EXPIRE_DAYS: int = 60 # default 60 days
    ENCRYPTION_KEY: str = ""

    API_BASE_URL: str = "" # base url
    TELEGRAM_REDIS_KEY: str = "deepcore_tg_bot_conf_all"
    
    # Default tool icon URL
    DEFAULT_TOOL_ICON: str = "https://deepweb3.s3.ap-southeast-2.amazonaws.com/static_images/icons/default.webp"

    DATA_API_BASE: str = ""
    DATA_API_KEY: str = ""

    COINMARKETCAP_BASE_HOST: str = ""
    COINMARKETCAP_API_KEY: str = ""

    MONGO_STRING: str = ""
    IMGAI_API_KEY: str = ""
    IMGAI_URL: str = ""
    C_PRO_API_KEY: str = ""
    CRYPTOCURRENCY_LATEST_URL = ""

    # Agent balance check switch
    AGENT_BALANCE_CHECK_ENABLED: bool = False

    MASTER_ADDRESS: str = ""
    SOLANA_RPC_URL: str = ""



SETTINGS = Settings()
