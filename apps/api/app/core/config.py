from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATABASE_URL = (
    "mysql+pymysql://root:CHANGE_ME@127.0.0.1:3306/"
    "offermaster?charset=utf8mb4"
)
DEFAULT_BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="JOBPILOT_",
        extra="ignore",
        populate_by_name=True,
    )

    env: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = Field(default="INFO", validation_alias="JOBPILOT_LOG_LEVEL")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", validation_alias="JOBPILOT_LOG_DIR")
    uploads_path: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads",
        validation_alias="JOBPILOT_UPLOADS_PATH",
    )
    imports_path: Path = Field(
        default=PROJECT_ROOT / "data" / "imports",
        validation_alias="JOBPILOT_IMPORTS_PATH",
    )
    exports_path: Path = Field(
        default=PROJECT_ROOT / "data" / "exports",
        validation_alias="JOBPILOT_EXPORTS_PATH",
    )
    database_url_string: str = Field(
        default=DEFAULT_DATABASE_URL,
        validation_alias="JOBPILOT_DATABASE_URL",
    )
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    vector_store_provider: str = Field(
        default="deferred",
        validation_alias="JOBPILOT_VECTOR_STORE_PROVIDER",
    )
    mcp_enabled: bool = False
    mcp_server_url: str | None = Field(default=None, validation_alias="JOBPILOT_MCP_SERVER_URL")
    mcp_tool_allowlist: str = Field(
        default="open_page,read_page,fill_form",
        validation_alias="JOBPILOT_MCP_TOOL_ALLOWLIST",
    )
    xiaohongshu_mcp_base_url: str | None = Field(
        default=None,
        validation_alias="JOBPILOT_XIAOHONGSHU_MCP_BASE_URL",
    )
    xiaohongshu_mcp_auth_token: SecretStr | None = Field(
        default=None,
        validation_alias="JOBPILOT_XIAOHONGSHU_MCP_AUTH_TOKEN",
    )
    job_providers: str = Field(
        default="mock,import_file",
        validation_alias="JOBPILOT_JOB_PROVIDERS",
    )
    worker_poll_interval_seconds: int = Field(
        default=30,
        validation_alias="JOBPILOT_WORKER_POLL_INTERVAL_SECONDS",
    )
    worker_max_retries: int = Field(default=3, validation_alias="JOBPILOT_WORKER_MAX_RETRIES")
    external_agent_auto_dispatch: bool = Field(
        default=False,
        validation_alias="JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH",
    )
    external_web_search_provider: str = Field(
        default="auto",
        validation_alias="JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER",
    )
    claude_sdk_agent_base_url: str | None = Field(
        default=None,
        validation_alias="JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL",
    )
    claude_sdk_agent_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="JOBPILOT_CLAUDE_SDK_AGENT_API_KEY",
    )
    claude_sdk_agent_model: str = Field(
        default="MiniMax-M2.7",
        validation_alias="JOBPILOT_CLAUDE_SDK_AGENT_MODEL",
    )
    claude_sdk_agent_timeout_seconds: float = Field(
        default=300.0,
        validation_alias="JOBPILOT_CLAUDE_SDK_AGENT_TIMEOUT_SECONDS",
    )
    openai_sdk_agent_enabled: bool = Field(
        default=False,
        validation_alias="JOBPILOT_OPENAI_SDK_AGENT_ENABLED",
    )
    openai_sdk_agent_base_url: str | None = Field(
        default=None,
        validation_alias="JOBPILOT_OPENAI_SDK_AGENT_BASE_URL",
    )
    openai_sdk_agent_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="JOBPILOT_OPENAI_SDK_AGENT_API_KEY",
    )
    openai_sdk_agent_model: str | None = Field(
        default=None,
        validation_alias="JOBPILOT_OPENAI_SDK_AGENT_MODEL",
    )
    openai_sdk_agent_timeout_seconds: float = Field(
        default=120.0,
        validation_alias="JOBPILOT_OPENAI_SDK_AGENT_TIMEOUT_SECONDS",
    )
    speech_provider: str = Field(
        default="web_speech",
        validation_alias="JOBPILOT_SPEECH_PROVIDER",
    )
    llm_provider: str = Field(default="bailian", validation_alias="JOBPILOT_LLM_PROVIDER")
    llm_base_url: str = Field(
        default=DEFAULT_BAILIAN_BASE_URL,
        validation_alias="JOBPILOT_LLM_BASE_URL",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="JOBPILOT_LLM_API_KEY",
    )
    llm_model: str = Field(default="qwen-plus", validation_alias="JOBPILOT_LLM_MODEL")
    llm_timeout_seconds: float = Field(
        default=60.0,
        validation_alias="JOBPILOT_LLM_TIMEOUT_SECONDS",
    )
    llm_max_retries: int = Field(default=2, validation_alias="JOBPILOT_LLM_MAX_RETRIES")
    execution_planner_enabled: bool = Field(default=False, validation_alias="JOBPILOT_EXECUTION_PLANNER_ENABLED")
    intent_llm_provider: str | None = Field(default=None, validation_alias="JOBPILOT_INTENT_LLM_PROVIDER")
    intent_llm_base_url: str | None = Field(default=None, validation_alias="JOBPILOT_INTENT_LLM_BASE_URL")
    intent_llm_api_key: SecretStr | None = Field(default=None, validation_alias="JOBPILOT_INTENT_LLM_API_KEY")
    intent_llm_model: str | None = Field(default=None, validation_alias="JOBPILOT_INTENT_LLM_MODEL")
    intent_llm_timeout_seconds: float | None = Field(
        default=None,
        validation_alias="JOBPILOT_INTENT_LLM_TIMEOUT_SECONDS",
    )
    intent_llm_max_retries: int | None = Field(default=None, validation_alias="JOBPILOT_INTENT_LLM_MAX_RETRIES")
    embedding_provider: str = Field(
        default="disabled",
        validation_alias="JOBPILOT_EMBEDDING_PROVIDER",
    )

    @property
    def database_url(self) -> URL:
        return make_url(self.database_url_string)

    @property
    def enabled_job_providers(self) -> list[str]:
        return [provider.strip() for provider in self.job_providers.split(",") if provider.strip()]

    @property
    def allowed_mcp_tools(self) -> list[str]:
        return [tool.strip() for tool in self.mcp_tool_allowlist.split(",") if tool.strip()]

    @field_validator(
        "log_dir",
        "uploads_path",
        "imports_path",
        "exports_path",
        "vector_store_path",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @field_validator(
        "llm_base_url",
        "intent_llm_base_url",
        "xiaohongshu_mcp_base_url",
        "claude_sdk_agent_base_url",
        "openai_sdk_agent_base_url",
        mode="after",
    )
    @classmethod
    def strip_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
