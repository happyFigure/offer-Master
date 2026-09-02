from dataclasses import dataclass

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int

    def safe_summary(self) -> dict[str, str | float | int]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": "**********",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def build_llm_runtime_config(settings: Settings | None = None) -> LLMRuntimeConfig:
    runtime_settings = settings or get_settings()
    if runtime_settings.llm_api_key is None:
        raise ValueError("JOBPILOT_LLM_API_KEY is required to call the LLM provider")

    api_key = runtime_settings.llm_api_key.get_secret_value().strip()
    if not api_key:
        raise ValueError("JOBPILOT_LLM_API_KEY cannot be empty")

    return LLMRuntimeConfig(
        provider=runtime_settings.llm_provider,
        base_url=runtime_settings.llm_base_url,
        api_key=api_key,
        model=runtime_settings.llm_model,
        timeout_seconds=runtime_settings.llm_timeout_seconds,
        max_retries=runtime_settings.llm_max_retries,
    )


def build_intent_llm_runtime_config(settings: Settings | None = None) -> LLMRuntimeConfig:
    runtime_settings = settings or get_settings()
    api_key_secret = runtime_settings.intent_llm_api_key or runtime_settings.llm_api_key
    if api_key_secret is None:
        raise ValueError("JOBPILOT_INTENT_LLM_API_KEY or JOBPILOT_LLM_API_KEY is required to call the intent LLM provider")

    api_key = api_key_secret.get_secret_value().strip()
    if not api_key:
        raise ValueError("Intent LLM API key cannot be empty")

    return LLMRuntimeConfig(
        provider=runtime_settings.intent_llm_provider or runtime_settings.llm_provider,
        base_url=runtime_settings.intent_llm_base_url or runtime_settings.llm_base_url,
        api_key=api_key,
        model=runtime_settings.intent_llm_model or runtime_settings.llm_model,
        timeout_seconds=runtime_settings.intent_llm_timeout_seconds or runtime_settings.llm_timeout_seconds,
        max_retries=runtime_settings.intent_llm_max_retries if runtime_settings.intent_llm_max_retries is not None else runtime_settings.llm_max_retries,
    )
