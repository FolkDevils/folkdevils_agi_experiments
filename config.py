# config.py
"""Centralized settings for Son of Andrew using credential vault."""
from credentials import credentials
import os

class _Settings:
    """Settings powered by the credential vault system."""
    
    # Model & keys
    @property
    def OPENAI_API_KEY(self) -> str:
        return credentials.get("OPENAI_API_KEY")
    
    @property
    def MODEL_NAME(self) -> str:
        return credentials.get("OPENAI_MODEL")

    # Runtime knobs
    @property
    def STREAM(self) -> bool:
        return credentials.get_bool("OPENAI_STREAM")
    
    @property
    def MAX_TOKENS(self) -> int:
        return credentials.get_int("OPENAI_MAX_TOKENS")
    
    @property
    def MAX_STEPS(self) -> int:
        return credentials.get_int("SON_OF_ANDREW_MAX_STEPS")

    # Per-agent temps
    @property
    def WRITER_TEMP(self) -> float:
        return credentials.get_float("WRITER_TEMPERATURE")
    
    @property
    def EDITOR_TEMP(self) -> float:
        return credentials.get_float("EDITOR_TEMPERATURE")
    
    @property
    def PERF_TEMP(self) -> float:
        return credentials.get_float("PERFORMANCE_TEMPERATURE")
    
    @property
    def LEARN_TEMP(self) -> float:
        return credentials.get_float("LEARNING_TEMPERATURE")

    # Memory back-end
    @property
    def USE_ZEP(self) -> bool:
        return credentials.get_bool("USE_ZEP")
    
    @property
    def ZEP_PROJECT_KEY(self) -> str:
        return credentials.get("ZEP_PROJECT_KEY")
    
    @property
    def ZEP_USER_ID(self) -> str:
        return credentials.get("ZEP_USER_ID")
    
    # Performance Settings
    @property
    def ENABLE_RESPONSE_CACHING(self) -> bool:
        return credentials.get_bool("ENABLE_RESPONSE_CACHING")
    
    @property
    def CACHE_TTL_SECONDS(self) -> int:
        return credentials.get_int("CACHE_TTL_SECONDS")
    
    @property
    def FAST_ROUTING(self) -> bool:
        return credentials.get_bool("FAST_ROUTING")
    
    def validate_all_credentials(self) -> None:
        """Validate all required credentials are present."""
        credentials.validate_all_required()
    
    def get_credential_summary(self) -> dict:
        """Get summary of all credentials for debugging."""
        return credentials.get_summary()

settings = _Settings() 