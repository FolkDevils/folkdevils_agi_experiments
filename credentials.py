"""
Credential Vault for Son of Andrew
Centralized, secure credential management with validation and error handling.
"""
import os
import logging
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class CredentialSpec:
    """Specification for a credential with validation and metadata."""
    key: str
    required: bool = False
    default: Optional[str] = None
    validator: Optional[Callable[[str], bool]] = None
    description: str = ""
    sensitive: bool = True  # Whether to mask in logs

class CredentialError(Exception):
    """Raised when credential validation or retrieval fails."""
    pass

class CredentialVault:
    """
    Centralized credential management system.
    
    Features:
    - Environment variable loading with validation
    - Required vs optional credentials
    - Custom validators for credential formats
    - Helpful error messages for missing/invalid credentials
    - Secure logging (masks sensitive values)
    - Extensible for future backends (AWS Secrets Manager, etc.)
    """
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._specs: Dict[str, CredentialSpec] = {}
        self._setup_credential_specs()
    
    def _setup_credential_specs(self):
        """Define all credential specifications."""
        
        # OpenAI credentials
        self.register_credential(CredentialSpec(
            key="OPENAI_API_KEY",
            required=True,
            validator=self._validate_openai_key,
            description="OpenAI API key for GPT models"
        ))
        
        # Zep Cloud credentials
        self.register_credential(CredentialSpec(
            key="ZEP_PROJECT_KEY", 
            required=False,
            default="",
            validator=self._validate_zep_key,
            description="Zep Cloud project key for memory storage"
        ))
        
        self.register_credential(CredentialSpec(
            key="ZEP_USER_ID",
            required=False,
            default="son_of_andrew_user",
            validator=self._validate_user_id,
            description="Consistent user ID for Zep Cloud sessions",
            sensitive=False
        ))
        
        # Configuration values (not sensitive)
        self.register_credential(CredentialSpec(
            key="OPENAI_MODEL",
            required=False,
            default="gpt-4o",
            validator=self._validate_model_name,
            description="OpenAI model name",
            sensitive=False
        ))
        
        self.register_credential(CredentialSpec(
            key="OPENAI_STREAM",
            required=False,
            default="false",
            validator=self._validate_boolean,
            description="Enable streaming responses",
            sensitive=False
        ))
        
        self.register_credential(CredentialSpec(
            key="USE_ZEP",
            required=False,
            default="true",
            validator=self._validate_boolean,
            description="Enable Zep Cloud integration",
            sensitive=False
        ))
        
        # Numeric configuration
        self.register_credential(CredentialSpec(
            key="OPENAI_MAX_TOKENS",
            required=False,
            default="4096",
            validator=self._validate_positive_int,
            description="Maximum tokens for OpenAI responses",
            sensitive=False
        ))
        
        self.register_credential(CredentialSpec(
            key="SON_OF_ANDREW_MAX_STEPS",
            required=False,
            default="7",
            validator=self._validate_positive_int,
            description="Maximum workflow steps",
            sensitive=False
        ))
        
        # Temperature settings
        for agent, default_temp in [
            ("WRITER", "0.7"),
            ("EDITOR", "0.3"), 
            ("PERFORMANCE", "0.2"),
            ("LEARNING", "0.1")
        ]:
            self.register_credential(CredentialSpec(
                key=f"{agent}_TEMPERATURE",
                required=False,
                default=default_temp,
                validator=self._validate_temperature,
                description=f"Temperature setting for {agent.lower()} agent",
                sensitive=False
            ))
        
        # Performance optimization settings
        self.register_credential(CredentialSpec(
            key="ENABLE_RESPONSE_CACHING",
            required=False,
            default="true",
            validator=self._validate_boolean,
            description="Enable response caching for performance",
            sensitive=False
        ))
        
        self.register_credential(CredentialSpec(
            key="CACHE_TTL_SECONDS",
            required=False,
            default="300",
            validator=self._validate_positive_int,
            description="Cache time-to-live in seconds",
            sensitive=False
        ))
        
        self.register_credential(CredentialSpec(
            key="FAST_ROUTING",
            required=False,
            default="true",
            validator=self._validate_boolean,
            description="Enable fast pattern-based routing",
            sensitive=False
        ))
    
    def register_credential(self, spec: CredentialSpec) -> None:
        """Register a new credential specification."""
        self._specs[spec.key] = spec
        logger.debug(f"Registered credential: {spec.key}")
    
    def get(self, key: str, required: Optional[bool] = None) -> str:
        """
        Get a credential value with validation.
        
        Args:
            key: Environment variable key
            required: Override the spec's required setting
            
        Returns:
            The credential value
            
        Raises:
            CredentialError: If credential is missing, invalid, or validation fails
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Get specification
        spec = self._specs.get(key)
        if not spec:
            # Allow dynamic credentials not in specs
            value = os.getenv(key)
            if value is None:
                if required:
                    raise CredentialError(f"Required credential '{key}' not found in environment")
                return ""
            return value
        
        # Use override or spec requirement
        is_required = required if required is not None else spec.required
        
        # Get value from environment
        value = os.getenv(key, spec.default)
        
        # Check if required but missing
        if is_required and (value is None or value == ""):
            raise CredentialError(
                f"Required credential '{key}' is missing.\n"
                f"Description: {spec.description}\n"
                f"Please set the {key} environment variable."
            )
        
        # Use default if not provided
        if value is None:
            value = spec.default or ""
        
        # Validate if validator provided
        if spec.validator and value:
            try:
                if not spec.validator(value):
                    raise CredentialError(
                        f"Credential '{key}' failed validation.\n"
                        f"Description: {spec.description}\n"
                        f"Value: {'[REDACTED]' if spec.sensitive else value}"
                    )
            except Exception as e:
                raise CredentialError(
                    f"Credential '{key}' validation error: {e}\n"
                    f"Description: {spec.description}"
                )
        
        # Cache and return
        self._cache[key] = value
        
        # Log success (masking sensitive values)
        log_value = "[REDACTED]" if spec.sensitive and value else value
        logger.debug(f"Loaded credential '{key}': {log_value}")
        
        return value
    
    def get_bool(self, key: str, required: Optional[bool] = None) -> bool:
        """Get a credential as a boolean."""
        value = self.get(key, required)
        return value.lower() in ("true", "1", "yes", "on")
    
    def get_int(self, key: str, required: Optional[bool] = None) -> int:
        """Get a credential as an integer."""
        value = self.get(key, required)
        try:
            return int(value)
        except ValueError:
            spec = self._specs.get(key, CredentialSpec(key=key))
            raise CredentialError(
                f"Credential '{key}' must be an integer, got: {value}\n"
                f"Description: {spec.description}"
            )
    
    def get_float(self, key: str, required: Optional[bool] = None) -> float:
        """Get a credential as a float."""
        value = self.get(key, required)
        try:
            return float(value)
        except ValueError:
            spec = self._specs.get(key, CredentialSpec(key=key))
            raise CredentialError(
                f"Credential '{key}' must be a number, got: {value}\n"
                f"Description: {spec.description}"
            )
    
    def validate_all_required(self) -> None:
        """Validate all required credentials are present and valid."""
        errors = []
        
        for key, spec in self._specs.items():
            if spec.required:
                try:
                    self.get(key)
                except CredentialError as e:
                    errors.append(str(e))
        
        if errors:
            raise CredentialError(
                f"Missing required credentials:\n" + 
                "\n---\n".join(errors)
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all credentials (for debugging)."""
        summary = {}
        
        for key, spec in self._specs.items():
            try:
                value = self.get(key)
                summary[key] = {
                    "present": bool(value),
                    "value": "[REDACTED]" if spec.sensitive and value else value,
                    "required": spec.required,
                    "description": spec.description
                }
            except CredentialError:
                summary[key] = {
                    "present": False,
                    "value": None,
                    "required": spec.required,
                    "description": spec.description,
                    "error": True
                }
        
        return summary
    
    # Validation methods
    def _validate_openai_key(self, value: str) -> bool:
        """Validate OpenAI API key format."""
        return value.startswith("sk-") and len(value) > 20
    
    def _validate_zep_key(self, value: str) -> bool:
        """Validate Zep project key format."""
        return len(value) > 10  # Basic length check
    
    def _validate_user_id(self, value: str) -> bool:
        """Validate user ID format."""
        return len(value) > 0 and value.replace("_", "").replace("-", "").isalnum()
    
    def _validate_model_name(self, value: str) -> bool:
        """Validate OpenAI model name."""
        valid_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        return any(model in value for model in valid_models)
    
    def _validate_boolean(self, value: str) -> bool:
        """Validate boolean string."""
        return value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off")
    
    def _validate_positive_int(self, value: str) -> bool:
        """Validate positive integer."""
        try:
            return int(value) > 0
        except ValueError:
            return False
    
    def _validate_temperature(self, value: str) -> bool:
        """Validate temperature value (0.0 to 2.0)."""
        try:
            temp = float(value)
            return 0.0 <= temp <= 2.0
        except ValueError:
            return False

# Global credential vault instance
credentials = CredentialVault() 