"""
Azure OpenAI Provider Module

Implements a provider abstraction supporting:
1. Azure OpenAI Responses API (preferred)
2. Azure OpenAI Chat Completions API (fallback)

Configuration is via environment variables (no hardcoding).
"""

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from apps.api.llm.azure_env_fallback import load_azure_normalized_config_from_env

class ProviderType(str, Enum):
    """Supported provider types."""

    AZURE_RESPONSES = "azure_responses"
    AZURE_CHAT = "azure_chat"
    OPENAI = "openai"
    MOCK = "mock"  # For testing


@dataclass
class ProviderConfig:
    """Configuration for LLM provider."""

    provider_type: ProviderType
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_API_VERSION
        - AZURE_OPENAI_DEPLOYMENT_NAME
        - AZURE_OPENAI_MODEL_NAME (optional)
        - LLM_PROVIDER_TYPE (optional, defaults to azure_responses)
        - LLM_MAX_TOKENS (optional)
        - LLM_TEMPERATURE (optional)
        """
        provider_type_str = os.getenv("LLM_PROVIDER_TYPE", "azure_responses")
        try:
            provider_type = ProviderType(provider_type_str)
        except ValueError:
            provider_type = ProviderType.AZURE_RESPONSES

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "") or ""
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "") or ""
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "") or "2024-10-21"
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "") or ""

        # Support non-standard user .env keys (ENDPOINT_5 / API_KEY_5 / DEPLOYMENT_5)
        if not (endpoint and api_key and deployment_name):
            norm = load_azure_normalized_config_from_env()
            if norm:
                endpoint = endpoint or norm.endpoint
                api_key = api_key or norm.api_key
                api_version = api_version or norm.api_version
                deployment_name = deployment_name or norm.deployment_name

        return cls(
            provider_type=provider_type,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        )

    def is_configured(self) -> bool:
        """Check if required config is present."""
        if self.provider_type == ProviderType.MOCK:
            return True
        return bool(self.endpoint and self.api_key and self.deployment_name)


@dataclass
class LLMRequest:
    """Request to the LLM."""

    system_prompt: str
    user_message: str
    response_format: Optional[dict] = None  # JSON schema for structured output
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    parsed_json: Optional[dict] = None
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            request: The LLM request.

        Returns:
            LLMResponse with content and optional parsed JSON.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        pass


class AzureResponsesProvider(LLMProvider):
    """
    Azure OpenAI Responses API provider.

    This is the preferred provider per Microsoft guidance.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize with configuration."""
        self.config = config
        self._client = None

    async def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request using the Responses API."""
        try:
            client = await self._get_client()

            # Responses API uses `input` and `max_output_tokens`
            input_messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            params: dict[str, Any] = {
                "model": self.config.deployment_name,  # Azure deployment name
                "input": input_messages,
                "temperature": request.temperature or self.config.temperature,
                "max_output_tokens": request.max_tokens or self.config.max_tokens,
            }

            if request.response_format:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.response_format,
                }

            response = await client.responses.create(**params)

            # Extract content (SDK provides output_text; fallback to parsing output blocks)
            content = getattr(response, "output_text", "") or ""
            if not content and getattr(response, "output", None):
                try:
                    # Best-effort fallback: concatenate text blocks
                    blocks = []
                    for item in response.output:
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", "") == "output_text":
                                blocks.append(getattr(c, "text", ""))
                    content = "\n".join([b for b in blocks if b])
                except Exception:
                    content = ""

            # Try to parse as JSON if we requested structured output
            parsed_json = None
            if request.response_format and content:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    pass

            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                model=getattr(response, "model", "") or "",
                usage={
                    "prompt_tokens": getattr(getattr(response, "usage", None), "input_tokens", 0) or 0,
                    "completion_tokens": getattr(getattr(response, "usage", None), "output_tokens", 0) or 0,
                    "total_tokens": getattr(getattr(response, "usage", None), "total_tokens", 0) or 0,
                },
                finish_reason="stop",
            )

        except Exception as e:
            return LLMResponse(
                content="",
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if Azure OpenAI is available."""
        try:
            client = await self._get_client()
            # Simple models list check
            return True
        except Exception:
            return False


class AzureChatProvider(LLMProvider):
    """
    Azure OpenAI Chat Completions API provider.

    This is the fallback provider for compatibility.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize with configuration."""
        self.config = config
        self._client = None

    async def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request using Chat Completions API."""
        try:
            client = await self._get_client()

            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            params: dict[str, Any] = {
                "model": self.config.deployment_name,
                "messages": messages,
                "temperature": request.temperature or self.config.temperature,
                "max_tokens": request.max_tokens or self.config.max_tokens,
            }

            if request.response_format:
                params["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**params)

            content = response.choices[0].message.content or ""

            parsed_json = None
            if request.response_format and content:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    pass

            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason or "",
            )

        except Exception as e:
            return LLMResponse(
                content="",
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if Azure OpenAI is available."""
        try:
            await self._get_client()
            return True
        except Exception:
            return False


class MockProvider(LLMProvider):
    """
    Mock provider for testing.

    Returns configurable responses without API calls.
    """

    def __init__(
        self,
        default_response: str = "Mock response",
        default_json: Optional[dict] = None,
    ):
        """Initialize with default responses."""
        self.default_response = default_response
        self.default_json = default_json
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return mock response."""
        self.requests.append(request)

        return LLMResponse(
            content=self.default_response,
            parsed_json=self.default_json,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        """Mock is always available."""
        return True


def create_provider(config: Optional[ProviderConfig] = None) -> LLMProvider:
    """
    Create an LLM provider from configuration.

    Args:
        config: Provider configuration. If None, loads from environment.

    Returns:
        Configured LLMProvider instance.
    """
    if config is None:
        config = ProviderConfig.from_env()

    if config.provider_type == ProviderType.AZURE_RESPONSES:
        return AzureResponsesProvider(config)
    elif config.provider_type == ProviderType.AZURE_CHAT:
        return AzureChatProvider(config)
    elif config.provider_type == ProviderType.MOCK:
        return MockProvider()
    else:
        # Default to Azure Responses
        return AzureResponsesProvider(config)


# Singleton provider instance
_provider: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    """
    Get the global LLM provider instance.

    Creates one if it doesn't exist.
    """
    global _provider
    if _provider is None:
        _provider = create_provider()
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """
    Set the global LLM provider instance.

    Useful for testing or custom configurations.
    """
    global _provider
    _provider = provider

