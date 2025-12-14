"""
Tests for Azure OpenAI provider abstraction.

Tests configuration, provider creation, and mock provider.
"""

import pytest
import os
from unittest.mock import patch, AsyncMock

from apps.api.llm.gpt5_client_azure import (
    ProviderConfig,
    ProviderType,
    LLMRequest,
    LLMResponse,
    MockProvider,
    create_provider,
    get_provider,
    set_provider,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_from_env_with_defaults(self):
        """Test configuration from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = ProviderConfig.from_env()

            # Default provider is azure_chat (supports response_format with JSON schema broadly)
            assert config.provider_type == ProviderType.AZURE_CHAT
            assert config.api_version == "2024-10-21"
            assert config.max_tokens == 4096
            assert config.temperature == 0.0

    def test_from_env_with_values(self):
        """Test configuration with environment values."""
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2025-01-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5",
            "AZURE_OPENAI_MODEL_NAME": "gpt-5-turbo",
            "LLM_PROVIDER_TYPE": "azure_chat",
            "LLM_MAX_TOKENS": "2048",
            "LLM_TEMPERATURE": "0.5",
        }

        with patch.dict(os.environ, env, clear=True):
            config = ProviderConfig.from_env()

            assert config.endpoint == "https://test.openai.azure.com/"
            assert config.api_key == "test-key"
            assert config.api_version == "2025-01-01"
            assert config.deployment_name == "gpt-5"
            assert config.model_name == "gpt-5-turbo"
            assert config.provider_type == ProviderType.AZURE_CHAT
            assert config.max_tokens == 2048
            assert config.temperature == 0.5

    def test_is_configured_true(self):
        """Test is_configured returns True when required fields present."""
        config = ProviderConfig(
            provider_type=ProviderType.AZURE_RESPONSES,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-10-21",
            deployment_name="gpt-5",
        )

        assert config.is_configured() is True

    def test_is_configured_false_missing_endpoint(self):
        """Test is_configured returns False when endpoint missing."""
        config = ProviderConfig(
            provider_type=ProviderType.AZURE_RESPONSES,
            endpoint="",
            api_key="test-key",
            api_version="2024-10-21",
            deployment_name="gpt-5",
        )

        assert config.is_configured() is False

    def test_is_configured_mock_always_true(self):
        """Test that mock provider is always configured."""
        config = ProviderConfig(
            provider_type=ProviderType.MOCK,
            endpoint="",
            api_key="",
            api_version="",
            deployment_name="",
        )

        assert config.is_configured() is True


class TestMockProvider:
    """Tests for MockProvider."""

    @pytest.mark.asyncio
    async def test_complete_returns_default(self):
        """Test that mock provider returns default response."""
        provider = MockProvider(
            default_response="Test response",
            default_json={"answer": "test"},
        )

        request = LLMRequest(
            system_prompt="You are a test assistant",
            user_message="Hello",
        )

        response = await provider.complete(request)

        assert response.content == "Test response"
        assert response.parsed_json == {"answer": "test"}
        assert response.model == "mock-model"
        assert response.error is None

    @pytest.mark.asyncio
    async def test_complete_tracks_requests(self):
        """Test that mock provider tracks requests."""
        provider = MockProvider()

        request1 = LLMRequest(system_prompt="Sys1", user_message="Msg1")
        request2 = LLMRequest(system_prompt="Sys2", user_message="Msg2")

        await provider.complete(request1)
        await provider.complete(request2)

        assert len(provider.requests) == 2
        assert provider.requests[0].user_message == "Msg1"
        assert provider.requests[1].user_message == "Msg2"

    @pytest.mark.asyncio
    async def test_health_check_always_true(self):
        """Test that mock provider health check returns True."""
        provider = MockProvider()

        result = await provider.health_check()

        assert result is True


class TestCreateProvider:
    """Tests for create_provider function."""

    def test_creates_mock_provider(self):
        """Test creating a mock provider."""
        config = ProviderConfig(
            provider_type=ProviderType.MOCK,
            endpoint="",
            api_key="",
            api_version="",
            deployment_name="",
        )

        provider = create_provider(config)

        assert isinstance(provider, MockProvider)

    def test_creates_azure_responses_provider(self):
        """Test creating an Azure Responses provider."""
        from apps.api.llm.gpt5_client_azure import AzureResponsesProvider

        config = ProviderConfig(
            provider_type=ProviderType.AZURE_RESPONSES,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-10-21",
            deployment_name="gpt-5",
        )

        provider = create_provider(config)

        assert isinstance(provider, AzureResponsesProvider)

    def test_creates_azure_chat_provider(self):
        """Test creating an Azure Chat provider."""
        from apps.api.llm.gpt5_client_azure import AzureChatProvider

        config = ProviderConfig(
            provider_type=ProviderType.AZURE_CHAT,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-10-21",
            deployment_name="gpt-5",
        )

        provider = create_provider(config)

        assert isinstance(provider, AzureChatProvider)


class TestGlobalProvider:
    """Tests for global provider singleton."""

    def test_get_provider_creates_default(self):
        """Test that get_provider creates a provider from env."""
        # Reset global state
        import apps.api.llm.gpt5_client_azure as module

        module._provider = None

        with patch.dict(os.environ, {"LLM_PROVIDER_TYPE": "mock"}):
            provider = get_provider()

            assert provider is not None

    def test_set_provider_overrides(self):
        """Test that set_provider sets the global provider."""
        mock = MockProvider(default_response="Custom")
        set_provider(mock)

        provider = get_provider()

        assert provider is mock


class TestLLMRequest:
    """Tests for LLMRequest."""

    def test_request_with_response_format(self):
        """Test creating request with JSON schema."""
        schema = {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
            },
        }

        request = LLMRequest(
            system_prompt="System",
            user_message="User",
            response_format=schema,
        )

        assert request.response_format == schema

    def test_request_with_overrides(self):
        """Test creating request with temperature/token overrides."""
        request = LLMRequest(
            system_prompt="System",
            user_message="User",
            temperature=0.7,
            max_tokens=1024,
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 1024


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_with_error(self):
        """Test response with error."""
        response = LLMResponse(
            content="",
            error="Connection failed",
        )

        assert response.error == "Connection failed"
        assert response.content == ""

    def test_response_with_parsed_json(self):
        """Test response with parsed JSON."""
        response = LLMResponse(
            content='{"answer": "test"}',
            parsed_json={"answer": "test"},
            model="gpt-5",
            finish_reason="stop",
        )

        assert response.parsed_json["answer"] == "test"
        assert response.model == "gpt-5"

