import pytest

from app.models.core import AzureOpenAIConfig, resolve_embedding_dimensions


def test_resolve_embedding_dimensions_uses_known_model_defaults():
    assert resolve_embedding_dimensions("text-embedding-3-large") == 3072
    assert resolve_embedding_dimensions("text-embedding-3-small") == 1536
    assert resolve_embedding_dimensions("text-embedding-ada-002") == 1536


def test_resolve_embedding_dimensions_falls_back_for_unknown_models():
    assert resolve_embedding_dimensions("custom-embedding-model") == 1536


def test_resolve_embedding_dimensions_accepts_configured_string_or_int():
    assert resolve_embedding_dimensions("text-embedding-3-large", "1024") == 1024
    assert resolve_embedding_dimensions("text-embedding-3-large", 768) == 768


@pytest.mark.parametrize("configured", ["0", "-1"])
def test_resolve_embedding_dimensions_rejects_non_positive_values(configured):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_embedding_dimensions("text-embedding-3-large", configured)


def test_azure_openai_config_resolves_embedding_dimensions_on_init():
    config = AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com",
        api_key="key",
        embedding_deployment="embedding-deployment",
        embedding_model="text-embedding-3-large",
        chat_deployment="chat-deployment",
        chat_model="gpt-5",
    )

    assert config.embedding_dimensions == 3072
