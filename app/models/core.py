from dataclasses import dataclass
from typing import Optional


def resolve_embedding_dimensions(model_name: str, configured_dimensions: Optional[str | int] = None) -> int:
    if configured_dimensions is not None and str(configured_dimensions).strip():
        dimensions = int(configured_dimensions)
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be a positive integer.")
        return dimensions

    defaults_by_model = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    return defaults_by_model.get(model_name, 1536)


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    embedding_deployment: str
    embedding_model: str
    chat_deployment: str  
    chat_model: str
    embedding_dimensions: Optional[int] = None

    def __post_init__(self):
        self.embedding_dimensions = resolve_embedding_dimensions(
            self.embedding_model,
            self.embedding_dimensions,
        )

@dataclass
class AzureSearchConfig:
    """
    Azure Config Helper.

    ## Paramaters
    **service_endpoint**: *str* The URL to your Azure AI Search resource.

    index_name: str
    key: str
    """
    service_endpoint: str
    index_name: str
    key: str
