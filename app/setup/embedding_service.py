from models.core import resolve_embedding_dimensions


class EmbeddingService:
    def __init__(self, endpoint, deployment, model_name, dimensions=None, api_key=None):
        self.endpoint = endpoint
        self.deployment = deployment
        self.model_name = model_name
        self.dimensions = resolve_embedding_dimensions(model_name, dimensions)
        self.api_key = api_key

    
