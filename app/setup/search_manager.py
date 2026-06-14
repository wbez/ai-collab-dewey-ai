import json
import logging
from typing import List, Optional

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchVectorizer,
    AzureOpenAIEmbeddingSkill,
    SearchIndexerSkillset,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexer,
    FieldMapping,
)

from azure.search.documents.indexes.aio import SearchIndexerClient
from .search_service import SearchInfo
from .embedding_service import EmbeddingService


logger = logging.getLogger("scripts")


class SearchManager:
    """
    Class to manage a search service. It can create indexes, and update or remove sections stored in these indexes
    To learn more, please visit https://learn.microsoft.com/azure/search/search-what-is-azure-search
    """

    def __init__(
        self,
        search_info: SearchInfo,
        embeddings: EmbeddingService,
        blob_connection_string: str,
        blob_container_name: str,
    ):
        self.search_info = search_info
        self.embeddings = embeddings
        self.embedding_dimensions = self.embeddings.dimensions
        self.blob_connection_string = blob_connection_string
        self.blob_container_name = blob_container_name

    async def create_index(self, vectorizers: Optional[List[VectorSearchVectorizer]] = None):
        logger.info("Checking whether search index %s exists...", self.search_info.index_name)

        async with self.search_info.create_search_index_client() as search_index_client:
            existing_index_names = [name async for name in search_index_client.list_index_names()]
            if self.search_info.index_name not in existing_index_names:
                logger.info("Creating new search index %s", self.search_info.index_name)
                fields = [
                    SearchField(
                        name="chunk_id", 
                        type="Edm.String", 
                        key=True,
                        filterable=True,
                        sortable=True,
                        facetable=True,
                        analyzer_name="keyword",
                    ),
                    SearchableField(
                        name="content",
                        type="Edm.String",
                        analyzer_name="standard.lucene",
                    ),
                    SearchableField(
                        name="headline",
                        type="Edm.String",
                        analyzer_name="standard.lucene",
                    ),
                    SearchField(
                        name="content_vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        hidden=False,
                        searchable=True,
                        filterable=False,
                        sortable=False,
                        facetable=False,
                        vector_search_dimensions=self.embedding_dimensions,
                        vector_search_profile_name="embedding_config",
                    ),
                    SimpleField(
                        name="url",
                        type="Edm.String",
                    ),
                    SimpleField(
                        name="authors",
                        type="Collection(Edm.String)",
                        filterable=True,
                        facetable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="publish_date",
                        type="Edm.DateTimeOffset",
                        filterable=True,
                        sortable=True,
                        facetable=True,
                        retrievable=True,
                        searchable=False
                    ),
                    SimpleField(
                        name="sourcepage",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                    ),
                    SearchableField(
                        name="parent_id", 
                        type="Edm.String",
                        analyzer_name="standard.lucene",
                        filterable=True,
                        sortable=False,
                        facetable=False,
                        retrievable=True,
                    )
                ]

                vectorizers = [
                    AzureOpenAIVectorizer(
                        vectorizer_name=f"{self.search_info.index_name}-vectorizer",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.embeddings.endpoint,
                            deployment_name=self.embeddings.deployment,
                            api_key=self.embeddings.api_key,
                            model_name=self.embeddings.model_name,
                        ),
                    )
                ]

                index = SearchIndex(
                    name=self.search_info.index_name,
                    fields=fields,
                    semantic_search=SemanticSearch(
                        configurations=[
                            SemanticConfiguration(
                                name="default",
                                prioritized_fields=SemanticPrioritizedFields(
                                    title_field=SemanticField(field_name="headline"), 
                                    content_fields=[SemanticField(field_name="content")],
                                    keywords_fields=[SemanticField(field_name="content")],
                                ),
                            )
                        ]
                    ),
                    vector_search=VectorSearch(
                        algorithms=[
                            HnswAlgorithmConfiguration(
                                name="hnsw_config",
                                parameters=HnswParameters(metric="cosine"),
                            )
                        ],
                        profiles=[
                            VectorSearchProfile(
                                name="embedding_config",
                                algorithm_configuration_name="hnsw_config",
                                vectorizer_name=(
                                    f"{self.search_info.index_name}-vectorizer"
                                ),
                            ),
                        ],
                        vectorizers=vectorizers,
                    ),
                )

                await search_index_client.create_index(index)
                return

            existing_index = await search_index_client.get_index(self.search_info.index_name)
            vector_field = next(
                (field for field in existing_index.fields if field.name == "content_vector"),
                None,
            )
            existing_dimensions = getattr(vector_field, "vector_search_dimensions", None)

            if existing_dimensions != self.embedding_dimensions:
                raise ValueError(
                    f"Existing index '{self.search_info.index_name}' uses "
                    f"{existing_dimensions}-dimensional vectors for 'content_vector', "
                    f"but the configured embedding deployment uses {self.embedding_dimensions}. "
                    "Delete the index or choose a new AZURE_SEARCH_INDEX_NAME before rerunning setup."
                )

            logger.info(
                "Search index %s already exists with matching vector dimensions (%s).",
                self.search_info.index_name,
                self.embedding_dimensions,
            )

    async def create_blob_data_source(self):
        """Create a blob data source for the indexer."""
        data_source_name = f"{self.search_info.index_name}-blob-ds"
        
        data_source = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=self.blob_connection_string,
            container=SearchIndexerDataContainer(name=self.blob_container_name)
        )
        
        return data_source, data_source_name

    async def create_index_skills(self):
        skillset_name = f"{self.search_info.index_name}-skillset"

        split_skill_payload = {
            "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
            "name": f"{self.search_info.index_name}-split-skill",
            "description": "Split skill to chunk documents",
            "textSplitMode": "pages",
            "context": "/document",
            "maximumPageLength": 512,
            "pageOverlapLength": 96,
            "maximumPagesToTake": 0,
            "unit": "azureOpenAITokens",
            "azureOpenAITokenizerParameters": {
                "encoderModelName": "cl100k_base",
            },
            "inputs": [
                {"name": "text", "source": "/document/content"},
            ],
            "outputs": [
                {"name": "textItems", "targetName": "pages"},
            ],
        }

        text_embedding_skill = AzureOpenAIEmbeddingSkill(
            name=f"{self.search_info.index_name}-text-embedding-skill",
            description="Embedding skill to generate embeddings",
            context="/document/pages/*",
            resource_url=self.embeddings.endpoint,
            deployment_name=self.embeddings.deployment,
            api_key=self.embeddings.api_key,
            model_name=self.embeddings.model_name,
            dimensions=self.embeddings.dimensions,
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/pages/*"),
            ],
            outputs=[
                OutputFieldMappingEntry(name="embedding", target_name="content_vector")
            ],
        )

        index_projection = SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=self.search_info.index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(name="content", source="/document/pages/*"),
                        InputFieldMappingEntry(name="headline", source="/document/headline"),
                        InputFieldMappingEntry(name="content_vector", source="/document/pages/*/content_vector"),
                        InputFieldMappingEntry(name="url", source="/document/url"),
                        InputFieldMappingEntry(name="authors", source="/document/authors"),
                        InputFieldMappingEntry(name="publish_date", source="/document/publish_date"),
                        InputFieldMappingEntry(name="sourcepage", source="/document/metadata_storage_name"),
                    ],
                ),
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        skillset = SearchIndexerSkillset(
            name=skillset_name,
            description="Skillset to process documents and generate embeddings",
            skills=[text_embedding_skill],
            index_projection=index_projection,
        )

        skillset_payload = skillset.serialize()
        skillset_payload["skills"].insert(0, split_skill_payload)

        return skillset_payload

    async def create_indexer(self, skillset_name: str, data_source_name: str):
        """Create an indexer to connect data source through skillset to index."""
        indexer_name = f"{self.search_info.index_name}-indexer"
        
        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer to automatically process documents through skillset",
            skillset_name=skillset_name,
            target_index_name=self.search_info.index_name,
            data_source_name=data_source_name,
            field_mappings=[
                FieldMapping(source_field_name="content", target_field_name="content"),
                FieldMapping(source_field_name="headline", target_field_name="headline"),
                FieldMapping(source_field_name="url", target_field_name="url"),
                FieldMapping(source_field_name="authors", target_field_name="authors"),
                FieldMapping(source_field_name="publish_date", target_field_name="publish_date"),
            ],
            parameters={
                "configuration": {
                    "parsingMode": "json",
                    "dataToExtract": "contentAndMetadata"
                }
            }
        )
        
        return indexer, indexer_name

    async def create_or_update_preview_skillset(self, skillset_payload: dict):
        skillset_name = skillset_payload["name"]
        url = f"{self.search_info.endpoint}/skillsets/{skillset_name}?api-version=2024-09-01-preview"

        if not isinstance(self.search_info.credential, AzureKeyCredential):
            raise TypeError("Preview skillset update currently requires an AzureKeyCredential.")

        headers = {
            "Content-Type": "application/json",
            "Prefer": "return=representation",
            "api-key": self.search_info.credential.key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.put(
                url,
                headers=headers,
                data=json.dumps(skillset_payload).encode("utf-8"),
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(
                        f"Failed to create or update preview skillset {skillset_name}: "
                        f"{response.status} {await response.text()}"
                    )

    async def setup(self):
        ds_client = SearchIndexerClient(endpoint=self.search_info.endpoint, credential=self.search_info.credential)

        # Create blob data source
        data_source, data_source_name = await self.create_blob_data_source()
        await ds_client.create_or_update_data_source_connection(data_source)

        # Create skillset
        skillset_payload = await self.create_index_skills()
        skillset_name = skillset_payload["name"]
        await self.create_or_update_preview_skillset(skillset_payload)

        # Create indexer
        indexer, indexer_name = await self.create_indexer(skillset_name, data_source_name)
        await ds_client.create_or_update_indexer(indexer)

        await ds_client.close()

        return indexer_name


async def main(search_info, embeddings):
    search_manager = SearchManager(
        search_info,
        embeddings
    )

    await search_manager.create_index()
    await search_manager.setup()
