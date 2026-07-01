import json
import logging
from typing import Callable, Dict, List, Optional

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchVectorizer,
)

from .embedding_service import EmbeddingService
from .search_service import SearchInfo


logger = logging.getLogger("scripts")


class SearchManager:
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

    def resource_names(self) -> Dict[str, str]:
        return {
            "data_source": f"{self.search_info.index_name}-blob-ds",
            "skillset": f"{self.search_info.index_name}-skillset",
            "indexer": f"{self.search_info.index_name}-indexer",
            "index": self.search_info.index_name,
        }

    async def create_index(self, vectorizers: Optional[List[VectorSearchVectorizer]] = None):
        logger.info("Checking whether search index %s exists...", self.search_info.index_name)

        async with self.search_info.create_search_index_client() as search_index_client:
            existing_index_names = [name async for name in search_index_client.list_index_names()]
            if self.search_info.index_name not in existing_index_names:
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
                    SearchableField(name="content", type="Edm.String", analyzer_name="standard.lucene"),
                    SearchableField(name="search_text", type="Edm.String", analyzer_name="standard.lucene"),
                    SearchableField(name="headline", type="Edm.String", analyzer_name="standard.lucene"),
                    SearchableField(name="description", type="Edm.String", analyzer_name="standard.lucene"),
                    SearchableField(
                        name="author_search_text",
                        type="Edm.String",
                        analyzer_name="standard.lucene",
                    ),
                    SearchableField(
                        name="speaker_search_text",
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
                    SimpleField(name="url", type="Edm.String"),
                    SimpleField(
                        name="authors",
                        type="Collection(Edm.String)",
                        filterable=True,
                        facetable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="speakers",
                        type="Collection(Edm.String)",
                        filterable=True,
                        facetable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="guests",
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
                        searchable=False,
                    ),
                    SimpleField(
                        name="recording_date",
                        type="Edm.DateTimeOffset",
                        filterable=True,
                        sortable=True,
                        facetable=True,
                        retrievable=True,
                        searchable=False,
                    ),
                    SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
                    SearchableField(
                        name="parent_id",
                        type="Edm.String",
                        analyzer_name="standard.lucene",
                        filterable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="content_type",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="program",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                        retrievable=True,
                    ),
                    SimpleField(name="transcript_url", type="Edm.String", retrievable=True),
                    SimpleField(
                        name="recording_urls",
                        type="Collection(Edm.String)",
                        retrievable=True,
                    ),
                    SimpleField(name="timestamp_label", type="Edm.String", retrievable=True),
                    SimpleField(
                        name="start_seconds",
                        type="Edm.Double",
                        filterable=True,
                        sortable=True,
                        retrievable=True,
                    ),
                    SimpleField(
                        name="end_seconds",
                        type="Edm.Double",
                        filterable=True,
                        sortable=True,
                        retrievable=True,
                    ),
                    SimpleField(name="raw_metadata_json", type="Edm.String", retrievable=True),
                ]

                vectorizers = vectorizers or [
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
                                    content_fields=[
                                        SemanticField(field_name="search_text"),
                                        SemanticField(field_name="content"),
                                        SemanticField(field_name="description"),
                                    ],
                                    keywords_fields=[
                                        SemanticField(field_name="author_search_text"),
                                        SemanticField(field_name="speaker_search_text"),
                                    ],
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
                                vectorizer_name=f"{self.search_info.index_name}-vectorizer",
                            )
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
                    f"{existing_dimensions}-dimensional vectors, but the configured "
                    f"embedding deployment uses {self.embedding_dimensions}."
                )

    async def create_blob_data_source(self):
        data_source_name = f"{self.search_info.index_name}-blob-ds"
        data_source = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=self.blob_connection_string,
            container=SearchIndexerDataContainer(name=self.blob_container_name),
        )
        return data_source, data_source_name

    async def create_index_skills(self):
        skillset_name = f"{self.search_info.index_name}-skillset"

        split_skill = SplitSkill(
            name=f"{self.search_info.index_name}-split-skill",
            description="Split article documents into chunks",
            text_split_mode="pages",
            context="/document",
            maximum_page_length=512,
            page_overlap_length=96,
            maximum_pages_to_take=0,
            unit="azureOpenAITokens",
            inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
            outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
        )

        text_embedding_skill = AzureOpenAIEmbeddingSkill(
            name=f"{self.search_info.index_name}-text-embedding-skill",
            description="Embedding skill to generate embeddings for article chunks",
            context="/document/pages/*",
            resource_url=self.embeddings.endpoint,
            deployment_name=self.embeddings.deployment,
            api_key=self.embeddings.api_key,
            model_name=self.embeddings.model_name,
            dimensions=self.embedding_dimensions,
            inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
            outputs=[OutputFieldMappingEntry(name="embedding", target_name="content_vector")],
        )

        index_projection = SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=self.search_info.index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(name="content", source="/document/pages/*"),
                        InputFieldMappingEntry(name="search_text", source="/document/pages/*"),
                        InputFieldMappingEntry(name="headline", source="/document/headline"),
                        InputFieldMappingEntry(name="description", source="/document/description"),
                        InputFieldMappingEntry(
                            name="content_vector",
                            source="/document/pages/*/content_vector",
                        ),
                        InputFieldMappingEntry(name="url", source="/document/url"),
                        InputFieldMappingEntry(name="authors", source="/document/authors"),
                        InputFieldMappingEntry(
                            name="author_search_text",
                            source="/document/author_search_text",
                        ),
                        InputFieldMappingEntry(name="publish_date", source="/document/publish_date"),
                        InputFieldMappingEntry(name="sourcepage", source="/document/metadata_storage_name"),
                        InputFieldMappingEntry(name="content_type", source="/document/content_type"),
                    ],
                )
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        return SearchIndexerSkillset(
            name=skillset_name,
            description="Skillset to process article documents and generate embeddings",
            skills=[split_skill, text_embedding_skill],
            index_projection=index_projection,
        )

    async def create_indexer(self, skillset_name: str, data_source_name: str):
        indexer_name = f"{self.search_info.index_name}-indexer"
        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer to process article documents through skillset",
            skillset_name=skillset_name,
            target_index_name=self.search_info.index_name,
            data_source_name=data_source_name,
            field_mappings=[
                FieldMapping(source_field_name="content", target_field_name="content"),
                FieldMapping(source_field_name="headline", target_field_name="headline"),
                FieldMapping(source_field_name="description", target_field_name="description"),
                FieldMapping(source_field_name="url", target_field_name="url"),
                FieldMapping(source_field_name="authors", target_field_name="authors"),
                FieldMapping(
                    source_field_name="author_search_text",
                    target_field_name="author_search_text",
                ),
                FieldMapping(source_field_name="publish_date", target_field_name="publish_date"),
                FieldMapping(source_field_name="content_type", target_field_name="content_type"),
            ],
            parameters={
                "configuration": {
                    "parsingMode": "json",
                    "dataToExtract": "contentAndMetadata",
                }
            },
        )
        return indexer, indexer_name

    async def setup(self):
        ds_client = SearchIndexerClient(
            endpoint=self.search_info.endpoint,
            credential=self.search_info.credential,
        )
        try:
            data_source, data_source_name = await self.create_blob_data_source()
            await ds_client.create_or_update_data_source_connection(data_source)

            embedding_skillset = await self.create_index_skills()
            await ds_client.create_or_update_skillset(embedding_skillset)

            indexer, indexer_name = await self.create_indexer(
                embedding_skillset.name,
                data_source_name,
            )
            await ds_client.create_or_update_indexer(indexer)
            return indexer_name
        finally:
            await ds_client.close()

    async def cleanup_search_resources(self):
        names = self.resource_names()

        async with self.search_info.create_search_indexer_client() as indexer_client:
            for delete_fn, name in [
                (indexer_client.delete_indexer, names["indexer"]),
                (indexer_client.delete_skillset, names["skillset"]),
                (indexer_client.delete_data_source_connection, names["data_source"]),
            ]:
                try:
                    await delete_fn(name)
                except Exception:
                    pass

        async with self.search_info.create_search_index_client() as index_client:
            try:
                await index_client.delete_index(names["index"])
            except Exception:
                pass

    async def list_chunk_ids_by_parent(self, parent_id: str) -> List[str]:
        if not isinstance(self.search_info.credential, AzureKeyCredential):
            raise TypeError("Transcript deletion currently requires an AzureKeyCredential.")

        endpoint = self.search_info.endpoint.rstrip("/")
        url = (
            f"{endpoint}/indexes/{self.search_info.index_name}/docs/search"
            "?api-version=2024-07-01"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self.search_info.credential.key,
        }

        chunk_ids: List[str] = []
        skip = 0
        page_size = 1000
        safe_parent_id = parent_id.replace("'", "''")
        while True:
            payload = {
                "search": "*",
                "filter": f"parent_id eq '{safe_parent_id}'",
                "select": "chunk_id",
                "top": page_size,
                "skip": skip,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload).encode("utf-8"),
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            "Failed to list transcript chunks from Azure Search: "
                            f"{response.status} {await response.text()}"
                        )
                    body = await response.json()

            page_ids = [item["chunk_id"] for item in body.get("value", []) if item.get("chunk_id")]
            chunk_ids.extend(page_ids)
            if len(page_ids) < page_size:
                break
            skip += page_size

        return chunk_ids

    async def delete_transcript_chunks(self, parent_id: str) -> int:
        chunk_ids = await self.list_chunk_ids_by_parent(parent_id)
        if not chunk_ids:
            return 0

        endpoint = self.search_info.endpoint.rstrip("/")
        url = (
            f"{endpoint}/indexes/{self.search_info.index_name}/docs/index"
            "?api-version=2024-07-01"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self.search_info.credential.key,
        }

        batch_size = 500
        async with aiohttp.ClientSession() as session:
            for start in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[start : start + batch_size]
                payload = {
                    "value": [
                        {
                            "@search.action": "delete",
                            "chunk_id": chunk_id,
                        }
                        for chunk_id in batch
                    ]
                }
                async with session.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload).encode("utf-8"),
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            "Failed to delete transcript chunks from Azure Search: "
                            f"{response.status} {await response.text()}"
                        )
        return len(chunk_ids)

    async def upload_transcript_chunks(
        self,
        transcript_chunks: List[Dict[str, object]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ):
        if not transcript_chunks:
            return []

        if not isinstance(self.search_info.credential, AzureKeyCredential):
            raise TypeError("Transcript upload currently requires an AzureKeyCredential.")

        endpoint = self.search_info.endpoint.rstrip("/")
        url = (
            f"{endpoint}/indexes/{self.search_info.index_name}/docs/index"
            "?api-version=2024-07-01"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self.search_info.credential.key,
        }

        batch_size = 500
        async with aiohttp.ClientSession() as session:
            for start in range(0, len(transcript_chunks), batch_size):
                batch = transcript_chunks[start : start + batch_size]
                payload = {
                    "value": [
                        {
                            "@search.action": "upload",
                            **document,
                        }
                        for document in batch
                    ]
                }
                async with session.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload).encode("utf-8"),
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            "Failed to upload transcript chunks to Azure Search: "
                            f"{response.status} {await response.text()}"
                        )

                    response_payload = await response.json()
                    failed_items = [
                        item
                        for item in response_payload.get("value", [])
                        if not item.get("status", False)
                    ]
                    if failed_items:
                        first_error = failed_items[0]
                        raise RuntimeError(
                            "Failed to upload transcript chunk batch to Azure Search: "
                            f"{first_error.get('errorMessage', 'unknown error')}"
                        )
                if progress_callback is not None:
                    progress_callback(len(batch))
        return transcript_chunks
