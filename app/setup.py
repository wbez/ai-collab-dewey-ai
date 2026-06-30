import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

from models.core import resolve_embedding_dimensions
from setup import EmbeddingService, SearchInfo, SearchManager
from transcripts import (
    build_name_search_text,
    build_transcript_chunks,
    load_transcript_document,
)


class ConfigurationError(Exception):
    pass


class SetupManager:
    def __init__(self):
        self.data_folder = Path(__file__).parent.parent / "data"
        self.env_file = Path(__file__).parent.parent / ".env"
        load_dotenv(self.env_file)

    def validate_configuration(self) -> Dict[str, str]:
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": "Azure OpenAI service endpoint",
            "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
            "EMBEDDING_DEPLOYMENT_NAME": "Azure OpenAI embedding deployment name",
            "EMBEDDING_MODEL_NAME": "Azure OpenAI embedding model name",
            "AZURE_SEARCH_ENDPOINT": "Azure AI Search service endpoint",
            "AZURE_SEARCH_API_KEY": "Azure AI Search API key",
            "AZURE_SEARCH_INDEX_NAME": "Azure AI Search index name",
            "AZURE_STORAGE_CONNECTION_STRING": "Azure Storage connection string",
            "AZURE_STORAGE_CONTAINER_NAME": "Azure Storage container name",
        }

        config: Dict[str, str] = {}
        missing_vars = []
        for var_name, description in required_vars.items():
            value = os.getenv(var_name)
            if not value or not value.strip():
                missing_vars.append(f"  - {var_name}: {description}")
            else:
                config[var_name] = value.strip()

        if missing_vars:
            raise ConfigurationError(
                "Missing required environment variables in .env file:\n"
                + "\n".join(missing_vars)
                + f"\n\nPlease update your .env file at: {self.env_file}"
            )

        configured_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        try:
            config["EMBEDDING_DIMENSIONS"] = str(
                resolve_embedding_dimensions(
                    config["EMBEDDING_MODEL_NAME"],
                    configured_dimensions,
                )
            )
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid EMBEDDING_DIMENSIONS value in .env file. "
                "It must be a positive integer."
            ) from exc

        return config

    def discover_inputs(self) -> Tuple[List[Path], List[Tuple[Path, Path]]]:
        if not self.data_folder.exists():
            print(f"📁 Creating data folder at {self.data_folder}")
            self.data_folder.mkdir(exist_ok=True)
            return [], []

        json_files = sorted(self.data_folder.glob("*.json"))
        vtt_files = sorted(self.data_folder.glob("*.vtt"))
        vtt_stems = {path.stem for path in vtt_files}
        article_json_files = [path for path in json_files if path.stem not in vtt_stems]

        transcript_pairs: List[Tuple[Path, Path]] = []
        for vtt_path in vtt_files:
            sidecar_path = vtt_path.with_suffix(".json")
            if sidecar_path.exists():
                transcript_pairs.append((vtt_path, sidecar_path))
            else:
                print(f"⚠️  Skipping {vtt_path.name}: expected sidecar {sidecar_path.name}")

        if article_json_files:
            print(f"📄 Found {len(article_json_files)} article JSON file(s)")
        if transcript_pairs:
            print(f"🎙️  Found {len(transcript_pairs)} transcript bundle(s)")
        if not article_json_files and not transcript_pairs:
            print(f"📁 No supported input files found in {self.data_folder}")

        return article_json_files, transcript_pairs

    def load_article_documents(self, json_files: List[Path]) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        required_fields = ["headline", "content", "url", "authors", "publish_date"]

        for file_path in json_files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"❌ Error loading {file_path.name}: {exc}")
                continue

            file_documents = data if isinstance(data, list) else [data]
            valid_count = 0
            for index, doc in enumerate(file_documents):
                missing_fields = [
                    field
                    for field in required_fields
                    if field not in doc or doc[field] in ("", None, [])
                ]
                if missing_fields:
                    print(
                        f"⚠️  Skipping document {index} in {file_path.name}: "
                        f"missing required fields: {', '.join(missing_fields)}"
                    )
                    continue

                authors = [str(author).strip() for author in doc.get("authors", []) if str(author).strip()]
                normalized_doc = {
                    **doc,
                    "authors": authors,
                    "content_type": "article",
                    "author_search_text": build_name_search_text(authors),
                    "description": doc.get("description"),
                }
                valid_count += 1
                documents.append(normalized_doc)

            print(f"✅ Loaded {valid_count} valid article document(s) from {file_path.name}")

        return documents

    def load_transcript_documents(self, transcript_pairs: List[Tuple[Path, Path]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for vtt_path, sidecar_path in transcript_pairs:
            try:
                transcript_document = load_transcript_document(vtt_path, sidecar_path)
                transcript_chunks = build_transcript_chunks(transcript_document)
                chunks.extend(transcript_chunks)
                print(
                    f"✅ Loaded {len(transcript_chunks)} transcript chunk(s) from "
                    f"{vtt_path.name}"
                )
            except Exception as exc:
                print(f"❌ Error loading transcript {vtt_path.name}: {exc}")
        return chunks

    async def setup_azure_resources(self, config: Dict[str, str]) -> tuple:
        search_info = SearchInfo(
            endpoint=config["AZURE_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(config["AZURE_SEARCH_API_KEY"]),
            index_name=config["AZURE_SEARCH_INDEX_NAME"],
        )

        embeddings = EmbeddingService(
            endpoint=config["AZURE_OPENAI_ENDPOINT"],
            deployment=config["EMBEDDING_DEPLOYMENT_NAME"],
            model_name=config["EMBEDDING_MODEL_NAME"],
            dimensions=config["EMBEDDING_DIMENSIONS"],
            api_key=config["AZURE_OPENAI_API_KEY"],
        )

        search_manager = SearchManager(
            search_info,
            embeddings,
            config["AZURE_STORAGE_CONNECTION_STRING"],
            config["AZURE_STORAGE_CONTAINER_NAME"],
        )

        print("🔧 Creating Azure AI Search index...")
        print(f"   Embedding dimensions: {embeddings.dimensions}")
        await search_manager.create_index()
        print("✅ Index created successfully")

        print("⚙️  Setting up article skillset and indexer...")
        indexer_name = await search_manager.setup()
        print(f"✅ Skillset and indexer '{indexer_name}' created successfully")
        return search_info, embeddings, search_manager

    async def upload_documents_to_blob(
        self,
        blob_connection_string: str,
        container_name: str,
        documents: List[Dict[str, Any]],
    ):
        if not documents:
            print("📄 No article documents to upload")
            return

        print(f"📤 Uploading {len(documents)} article document(s) to blob storage...")
        async with BlobServiceClient.from_connection_string(blob_connection_string) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            try:
                await container_client.create_container()
                print(f"✅ Created container '{container_name}'")
            except Exception:
                pass

            success_count = 0
            for index, doc in enumerate(documents):
                try:
                    blob_name = f"doc_{doc.get('id', index)}.json"
                    blob_data = json.dumps(doc, ensure_ascii=False, indent=2)
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_name,
                    )
                    await blob_client.upload_blob(blob_data, overwrite=True)
                    success_count += 1
                except Exception as exc:
                    print(f"❌ Error uploading article document {index}: {exc}")

            print(f"✅ Successfully uploaded {success_count} article document(s) to blob storage")

    async def run_indexer(self, search_info: SearchInfo, indexer_name: str):
        print(f"🔄 Running indexer '{indexer_name}' to process article documents...")
        from azure.search.documents.indexes.aio import SearchIndexerClient

        async with SearchIndexerClient(
            endpoint=search_info.endpoint,
            credential=search_info.credential,
        ) as indexer_client:
            try:
                await indexer_client.run_indexer(indexer_name)
                print(f"✅ Indexer '{indexer_name}' started successfully")
            except Exception as exc:
                print(f"❌ Error running indexer: {exc}")

    async def upload_transcript_chunks(
        self,
        search_manager: SearchManager,
        transcript_chunks: List[Dict[str, Any]],
    ):
        if not transcript_chunks:
            print("🎙️  No transcript chunks to upload")
            return
        print(f"📤 Uploading {len(transcript_chunks)} transcript chunk(s) directly to search...")
        await search_manager.upload_transcript_chunks(transcript_chunks)
        print("✅ Transcript chunks uploaded successfully")

    async def run_setup(self):
        print("=" * 60)
        print("🔍 Azure AI Search Setup")
        print("=" * 60)

        try:
            print("🔧 Validating configuration...")
            config = self.validate_configuration()
            print("✅ Configuration validated successfully")

            article_files, transcript_pairs = self.discover_inputs()
            article_documents = self.load_article_documents(article_files) if article_files else []
            transcript_chunks = self.load_transcript_documents(transcript_pairs) if transcript_pairs else []

            print("\n🚀 Setting up Azure AI Search resources...")
            search_info, embeddings, search_manager = await self.setup_azure_resources(config)
            indexer_name = f"{config['AZURE_SEARCH_INDEX_NAME']}-indexer"

            if article_documents:
                await self.upload_documents_to_blob(
                    config["AZURE_STORAGE_CONNECTION_STRING"],
                    config["AZURE_STORAGE_CONTAINER_NAME"],
                    article_documents,
                )
                print("\n🔄 Processing article documents through skillset...")
                await self.run_indexer(search_info, indexer_name)

            if transcript_chunks:
                print("\n🔄 Uploading transcript chunks...")
                await self.upload_transcript_chunks(search_manager, transcript_chunks)

            print("\n🎉 Setup completed successfully!")
            print("=" * 60)
            print("Your Azure AI Search environment is ready to use.")
            print(f"Index: {config['AZURE_SEARCH_INDEX_NAME']}")
            print(f"Article documents indexed: {len(article_documents)}")
            print(f"Transcript chunks indexed: {len(transcript_chunks)}")
            print("=" * 60)

        except ConfigurationError as exc:
            print(f"❌ Configuration Error:\n{exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"❌ Setup failed: {exc}")
            sys.exit(1)


async def main():
    setup_manager = SetupManager()
    await setup_manager.run_setup()


if __name__ == "__main__":
    asyncio.run(main())
