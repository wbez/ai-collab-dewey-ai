import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

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
        self.transcript_state_path = self.data_folder / ".ingest-state.json"
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

    def load_transcript_state(self) -> Dict[str, Dict[str, str]]:
        if not self.transcript_state_path.exists():
            return {"transcripts": {}}
        try:
            return json.loads(self.transcript_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"transcripts": {}}

    def save_transcript_state(self, state: Dict[str, Dict[str, str]]):
        self.transcript_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def clear_transcript_state(self):
        if self.transcript_state_path.exists():
            self.transcript_state_path.unlink()

    def compute_stable_hash(self, payload: Any) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_transcript_source_hash(self, vtt_path: Path, sidecar_path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(b"transcript-chunker-v1\n")
        digest.update(vtt_path.read_bytes())
        digest.update(b"\n--sidecar--\n")
        digest.update(sidecar_path.read_bytes())
        return digest.hexdigest()

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
        bundles: List[Dict[str, Any]] = []
        for vtt_path, sidecar_path in transcript_pairs:
            try:
                transcript_document = load_transcript_document(vtt_path, sidecar_path)
                transcript_chunks = build_transcript_chunks(transcript_document)
                print(
                    f"✅ Loaded {len(transcript_chunks)} transcript chunk(s) from "
                    f"{vtt_path.name}"
                )
                bundles.append(
                    {
                        "document": transcript_document,
                        "chunks": transcript_chunks,
                        "source_hash": self.compute_transcript_source_hash(vtt_path, sidecar_path),
                        "vtt_path": vtt_path,
                        "sidecar_path": sidecar_path,
                    }
                )
            except Exception as exc:
                print(f"❌ Error loading transcript {vtt_path.name}: {exc}")
        return bundles

    def create_services(self, config: Dict[str, str]):
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
        return search_info, embeddings, search_manager

    async def setup_azure_resources(self, config: Dict[str, str]) -> tuple:
        search_info, embeddings, search_manager = self.create_services(config)

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
            return {"uploaded": 0, "skipped": 0, "failed": 0}

        print(f"📤 Uploading {len(documents)} article document(s) to blob storage...")
        async with BlobServiceClient.from_connection_string(blob_connection_string) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            try:
                await container_client.create_container()
                print(f"✅ Created container '{container_name}'")
            except Exception:
                pass

            success_count = 0
            skipped_count = 0
            failed_count = 0
            progress = tqdm(
                enumerate(documents),
                total=len(documents),
                desc="Uploading articles",
                unit="doc",
                smoothing=0.6
            )
            for index, doc in progress:
                try:
                    blob_name = f"doc_{doc.get('id', index)}.json"
                    blob_data = json.dumps(doc, ensure_ascii=False, indent=2)
                    content_hash = self.compute_stable_hash(doc)
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_name,
                    )
                    existing_hash = None
                    if await blob_client.exists():
                        properties = await blob_client.get_blob_properties()
                        existing_hash = (properties.metadata or {}).get("content_hash")
                    if existing_hash == content_hash:
                        skipped_count += 1
                        progress.set_postfix(
                            uploaded=success_count,
                            skipped=skipped_count,
                            refresh=False,
                        )
                        continue
                    await blob_client.upload_blob(
                        blob_data,
                        overwrite=True,
                        metadata={"content_hash": content_hash},
                    )
                    success_count += 1
                    progress.set_postfix(
                        uploaded=success_count,
                        skipped=skipped_count,
                        refresh=False,
                    )
                except Exception as exc:
                    failed_count += 1
                    print(f"❌ Error uploading article document {index}: {exc}")

            print(
                "✅ Article blob sync complete: "
                f"{success_count} uploaded, {skipped_count} skipped, {failed_count} failed"
            )
            return {"uploaded": success_count, "skipped": skipped_count, "failed": failed_count}

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
                return "started"
            except Exception as exc:
                if "Another indexer invocation is currently in progress" in str(exc):
                    print(f"⚠️  Indexer '{indexer_name}' is already running")
                    return "already_running"
                print(f"❌ Error running indexer: {exc}")
                return "failed"

    async def delete_all_blobs(self, blob_connection_string: str, container_name: str):
        async with BlobServiceClient.from_connection_string(blob_connection_string) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            try:
                await container_client.delete_container()
                print(f"🧹 Deleted blob container '{container_name}'")
            except Exception:
                print(f"🧹 Blob container '{container_name}' was already empty or missing")

    async def upload_transcript_chunks(
        self,
        search_manager: SearchManager,
        transcript_bundles: List[Dict[str, Any]],
        state: Dict[str, Dict[str, str]],
    ):
        if not transcript_bundles:
            print("🎙️  No transcript chunks to upload")
            return {"uploaded": 0, "skipped": 0, "deleted": 0, "failed": 0}

        transcript_state = state.setdefault("transcripts", {})
        bundles_to_upload = []
        uploaded_bundle_count = 0
        skipped_bundle_count = 0
        deleted_chunk_count = 0
        failed_bundle_count = 0

        for bundle in transcript_bundles:
            parent_id = bundle["document"]["id"]
            source_hash = bundle["source_hash"]
            if transcript_state.get(parent_id) == source_hash:
                skipped_bundle_count += 1
                continue
            bundles_to_upload.append(bundle)

        total_chunk_count = sum(len(bundle["chunks"]) for bundle in bundles_to_upload)
        if not bundles_to_upload:
            print(f"🎙️  Transcript sync complete: 0 uploaded, {skipped_bundle_count} skipped")
            return {"uploaded": 0, "skipped": skipped_bundle_count, "deleted": 0, "failed": 0}

        print(f"📤 Uploading {total_chunk_count} transcript chunk(s) directly to search...")
        progress = tqdm(total=total_chunk_count, desc="Uploading transcripts", unit="chunk", smoothing=0.6)

        try:
            for bundle in bundles_to_upload:
                parent_id = bundle["document"]["id"]
                deleted_chunk_count += await search_manager.delete_transcript_chunks(parent_id)
                await search_manager.upload_transcript_chunks(
                    bundle["chunks"],
                    progress_callback=progress.update,
                )
                transcript_state[parent_id] = bundle["source_hash"]
                uploaded_bundle_count += 1
        except Exception:
            failed_bundle_count += 1
            raise
        finally:
            progress.close()

        self.save_transcript_state(state)
        print(
            "✅ Transcript sync complete: "
            f"{uploaded_bundle_count} bundle(s) uploaded, "
            f"{skipped_bundle_count} skipped, "
            f"{deleted_chunk_count} old chunk(s) deleted"
        )
        return {
            "uploaded": uploaded_bundle_count,
            "skipped": skipped_bundle_count,
            "deleted": deleted_chunk_count,
            "failed": failed_bundle_count,
        }

    async def run_setup(self, clean: bool = False):
        print("=" * 60)
        print("🔍 Azure AI Search Setup")
        print("=" * 60)

        try:
            print("🔧 Validating configuration...")
            config = self.validate_configuration()
            print("✅ Configuration validated successfully")

            if clean:
                print("\n🧹 Cleaning existing Azure resources and local transcript state...")
                _, _, clean_search_manager = self.create_services(config)
                await self.delete_all_blobs(
                    config["AZURE_STORAGE_CONNECTION_STRING"],
                    config["AZURE_STORAGE_CONTAINER_NAME"],
                )
                await clean_search_manager.cleanup_search_resources()
                self.clear_transcript_state()
                print("✅ Clean completed")

            article_files, transcript_pairs = self.discover_inputs()
            article_documents = self.load_article_documents(article_files) if article_files else []
            transcript_bundles = self.load_transcript_documents(transcript_pairs) if transcript_pairs else []
            transcript_state = self.load_transcript_state()

            print("\n🚀 Setting up Azure AI Search resources...")
            search_info, embeddings, search_manager = await self.setup_azure_resources(config)
            indexer_name = f"{config['AZURE_SEARCH_INDEX_NAME']}-indexer"
            article_sync = {"uploaded": 0, "skipped": 0, "failed": 0}
            indexer_status = "not_needed"
            transcript_sync = {"uploaded": 0, "skipped": 0, "deleted": 0, "failed": 0}

            if article_documents:
                article_sync = await self.upload_documents_to_blob(
                    config["AZURE_STORAGE_CONNECTION_STRING"],
                    config["AZURE_STORAGE_CONTAINER_NAME"],
                    article_documents,
                )
                if article_sync["uploaded"] > 0 or clean:
                    print("\n🔄 Processing article documents through skillset...")
                    indexer_status = await self.run_indexer(search_info, indexer_name)
                else:
                    indexer_status = "skipped_no_changes"

            if transcript_bundles:
                print("\n🔄 Uploading transcript chunks...")
                transcript_sync = await self.upload_transcript_chunks(
                    search_manager,
                    transcript_bundles,
                    transcript_state,
                )

            print("\n🎉 Setup completed successfully!")
            print("=" * 60)
            print("Your Azure AI Search environment is ready to use.")
            print(f"Index: {config['AZURE_SEARCH_INDEX_NAME']}")
            print(
                "Article blob sync: "
                f"{article_sync['uploaded']} uploaded, {article_sync['skipped']} skipped, "
                f"{article_sync['failed']} failed"
            )
            print(f"Article indexer status: {indexer_status}")
            print(
                "Transcript sync: "
                f"{transcript_sync['uploaded']} bundle(s) uploaded, "
                f"{transcript_sync['skipped']} skipped, "
                f"{transcript_sync['deleted']} old chunk(s) deleted, "
                f"{transcript_sync['failed']} failed"
            )
            print("=" * 60)

        except ConfigurationError as exc:
            print(f"❌ Configuration Error:\n{exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"❌ Setup failed: {exc}")
            sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="Set up and sync Dewey search content.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing blobs, search index resources, and transcript ingest state before setup.",
    )
    args = parser.parse_args()

    setup_manager = SetupManager()
    await setup_manager.run_setup(clean=args.clean)


if __name__ == "__main__":
    asyncio.run(main())
