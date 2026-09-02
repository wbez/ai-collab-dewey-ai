import argparse
import asyncio
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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
from scripts_content import build_script_chunks, load_script_document


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

    def discover_inputs(
        self, modified_since: Optional[datetime] = None
    ) -> Tuple[List[Path], List[Tuple[Path, Path]], List[Path]]:
        if not self.data_folder.exists():
            print(f"📁 Creating data folder at {self.data_folder}")
            self.data_folder.mkdir(exist_ok=True)
            return [], [], []

        json_files = sorted(self.data_folder.rglob("*.json"))
        vtt_files = sorted(self.data_folder.glob("*.vtt"))
        vtt_stems = {path.stem for path in vtt_files}
        if modified_since is not None:
            def changed_since(path: Path) -> bool:
                return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= modified_since
            json_files = [path for path in json_files if changed_since(path)]
            vtt_files = [path for path in vtt_files if changed_since(path)]
        script_json_files: List[Path] = []
        article_json_files: List[Path] = []
        scripts_folder = self.data_folder / "scripts"
        for path in json_files:
            if path.stem in vtt_stems:
                continue
            if path.parent == scripts_folder:
                script_json_files.append(path)
                continue
            try:
                # Article exports can be large; inspect only a small prefix unless the
                # document lives in the dedicated scripts directory above.
                with path.open(encoding="utf-8") as handle:
                    payload = json.loads(handle.read(4096))
            except Exception:
                article_json_files.append(path)
                continue
            if isinstance(payload, dict) and payload.get("content_type") == "script":
                script_json_files.append(path)
            else:
                article_json_files.append(path)

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
        if script_json_files:
            print(f"📝 Found {len(script_json_files)} script document(s)")
        if not article_json_files and not transcript_pairs and not script_json_files:
            print(f"📁 No supported input files found in {self.data_folder}")

        return article_json_files, transcript_pairs, script_json_files

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

    async def clean_article_content(self, config: Dict[str, str]):
        print("\n🧹 Cleaning article blobs and indexed article chunks...")
        _, _, search_manager = self.create_services(config)
        await self.delete_all_blobs(
            config["AZURE_STORAGE_CONNECTION_STRING"],
            config["AZURE_STORAGE_CONTAINER_NAME"],
        )
        deleted_count = await search_manager.delete_chunks_by_content_type("article")
        print(f"✅ Removed {deleted_count} indexed article chunk(s)")

    async def clean_transcript_content(self, config: Dict[str, str]):
        print("\n🧹 Cleaning indexed transcript chunks and local transcript state...")
        _, _, search_manager = self.create_services(config)
        deleted_count = await search_manager.delete_chunks_by_content_type("transcript")
        self.clear_transcript_state()
        print(f"✅ Removed {deleted_count} indexed transcript chunk(s)")

    async def clean_script_content(self, config: Dict[str, str]):
        print("\n🧹 Cleaning indexed script chunks and local script state...")
        _, _, search_manager = self.create_services(config)
        deleted_count = await search_manager.delete_chunks_by_content_type("script")
        state = self.load_transcript_state()
        state.pop("scripts", None)
        self.save_transcript_state(state)
        print(f"✅ Removed {deleted_count} indexed script chunk(s)")

    def compute_stable_hash(self, payload: Any) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def compute_file_md5(self, path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def compute_transcript_ingest_hash(self, vtt_path: Path, sidecar_path: Path) -> str:
        digest = hashlib.md5()
        digest.update(b"transcript-chunker-v1\n")
        with vtt_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\n--sidecar--\n")
        with sidecar_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def iter_json_documents_from_file(self, file_path: Path) -> Iterator[Any]:
        decoder = json.JSONDecoder()
        chunk_size = 1024 * 1024

        with file_path.open(encoding="utf-8") as handle:
            buffer = ""
            eof = False

            def read_more() -> bool:
                nonlocal buffer, eof
                chunk = handle.read(chunk_size)
                if not chunk:
                    eof = True
                    return False
                buffer += chunk
                return True

            def discard_leading_whitespace() -> None:
                nonlocal buffer
                buffer = buffer.lstrip()

            def decode_value() -> Any:
                nonlocal buffer
                while True:
                    try:
                        document, consumed = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if eof or not read_more():
                            raise
                        continue
                    buffer = buffer[consumed:]
                    return document

            while not buffer and read_more():
                pass
            discard_leading_whitespace()
            if not buffer:
                return

            if buffer[0] != "[":
                document = decode_value()
                discard_leading_whitespace()
                while not buffer and not eof:
                    read_more()
                    discard_leading_whitespace()
                if buffer:
                    raise json.JSONDecodeError("Extra data", buffer, 0)
                yield document
                return

            buffer = buffer[1:]
            expect_value = True
            seen_value = False
            while True:
                discard_leading_whitespace()
                while not buffer and not eof:
                    read_more()
                    discard_leading_whitespace()

                if not buffer:
                    if eof:
                        raise json.JSONDecodeError("Unterminated JSON array", "", 0)
                    continue
                if buffer[0] == "]":
                    if expect_value and seen_value:
                        raise json.JSONDecodeError("Trailing comma in JSON array", buffer, 0)
                    return
                if not expect_value:
                    if buffer[0] != ",":
                        raise json.JSONDecodeError("Expecting ',' delimiter", buffer, 0)
                    buffer = buffer[1:]
                    expect_value = True
                    continue

                if buffer[0] == ",":
                    raise json.JSONDecodeError("Expecting value", buffer, 0)
                yield decode_value()
                expect_value = False
                seen_value = True

    def iter_article_documents(self, json_files: List[Path]) -> Iterator[Dict[str, Any]]:
        required_fields = ["headline", "content", "url", "authors", "publish_date"]

        for file_path in json_files:
            try:
                file_documents = self.iter_json_documents_from_file(file_path)
            except Exception as exc:
                print(f"❌ Error loading {file_path.name}: {exc}")
                continue

            valid_count = 0
            try:
                for index, doc in enumerate(file_documents):
                    if not isinstance(doc, dict):
                        print(
                            f"⚠️  Skipping document {index} in {file_path.name}: "
                            "expected a JSON object"
                        )
                        continue

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
                    normalized_doc["source_id"] = self.compute_stable_hash(normalized_doc)
                    valid_count += 1
                    yield normalized_doc
            except Exception as exc:
                print(f"❌ Error loading {file_path.name}: {exc}")
                continue

            print(f"✅ Loaded {valid_count} valid article document(s) from {file_path.name}")

    def load_transcript_bundle(self, vtt_path: Path, sidecar_path: Path) -> Dict[str, Any]:
        source_hash = self.compute_file_md5(vtt_path)
        transcript_document = load_transcript_document(vtt_path, sidecar_path)
        transcript_chunks = build_transcript_chunks(transcript_document, source_hash=source_hash)
        print(f"✅ Loaded {len(transcript_chunks)} transcript chunk(s) from {vtt_path.name}")
        return {
            "document": transcript_document,
            "chunks": transcript_chunks,
            "source_hash": source_hash,
            "ingest_hash": self.compute_transcript_ingest_hash(vtt_path, sidecar_path),
            "vtt_path": vtt_path,
            "sidecar_path": sidecar_path,
        }

    def load_script_bundle(self, script_path: Path) -> Dict[str, Any]:
        document = load_script_document(script_path)
        return {
            "document": document,
            "chunks": build_script_chunks(document),
            "source_hash": self.compute_stable_hash(document),
        }

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

    async def setup_azure_resources(self, config: Dict[str, str], *, update_existing_indexer: bool = False) -> tuple:
        search_info, embeddings, search_manager = self.create_services(config)

        print("🔧 Creating Azure AI Search index...")
        print(f"   Embedding dimensions: {embeddings.dimensions}")
        await search_manager.create_index()
        print("✅ Index created successfully")

        await self.ensure_blob_container(
            config["AZURE_STORAGE_CONNECTION_STRING"],
            config["AZURE_STORAGE_CONTAINER_NAME"],
        )

        print("⚙️  Checking article skillset and indexer...")
        indexer_name, indexer_updated = await search_manager.setup(
            update_existing_indexer=update_existing_indexer
        )
        if indexer_updated:
            print(f"✅ Skillset and indexer '{indexer_name}' created or updated")
        else:
            print(f"✅ Preserved existing indexer '{indexer_name}'")
        return search_info, embeddings, search_manager, indexer_updated

    async def ensure_blob_container(
        self,
        blob_connection_string: str,
        container_name: str,
    ):
        async with BlobServiceClient.from_connection_string(blob_connection_string) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            try:
                await container_client.create_container()
                print(f"✅ Created container '{container_name}'")
            except Exception:
                pass

    async def upload_documents_to_blob(
        self,
        blob_connection_string: str,
        container_name: str,
        documents: Iterable[Dict[str, Any]],
        total_documents: Optional[int] = None,
    ):
        print("📤 Uploading article document(s) to blob storage...")
        async with BlobServiceClient.from_connection_string(blob_connection_string) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            try:
                await container_client.create_container()
            except Exception:
                pass

            existing_hashes_by_blob_name = await self.load_blob_content_hashes(container_client)
            accepted_content_hashes: set[str] = set()
            success_count = 0
            skipped_count = 0
            failed_count = 0
            progress = tqdm(
                documents,
                total=total_documents,
                desc="Uploading articles",
                unit="doc",
                smoothing=0.6,
            )
            try:
                for index, doc in enumerate(progress):
                    try:
                        blob_name = f"doc_{doc.get('id', index)}.json"
                        blob_data = json.dumps(doc, ensure_ascii=False, indent=2)
                        content_hash = self.compute_stable_hash(doc)
                        if content_hash in accepted_content_hashes:
                            skipped_count += 1
                            progress.set_postfix(
                                uploaded=success_count,
                                skipped=skipped_count,
                                refresh=False,
                            )
                            continue
                        existing_hash = existing_hashes_by_blob_name.get(blob_name)
                        if existing_hash == content_hash:
                            accepted_content_hashes.add(content_hash)
                            skipped_count += 1
                            progress.set_postfix(
                                uploaded=success_count,
                                skipped=skipped_count,
                                refresh=False,
                            )
                            continue
                        blob_client = blob_service_client.get_blob_client(
                            container=container_name,
                            blob=blob_name,
                        )
                        await blob_client.upload_blob(
                            blob_data,
                            overwrite=True,
                            metadata={"content_hash": content_hash},
                        )
                        existing_hashes_by_blob_name[blob_name] = content_hash
                        accepted_content_hashes.add(content_hash)
                        success_count += 1
                        progress.set_postfix(
                            uploaded=success_count,
                            skipped=skipped_count,
                            refresh=False,
                        )
                    except Exception as exc:
                        failed_count += 1
                        print(f"❌ Error uploading article document {index}: {exc}")
            finally:
                progress.close()

            if success_count == 0 and skipped_count == 0 and failed_count == 0:
                print("📄 No article documents to upload")
                return {"uploaded": 0, "skipped": 0, "failed": 0}

            print(
                "✅ Article blob sync complete: "
                f"{success_count} uploaded, {skipped_count} skipped, {failed_count} failed"
            )
            return {"uploaded": success_count, "skipped": skipped_count, "failed": failed_count}

    async def load_blob_content_hashes(self, container_client: Any) -> Dict[str, str]:
        hashes_by_blob_name: Dict[str, str] = {}
        async for blob in container_client.list_blobs(include=["metadata"]):
            metadata = getattr(blob, "metadata", None) or {}
            content_hash = metadata.get("content_hash")
            if content_hash:
                hashes_by_blob_name[blob.name] = content_hash
        return hashes_by_blob_name

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
        transcript_pairs: List[Tuple[Path, Path]],
        state: Dict[str, Dict[str, str]],
    ):
        if not transcript_pairs:
            print("🎙️  No transcript chunks to upload")
            return {"uploaded": 0, "skipped": 0, "deleted": 0, "failed": 0}

        transcript_state = state.setdefault("transcripts", {})
        uploaded_bundle_count = 0
        skipped_bundle_count = 0
        deleted_chunk_count = 0
        failed_bundle_count = 0
        print("📤 Uploading transcript chunk(s) directly to search...")
        progress = tqdm(desc="Uploading transcripts", unit="chunk", smoothing=0.6)

        try:
            for vtt_path, sidecar_path in transcript_pairs:
                try:
                    bundle = self.load_transcript_bundle(vtt_path, sidecar_path)
                except Exception as exc:
                    failed_bundle_count += 1
                    print(f"❌ Error loading transcript {vtt_path.name}: {exc}")
                    continue

                parent_id = bundle["source_hash"]
                state_key = str(bundle["document"]["id"])
                ingest_hash = bundle.get("ingest_hash", bundle["source_hash"])
                previous_state = transcript_state.get(state_key)
                if isinstance(previous_state, dict):
                    previous_ingest_hash = previous_state.get("ingest_hash")
                    previous_parent_id = previous_state.get("parent_id")
                else:
                    previous_ingest_hash = previous_state
                    previous_parent_id = state_key if previous_state is not None else None
                if previous_ingest_hash == ingest_hash:
                    skipped_bundle_count += 1
                    continue

                try:
                    if previous_parent_id and previous_parent_id != parent_id:
                        deleted_chunk_count += await search_manager.delete_transcript_chunks(str(previous_parent_id))
                    deleted_chunk_count += await search_manager.delete_transcript_chunks(parent_id)
                    await search_manager.upload_transcript_chunks(
                        bundle["chunks"],
                        progress_callback=progress.update,
                    )
                except Exception:
                    failed_bundle_count += 1
                    raise

                transcript_state[state_key] = {
                    "ingest_hash": ingest_hash,
                    "parent_id": parent_id,
                }
                self.save_transcript_state(state)
                uploaded_bundle_count += 1
        finally:
            progress.close()
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

    async def upload_script_chunks(
        self, search_manager: SearchManager, script_paths: List[Path], state: Dict[str, Dict[str, str]]
    ):
        script_state = state.setdefault("scripts", {})
        uploaded = skipped = deleted = failed = 0
        for script_path in script_paths:
            try:
                bundle = self.load_script_bundle(script_path)
                parent_id = str(bundle["document"]["id"])
                if script_state.get(parent_id) == bundle["source_hash"]:
                    skipped += 1
                    continue
                deleted += await search_manager.delete_transcript_chunks(parent_id)
                await search_manager.upload_transcript_chunks(bundle["chunks"])
                script_state[parent_id] = bundle["source_hash"]
                self.save_transcript_state(state)
                uploaded += 1
            except Exception as exc:
                failed += 1
                print(f"❌ Error loading script {script_path.name}: {exc}")
        print(f"✅ Script sync complete: {uploaded} uploaded, {skipped} skipped, {deleted} old chunk(s) deleted")
        return {"uploaded": uploaded, "skipped": skipped, "deleted": deleted, "failed": failed}

    async def run_setup(
        self,
        clean: bool = False,
        clean_articles: bool = False,
        clean_transcripts: bool = False,
        clean_scripts: bool = False,
        modified_since: Optional[datetime] = None,
        update_existing_indexer: bool = False,
    ):
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
            else:
                if clean_articles:
                    await self.clean_article_content(config)
                if clean_transcripts:
                    await self.clean_transcript_content(config)
                if clean_scripts:
                    await self.clean_script_content(config)

            article_files, transcript_pairs, script_paths = self.discover_inputs(modified_since)
            transcript_state = self.load_transcript_state()

            print("\n🚀 Setting up Azure AI Search resources...")
            search_info, embeddings, search_manager, indexer_updated = await self.setup_azure_resources(
                config, update_existing_indexer=update_existing_indexer
            )
            indexer_name = f"{config['AZURE_SEARCH_INDEX_NAME']}-indexer"
            article_sync = {"uploaded": 0, "skipped": 0, "failed": 0}
            indexer_status = "not_needed"
            transcript_sync = {"uploaded": 0, "skipped": 0, "deleted": 0, "failed": 0}
            script_sync = {"uploaded": 0, "skipped": 0, "deleted": 0, "failed": 0}

            if article_files:
                article_sync = await self.upload_documents_to_blob(
                    config["AZURE_STORAGE_CONNECTION_STRING"],
                    config["AZURE_STORAGE_CONTAINER_NAME"],
                    self.iter_article_documents(article_files),
                )
                if article_sync["uploaded"] > 0 or (clean and indexer_updated):
                    print("\n🔄 Processing article documents through skillset...")
                    indexer_status = await self.run_indexer(search_info, indexer_name)
                else:
                    indexer_status = "skipped_no_changes"

            if transcript_pairs:
                print("\n🔄 Uploading transcript chunks...")
                transcript_sync = await self.upload_transcript_chunks(
                    search_manager,
                    transcript_pairs,
                    transcript_state,
                )
            if script_paths:
                print("\n🔄 Uploading script chunks...")
                script_sync = await self.upload_script_chunks(search_manager, script_paths, transcript_state)

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
            print(
                "Script sync: "
                f"{script_sync['uploaded']} document(s) uploaded, {script_sync['skipped']} skipped, "
                f"{script_sync['deleted']} old chunk(s) deleted, {script_sync['failed']} failed"
            )
            print("=" * 60)

        except ConfigurationError as exc:
            print(f"❌ Configuration Error:\n{exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"❌ Setup failed: {exc}")
            traceback.print_exc()
            sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="Set up and sync Dewey search content.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing blobs, search index resources, and transcript ingest state before setup.",
    )
    parser.add_argument(
        "--modified-today",
        action="store_true",
        help="Only ingest files modified since 00:00 UTC today.",
    )
    parser.add_argument(
        "--update-existing-indexer",
        action="store_true",
        help="Explicitly replace the existing Azure indexer, data source, and skillset definitions.",
    )
    parser.add_argument(
        "--clean-articles",
        action="store_true",
        help="Delete existing article blobs and indexed article chunks before setup.",
    )
    parser.add_argument(
        "--clean-transcripts",
        action="store_true",
        help="Delete indexed transcript chunks and local transcript ingest state before setup.",
    )
    parser.add_argument(
        "--clean-scripts",
        action="store_true",
        help="Delete indexed script chunks and local script ingest state before setup.",
    )
    args = parser.parse_args()

    setup_manager = SetupManager()
    modified_since = (
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if args.modified_today else None
    )
    await setup_manager.run_setup(
        clean=args.clean,
        clean_articles=args.clean_articles,
        clean_transcripts=args.clean_transcripts,
        clean_scripts=args.clean_scripts,
        modified_since=modified_since,
        update_existing_indexer=args.update_existing_indexer,
    )


if __name__ == "__main__":
    asyncio.run(main())
