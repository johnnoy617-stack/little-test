from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import inspect, text

from app.config import Config
from app.extensions import db
from app.routes import register_routes
from app.services.llm import AIClient
from app.services.vector_store import QdrantVectorStore


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    _ensure_runtime_directories(app)

    db.init_app(app)

    with app.app_context():
        from app import models  # noqa: F401

        db.create_all()
        _migrate_legacy_schema()

    app.extensions["vector_store"] = QdrantVectorStore(
        url=app.config["QDRANT_URL"],
        api_key=app.config["QDRANT_API_KEY"],
        collection_name=app.config["QDRANT_COLLECTION"],
    )
    app.extensions["ai_client"] = AIClient(
        llm_api_key=app.config["LLM_API_KEY"],
        llm_base_url=app.config["LLM_BASE_URL"],
        llm_model=app.config["LLM_MODEL"],
        embedding_api_key=app.config["EMBEDDING_API_KEY"],
        embedding_base_url=app.config["EMBEDDING_BASE_URL"],
        embedding_model=app.config["EMBEDDING_MODEL"],
        rerank_enabled=app.config["RERANK_ENABLED"],
        rerank_api_key=app.config["RERANK_API_KEY"],
        rerank_base_url=app.config["RERANK_BASE_URL"],
        rerank_model=app.config["RERANK_MODEL"],
        timeout=app.config["REQUEST_TIMEOUT"],
    )

    register_routes(app)
    return app


def _ensure_runtime_directories(app: Flask) -> None:
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)


def _migrate_legacy_schema() -> None:
    inspector = inspect(db.engine)

    if inspector.has_table("documents"):
        document_columns = {column["name"] for column in inspector.get_columns("documents")}
        _maybe_add_column("documents", "file_type", "TEXT DEFAULT 'pdf'") if "file_type" not in document_columns else None
        _maybe_add_column("documents", "storage_path", "TEXT") if "storage_path" not in document_columns else None
        _maybe_add_column("documents", "chunk_count", "INTEGER DEFAULT 0") if "chunk_count" not in document_columns else None
        _maybe_add_column("documents", "error_message", "TEXT") if "error_message" not in document_columns else None

    if inspector.has_table("chunks"):
        chunk_columns = {column["name"] for column in inspector.get_columns("chunks")}
        _maybe_add_column("chunks", "position_label", "TEXT") if "position_label" not in chunk_columns else None
        _maybe_add_column("chunks", "qdrant_point_id", "TEXT") if "qdrant_point_id" not in chunk_columns else None
        _maybe_add_column("chunks", "embedding_model", "TEXT") if "embedding_model" not in chunk_columns else None


def _maybe_add_column(table_name: str, column_name: str, column_sql: str) -> None:
    db.session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))
    db.session.commit()
