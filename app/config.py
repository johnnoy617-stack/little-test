import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"


def _build_database_uri() -> str:
    configured = os.getenv("DATABASE_URL", "").strip()
    if not configured:
        return f"sqlite:///{(INSTANCE_DIR / 'app.db').as_posix()}"

    if configured.startswith("sqlite:///") and not configured.startswith("sqlite:////"):
        raw_path = configured.removeprefix("sqlite:///")
        path = Path(raw_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return f"sqlite:///{path.resolve().as_posix()}"

    return configured


def _pick_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _pick_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = _build_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(INSTANCE_DIR / "uploads"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "8"))
    TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
    VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.25"))

    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "course_kb")

    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL = os.getenv(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
    )

    LLM_API_KEY = _pick_env("LLM_API_KEY", "SILICONFLOW_API_KEY")
    LLM_BASE_URL = _pick_env("LLM_BASE_URL", "SILICONFLOW_BASE_URL", default=SILICONFLOW_BASE_URL)
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    EMBEDDING_API_KEY = _pick_env("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY")
    EMBEDDING_BASE_URL = _pick_env(
        "EMBEDDING_BASE_URL", "SILICONFLOW_BASE_URL", default=SILICONFLOW_BASE_URL
    )
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    RERANK_ENABLED = _pick_bool("RERANK_ENABLED", False)
    RERANK_API_KEY = _pick_env("RERANK_API_KEY", "EMBEDDING_API_KEY", "SILICONFLOW_API_KEY")
    RERANK_BASE_URL = _pick_env(
        "RERANK_BASE_URL", "EMBEDDING_BASE_URL", "SILICONFLOW_BASE_URL", default=SILICONFLOW_BASE_URL
    )
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))

    SITE_TITLE = os.getenv("SITE_TITLE", "HKU Knowledge Companion")
    SITE_SUBTITLE = os.getenv("SITE_SUBTITLE", "Course Design Demonstration")
