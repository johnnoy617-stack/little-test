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


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = _build_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(INSTANCE_DIR / "uploads"))
    INDEX_DIR = os.getenv("INDEX_DIR", str(INSTANCE_DIR / "index"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.08"))

    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL = os.getenv(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
    )
    SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
