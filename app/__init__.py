from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from app.config import Config
from app.extensions import db
from app.routes import register_routes
from app.services.llm import SiliconFlowClient
from app.services.retrieval import TfidfRetriever


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    _ensure_runtime_directories(app)

    db.init_app(app)

    with app.app_context():
        from app import models  # noqa: F401

        db.create_all()

    app.extensions["retriever"] = TfidfRetriever(app.config["INDEX_DIR"])
    app.extensions["llm_client"] = SiliconFlowClient(
        api_key=app.config["SILICONFLOW_API_KEY"],
        base_url=app.config["SILICONFLOW_BASE_URL"],
        model=app.config["SILICONFLOW_MODEL"],
        timeout=app.config["REQUEST_TIMEOUT"],
    )

    register_routes(app)
    return app


def _ensure_runtime_directories(app: Flask) -> None:
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["INDEX_DIR"]).mkdir(parents=True, exist_ok=True)
