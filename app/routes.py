from pathlib import Path

from flask import current_app, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from app.extensions import db
from app.models import Chunk, Document
from app.services.documents import (
    DocumentProcessingError,
    delete_document_file,
    delete_document_record,
    finalize_document,
    mark_document_failed,
    save_and_parse_document,
)
from app.services.llm import EmbeddingError, LLMError
from app.services.vector_store import VectorStoreError


NOT_ENOUGH_EVIDENCE_ANSWER = "There is not enough evidence in the current knowledge base to answer this question."


def register_routes(app):
    @app.get("/")
    def index():
        return render_template("index.html", stats=_build_stats())

    @app.post("/ask")
    def ask():
        question = request.form.get("question", "").strip()
        stats = _build_stats()
        result = _default_result(question)

        if not question:
            result["error"] = "请输入问题后再提交。"
            return render_template("index.html", stats=stats, result=result), 400
        if len(question) > 1000:
            result["error"] = "问题长度不能超过 1000 个字符。"
            return render_template("index.html", stats=stats, result=result), 400

        ai_client = current_app.extensions["ai_client"]
        vector_store = current_app.extensions["vector_store"]

        try:
            query_vector = ai_client.embed_texts([question])[0]
            matches = vector_store.search(
                query_vector=query_vector,
                limit=current_app.config["TOP_K_INITIAL"],
            )
        except (EmbeddingError, VectorStoreError) as exc:
            result["error"] = str(exc)
            return render_template("index.html", stats=stats, result=result), 500

        if not matches:
            result["answer"] = NOT_ENOUGH_EVIDENCE_ANSWER
            return render_template("index.html", stats=stats, result=result)

        filtered = [m for m in matches if m["score"] >= current_app.config["VECTOR_SCORE_THRESHOLD"]]
        if not filtered:
            result["answer"] = NOT_ENOUGH_EVIDENCE_ANSWER
            result["matched_chunks"] = matches[: current_app.config["TOP_K_FINAL"]]
            result["citations"] = _build_citations(result["matched_chunks"])
            return render_template("index.html", stats=stats, result=result)

        reranked = ai_client.rerank_matches(
            question,
            filtered,
            top_n=current_app.config["TOP_K_FINAL"],
        )

        try:
            answer = ai_client.generate_answer(question, reranked)
            result["answer"] = answer
            result["has_enough_context"] = True
        except LLMError as exc:
            result["error"] = str(exc)

        result["matched_chunks"] = reranked
        result["citations"] = _build_citations(reranked)
        return render_template("index.html", stats=stats, result=result)

    @app.get("/admin")
    def admin():
        return render_template("admin.html", documents=_list_documents(), stats=_build_stats())

    @app.post("/admin/upload")
    def upload():
        uploaded_file = request.files.get("knowledge_file")
        try:
            document, chunks = save_and_parse_document(current_app, uploaded_file)
            ai_client = current_app.extensions["ai_client"]
            vector_store = current_app.extensions["vector_store"]
            embeddings = ai_client.embed_texts([chunk.content for chunk in chunks])
            vector_store.upsert_chunks(chunks, embeddings)
            finalize_document(document, chunks, ai_client.embedding_model)
            flash("文件上传、分块和向量入库已完成。", "success")
        except DocumentProcessingError as exc:
            flash(str(exc), "error")
        except (EmbeddingError, VectorStoreError, LLMError) as exc:
            if "document" in locals() and getattr(document, "id", None):
                mark_document_failed(document, str(exc))
            flash(str(exc), "error")
        except Exception as exc:
            if "document" in locals() and getattr(document, "id", None):
                mark_document_failed(document, str(exc))
            flash(f"上传失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.post("/admin/delete/<int:document_id>")
    def remove_document(document_id: int):
        document = Document.query.get_or_404(document_id)
        point_ids = [chunk.qdrant_point_id for chunk in document.chunks if chunk.qdrant_point_id]
        try:
            current_app.extensions["vector_store"].delete_points(point_ids)
            delete_document_file(document)
            delete_document_record(document)
            flash("文档与向量记录已删除。", "success")
        except Exception as exc:
            flash(f"删除失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.post("/admin/reindex")
    def reindex():
        try:
            _rebuild_vector_index()
            flash("已根据当前知识库重新写入 Qdrant。", "success")
        except Exception as exc:
            flash(f"重建失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(_error):
        flash("上传文件超过大小限制，请控制在 10MB 以内。", "error")
        return redirect(url_for("admin"))

    @app.context_processor
    def inject_site_meta():
        return {
            "site_title": current_app.config["SITE_TITLE"],
            "site_subtitle": current_app.config["SITE_SUBTITLE"],
        }


def _rebuild_vector_index() -> None:
    ai_client = current_app.extensions["ai_client"]
    vector_store = current_app.extensions["vector_store"]
    chunks = (
        Chunk.query.join(Document)
        .order_by(Document.uploaded_at.asc(), Chunk.page_number.asc(), Chunk.chunk_index.asc())
        .all()
    )
    if not chunks:
        return
    embeddings = ai_client.embed_texts([chunk.content for chunk in chunks])
    vector_store.recreate(chunks, embeddings)
    for chunk in chunks:
        chunk.qdrant_point_id = str(chunk.id)
        chunk.embedding_model = ai_client.embedding_model
    for document in Document.query.all():
        document.status = "ready"
        document.error_message = None
        document.chunk_count = len(document.chunks)
    db.session.commit()


def _list_documents() -> list[Document]:
    return Document.query.order_by(Document.uploaded_at.desc()).all()


def _build_stats() -> dict:
    documents_count = db.session.query(db.func.count(Document.id)).scalar() or 0
    chunks_count = db.session.query(db.func.count(Chunk.id)).scalar() or 0
    ready_count = db.session.query(db.func.count(Document.id)).filter(Document.status == "ready").scalar() or 0
    return {
        "documents_count": documents_count,
        "chunks_count": chunks_count,
        "ready_count": ready_count,
        "vector_provider": "Qdrant Cloud",
        "embedding_model": current_app.config["EMBEDDING_MODEL"],
        "llm_model": current_app.config["LLM_MODEL"],
        "upload_dir": Path(current_app.config["UPLOAD_DIR"]).name,
    }


def _build_citations(matches: list[dict]) -> list[str]:
    seen = set()
    citations = []
    for match in matches:
        key = (match["document_name"], match["position_label"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(f"{match['document_name']} {match['position_label']}")
    return citations


def _default_result(question: str) -> dict:
    return {
        "question": question,
        "answer": "",
        "error": None,
        "matched_chunks": [],
        "citations": [],
        "has_enough_context": False,
    }
