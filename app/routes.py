from pathlib import Path

from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from app.extensions import db
from app.models import Chunk, Document
from app.services.documents import (
    DocumentProcessingError,
    delete_document,
    ingest_pdf,
)
from app.services.llm import LLMError


def register_routes(app):
    @app.get("/")
    def index():
        stats = _build_stats()
        return render_template("index.html", stats=stats)

    @app.post("/ask")
    def ask():
        question = request.form.get("question", "").strip()
        stats = _build_stats()
        result = {
            "question": question,
            "answer": "",
            "error": None,
            "matched_chunks": [],
            "citations": [],
            "has_enough_context": False,
        }

        if not question:
            result["error"] = "请输入问题后再提交。"
            return render_template("index.html", stats=stats, result=result), 400

        retriever = current_app.extensions["retriever"]
        ranked_chunks = retriever.search(
            question,
            top_k=current_app.config["TOP_K"],
        )
        if not ranked_chunks:
            result["answer"] = "当前知识库中没有足够依据来回答这个问题。"
            return render_template("index.html", stats=stats, result=result)

        chunk_ids = [item["chunk_id"] for item in ranked_chunks]
        chunk_map = {chunk.id: chunk for chunk in Chunk.query.filter(Chunk.id.in_(chunk_ids)).all()}
        matches = []
        for item in ranked_chunks:
            chunk = chunk_map.get(item["chunk_id"])
            if not chunk:
                continue
            matches.append(
                {
                    "chunk_id": chunk.id,
                    "score": round(item["score"], 4),
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "document_name": chunk.document.original_name,
                }
            )

        if not matches or matches[0]["score"] < current_app.config["SCORE_THRESHOLD"]:
            result["answer"] = "当前知识库中没有足够依据来回答这个问题。"
            result["matched_chunks"] = matches
            result["citations"] = _build_citations(matches)
            return render_template("index.html", stats=stats, result=result)

        llm_client = current_app.extensions["llm_client"]
        try:
            answer = llm_client.generate_answer(question, matches)
            result["answer"] = answer
            result["has_enough_context"] = True
        except LLMError as exc:
            result["error"] = str(exc)

        result["matched_chunks"] = matches
        result["citations"] = _build_citations(matches)
        return render_template("index.html", stats=stats, result=result)

    @app.get("/admin")
    def admin():
        return render_template("admin.html", documents=_list_documents())

    @app.post("/admin/upload")
    def upload():
        uploaded_file = request.files.get("pdf_file")
        try:
            ingest_pdf(current_app, uploaded_file)
            rebuild_index()
            flash("PDF 上传并建库完成。", "success")
        except DocumentProcessingError as exc:
            flash(str(exc), "error")
        except Exception as exc:
            flash(f"上传失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.post("/admin/delete/<int:document_id>")
    def remove_document(document_id: int):
        document = Document.query.get_or_404(document_id)
        try:
            delete_document(current_app, document)
            rebuild_index()
            flash("文档已删除，索引已更新。", "success")
        except Exception as exc:
            flash(f"删除失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.post("/admin/reindex")
    def reindex():
        try:
            rebuild_index()
            flash("索引重建完成。", "success")
        except Exception as exc:
            flash(f"索引重建失败：{exc}", "error")
        return redirect(url_for("admin"))

    @app.context_processor
    def inject_admin_path():
        return {"admin_path": "/admin"}


def rebuild_index() -> None:
    retriever = current_app.extensions["retriever"]
    chunks = (
        Chunk.query.join(Document)
        .order_by(Document.uploaded_at.asc(), Chunk.page_number.asc(), Chunk.chunk_index.asc())
        .all()
    )
    retriever.build(chunks)


def _list_documents() -> list[Document]:
    return Document.query.order_by(Document.uploaded_at.desc()).all()


def _build_stats() -> dict:
    documents_count = db.session.query(db.func.count(Document.id)).scalar() or 0
    chunks_count = db.session.query(db.func.count(Chunk.id)).scalar() or 0
    index_ready = current_app.extensions["retriever"].ensure_loaded()
    return {
        "documents_count": documents_count,
        "chunks_count": chunks_count,
        "index_ready": index_ready,
        "model_name": current_app.config["SILICONFLOW_MODEL"],
    }


def _build_citations(matches: list[dict]) -> list[str]:
    seen = set()
    citations = []
    for match in matches:
        key = (match["document_name"], match["page_number"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(f"{match['document_name']} 第 {match['page_number']} 页")
    return citations
