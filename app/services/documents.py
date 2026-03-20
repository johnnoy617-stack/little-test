import re
import uuid
from pathlib import Path

from pypdf import PdfReader
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import Chunk, Document


class DocumentProcessingError(Exception):
    pass


def ingest_pdf(app, uploaded_file: FileStorage) -> Document:
    if not uploaded_file or not uploaded_file.filename:
        raise DocumentProcessingError("请选择要上传的 PDF 文件。")

    original_name = uploaded_file.filename
    if not original_name.lower().endswith(".pdf"):
        raise DocumentProcessingError("只支持 PDF 文件上传。")

    safe_name = secure_filename(original_name) or "document.pdf"
    generated_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = Path(app.config["UPLOAD_DIR"]) / generated_name

    uploaded_file.save(file_path)

    try:
        pages = extract_pdf_pages(file_path)
        if not pages:
            raise DocumentProcessingError("PDF 没有可读取的文本内容。")

        chunks = chunk_pages(
            pages,
            chunk_size=app.config["CHUNK_SIZE"],
            overlap=app.config["CHUNK_OVERLAP"],
        )
        if not chunks:
            raise DocumentProcessingError("未提取到文本内容，可能是扫描版 PDF。")

        document = Document(
            filename=generated_name,
            original_name=original_name,
            page_count=len(pages),
            status="ready",
        )
        db.session.add(document)
        db.session.flush()

        for chunk in chunks:
            db.session.add(
                Chunk(
                    document_id=document.id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    char_count=len(chunk["content"]),
                )
            )

        db.session.commit()
        return document
    except Exception:
        db.session.rollback()
        if file_path.exists():
            file_path.unlink()
        raise


def delete_document(app, document: Document) -> None:
    file_path = Path(app.config["UPLOAD_DIR"]) / document.filename
    db.session.delete(document)
    db.session.commit()
    if file_path.exists():
        file_path.unlink()


def extract_pdf_pages(file_path: Path) -> list[dict]:
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise DocumentProcessingError("PDF 文件无法解析，请确认文件未损坏。") from exc

    pages: list[dict] = []
    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = normalize_text(raw_text)
        if cleaned_text:
            pages.append({"page_number": page_index, "text": cleaned_text})

    return pages


def chunk_pages(
    pages: list[dict], chunk_size: int = 700, overlap: int = 120
) -> list[dict]:
    chunks: list[dict] = []
    effective_overlap = min(overlap, max(0, chunk_size - 50))

    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(
                    {
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "content": content,
                    }
                )
                chunk_index += 1

            if end >= len(text):
                break

            start = max(end - effective_overlap, start + 1)

    return chunks


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
