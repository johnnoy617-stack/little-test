from datetime import datetime

from app.extensions import db


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    original_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), default="pdf", nullable=False)
    storage_path = db.Column(db.String(500), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    page_count = db.Column(db.Integer, default=0, nullable=False)
    chunk_count = db.Column(db.Integer, default=0, nullable=False)
    status = db.Column(db.String(50), default="ready", nullable=False)
    error_message = db.Column(db.Text, nullable=True)

    chunks = db.relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Chunk(db.Model):
    __tablename__ = "chunks"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.Integer, db.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number = db.Column(db.Integer, nullable=False, default=1)
    position_label = db.Column(db.String(100), nullable=True)
    chunk_index = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    char_count = db.Column(db.Integer, nullable=False)
    qdrant_point_id = db.Column(db.String(100), nullable=True, unique=True)
    embedding_model = db.Column(db.String(255), nullable=True)

    document = db.relationship("Document", back_populates="chunks")
