from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStoreError(Exception):
    pass


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str, collection_name: str) -> None:
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self._client: QdrantClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.collection_name)

    @property
    def client(self) -> QdrantClient:
        if not self.enabled:
            raise VectorStoreError("尚未配置 Qdrant Cloud 的连接信息。")
        if self._client is None:
            self._client = QdrantClient(url=self.url, api_key=self.api_key or None)
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.collection_name for item in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list, embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError("向量数量与文本块数量不一致。")

        self.ensure_collection(len(embeddings[0]))
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = chunk.id
            chunk.qdrant_point_id = str(chunk.id)
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": chunk.document.original_name,
                "page_number": chunk.page_number,
                "position_label": chunk.position_label,
                "content": chunk.content,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], limit: int) -> list[dict]:
        self.ensure_collection(len(query_vector))
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        matches = []
        for point in points:
            payload = point.payload or {}
            matches.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "score": float(point.score),
                    "content": payload.get("content", ""),
                    "page_number": payload.get("page_number", 1),
                    "position_label": payload.get("position_label") or f"第 {payload.get('page_number', 1)} 页",
                    "document_name": payload.get("document_name", "未知文档"),
                    "qdrant_point_id": str(point.id),
                }
            )
        return matches

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids or not self.enabled:
            return
        normalized_ids = []
        for point_id in point_ids:
            if point_id is None:
                continue
            if isinstance(point_id, str) and point_id.isdigit():
                normalized_ids.append(int(point_id))
            else:
                normalized_ids.append(point_id)
        if not normalized_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=normalized_ids),
            wait=True,
        )

    def recreate(self, chunks: list, embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
        )
        self.upsert_chunks(chunks, embeddings)
