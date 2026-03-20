import json
from pathlib import Path

import joblib
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.models import Chunk


class TfidfRetriever:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.vectorizer_path = self.index_dir / "vectorizer.joblib"
        self.matrix_path = self.index_dir / "chunk_matrix.npz"
        self.chunk_ids_path = self.index_dir / "chunk_ids.json"
        self.vectorizer = None
        self.matrix = None
        self.chunk_ids: list[int] = []

    def load(self) -> bool:
        if not (
            self.vectorizer_path.exists()
            and self.matrix_path.exists()
            and self.chunk_ids_path.exists()
        ):
            self.clear_memory()
            return False

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.matrix = load_npz(self.matrix_path)
        self.chunk_ids = json.loads(self.chunk_ids_path.read_text(encoding="utf-8"))
        return True

    def ensure_loaded(self) -> bool:
        if self.vectorizer is not None and self.matrix is not None and self.chunk_ids:
            return True
        return self.load()

    def build(self, chunks: list[Chunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not chunks:
            self.clear_disk()
            self.clear_memory()
            return

        corpus = [chunk.content for chunk in chunks]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        chunk_ids = [chunk.id for chunk in chunks]

        joblib.dump(vectorizer, self.vectorizer_path)
        save_npz(self.matrix_path, matrix)
        self.chunk_ids_path.write_text(
            json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8"
        )

        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunk_ids = chunk_ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        if not self.ensure_loaded():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        if scores.size == 0:
            return []

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({"chunk_id": self.chunk_ids[int(idx)], "score": score})
        return results

    def clear_disk(self) -> None:
        for path in [self.vectorizer_path, self.matrix_path, self.chunk_ids_path]:
            if path.exists():
                path.unlink()

    def clear_memory(self) -> None:
        self.vectorizer = None
        self.matrix = None
        self.chunk_ids = []
