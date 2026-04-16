from __future__ import annotations

from collections import OrderedDict

import requests


class LLMError(Exception):
    pass


class EmbeddingError(Exception):
    pass


class RerankError(Exception):
    pass


class AIClient:
    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        embedding_api_key: str,
        embedding_base_url: str,
        embedding_model: str,
        rerank_enabled: bool,
        rerank_api_key: str,
        rerank_base_url: str,
        rerank_model: str,
        timeout: int = 45,
    ) -> None:
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.rerank_enabled = rerank_enabled
        self.rerank_api_key = rerank_api_key
        self.rerank_base_url = rerank_base_url.rstrip("/")
        self.rerank_model = rerank_model
        self.timeout = timeout

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @property
    def embedding_enabled(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_enabled:
            raise EmbeddingError("尚未配置向量化 API Key 或 embedding 模型。")
        if not texts:
            return []

        response = self._post(
            url=f"{self.embedding_base_url}/embeddings",
            api_key=self.embedding_api_key,
            payload={"model": self.embedding_model, "input": texts},
            error_cls=EmbeddingError,
            default_message="向量化请求失败。",
        )

        data = response.get("data") or []
        vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
        if len(vectors) != len(texts):
            raise EmbeddingError("向量化结果数量与输入数量不一致。")
        return vectors

    def rerank_matches(self, question: str, matches: list[dict], top_n: int) -> list[dict]:
        if not self.rerank_enabled or not matches:
            return matches[:top_n]
        if not self.rerank_api_key or not self.rerank_model:
            return matches[:top_n]

        documents = [match["content"] for match in matches]
        try:
            response = self._post(
                url=f"{self.rerank_base_url}/rerank",
                api_key=self.rerank_api_key,
                payload={
                    "model": self.rerank_model,
                    "query": question,
                    "documents": documents,
                    "top_n": top_n,
                },
                error_cls=RerankError,
                default_message="重排序请求失败。",
            )
        except RerankError:
            return matches[:top_n]

        items = response.get("results") or response.get("data") or []
        reranked = []
        for item in items:
            index = item.get("index")
            if index is None or index >= len(matches):
                continue
            match = dict(matches[index])
            match["rerank_score"] = float(item.get("relevance_score") or item.get("score") or 0.0)
            reranked.append(match)

        return reranked or matches[:top_n]

    def generate_answer(self, question: str, matches: list[dict]) -> str:
        if not self.llm_enabled:
            raise LLMError("尚未配置问答模型 API Key 或模型名称。")
        if not matches:
            raise LLMError("没有足够的检索上下文可供回答。")

        context_lines = []
        for index, match in enumerate(matches, start=1):
            context_lines.append(
                f"[Passage {index}] Source: {match['document_name']} {match['position_label']}\n{match['content']}"
            )

        unique_citations = OrderedDict()
        for match in matches:
            key = (match["document_name"], match["position_label"])
            unique_citations[key] = None
        citation_text = "; ".join(
            f"{doc} {position}" for doc, position in unique_citations.keys()
        )
        context_text = "\n\n".join(context_lines)

        system_prompt = (
            "You are an English-only knowledge base assistant for a course design website. "
            "Answer only in English. Use only the provided passages as evidence and do not invent facts. "
            "If the evidence is insufficient, clearly say: 'Based on the current knowledge base, this cannot be determined.' "
            "Write in a clear, formal tone and end with a separate line starting with 'Sources:'."
        )
        user_prompt = (
            f"User question: {question}\n\n"
            f"Knowledge base passages:\n{context_text}\n\n"
            f"Please answer in English only. End with a separate line in this format: Sources: {citation_text}"
        )

        response = self._post(
            url=f"{self.llm_base_url}/chat/completions",
            api_key=self.llm_api_key,
            payload={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            error_cls=LLMError,
            default_message="调用问答模型失败，请稍后重试。",
        )

        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型返回了无法识别的结果。") from exc

    def _post(self, url: str, api_key: str, payload: dict, error_cls, default_message: str) -> dict:
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise error_cls(default_message) from exc

        if response.status_code >= 400:
            message = _extract_error_message(response)
            raise error_cls(f"{default_message} {message}")

        payload = response.json()
        if not isinstance(payload, dict):
            raise error_cls("接口返回了无法识别的数据格式。")
        return payload


def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message") or error.get("type") or str(error)
        if error:
            return str(error)
        message = data.get("message")
        if message:
            return str(message)
    return f"HTTP {response.status_code}"
