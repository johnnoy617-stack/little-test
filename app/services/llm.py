from __future__ import annotations

from collections import OrderedDict

import requests


class LLMError(Exception):
    pass


class SiliconFlowClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 45):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def generate_answer(self, question: str, matches: list[dict]) -> str:
        if not self.enabled:
            raise LLMError("尚未配置 SiliconFlow API Key 或模型名称。")
        if not matches:
            raise LLMError("没有可用于回答的检索上下文。")

        context_lines = []
        for index, match in enumerate(matches, start=1):
            context_lines.append(
                f"[片段{index}] 来源：{match['document_name']} 第 {match['page_number']} 页\n"
                f"{match['content']}"
            )

        unique_citations = OrderedDict()
        for match in matches:
            key = (match["document_name"], match["page_number"])
            unique_citations[key] = None
        citation_text = "；".join(
            f"{doc} 第 {page} 页" for doc, page in unique_citations.keys()
        )

        system_prompt = (
            "你是一个严格基于知识库回答问题的中文助手。"
            "你只能使用提供的资料片段回答，不能编造。"
            "如果资料不足，请明确回答“根据当前知识库内容，无法确定”。"
            "答案末尾加一行“参考来源：...”并引用给定资料中的文件名与页码。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"知识库片段：\n{'\n\n'.join(context_lines)}\n\n"
            f"请用简洁、准确、自然的中文回答。"
            f"如果能够回答，请在最后写：参考来源：{citation_text}"
        )

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError("调用大模型失败，请稍后重试。") from exc

        if response.status_code >= 400:
            message = _extract_error_message(response)
            raise LLMError(f"模型接口返回错误：{message}")

        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型返回了无法识别的结果。") from exc


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
