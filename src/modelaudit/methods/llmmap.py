"""LLMmap 黑盒指纹方法.

参考 LLMmap (USENIX Security 2025, MIT License):
通过发送精心设计的探测 prompt 并分析响应模式来识别模型身份。

核心思路:
1. 发送 N 个探测 prompt 到目标模型
2. 收集响应文本
3. 提取响应特征向量（词频、长度、结构等）
4. 与已知模型模板比对，计算相似度
"""

import hashlib
import json
import logging
import re
import time
from typing import Any

from modelaudit.base import BlackBoxFingerprinter
from modelaudit.models import ComparisonResult, Fingerprint
from modelaudit.probes import get_probes
from modelaudit.registry import register

logger = logging.getLogger(__name__)


def _extract_response_features(response: str) -> dict[str, Any]:
    """从单条响应中提取特征向量."""
    words = response.split()
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_words = len(words) or 1

    # 常见 LLM 风格标记词
    style_markers = {
        "apologetic": ["sorry", "apologize", "unfortunately", "cannot", "can't", "i'm unable"],
        "helpful": ["certainly", "sure", "absolutely", "of course", "happy to", "glad to"],
        "hedging": ["however", "although", "perhaps", "might", "could", "may"],
        "structured": ["first", "second", "third", "finally", "additionally", "moreover"],
        "ai_aware": ["as an ai", "language model", "i don't have", "i'm not able", "trained"],
    }

    marker_scores = {}
    for category, markers in style_markers.items():
        count = sum(response.lower().count(m) for m in markers)
        marker_scores[category] = count / total_words

    return {
        "length_chars": len(response),
        "length_words": len(words),
        "length_sentences": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / total_words,
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "unique_word_ratio": len({w.lower() for w in words}) / total_words,
        "punctuation_ratio": sum(1 for c in response if c in ".,;:!?") / max(len(response), 1),
        "newline_ratio": response.count("\n") / max(len(response), 1),
        "has_bullet_points": bool(re.search(r"^[\s]*[-*•]\s", response, re.MULTILINE)),
        "has_numbered_list": bool(re.search(r"^[\s]*\d+[.)]\s", response, re.MULTILINE)),
        "has_markdown_headers": bool(re.search(r"^#+\s", response, re.MULTILINE)),
        "has_code_blocks": "```" in response,
        "starts_with_refusal": any(
            response.lower().startswith(w) for w in ["i cannot", "i can't", "sorry", "i apologize"]
        ),
        "marker_scores": marker_scores,
    }


def _compute_fingerprint_vector(probe_features: list[dict[str, Any]]) -> dict[str, float]:
    """将多个探测的特征合并为指纹向量."""
    vector: dict[str, float] = {}
    n = len(probe_features) or 1

    # 聚合数值特征
    numeric_keys = [
        "length_chars", "length_words", "length_sentences",
        "avg_word_length", "avg_sentence_length", "unique_word_ratio",
        "punctuation_ratio", "newline_ratio",
    ]
    for key in numeric_keys:
        values = [f.get(key, 0) for f in probe_features]
        vector[f"avg_{key}"] = sum(values) / n

    # 聚合布尔特征
    bool_keys = [
        "has_bullet_points", "has_numbered_list", "has_markdown_headers",
        "has_code_blocks", "starts_with_refusal",
    ]
    for key in bool_keys:
        vector[f"ratio_{key}"] = sum(1 for f in probe_features if f.get(key)) / n

    # 聚合风格标记
    all_categories = set()
    for f in probe_features:
        all_categories.update(f.get("marker_scores", {}).keys())
    for cat in all_categories:
        values = [f.get("marker_scores", {}).get(cat, 0) for f in probe_features]
        vector[f"style_{cat}"] = sum(values) / n

    return vector


# 各特征维度的典型范围，用于归一化到 0-1
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "avg_length_chars": (50, 3000),
    "avg_length_words": (10, 500),
    "avg_length_sentences": (1, 20),
    "avg_avg_word_length": (3, 8),
    "avg_avg_sentence_length": (5, 40),
    "avg_unique_word_ratio": (0, 1),
    "avg_punctuation_ratio": (0, 0.1),
    "avg_newline_ratio": (0, 0.05),
}


def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    """将指纹向量归一化到 0-1 范围，消除量纲差异."""
    normalized = {}
    for key, value in vector.items():
        if key in _FEATURE_RANGES:
            lo, hi = _FEATURE_RANGES[key]
            normalized[key] = max(0, min(1, (value - lo) / (hi - lo))) if hi > lo else 0
        else:
            # ratio_ 和 style_ 特征本身已在 0-1 范围
            normalized[key] = value
    return normalized


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度（归一化后比对）."""
    a = _normalize_vector(a)
    b = _normalize_vector(b)

    all_keys = set(a.keys()) | set(b.keys())
    if not all_keys:
        return 0.0

    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    norm_a = sum(a.get(k, 0) ** 2 for k in all_keys) ** 0.5
    norm_b = sum(b.get(k, 0) ** 2 for k in all_keys) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _call_model_api(
    model: str,
    prompt: str,
    provider: str = "openai",
    api_key: str = "",
    api_base: str = "",
    max_retries: int = 3,
    api_timeout: int = 60,
) -> str:
    """调用模型 API 获取响应，支持指数退避重试."""
    for attempt in range(max_retries):
        try:
            text = _call_model_api_once(model, prompt, provider, api_key, api_base, api_timeout)
            if not text or not text.strip():
                logger.warning("API 返回空响应 (model=%s, attempt=%d)", model, attempt + 1)
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt)
                    continue
            return text
        except (ImportError, ValueError):
            raise
        except Exception as e:
            err_str = str(e).lower()
            # 认证/权限错误 — 不重试
            if any(kw in err_str for kw in ("401", "403", "unauthorized", "forbidden",
                                             "invalid api key", "authentication")):
                raise ValueError(f"API 认证失败 (model={model}): {e}") from e
            # 速率限制 — 加长退避
            if "429" in err_str or "rate" in err_str:
                logger.warning("API 速率限制 (model=%s, attempt=%d/%d)", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt + 1)  # 加长退避
                    continue
            logger.warning(
                "API 调用失败 (model=%s, attempt=%d/%d): %s",
                model, attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)
            else:
                raise
    return ""


def _backoff_sleep(attempt: int) -> None:
    """指数退避等待."""
    delay = min(2 ** attempt, 30)
    logger.info("等待 %ds 后重试...", delay)
    time.sleep(delay)


def _call_model_api_once(
    model: str,
    prompt: str,
    provider: str = "openai",
    api_key: str = "",
    api_base: str = "",
    api_timeout: int = 60,
) -> str:
    """单次调用模型 API."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要 openai 库。请运行: pip install knowlyr-modelaudit[blackbox]")

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base

        client_kwargs["timeout"] = float(api_timeout)
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "需要 anthropic 库。请运行: pip install knowlyr-modelaudit[blackbox]"
            )

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        client = Anthropic(**client_kwargs)
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    elif provider == "custom":
        try:
            import httpx
        except ImportError:
            raise ImportError("需要 httpx 库。请运行: pip install knowlyr-modelaudit[blackbox]")

        if not api_base:
            raise ValueError("自定义 provider 需要指定 api_base")

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = httpx.post(
            f"{api_base.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.0,
            },
            headers=headers,
            timeout=api_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"不支持的 provider: {provider}")


@register("llmmap")
class LLMmapFingerprinter(BlackBoxFingerprinter):
    """LLMmap 黑盒指纹 — 基于探测 prompt 响应模式.

    通过发送精心设计的探测 prompt 并分析响应特征来识别模型身份。
    参考: LLMmap (USENIX Security 2025), MIT License
    """

    @property
    def name(self) -> str:
        return "llmmap"

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        api_base: str = "",
        num_probes: int = 8,
        api_timeout: int = 60,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.num_probes = num_probes
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self._model: str = ""
        self._responses: list[str] = []

    def prepare(self, model: str, **kwargs) -> None:
        """设置目标模型."""
        self._model = model
        self._responses = []
        # 覆盖设置
        if "provider" in kwargs:
            self.provider = kwargs["provider"]
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]

    def get_fingerprint(self) -> Fingerprint:
        """发送探测 prompt 并提取指纹."""
        if not self._model:
            raise RuntimeError("请先调用 prepare() 设置目标模型")

        probes = get_probes(count=self.num_probes)
        responses: list[str] = []
        probe_features: list[dict[str, Any]] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _call_probe(probe):
            return _call_model_api(
                model=self._model,
                prompt=probe.prompt,
                provider=self.provider,
                api_key=self.api_key,
                api_base=self.api_base,
                max_retries=self.max_retries,
                api_timeout=self.api_timeout,
            )

        # 并发发送探测 (最多 4 个并发)
        probe_response_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_probe = {
                executor.submit(_call_probe, probe): probe for probe in probes
            }
            for future in as_completed(future_to_probe):
                probe = future_to_probe[future]
                probe_response_map[probe.id] = future.result()

        # 按原始顺序组装结果
        for probe in probes:
            response = probe_response_map[probe.id]
            responses.append(response)
            features = _extract_response_features(response)
            probe_features.append(features)

        self._responses = responses
        vector = _compute_fingerprint_vector(probe_features)

        # 生成指纹哈希（用于快速比对）
        fp_hash = hashlib.md5(json.dumps(vector, sort_keys=True).encode()).hexdigest()[:16]

        return Fingerprint(
            model_id=self._model,
            method="llmmap",
            fingerprint_type="blackbox",
            data={
                "vector": vector,
                "hash": fp_hash,
                "num_probes": len(probes),
                "probe_ids": [p.id for p in probes],
                "raw_responses": responses,
            },
        )

    def compare(self, fp_a: Fingerprint, fp_b: Fingerprint) -> ComparisonResult:
        """比对两个 LLMmap 指纹."""
        vec_a = fp_a.data.get("vector", {})
        vec_b = fp_b.data.get("vector", {})

        similarity = _cosine_similarity(vec_a, vec_b)
        threshold = 0.85

        return ComparisonResult(
            model_a=fp_a.model_id,
            model_b=fp_b.model_id,
            method="llmmap",
            similarity=similarity,
            is_derived=similarity >= threshold,
            threshold=threshold,
            confidence=min(abs(similarity - threshold) / 0.15, 1.0),
            details={
                "hash_a": fp_a.data.get("hash", ""),
                "hash_b": fp_b.data.get("hash", ""),
                "hash_match": fp_a.data.get("hash") == fp_b.data.get("hash"),
            },
        )
