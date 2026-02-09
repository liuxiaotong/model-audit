"""DLI 蒸馏检测方法 — Distillation Lineage Inference.

基于行为签名的蒸馏血缘推断。
不需要模型权重，通过统计分析两个模型在同一组探测 prompt 上的
输出分布差异来判断是否存在蒸馏关系。

核心思路:
1. 用同一组探测 prompt 收集两个模型的响应
2. 提取多维行为签名 (n-gram 分布、拒绝模式、格式偏好、词汇重叠)
3. 用 Jensen-Shannon 散度衡量分布差异
4. 综合判定是否存在蒸馏关系
"""

import logging
import math
import re
from collections import Counter
from typing import Any

from modelaudit.base import BlackBoxFingerprinter
from modelaudit.models import ComparisonResult, Fingerprint
from modelaudit.registry import register

logger = logging.getLogger(__name__)


def _extract_ngrams(text: str, n: int = 2) -> Counter:
    """提取文本的 n-gram 频率分布."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return Counter()
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """计算 Jensen-Shannon 散度 (对称 KL 散度)."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0

    # 归一化
    p_sum = sum(p.values()) or 1
    q_sum = sum(q.values()) or 1
    p_norm = {k: p.get(k, 0) / p_sum for k in all_keys}
    q_norm = {k: q.get(k, 0) / q_sum for k in all_keys}

    # M = (P + Q) / 2
    m = {k: (p_norm[k] + q_norm[k]) / 2 for k in all_keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(
            a[k] * math.log(a[k] / b[k]) if a[k] > 0 and b[k] > 0 else 0
            for k in all_keys
        )

    return (_kl(p_norm, m) + _kl(q_norm, m)) / 2


def _extract_behavior_signature(responses: list[str]) -> dict[str, Any]:
    """从响应列表中提取行为签名."""
    if not responses:
        return {"bigram_dist": {}, "features": {}}

    # 合并所有响应的 bigram 分布
    total_bigrams: Counter = Counter()
    for r in responses:
        total_bigrams.update(_extract_ngrams(r, n=2))

    # 归一化 bigram 分布（取 top 100）
    top_bigrams = total_bigrams.most_common(100)
    total = sum(c for _, c in top_bigrams) or 1
    bigram_dist = {ng: c / total for ng, c in top_bigrams}

    # 行为特征
    total_responses = len(responses)
    combined = " ".join(responses).lower()
    words = re.findall(r"\w+", combined)
    total_words = len(words) or 1

    features = {
        # 拒绝率
        "refusal_rate": sum(
            1 for r in responses
            if any(p in r.lower() for p in [
                "i cannot", "i can't", "i'm unable", "i apologize",
                "i don't think i should", "i'd rather not",
            ])
        ) / total_responses,
        # 平均响应长度
        "avg_length": sum(len(r.split()) for r in responses) / total_responses,
        # 词汇多样性
        "vocab_diversity": len(set(words)) / total_words,
        # 格式偏好
        "markdown_rate": sum(
            1 for r in responses if re.search(r"^#+\s", r, re.MULTILINE)
        ) / total_responses,
        "list_rate": sum(
            1 for r in responses if re.search(r"^[\s]*[-*•]\s", r, re.MULTILINE)
        ) / total_responses,
        "code_block_rate": sum(
            1 for r in responses if "```" in r
        ) / total_responses,
        # 语气标记
        "hedging_rate": sum(
            combined.count(w) for w in ["perhaps", "maybe", "might", "could", "possibly"]
        ) / total_words,
        "certainty_rate": sum(
            combined.count(w) for w in ["certainly", "definitely", "absolutely", "clearly"]
        ) / total_words,
    }

    return {"bigram_dist": bigram_dist, "features": features}


def _compute_behavior_similarity(sig_a: dict[str, Any], sig_b: dict[str, Any]) -> float:
    """计算两个行为签名的相似度 [0, 1]."""
    # 1. bigram 分布的 JS 散度 (权重 0.4)
    js_div = _js_divergence(sig_a.get("bigram_dist", {}), sig_b.get("bigram_dist", {}))
    # JS 散度范围 [0, ln2]，归一化到 [0, 1] 并转换为相似度
    bigram_sim = 1.0 - min(js_div / math.log(2), 1.0)

    # 2. 行为特征的余弦相似度 (权重 0.6)
    feat_a = sig_a.get("features", {})
    feat_b = sig_b.get("features", {})
    all_keys = set(feat_a.keys()) | set(feat_b.keys())

    if not all_keys:
        return bigram_sim

    dot = sum(feat_a.get(k, 0) * feat_b.get(k, 0) for k in all_keys)
    norm_a = sum(feat_a.get(k, 0) ** 2 for k in all_keys) ** 0.5
    norm_b = sum(feat_b.get(k, 0) ** 2 for k in all_keys) ** 0.5

    if norm_a == 0 or norm_b == 0:
        feat_sim = 0.0
    else:
        feat_sim = dot / (norm_a * norm_b)

    return bigram_sim * 0.4 + feat_sim * 0.6


@register("dli")
class DLIFingerprinter(BlackBoxFingerprinter):
    """DLI 蒸馏血缘推断 — 基于行为签名的统计分析.

    通过比对两个模型在探测 prompt 上的输出分布差异，
    判断是否存在蒸馏关系。不需要模型权重。
    """

    @property
    def name(self) -> str:
        return "dli"

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        api_base: str = "",
        num_probes: int = 8,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.num_probes = num_probes
        self._model: str = ""
        self._responses: list[str] = []

    def prepare(self, model: str, **kwargs) -> None:
        """设置目标模型."""
        self._model = model
        self._responses = []
        if "provider" in kwargs:
            self.provider = kwargs["provider"]
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]

    def get_fingerprint(self) -> Fingerprint:
        """发送探测 prompt 并提取行为签名."""
        if not self._model:
            raise RuntimeError("请先调用 prepare() 设置目标模型")

        from modelaudit.methods.llmmap import _call_model_api
        from modelaudit.probes import get_probes

        probes = get_probes(count=self.num_probes)
        responses: list[str] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _call_probe(probe):
            return _call_model_api(
                model=self._model,
                prompt=probe.prompt,
                provider=self.provider,
                api_key=self.api_key,
                api_base=self.api_base,
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
            responses.append(probe_response_map[probe.id])

        self._responses = responses
        signature = _extract_behavior_signature(responses)

        return Fingerprint(
            model_id=self._model,
            method="dli",
            fingerprint_type="blackbox",
            data={
                "signature": signature,
                "num_probes": len(probes),
                "probe_ids": [p.id for p in probes],
                "raw_responses": responses,
            },
        )

    def compare(self, fp_a: Fingerprint, fp_b: Fingerprint) -> ComparisonResult:
        """比对两个 DLI 指纹."""
        sig_a = fp_a.data.get("signature", {})
        sig_b = fp_b.data.get("signature", {})

        similarity = _compute_behavior_similarity(sig_a, sig_b)
        threshold = 0.80  # DLI 使用稍低的阈值，因为行为签名粒度更粗

        return ComparisonResult(
            model_a=fp_a.model_id,
            model_b=fp_b.model_id,
            method="dli",
            similarity=round(similarity, 6),
            is_derived=similarity >= threshold,
            threshold=threshold,
            confidence=min(abs(similarity - threshold) / 0.2, 1.0),
            details={
                "bigram_js_divergence": _js_divergence(
                    sig_a.get("bigram_dist", {}), sig_b.get("bigram_dist", {}),
                ),
                "feature_diff": {
                    k: abs(
                        sig_a.get("features", {}).get(k, 0)
                        - sig_b.get("features", {}).get(k, 0)
                    )
                    for k in set(
                        list(sig_a.get("features", {}).keys())
                        + list(sig_b.get("features", {}).keys())
                    )
                },
            },
        )
