"""审计引擎 — 组合多种方法进行综合判定."""

from typing import Any

from modelaudit.config import AuditConfig
from modelaudit.models import AuditResult, ComparisonResult, DetectionResult, Fingerprint
from modelaudit.registry import get_fingerprinter


class AuditEngine:
    """模型审计引擎.

    提供三个核心功能:
    1. detect() — 检测文本来源
    2. verify() — 验证模型身份
    3. compare() — 比对两个模型
    """

    def __init__(self, config: AuditConfig | None = None):
        self.config = config or AuditConfig()
        # 确保方法模块已注册
        import modelaudit.methods  # noqa: F401

    def fingerprint(self, model: str, method: str = "llmmap", **kwargs) -> Fingerprint:
        """提取单个模型的指纹.

        Args:
            model: 模型名称或路径
            method: 指纹方法名称
            **kwargs: 传递给指纹方法的额外参数
        """
        fp_kwargs: dict[str, Any] = {}
        if method == "llmmap":
            fp_kwargs["provider"] = kwargs.get("provider", self.config.provider)
            fp_kwargs["api_key"] = kwargs.get("api_key", self.config.api_key)
            fp_kwargs["api_base"] = kwargs.get("api_base", self.config.api_base)
            fp_kwargs["num_probes"] = kwargs.get("num_probes", self.config.num_probes)

        fingerprinter = get_fingerprinter(method, **fp_kwargs)
        fingerprinter.prepare(model, **kwargs)
        return fingerprinter.get_fingerprint()

    def compare(
        self,
        model_a: str,
        model_b: str,
        method: str = "llmmap",
        **kwargs,
    ) -> ComparisonResult:
        """比对两个模型.

        Args:
            model_a: 模型 A 名称
            model_b: 模型 B 名称
            method: 使用的指纹方法
        """
        fp_a = self.fingerprint(model_a, method=method, **kwargs)
        fp_b = self.fingerprint(model_b, method=method, **kwargs)

        fingerprinter = get_fingerprinter(method)
        return fingerprinter.compare(fp_a, fp_b)

    def verify(self, model: str, **kwargs) -> dict[str, Any]:
        """验证模型身份 — 检查 API 背后是不是声称的模型.

        发送探测 prompt，分析响应是否与声称的模型一致。

        Args:
            model: 声称的模型名称
        """
        fp = self.fingerprint(model, method="llmmap", **kwargs)

        # 分析指纹中的风格标记
        from modelaudit.methods.style import _compute_style_scores

        responses = fp.data.get("raw_responses", [])
        if not responses:
            return {
                "model": model,
                "verified": False,
                "reason": "无法获取模型响应",
            }

        combined_text = "\n".join(responses)
        scores = _compute_style_scores(combined_text)

        # 找到最匹配的模型家族
        if scores:
            best_match = max(scores, key=lambda k: scores[k])
            best_score = scores[best_match]
        else:
            best_match = "unknown"
            best_score = 0.0

        # 检查声称的模型是否匹配
        model_lower = model.lower()
        claimed_family = None
        for family in scores:
            if family in model_lower:
                claimed_family = family
                break

        if claimed_family:
            claimed_score = scores.get(claimed_family, 0.0)
            is_match = claimed_family == best_match
        else:
            claimed_score = 0.0
            is_match = False

        return {
            "model": model,
            "verified": is_match,
            "claimed_family": claimed_family or "unknown",
            "best_match": best_match,
            "claimed_score": round(claimed_score, 4),
            "best_score": round(best_score, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
            "fingerprint": fp,
        }

    def detect(self, texts: list[str]) -> list[DetectionResult]:
        """检测文本来源 — 判断文本是哪个模型生成的.

        Args:
            texts: 待检测的文本列表
        """
        from modelaudit.methods.style import detect_text_source

        return detect_text_source(texts)

    def audit(self, teacher: str, student: str, **kwargs) -> AuditResult:
        """完整审计 — 综合指纹比对 + 风格分析.

        Args:
            teacher: 教师模型（疑似被蒸馏的源模型）
            student: 学生模型（疑似蒸馏产物）
        """
        comparisons: list[ComparisonResult] = []

        # 使用 LLMmap 黑盒比对
        try:
            result = self.compare(teacher, student, method="llmmap", **kwargs)
            comparisons.append(result)
        except Exception:
            pass  # 方法不可用时跳过

        # 综合判定
        if comparisons:
            avg_similarity = sum(c.similarity for c in comparisons) / len(comparisons)
            derived_votes = sum(1 for c in comparisons if c.is_derived)
            total_votes = len(comparisons)

            if derived_votes > total_votes / 2:
                verdict = "likely_derived"
            elif avg_similarity < 0.5:
                verdict = "independent"
            else:
                verdict = "inconclusive"

            confidence = avg_similarity
        else:
            avg_similarity = 0.0
            verdict = "inconclusive"
            confidence = 0.0

        return AuditResult(
            model_a=teacher,
            model_b=student,
            comparisons=comparisons,
            verdict=verdict,
            confidence=round(confidence, 4),
            summary=self._generate_summary(teacher, student, verdict, comparisons),
        )

    def _generate_summary(
        self,
        teacher: str,
        student: str,
        verdict: str,
        comparisons: list[ComparisonResult],
    ) -> str:
        """生成审计摘要."""
        verdict_text = {
            "likely_derived": "可能存在蒸馏关系",
            "independent": "两个模型独立",
            "inconclusive": "无法确定",
        }

        lines = [
            f"审计对象: {teacher} vs {student}",
            f"判定结果: {verdict_text.get(verdict, verdict)}",
        ]

        for c in comparisons:
            lines.append(f"  [{c.method}] 相似度: {c.similarity:.4f} (阈值: {c.threshold})")

        return "\n".join(lines)
