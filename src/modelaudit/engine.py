"""审计引擎 — 组合多种方法进行综合判定."""

import logging
from typing import Any

from modelaudit.cache import FingerprintCache
from modelaudit.config import AuditConfig
from modelaudit.models import AuditResult, ComparisonResult, DetectionResult, Fingerprint
from modelaudit.registry import get_fingerprinter

logger = logging.getLogger(__name__)


class AuditEngine:
    """模型审计引擎.

    提供四个核心功能:
    1. detect() — 检测文本来源
    2. verify() — 验证模型身份
    3. compare() — 比对两个模型
    4. audit() — 完整蒸馏审计（含详细报告数据）
    """

    def __init__(self, config: AuditConfig | None = None, use_cache: bool = True):
        self.config = config or AuditConfig()
        self.cache = (
            FingerprintCache(self.config.cache_dir, ttl=self.config.cache_ttl)
            if use_cache
            else None
        )
        # 确保方法模块已注册
        import modelaudit.methods  # noqa: F401

    def fingerprint(self, model: str, method: str = "llmmap", **kwargs) -> Fingerprint:
        """提取单个模型的指纹.

        优先从缓存读取，缓存未命中时调用 API 并写入缓存。

        Args:
            model: 模型名称或路径
            method: 指纹方法名称
            **kwargs: 传递给指纹方法的额外参数
        """
        provider = kwargs.get("provider", self.config.provider)

        # 尝试从缓存读取
        if self.cache:
            cached = self.cache.get(model, method, provider)
            if cached:
                logger.info("缓存命中: model=%s, method=%s", model, method)
                return cached
            logger.debug("缓存未命中: model=%s, method=%s", model, method)

        fp_kwargs: dict[str, Any] = {}
        if method == "llmmap":
            fp_kwargs["provider"] = provider
            fp_kwargs["api_key"] = kwargs.get("api_key", self.config.api_key)
            fp_kwargs["api_base"] = kwargs.get("api_base", self.config.api_base)
            fp_kwargs["num_probes"] = kwargs.get("num_probes", self.config.num_probes)
        elif method == "dli":
            fp_kwargs["provider"] = provider
            fp_kwargs["api_key"] = kwargs.get("api_key", self.config.api_key)
            fp_kwargs["api_base"] = kwargs.get("api_base", self.config.api_base)
            fp_kwargs["num_probes"] = kwargs.get("num_probes", 8)
        elif method == "reef":
            fp_kwargs["device"] = kwargs.get("device", "cpu")

        logger.info("提取指纹: model=%s, method=%s, provider=%s", model, method, provider)
        fingerprinter = get_fingerprinter(method, **fp_kwargs)
        fingerprinter.prepare(model, **kwargs)
        fp = fingerprinter.get_fingerprint()

        # 写入缓存
        if self.cache:
            self.cache.put(model, method, provider, fp)
            logger.debug("指纹已写入缓存: model=%s", model)

        return fp

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

    def audit(
        self,
        teacher: str,
        student: str,
        *,
        teacher_provider: str | None = None,
        teacher_api_key: str | None = None,
        teacher_api_base: str | None = None,
        student_provider: str | None = None,
        student_api_key: str | None = None,
        student_api_base: str | None = None,
        **kwargs,
    ) -> AuditResult:
        """完整审计 — 综合指纹比对 + 风格分析，生成详细报告数据.

        支持双模型分别配置不同 provider/api_key/api_base，
        适用于跨平台审计（如 Kimi API + Claude API）。

        Args:
            teacher: 教师模型（疑似被蒸馏的源模型）
            student: 学生模型（疑似蒸馏产物）
            teacher_provider: 教师模型 API 提供商
            teacher_api_key: 教师模型 API Key
            teacher_api_base: 教师模型 API 地址
            student_provider: 学生模型 API 提供商
            student_api_key: 学生模型 API Key
            student_api_base: 学生模型 API 地址
        """
        # 确定各模型的 provider 配置（专属配置 > 通用配置 > 默认配置）
        t_provider = teacher_provider or kwargs.get("provider", self.config.provider)
        t_api_key = teacher_api_key or kwargs.get("api_key", self.config.api_key)
        t_api_base = teacher_api_base or kwargs.get("api_base", self.config.api_base)

        s_provider = student_provider or kwargs.get("provider", self.config.provider)
        s_api_key = student_api_key or kwargs.get("api_key", self.config.api_key)
        s_api_base = student_api_base or kwargs.get("api_base", self.config.api_base)

        num_probes = kwargs.get("num_probes", self.config.num_probes)

        # ── 1. 分别提取指纹 ──
        fp_teacher = self.fingerprint(
            teacher, method="llmmap",
            provider=t_provider, api_key=t_api_key, api_base=t_api_base,
            num_probes=num_probes,
        )
        fp_student = self.fingerprint(
            student, method="llmmap",
            provider=s_provider, api_key=s_api_key, api_base=s_api_base,
            num_probes=num_probes,
        )

        # ── 2. 指纹比对 ──
        fingerprinter = get_fingerprinter("llmmap")
        comparison = fingerprinter.compare(fp_teacher, fp_student)
        comparisons: list[ComparisonResult] = [comparison]

        # ── 2b. DLI 行为签名比对 (复用 LLMmap 已收集的响应) ──
        skipped: list[str] = []
        try:
            from modelaudit.methods.dli import (
                _compute_behavior_similarity,
                _extract_behavior_signature,
            )

            teacher_responses = fp_teacher.data.get("raw_responses", [])
            student_responses = fp_student.data.get("raw_responses", [])

            if teacher_responses and student_responses:
                sig_teacher = _extract_behavior_signature(teacher_responses)
                sig_student = _extract_behavior_signature(student_responses)
                dli_similarity = _compute_behavior_similarity(sig_teacher, sig_student)
                dli_threshold = 0.80

                dli_comparison = ComparisonResult(
                    model_a=teacher,
                    model_b=student,
                    method="dli",
                    similarity=round(dli_similarity, 6),
                    is_derived=dli_similarity >= dli_threshold,
                    threshold=dli_threshold,
                    confidence=min(abs(dli_similarity - dli_threshold) / 0.2, 1.0),
                    details={"reused_from": "llmmap_responses"},
                )
                comparisons.append(dli_comparison)
            else:
                reason = "teacher 响应为空" if not teacher_responses else "student 响应为空"
                skipped.append(f"DLI ({reason})")
                logger.info("DLI 比对跳过: %s", reason)
        except Exception as exc:
            skipped.append(f"DLI ({exc})")
            logger.debug("DLI 比对跳过: %s", exc)

        # ── 3. 逐条探测的风格分析 ──
        from modelaudit.methods.style import _compute_style_scores
        from modelaudit.probes import get_probes

        probes = get_probes(count=num_probes)
        teacher_responses = fp_teacher.data.get("raw_responses", [])
        student_responses = fp_student.data.get("raw_responses", [])

        probe_details: list[dict[str, Any]] = []
        for i, probe in enumerate(probes):
            t_response = teacher_responses[i] if i < len(teacher_responses) else ""
            s_response = student_responses[i] if i < len(student_responses) else ""

            t_scores = _compute_style_scores(t_response) if t_response else {}
            s_scores = _compute_style_scores(s_response) if s_response else {}

            t_best = max(t_scores, key=lambda k: t_scores[k]) if t_scores else "unknown"
            s_best = max(s_scores, key=lambda k: s_scores[k]) if s_scores else "unknown"

            probe_details.append({
                "probe_id": probe.id,
                "category": probe.category,
                "teacher_style": t_best,
                "student_style": s_best,
                "is_consistent": t_best == s_best,
            })

        # ── 4. 综合判定 ──
        avg_similarity = sum(c.similarity for c in comparisons) / len(comparisons)
        derived_votes = sum(1 for c in comparisons if c.is_derived)
        total_votes = len(comparisons)

        if derived_votes > total_votes / 2:
            verdict = "likely_derived"
        elif avg_similarity < 0.5:
            verdict = "independent"
        else:
            verdict = "inconclusive"

        confidence = min(abs(avg_similarity - comparison.threshold) / 0.15, 1.0)

        # ── 5. 打包完整详情（供 report.py 使用） ──
        details: dict[str, Any] = {
            "fingerprints": {
                "teacher": fp_teacher.model_dump(),
                "student": fp_student.model_dump(),
            },
            "probe_details": probe_details,
            "teacher_info": {
                "model": teacher,
                "provider": t_provider,
                "api_base": t_api_base,
            },
            "student_info": {
                "model": student,
                "provider": s_provider,
                "api_base": s_api_base,
            },
        }
        if skipped:
            details["skipped_methods"] = skipped

        return AuditResult(
            model_a=teacher,
            model_b=student,
            comparisons=comparisons,
            verdict=verdict,
            confidence=round(confidence, 4),
            summary=self._generate_summary(teacher, student, verdict, comparisons),
            details=details,
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
