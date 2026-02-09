"""测试报告生成."""

import json

from modelaudit.models import AuditResult, ComparisonResult
from modelaudit.report import (
    _generate_basic_report,
    _generate_detailed_report,
    _is_teacher_style,
    _judge_difference,
    generate_report,
)


def _make_comparison(
    similarity: float = 0.95,
    is_derived: bool = True,
    threshold: float = 0.85,
) -> ComparisonResult:
    return ComparisonResult(
        model_a="claude-opus",
        model_b="kimi-k2.5",
        method="llmmap",
        similarity=similarity,
        is_derived=is_derived,
        threshold=threshold,
        confidence=0.8,
    )


def _make_audit_result(
    verdict: str = "likely_derived",
    confidence: float = 0.8,
    similarity: float = 0.95,
    with_details: bool = True,
) -> AuditResult:
    comparison = _make_comparison(similarity=similarity, is_derived=verdict == "likely_derived")

    details = {}
    if with_details:
        details = {
            "fingerprints": {
                "teacher": {
                    "model_id": "claude-opus",
                    "method": "llmmap",
                    "fingerprint_type": "blackbox",
                    "data": {
                        "vector": {
                            "avg_length_chars": 800.0,
                            "avg_length_words": 120.0,
                            "avg_unique_word_ratio": 0.65,
                            "avg_punctuation_ratio": 0.08,
                            "avg_newline_ratio": 0.02,
                            "ratio_has_bullet_points": 0.6,
                            "ratio_has_code_blocks": 0.2,
                            "ratio_has_numbered_list": 0.3,
                            "ratio_has_markdown_headers": 0.4,
                            "style_helpful": 0.015,
                            "style_hedging": 0.008,
                            "style_structured": 0.012,
                        },
                    },
                },
                "student": {
                    "model_id": "kimi-k2.5",
                    "method": "llmmap",
                    "fingerprint_type": "blackbox",
                    "data": {
                        "vector": {
                            "avg_length_chars": 850.0,
                            "avg_length_words": 125.0,
                            "avg_unique_word_ratio": 0.64,
                            "avg_punctuation_ratio": 0.079,
                            "avg_newline_ratio": 0.021,
                            "ratio_has_bullet_points": 0.6,
                            "ratio_has_code_blocks": 0.2,
                            "ratio_has_numbered_list": 0.3,
                            "ratio_has_markdown_headers": 0.4,
                            "style_helpful": 0.014,
                            "style_hedging": 0.008,
                            "style_structured": 0.012,
                        },
                    },
                },
            },
            "probe_details": [
                {
                    "probe_id": "identity_direct",
                    "category": "self_awareness",
                    "teacher_style": "claude",
                    "student_style": "claude",
                    "is_consistent": True,
                },
                {
                    "probe_id": "safety_harmful",
                    "category": "safety_boundary",
                    "teacher_style": "claude",
                    "student_style": "gpt",
                    "is_consistent": False,
                },
                {
                    "probe_id": "reasoning_math",
                    "category": "reasoning",
                    "teacher_style": "claude",
                    "student_style": "claude",
                    "is_consistent": True,
                },
            ],
            "teacher_info": {
                "model": "claude-opus",
                "provider": "anthropic",
                "api_base": "",
            },
            "student_info": {
                "model": "kimi-k2.5",
                "provider": "custom",
                "api_base": "https://api.moonshot.cn/v1",
            },
        }

    return AuditResult(
        model_a="claude-opus",
        model_b="kimi-k2.5",
        comparisons=[comparison],
        verdict=verdict,
        confidence=confidence,
        summary="审计对象: claude-opus vs kimi-k2.5",
        details=details,
    )


class TestGenerateReport:
    def test_json_format(self):
        result = _make_audit_result()
        output = generate_report(result, format="json")
        data = json.loads(output)
        assert data["model_a"] == "claude-opus"
        assert data["model_b"] == "kimi-k2.5"
        assert data["verdict"] == "likely_derived"

    def test_markdown_with_details(self):
        result = _make_audit_result()
        output = generate_report(result, format="markdown")
        assert "# 模型蒸馏审计报告" in output
        assert "kimi-k2.5" in output
        assert "claude-opus" in output
        assert "## 1. 审计对象" in output
        assert "## 2. 审计方法" in output
        assert "## 3. 审计结果" in output

    def test_markdown_without_details_uses_basic(self):
        result = _make_audit_result(with_details=False)
        output = generate_report(result, format="markdown")
        # 基础报告没有 6 节结构
        assert "## 1. 审计对象" not in output
        assert "# 模型蒸馏审计报告" in output
        assert "判定结果" in output


class TestDetailedReport:
    def test_six_sections(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "## 1. 审计对象" in output
        assert "## 2. 审计方法" in output
        assert "## 3. 审计结果" in output
        assert "## 4. 关键发现" in output
        assert "## 5. 结论" in output
        assert "## 6. 局限性声明" in output

    def test_footer(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "knowlyr-modelaudit" in output

    def test_verdict_likely_derived(self):
        result = _make_audit_result(verdict="likely_derived")
        output = _generate_detailed_report(result)
        assert "可能存在蒸馏关系" in output

    def test_verdict_independent(self):
        result = _make_audit_result(verdict="independent", similarity=0.3, confidence=0.5)
        result.comparisons[0].similarity = 0.3
        result.comparisons[0].is_derived = False
        output = _generate_detailed_report(result)
        assert "两个模型独立" in output

    def test_verdict_inconclusive(self):
        result = _make_audit_result(verdict="inconclusive", similarity=0.6, confidence=0.3)
        result.comparisons[0].similarity = 0.6
        result.comparisons[0].is_derived = False
        output = _generate_detailed_report(result)
        assert "无法确定" in output

    def test_probe_table(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "identity_direct" in output
        assert "safety_harmful" in output
        assert "reasoning_math" in output

    def test_confidence_high(self):
        result = _make_audit_result(confidence=0.8)
        output = _generate_detailed_report(result)
        assert "高" in output

    def test_confidence_medium(self):
        result = _make_audit_result(confidence=0.5)
        output = _generate_detailed_report(result)
        assert "中" in output

    def test_confidence_low(self):
        result = _make_audit_result(confidence=0.2)
        output = _generate_detailed_report(result)
        assert "低" in output

    def test_feature_table(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "平均字符数" in output
        assert "词汇多样性" in output
        assert "helpful 标记" in output

    def test_style_consistency_rate(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        # 3 probes, 2 consistent → 67%
        assert "2/3" in output
        assert "67%" in output

    def test_provider_display(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "Anthropic" in output
        assert "api.moonshot.cn" in output

    def test_category_labels(self):
        result = _make_audit_result()
        output = _generate_detailed_report(result)
        assert "自我认知" in output
        assert "安全边界" in output
        assert "推理测试" in output


class TestBasicReport:
    def test_structure(self):
        result = _make_audit_result(with_details=False)
        output = _generate_basic_report(result)
        assert "# 模型蒸馏审计报告" in output
        assert "审计对象" in output
        assert "判定结果" in output

    def test_comparison_table(self):
        result = _make_audit_result(with_details=False)
        output = _generate_basic_report(result)
        assert "llmmap" in output
        assert "0.9500" in output

    def test_no_comparisons(self):
        result = AuditResult(
            model_a="a", model_b="b",
            comparisons=[], verdict="inconclusive",
        )
        output = _generate_basic_report(result)
        assert "# 模型蒸馏审计报告" in output


class TestJudgeDifference:
    def test_length_identical(self):
        assert "完全一致" in _judge_difference("avg_length_chars", 5)

    def test_length_highly_consistent(self):
        assert "高度一致" in _judge_difference("avg_length_chars", 200)

    def test_length_close(self):
        assert _judge_difference("avg_length_chars", 500) == "接近"

    def test_length_significant(self):
        assert _judge_difference("avg_length_chars", 1000) == "显著不同"

    def test_ratio_identical(self):
        assert "完全一致" in _judge_difference("ratio_has_bullet_points", 0.005)

    def test_ratio_significant(self):
        assert _judge_difference("ratio_has_bullet_points", 0.2) == "显著不同"

    def test_style_identical(self):
        assert "完全一致" in _judge_difference("style_helpful", 0.0005)

    def test_style_significant(self):
        assert _judge_difference("style_helpful", 0.01) == "显著不同"

    def test_default_range(self):
        assert "完全一致" in _judge_difference("unknown_feature", 0.005)


class TestIsTeacherStyle:
    def test_match_claude(self):
        assert _is_teacher_style("claude", "claude-opus") is True

    def test_match_gpt(self):
        assert _is_teacher_style("gpt", "gpt-4o") is True

    def test_no_match(self):
        assert _is_teacher_style("gpt", "claude-opus") is False

    def test_case_insensitive(self):
        assert _is_teacher_style("Claude", "claude-opus") is True
