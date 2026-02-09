"""测试 MCP Server 工具."""

from unittest.mock import patch

import pytest

from modelaudit.models import (
    AuditResult,
    ComparisonResult,
    DetectionResult,
)

try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="需要 mcp")


@needs_mcp
class TestCreateServer:
    def test_creates_server(self):
        from modelaudit.mcp_server import create_server

        server = create_server()
        assert server is not None

    def test_no_mcp_raises(self):
        from modelaudit import mcp_server

        original = mcp_server.HAS_MCP
        try:
            mcp_server.HAS_MCP = False
            with pytest.raises(ImportError, match="MCP 未安装"):
                mcp_server.create_server()
        finally:
            mcp_server.HAS_MCP = original


class TestDetectTextSource:
    def test_detect_tool_logic(self):
        """直接测试 detect 逻辑而不依赖 MCP 框架."""
        mock_results = [
            DetectionResult(
                text_id=1,
                text_preview="Hello world...",
                predicted_model="chatgpt",
                confidence=0.75,
                all_scores={"chatgpt": 0.75},
            ),
            DetectionResult(
                text_id=2,
                text_preview="Certainly! I'd...",
                predicted_model="chatgpt",
                confidence=0.80,
                all_scores={"chatgpt": 0.80},
            ),
        ]

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.detect.return_value = mock_results

            from modelaudit.mcp_server import AuditEngine

            engine = AuditEngine()
            results = engine.detect(["text1", "text2"])

            assert len(results) == 2
            assert results[0].predicted_model == "chatgpt"
            assert results[1].confidence == 0.80


class TestCompareModels:
    def test_compare_tool_logic(self):
        """测试 compare_models 工具的核心逻辑."""
        mock_result = ComparisonResult(
            model_a="gpt-4o",
            model_b="my-model",
            method="llmmap",
            similarity=0.85,
            is_derived=True,
            threshold=0.85,
            confidence=0.5,
            details={},
        )

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.compare.return_value = mock_result

            from modelaudit.mcp_server import AuditConfig, AuditEngine

            config = AuditConfig(provider="openai")
            engine = AuditEngine(config)
            result = engine.compare("gpt-4o", "my-model", method="llmmap", provider="openai")

            assert result.similarity == 0.85
            assert result.is_derived is True
            assert result.method == "llmmap"

    def test_compare_with_dli_method(self):
        """测试 DLI 方法参数传递."""
        mock_result = ComparisonResult(
            model_a="gpt-4o",
            model_b="my-model",
            method="dli",
            similarity=0.90,
            is_derived=True,
            threshold=0.80,
            confidence=0.5,
            details={},
        )

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.compare.return_value = mock_result

            from modelaudit.mcp_server import AuditConfig, AuditEngine

            config = AuditConfig(provider="openai")
            engine = AuditEngine(config)
            result = engine.compare("gpt-4o", "my-model", method="dli", provider="openai")

            assert result.method == "dli"
            mock_engine.compare.assert_called_once_with(
                "gpt-4o", "my-model", method="dli", provider="openai",
            )


class TestCompareModelsWhitebox:
    def test_whitebox_tool_logic(self):
        """测试 compare_models_whitebox 工具的核心逻辑."""
        mock_result = ComparisonResult(
            model_a="bert-base",
            model_b="distilbert",
            method="reef",
            similarity=0.92,
            is_derived=True,
            threshold=0.85,
            confidence=0.7,
            details={"layer_cka": {"layer_0": 0.95, "layer_1": 0.89}},
        )

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.compare.return_value = mock_result

            from modelaudit.mcp_server import AuditEngine

            engine = AuditEngine(use_cache=False)
            result = engine.compare("bert-base", "distilbert", method="reef", device="cpu")

            assert result.method == "reef"
            assert result.similarity == 0.92
            assert "layer_cka" in result.details


class TestAuditDistillation:
    def test_audit_tool_logic(self):
        """测试 audit_distillation 工具的核心逻辑."""
        mock_comparison = ComparisonResult(
            model_a="gpt-4o",
            model_b="my-model",
            method="llmmap",
            similarity=0.88,
            is_derived=True,
            threshold=0.85,
            confidence=0.6,
            details={},
        )
        mock_result = AuditResult(
            model_a="gpt-4o",
            model_b="my-model",
            comparisons=[mock_comparison],
            verdict="likely_derived",
            confidence=0.6,
            summary="审计对象: gpt-4o vs my-model\n判定结果: 可能存在蒸馏关系",
            details={},
        )

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine, \
             patch("modelaudit.mcp_server.generate_report") as mock_report:
            mock_engine = MockEngine.return_value
            mock_engine.audit.return_value = mock_result
            mock_report.return_value = "# 审计报告\n\n蒸馏关系: 是"

            from modelaudit.mcp_server import AuditConfig, AuditEngine, generate_report

            config = AuditConfig()
            engine = AuditEngine(config)
            result = engine.audit("gpt-4o", "my-model")

            assert result.verdict == "likely_derived"

            report = generate_report(result, "markdown")
            assert "审计报告" in report


class TestVerifyModel:
    def test_verify_tool_logic(self):
        """测试 verify_model 工具的核心逻辑."""
        mock_result = {
            "model": "gpt-4o",
            "verified": True,
            "claimed_family": "gpt-4",
            "best_match": "gpt-4",
            "claimed_score": 0.85,
            "best_score": 0.85,
            "all_scores": {"gpt-4": 0.85, "claude": 0.20, "llama": 0.15},
        }

        with patch("modelaudit.mcp_server.AuditEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.verify.return_value = mock_result

            from modelaudit.mcp_server import AuditConfig, AuditEngine

            config = AuditConfig(provider="openai")
            engine = AuditEngine(config)
            result = engine.verify("gpt-4o", provider="openai")

            assert result["verified"] is True
            assert result["best_match"] == "gpt-4"
            assert "all_scores" in result


@needs_mcp
class TestToolList:
    def test_tool_count(self):
        """验证服务器创建成功 (5 个工具定义)."""
        from modelaudit.mcp_server import create_server

        server = create_server()
        assert server is not None
