"""测试审计引擎."""

from unittest.mock import patch

from modelaudit.config import AuditConfig
from modelaudit.engine import AuditEngine
from modelaudit.models import Fingerprint


class TestAuditEngine:
    def test_init_default(self):
        engine = AuditEngine()
        assert engine.config.blackbox_method == "llmmap"

    def test_init_custom_config(self):
        config = AuditConfig(provider="anthropic", similarity_threshold=0.9)
        engine = AuditEngine(config)
        assert engine.config.provider == "anthropic"

    def test_detect(self):
        engine = AuditEngine()
        texts = [
            "Certainly! I'd be happy to help with that question.",
            "I think that's a very interesting perspective to consider.",
        ]
        results = engine.detect(texts)
        assert len(results) == 2
        assert all(r.predicted_model for r in results)

    def test_detect_empty(self):
        engine = AuditEngine()
        results = engine.detect([])
        assert results == []


class TestAuditConfig:
    def test_default_values(self):
        config = AuditConfig()
        assert config.provider == "openai"
        assert config.similarity_threshold == 0.85
        assert config.output_format == "markdown"
        assert config.cache_ttl == 0

    def test_custom_values(self):
        config = AuditConfig(
            provider="anthropic",
            similarity_threshold=0.9,
            num_probes=4,
        )
        assert config.provider == "anthropic"
        assert config.num_probes == 4

    def test_cache_ttl(self):
        config = AuditConfig(cache_ttl=3600)
        assert config.cache_ttl == 3600

    def test_cache_ttl_passed_to_engine(self):
        config = AuditConfig(cache_ttl=7200)
        engine = AuditEngine(config, use_cache=True)
        assert engine.cache is not None
        assert engine.cache.ttl == 7200


class TestAuditDLIIntegration:
    """测试 audit() 多方法集成."""

    def test_audit_includes_dli_comparison(self):
        """audit 结果应包含 LLMmap 和 DLI 两个比对."""
        # Mock fingerprint 返回带 raw_responses 的数据
        mock_fp = Fingerprint(
            model_id="test",
            method="llmmap",
            fingerprint_type="blackbox",
            data={
                "vector": {
                    "avg_length_chars": 100.0,
                    "avg_length_words": 20.0,
                    "avg_length_sentences": 3.0,
                    "avg_avg_word_length": 5.0,
                    "avg_avg_sentence_length": 15.0,
                    "avg_unique_word_ratio": 0.7,
                    "avg_punctuation_ratio": 0.03,
                    "avg_newline_ratio": 0.01,
                },
                "raw_responses": [
                    "Certainly! Here's the answer.",
                    "I'd be happy to help with that.",
                    "Let me explain this concept.",
                ],
                "probe_ids": ["p1", "p2", "p3"],
            },
        )

        with patch.object(AuditEngine, "fingerprint", return_value=mock_fp):
            engine = AuditEngine(use_cache=False)
            result = engine.audit("model-a", "model-b")

            methods = [c.method for c in result.comparisons]
            assert "llmmap" in methods
            assert "dli" in methods
            assert len(result.comparisons) >= 2
