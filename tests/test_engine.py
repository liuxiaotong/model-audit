"""测试审计引擎."""

from modelaudit.config import AuditConfig
from modelaudit.engine import AuditEngine


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
