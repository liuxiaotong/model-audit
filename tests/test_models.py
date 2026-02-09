"""测试数据模型."""

from modelaudit.models import (
    AuditResult,
    ComparisonResult,
    DetectionResult,
    Fingerprint,
)


class TestFingerprint:
    def test_create(self):
        fp = Fingerprint(
            model_id="gpt-4o",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {"a": 1.0, "b": 2.0}},
        )
        assert fp.model_id == "gpt-4o"
        assert fp.method == "llmmap"
        assert fp.fingerprint_type == "blackbox"

    def test_serialize(self):
        fp = Fingerprint(
            model_id="test",
            method="reef",
            fingerprint_type="whitebox",
            data={"cka": [0.99, 0.98]},
        )
        d = fp.model_dump()
        assert d["model_id"] == "test"
        assert d["data"]["cka"] == [0.99, 0.98]

    def test_metadata(self):
        fp = Fingerprint(
            model_id="test",
            method="llmmap",
            fingerprint_type="blackbox",
            data={},
            metadata={"source": "api"},
        )
        assert fp.metadata["source"] == "api"


class TestComparisonResult:
    def test_create_derived(self):
        result = ComparisonResult(
            model_a="gpt-4o",
            model_b="my-model",
            method="llmmap",
            similarity=0.95,
            is_derived=True,
            threshold=0.85,
            confidence=0.8,
        )
        assert result.is_derived
        assert result.similarity == 0.95

    def test_create_independent(self):
        result = ComparisonResult(
            model_a="gpt-4o",
            model_b="llama-3",
            method="llmmap",
            similarity=0.3,
            is_derived=False,
            threshold=0.85,
        )
        assert not result.is_derived


class TestDetectionResult:
    def test_create(self):
        result = DetectionResult(
            text_id=0,
            text_preview="Hello world...",
            predicted_model="gpt-4",
            confidence=0.75,
            scores={"gpt-4": 0.75, "claude": 0.6},
        )
        assert result.predicted_model == "gpt-4"
        assert result.scores["claude"] == 0.6


class TestAuditResult:
    def test_create(self):
        result = AuditResult(
            model_a="teacher",
            model_b="student",
            verdict="likely_derived",
            confidence=0.9,
            summary="test summary",
        )
        assert result.verdict == "likely_derived"

    def test_default_verdict(self):
        result = AuditResult(model_a="a", model_b="b")
        assert result.verdict == "inconclusive"
        assert result.confidence == 0.0
