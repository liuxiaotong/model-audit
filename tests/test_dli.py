"""测试 DLI 蒸馏检测方法."""

import pytest

from modelaudit.methods.dli import (
    DLIFingerprinter,
    _compute_behavior_similarity,
    _extract_behavior_signature,
    _extract_ngrams,
    _js_divergence,
)
from modelaudit.models import Fingerprint


class TestExtractNgrams:
    def test_basic(self):
        ngrams = _extract_ngrams("hello world foo bar", n=2)
        assert ngrams["hello world"] == 1
        assert ngrams["world foo"] == 1

    def test_empty(self):
        assert _extract_ngrams("", n=2) == {}

    def test_single_word(self):
        assert _extract_ngrams("hello", n=2) == {}

    def test_repeated(self):
        ngrams = _extract_ngrams("a b a b a b", n=2)
        assert ngrams["a b"] == 3
        assert ngrams["b a"] == 2


class TestJSDivergence:
    def test_identical(self):
        p = {"a": 0.5, "b": 0.5}
        assert abs(_js_divergence(p, p)) < 1e-10

    def test_different(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        js = _js_divergence(p, q)
        assert js > 0

    def test_empty(self):
        assert _js_divergence({}, {}) == 0.0

    def test_partial_overlap(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.3, "c": 0.7}
        js = _js_divergence(p, q)
        assert 0 < js


class TestExtractBehaviorSignature:
    def test_basic(self):
        responses = [
            "Certainly! Here's the answer to your question.",
            "I think that's a great question. Let me explain.",
        ]
        sig = _extract_behavior_signature(responses)
        assert "bigram_dist" in sig
        assert "features" in sig
        assert sig["features"]["avg_length"] > 0
        assert 0 <= sig["features"]["vocab_diversity"] <= 1

    def test_empty(self):
        sig = _extract_behavior_signature([])
        assert sig["bigram_dist"] == {}
        assert sig["features"] == {}

    def test_refusal_detection(self):
        responses = [
            "I cannot help with that.",
            "I apologize, but I'm unable to assist.",
            "Sure, here's the info.",
        ]
        sig = _extract_behavior_signature(responses)
        assert sig["features"]["refusal_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_markdown_detection(self):
        responses = [
            "# Title\n\nSome content",
            "Just plain text here",
        ]
        sig = _extract_behavior_signature(responses)
        assert sig["features"]["markdown_rate"] == 0.5


class TestBehaviorSimilarity:
    def test_identical_signatures(self):
        sig = _extract_behavior_signature([
            "Hello world, this is a test response.",
            "Another response with some content.",
        ])
        sim = _compute_behavior_similarity(sig, sig)
        assert abs(sim - 1.0) < 0.01

    def test_different_signatures(self):
        sig_a = _extract_behavior_signature([
            "Certainly! I'd be happy to help with that.",
        ])
        sig_b = _extract_behavior_signature([
            "I cannot help with that request. I apologize.",
        ])
        sim = _compute_behavior_similarity(sig_a, sig_b)
        assert sim < 1.0

    def test_empty_signatures(self):
        sig = {"bigram_dist": {}, "features": {}}
        sim = _compute_behavior_similarity(sig, sig)
        assert sim >= 0


class TestDLIFingerprinter:
    def test_init(self):
        fp = DLIFingerprinter(provider="openai")
        assert fp.name == "dli"
        assert fp.fingerprint_type == "blackbox"

    def test_prepare(self):
        fp = DLIFingerprinter()
        fp.prepare("gpt-4o")
        assert fp._model == "gpt-4o"

    def test_unprepared_raises(self):
        fp = DLIFingerprinter()
        with pytest.raises(RuntimeError, match="prepare"):
            fp.get_fingerprint()

    def test_compare_identical(self):
        fp = DLIFingerprinter()
        sig = _extract_behavior_signature([
            "Hello, this is a test.",
            "Another response here.",
        ])
        fingerprint = Fingerprint(
            model_id="test",
            method="dli",
            fingerprint_type="blackbox",
            data={"signature": sig},
        )
        result = fp.compare(fingerprint, fingerprint)
        assert result.similarity > 0.9
        assert result.method == "dli"

    def test_compare_empty(self):
        fp = DLIFingerprinter()
        fp_a = Fingerprint(
            model_id="a", method="dli",
            fingerprint_type="blackbox", data={},
        )
        fp_b = Fingerprint(
            model_id="b", method="dli",
            fingerprint_type="blackbox", data={},
        )
        result = fp.compare(fp_a, fp_b)
        assert 0 <= result.similarity <= 1
