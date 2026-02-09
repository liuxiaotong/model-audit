"""测试 LLMmap 黑盒指纹方法."""

from modelaudit.methods.llmmap import (
    LLMmapFingerprinter,
    _compute_fingerprint_vector,
    _cosine_similarity,
    _extract_response_features,
)
from modelaudit.models import Fingerprint


class TestExtractResponseFeatures:
    def test_basic(self):
        features = _extract_response_features("Hello, this is a simple test response.")
        assert features["length_words"] == 7
        assert features["length_chars"] > 0
        assert "marker_scores" in features

    def test_refusal(self):
        features = _extract_response_features(
            "I cannot help with that request. As an AI language model, I have limitations."
        )
        assert features["starts_with_refusal"] is True

    def test_markdown(self):
        features = _extract_response_features("# Title\n\nSome content\n\n- item 1\n- item 2")
        assert features["has_markdown_headers"] is True
        assert features["has_bullet_points"] is True

    def test_numbered_list(self):
        features = _extract_response_features("1. First\n2. Second\n3. Third")
        assert features["has_numbered_list"] is True

    def test_code_block(self):
        features = _extract_response_features("Here is code:\n```python\nprint('hi')\n```")
        assert features["has_code_blocks"] is True

    def test_empty(self):
        features = _extract_response_features("")
        assert features["length_words"] == 0


class TestCosineSimilarity:
    def test_identical(self):
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal(self):
        vec_a = {"a": 1.0, "b": 0.0}
        vec_b = {"a": 0.0, "b": 1.0}
        assert abs(_cosine_similarity(vec_a, vec_b)) < 1e-6

    def test_similar(self):
        vec_a = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec_b = {"a": 1.1, "b": 2.1, "c": 2.9}
        sim = _cosine_similarity(vec_a, vec_b)
        assert sim > 0.99

    def test_empty(self):
        assert _cosine_similarity({}, {}) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity({"a": 0}, {"a": 1}) == 0.0


class TestComputeFingerprintVector:
    def test_basic(self):
        features = [
            _extract_response_features("Hello world, this is a test."),
            _extract_response_features("Another response here with more words."),
        ]
        vector = _compute_fingerprint_vector(features)
        assert "avg_length_words" in vector
        assert "avg_avg_word_length" in vector
        assert "ratio_has_bullet_points" in vector

    def test_single_feature(self):
        features = [_extract_response_features("Short text.")]
        vector = _compute_fingerprint_vector(features)
        assert isinstance(vector, dict)
        assert len(vector) > 0


class TestLLMmapFingerprinter:
    def test_init(self):
        fp = LLMmapFingerprinter(provider="openai")
        assert fp.name == "llmmap"
        assert fp.fingerprint_type == "blackbox"

    def test_prepare(self):
        fp = LLMmapFingerprinter()
        fp.prepare("gpt-4o")
        assert fp._model == "gpt-4o"

    def test_compare_identical(self):
        fp = LLMmapFingerprinter()
        fingerprint = Fingerprint(
            model_id="test",
            method="llmmap",
            fingerprint_type="blackbox",
            data={
                "vector": {"a": 1.0, "b": 2.0},
                "hash": "abc123",
            },
        )
        result = fp.compare(fingerprint, fingerprint)
        assert abs(result.similarity - 1.0) < 1e-6
        assert result.is_derived is True

    def test_compare_different(self):
        fp = LLMmapFingerprinter()
        fp_a = Fingerprint(
            model_id="model_a",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {"a": 1.0, "b": 0.0}, "hash": "aaa"},
        )
        fp_b = Fingerprint(
            model_id="model_b",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {"a": 0.0, "b": 1.0}, "hash": "bbb"},
        )
        result = fp.compare(fp_a, fp_b)
        assert result.similarity < 0.5
        assert result.is_derived is False
