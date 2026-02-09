"""测试 LLMmap 黑盒指纹方法."""

from unittest.mock import patch

import pytest

from modelaudit.methods.llmmap import (
    LLMmapFingerprinter,
    _call_model_api,
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


class TestRetryLogic:
    @patch("modelaudit.methods.llmmap._call_model_api_once")
    @patch("modelaudit.methods.llmmap._backoff_sleep")
    def test_retry_on_exception(self, mock_sleep, mock_api):
        mock_api.side_effect = [ConnectionError("network"), "Success response"]
        result = _call_model_api("model", "prompt", max_retries=3)
        assert result == "Success response"
        assert mock_api.call_count == 2
        mock_sleep.assert_called_once()

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    @patch("modelaudit.methods.llmmap._backoff_sleep")
    def test_retry_on_empty_response(self, mock_sleep, mock_api):
        mock_api.side_effect = ["", "  ", "Valid response"]
        result = _call_model_api("model", "prompt", max_retries=3)
        assert result == "Valid response"
        assert mock_api.call_count == 3

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    @patch("modelaudit.methods.llmmap._backoff_sleep")
    def test_raise_after_max_retries(self, mock_sleep, mock_api):
        mock_api.side_effect = ConnectionError("network")
        with pytest.raises(ConnectionError):
            _call_model_api("model", "prompt", max_retries=2)
        assert mock_api.call_count == 2

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    def test_no_retry_on_import_error(self, mock_api):
        mock_api.side_effect = ImportError("no module")
        with pytest.raises(ImportError):
            _call_model_api("model", "prompt", max_retries=3)
        assert mock_api.call_count == 1

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    def test_no_retry_on_value_error(self, mock_api):
        mock_api.side_effect = ValueError("bad value")
        with pytest.raises(ValueError):
            _call_model_api("model", "prompt", max_retries=3)
        assert mock_api.call_count == 1

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    def test_success_on_first_try(self, mock_api):
        mock_api.return_value = "OK"
        result = _call_model_api("model", "prompt", max_retries=3)
        assert result == "OK"
        assert mock_api.call_count == 1

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    def test_auth_error_no_retry(self, mock_api):
        """401/403 认证错误应立即抛出, 不重试."""
        mock_api.side_effect = Exception("Error 401 Unauthorized")
        with pytest.raises(ValueError, match="API 认证失败"):
            _call_model_api("model", "prompt", max_retries=3)
        assert mock_api.call_count == 1

    @patch("modelaudit.methods.llmmap._call_model_api_once")
    @patch("modelaudit.methods.llmmap._backoff_sleep")
    def test_rate_limit_retries_with_longer_backoff(self, mock_sleep, mock_api):
        """429 速率限制应重试, 且退避更长."""
        mock_api.side_effect = [Exception("429 rate limit"), "OK"]
        result = _call_model_api("model", "prompt", max_retries=3)
        assert result == "OK"
        # 速率限制时 attempt+1 传入 backoff, 所以退避更长
        mock_sleep.assert_called_once_with(1)
