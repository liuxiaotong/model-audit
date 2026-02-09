"""测试指纹缓存."""

import json
import tempfile
from pathlib import Path

from modelaudit.cache import FingerprintCache
from modelaudit.models import Fingerprint


def _make_fp(model: str = "test-model", method: str = "llmmap") -> Fingerprint:
    """创建测试用指纹."""
    return Fingerprint(
        model_id=model,
        method=method,
        fingerprint_type="blackbox",
        data={"vector": {"avg_length_chars": 100.0}},
    )


class TestFingerprintCache:
    def test_put_and_get(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        fp = _make_fp()

        cache.put("test-model", "llmmap", "openai", fp)
        result = cache.get("test-model", "llmmap", "openai")

        assert result is not None
        assert result.model_id == "test-model"
        assert result.method == "llmmap"
        assert result.data["vector"]["avg_length_chars"] == 100.0

    def test_get_nonexistent(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        result = cache.get("no-such-model", "llmmap", "openai")
        assert result is None

    def test_get_corrupted_json(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # 写一个损坏的 JSON 文件
        bad_file = cache_dir / "llmmap_bad-model_openai.json"
        bad_file.write_text("not valid json{{{", encoding="utf-8")

        cache = FingerprintCache(str(cache_dir))
        result = cache.get("bad-model", "llmmap", "openai")
        assert result is None

    def test_list_entries_empty(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        entries = cache.list_entries()
        assert entries == []

    def test_list_entries(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        cache.put("model-a", "llmmap", "openai", _make_fp("model-a"))
        cache.put("model-b", "llmmap", "anthropic", _make_fp("model-b"))

        entries = cache.list_entries()
        assert len(entries) == 2
        models = {e["model"] for e in entries}
        assert "model-a" in models
        assert "model-b" in models

    def test_list_entries_with_corrupted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # 正常文件
        cache = FingerprintCache(str(cache_dir))
        cache.put("good-model", "llmmap", "openai", _make_fp("good-model"))
        # 损坏文件
        bad_file = cache_dir / "bad.json"
        bad_file.write_text("{invalid", encoding="utf-8")

        entries = cache.list_entries()
        assert len(entries) == 2
        # 损坏文件的字段应该是 "?"
        bad_entry = [e for e in entries if e["model"] == "?"]
        assert len(bad_entry) == 1

    def test_clear(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        cache.put("model-a", "llmmap", "openai", _make_fp("model-a"))
        cache.put("model-b", "llmmap", "openai", _make_fp("model-b"))

        count = cache.clear()
        assert count == 2
        assert cache.list_entries() == []

    def test_clear_empty(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        count = cache.clear()
        assert count == 0

    def test_key_sanitization(self):
        key = FingerprintCache._key("meta/llama:3.1", "llmmap", "openai")
        assert "/" not in key
        assert ":" not in key
        assert key == "llmmap_meta_llama_3.1_openai"

    def test_key_with_spaces(self):
        key = FingerprintCache._key("my model", "llmmap", "openai")
        assert " " not in key
        assert key == "llmmap_my_model_openai"

    def test_overwrite_existing(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        fp1 = _make_fp("model-x")
        fp2 = Fingerprint(
            model_id="model-x",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {"avg_length_chars": 999.0}},
        )

        cache.put("model-x", "llmmap", "openai", fp1)
        cache.put("model-x", "llmmap", "openai", fp2)

        result = cache.get("model-x", "llmmap", "openai")
        assert result is not None
        assert result.data["vector"]["avg_length_chars"] == 999.0

    def test_different_providers_different_keys(self, tmp_path):
        cache = FingerprintCache(str(tmp_path / "cache"))
        fp_openai = _make_fp("gpt-4o")
        fp_custom = Fingerprint(
            model_id="gpt-4o",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {"avg_length_chars": 200.0}},
        )

        cache.put("gpt-4o", "llmmap", "openai", fp_openai)
        cache.put("gpt-4o", "llmmap", "custom", fp_custom)

        r1 = cache.get("gpt-4o", "llmmap", "openai")
        r2 = cache.get("gpt-4o", "llmmap", "custom")

        assert r1 is not None
        assert r2 is not None
        assert r1.data["vector"]["avg_length_chars"] == 100.0
        assert r2.data["vector"]["avg_length_chars"] == 200.0
