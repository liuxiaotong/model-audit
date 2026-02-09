"""测试方法注册表."""

import pytest

import modelaudit.methods  # noqa: F401 触发注册
from modelaudit.registry import get_fingerprinter, list_methods


class TestRegistry:
    def test_llmmap_registered(self):
        methods = list_methods()
        assert "llmmap" in methods
        assert methods["llmmap"] == "blackbox"

    def test_get_llmmap(self):
        fp = get_fingerprinter("llmmap")
        assert fp.name == "llmmap"
        assert fp.fingerprint_type == "blackbox"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="未知指纹方法"):
            get_fingerprinter("nonexistent")

    def test_list_methods_not_empty(self):
        methods = list_methods()
        assert len(methods) > 0
