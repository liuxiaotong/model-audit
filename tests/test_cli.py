"""测试 CLI."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from modelaudit.cli import main


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.2" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ModelAudit" in result.output

    def test_methods(self):
        runner = CliRunner()
        result = runner.invoke(main, ["methods"])
        assert result.exit_code == 0
        assert "llmmap" in result.output

    def test_detect_jsonl(self):
        runner = CliRunner()

        # 创建临时 JSONL 文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "Certainly! I'd be happy to help."}) + "\n")
            f.write(json.dumps({"text": "I think that's an interesting question."}) + "\n")
            tmp_path = f.name

        try:
            result = runner.invoke(main, ["detect", tmp_path])
            assert result.exit_code == 0
            assert "来源分布" in result.output
        finally:
            Path(tmp_path).unlink()

    def test_detect_json(self):
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"text": "Hello world."},
                {"text": "Sure thing! Here is the answer."},
            ], f)
            tmp_path = f.name

        try:
            result = runner.invoke(main, ["detect", tmp_path])
            assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink()

    def test_detect_with_output(self):
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "Test text"}) + "\n")
            tmp_input = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_output = f.name

        try:
            result = runner.invoke(main, ["detect", tmp_input, "-o", tmp_output, "-f", "json"])
            assert result.exit_code == 0
            # 验证输出文件
            output_data = json.loads(Path(tmp_output).read_text())
            assert isinstance(output_data, list)
        finally:
            Path(tmp_input).unlink()
            Path(tmp_output).unlink(missing_ok=True)

    def test_detect_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["detect", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0

    def test_cache_list_empty(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(tmp_path / "empty")])
        assert result.exit_code == 0
        assert "缓存为空" in result.output

    def test_cache_list_with_entries(self, tmp_path):
        from modelaudit.cache import FingerprintCache
        from modelaudit.models import Fingerprint

        cache_dir = tmp_path / "cache"
        cache = FingerprintCache(str(cache_dir))
        fp = Fingerprint(
            model_id="test-model",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {}},
        )
        cache.put("test-model", "llmmap", "openai", fp)

        runner = CliRunner()
        result = runner.invoke(main, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "test-model" in result.output
        assert "1 条指纹" in result.output

    def test_cache_clear(self, tmp_path):
        from modelaudit.cache import FingerprintCache
        from modelaudit.models import Fingerprint

        cache_dir = tmp_path / "cache"
        cache = FingerprintCache(str(cache_dir))
        fp = Fingerprint(
            model_id="test-model",
            method="llmmap",
            fingerprint_type="blackbox",
            data={"vector": {}},
        )
        cache.put("test-model", "llmmap", "openai", fp)

        runner = CliRunner()
        result = runner.invoke(main, ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"])
        assert result.exit_code == 0
        assert "已清除 1 条缓存" in result.output

    def test_cache_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "--help"])
        assert result.exit_code == 0
        assert "管理指纹缓存" in result.output
