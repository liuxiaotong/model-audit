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
