"""测试 publish workflow YAML 有效性."""

from pathlib import Path

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

needs_yaml = pytest.mark.skipif(not HAS_YAML, reason="需要 PyYAML")


class TestPublishWorkflow:
    def test_workflow_exists(self):
        path = Path(__file__).parent.parent / ".github" / "workflows" / "publish.yml"
        assert path.exists()

    @needs_yaml
    def test_workflow_valid_yaml(self):
        path = Path(__file__).parent.parent / ".github" / "workflows" / "publish.yml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # PyYAML parses bare `on` as boolean True
        assert "on" in data or True in data
        assert "jobs" in data

    @needs_yaml
    def test_trigger_on_tags(self):
        path = Path(__file__).parent.parent / ".github" / "workflows" / "publish.yml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # PyYAML parses bare `on` as boolean True
        on_key = "on" if "on" in data else True
        assert "push" in data[on_key]
        assert "tags" in data[on_key]["push"]

    def test_ci_workflow_exists(self):
        path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        assert path.exists()
