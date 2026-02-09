"""测试风格分析方法."""

from modelaudit.methods.style import (
    _compute_style_scores,
    _detect_lang,
    compute_style_fingerprint,
    detect_text_source,
)


class TestComputeStyleScores:
    def test_gpt_style(self):
        text = (
            "Certainly! I'd be happy to help with that. Here's what you need to know. "
            "It's worth noting that this is a complex topic. In summary, the key points are: "
            "1. First point\n2. Second point\n3. Third point"
        )
        scores = _compute_style_scores(text)
        assert "gpt-4" in scores
        assert "claude" in scores
        assert scores["gpt-4"] > 0

    def test_claude_style(self):
        text = (
            "That's an interesting question. I think there are several perspectives to consider. "
            "That said, let me think about this carefully. Here's my analysis of the situation."
        )
        scores = _compute_style_scores(text)
        assert scores["claude"] > 0

    def test_refusal_style(self):
        text = "I cannot assist with that. As an AI language model, I don't have the ability to do that."
        scores = _compute_style_scores(text)
        # 应该有分数
        assert any(s > 0 for s in scores.values())

    def test_empty_text(self):
        scores = _compute_style_scores("")
        assert isinstance(scores, dict)


class TestDetectTextSource:
    def test_single_text(self):
        results = detect_text_source(["Certainly! Here's the answer to your question."])
        assert len(results) == 1
        assert results[0].predicted_model != ""
        assert 0 <= results[0].confidence <= 1

    def test_multiple_texts(self):
        texts = [
            "Certainly! I'd be happy to help with that.",
            "I think that's an interesting question. That said, let me share my perspective.",
            "Sure thing! No problem at all. Here is the information you need.",
        ]
        results = detect_text_source(texts)
        assert len(results) == 3
        for r in results:
            assert r.predicted_model in r.scores

    def test_text_preview(self):
        long_text = "A" * 200
        results = detect_text_source([long_text])
        assert len(results[0].text_preview) <= 83  # 80 + "..."


class TestComputeStyleFingerprint:
    def test_basic(self):
        texts = [
            "Certainly! Here's the answer.",
            "Of course! Here's what you need to know.",
        ]
        fingerprint = compute_style_fingerprint(texts)
        assert isinstance(fingerprint, dict)
        assert len(fingerprint) > 0
        assert all(isinstance(v, float) for v in fingerprint.values())


class TestDetectLang:
    def test_english(self):
        assert _detect_lang("Hello, this is a test.") == "en"

    def test_chinese(self):
        assert _detect_lang("这是一个中文测试文本，用来检测语言识别功能。") == "zh"

    def test_code_heavy_chinese(self):
        """代码多但包含中文注释的文本应识别为中文."""
        text = "```python\ndef foo():\n    pass\n```\n\n这个实现使用了动态规划的思路。"
        assert _detect_lang(text) == "zh"

    def test_empty(self):
        assert _detect_lang("") == "en"


class TestBenchmarkAccuracy:
    def test_all_benchmark_correct(self):
        """所有 benchmark 样本应被正确分类."""
        from modelaudit.benchmark import BENCHMARK_SAMPLES

        for sample in BENCHMARK_SAMPLES:
            scores = _compute_style_scores(sample.text)
            predicted = max(scores, key=lambda k: scores[k])
            assert predicted == sample.label, (
                f"Expected {sample.label}, got {predicted} "
                f"(category={sample.category})"
            )


class TestStructuralScoring:
    def test_plain_text_no_structural_bias(self):
        """纯文本不应因结构特征偏向任何模型."""
        scores = _compute_style_scores("A simple short text.")
        # 所有英文模型应只有 lang 分 (0.20), 无结构加分
        for model in ("mistral", "phi"):
            assert scores[model] == 0.20

    def test_code_blocks_boost(self):
        """含代码块的文本应给 tends_code_blocks=True 的模型加分."""
        text = "Here is some code:\n```python\nprint('hi')\n```"
        scores = _compute_style_scores(text)
        # gpt-4 tends_code_blocks=True, gemini tends_code_blocks=False
        assert scores["gpt-4"] > scores["gemini"]
