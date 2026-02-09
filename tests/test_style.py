"""测试风格分析方法."""

from modelaudit.methods.style import (
    _compute_style_scores,
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
