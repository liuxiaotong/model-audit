"""测试 benchmark 数据集与准确率评估."""

from modelaudit.benchmark import (
    BENCHMARK_SAMPLES,
    BenchmarkSample,
    evaluate_accuracy,
    get_benchmark_samples,
)


class TestBenchmarkSamples:
    def test_samples_not_empty(self):
        assert len(BENCHMARK_SAMPLES) > 0

    def test_sample_structure(self):
        for s in BENCHMARK_SAMPLES:
            assert isinstance(s, BenchmarkSample)
            assert s.text
            assert s.label
            assert s.category

    def test_multiple_labels(self):
        labels = {s.label for s in BENCHMARK_SAMPLES}
        assert len(labels) >= 4  # 至少 4 个模型家族

    def test_multiple_categories(self):
        categories = {s.category for s in BENCHMARK_SAMPLES}
        assert len(categories) >= 3  # 至少 3 个类别


class TestGetBenchmarkSamples:
    def test_all(self):
        samples = get_benchmark_samples()
        assert len(samples) == len(BENCHMARK_SAMPLES)

    def test_filter_by_category(self):
        qa = get_benchmark_samples(category="qa")
        assert all(s.category == "qa" for s in qa)
        assert len(qa) > 0

    def test_filter_by_label(self):
        gpt = get_benchmark_samples(label="gpt-4")
        assert all(s.label == "gpt-4" for s in gpt)
        assert len(gpt) > 0

    def test_filter_both(self):
        samples = get_benchmark_samples(category="qa", label="claude")
        assert all(s.category == "qa" and s.label == "claude" for s in samples)

    def test_no_match(self):
        assert get_benchmark_samples(label="nonexistent") == []


class TestEvaluateAccuracy:
    def test_perfect(self):
        preds = [("a", "a"), ("b", "b"), ("c", "c")]
        result = evaluate_accuracy(preds)
        assert result["accuracy"] == 1.0
        assert result["correct"] == 3

    def test_zero(self):
        preds = [("a", "b"), ("b", "c")]
        result = evaluate_accuracy(preds)
        assert result["accuracy"] == 0.0

    def test_partial(self):
        preds = [("a", "a"), ("b", "c")]
        result = evaluate_accuracy(preds)
        assert result["accuracy"] == 0.5

    def test_empty(self):
        result = evaluate_accuracy([])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_per_class(self):
        preds = [("a", "a"), ("a", "a"), ("b", "a")]
        result = evaluate_accuracy(preds)
        assert result["per_class"]["a"] == 2 / 3
