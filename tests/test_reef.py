"""测试 REEF 白盒指纹方法."""

import pytest

from modelaudit.methods.reef import _REEF_PROBES, REEFFingerprinter
from modelaudit.models import Fingerprint

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="需要 numpy")


@needs_numpy
class TestComputeCKA:
    def test_identical_representations(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cka = _compute_cka(X, X)
        assert abs(cka - 1.0) < 1e-6

    def test_different_representations(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        Y = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        cka = _compute_cka(X, Y)
        assert 0 <= cka <= 1

    def test_orthogonal(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        Y = [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        cka = _compute_cka(X, Y)
        assert 0 <= cka <= 1

    def test_single_sample(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[1.0, 2.0]]
        cka = _compute_cka(X, X)
        assert cka == 0.0  # n < 2

    def test_zero_matrix(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        cka = _compute_cka(X, X)
        assert cka == 0.0

    def test_different_dimensions(self):
        from modelaudit.methods.reef import _compute_cka

        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        Y = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cka = _compute_cka(X, Y)
        assert 0 <= cka <= 1


class TestREEFFingerprinter:
    def test_init(self):
        fp = REEFFingerprinter(device="cpu")
        assert fp.name == "reef"
        assert fp.fingerprint_type == "whitebox"

    def test_prepare(self):
        fp = REEFFingerprinter()
        fp.prepare("bert-base-uncased")
        assert fp._model == "bert-base-uncased"

    def test_prepare_with_device(self):
        fp = REEFFingerprinter()
        fp.prepare("bert-base-uncased", device="cuda")
        assert fp.device == "cuda"

    def test_unprepared_raises(self):
        fp = REEFFingerprinter()
        with pytest.raises(RuntimeError, match="prepare"):
            fp.get_fingerprint()

    def test_compare_empty_data(self):
        fp = REEFFingerprinter()
        fp_a = Fingerprint(
            model_id="model-a", method="reef",
            fingerprint_type="whitebox", data={},
        )
        fp_b = Fingerprint(
            model_id="model-b", method="reef",
            fingerprint_type="whitebox", data={},
        )
        result = fp.compare(fp_a, fp_b)
        assert result.similarity == 0.0
        assert result.is_derived is False

    @needs_numpy
    def test_compare_with_hidden_states(self):
        fp = REEFFingerprinter()
        # 模拟 2 层 x 3 样本 x 2 维 的隐藏状态
        hs = [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ]
        fp_a = Fingerprint(
            model_id="model-a", method="reef",
            fingerprint_type="whitebox",
            data={"hidden_states": hs, "hash": "aaa"},
        )
        fp_b = Fingerprint(
            model_id="model-b", method="reef",
            fingerprint_type="whitebox",
            data={"hidden_states": hs, "hash": "aaa"},
        )
        result = fp.compare(fp_a, fp_b)
        assert abs(result.similarity - 1.0) < 1e-6
        assert result.is_derived is True
        assert "layer_cka" in result.details

    @needs_numpy
    def test_compare_different_hidden_states(self):
        fp = REEFFingerprinter()
        hs_a = [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ]
        hs_b = [
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]
        fp_a = Fingerprint(
            model_id="model-a", method="reef",
            fingerprint_type="whitebox",
            data={"hidden_states": hs_a, "hash": "aaa"},
        )
        fp_b = Fingerprint(
            model_id="model-b", method="reef",
            fingerprint_type="whitebox",
            data={"hidden_states": hs_b, "hash": "bbb"},
        )
        result = fp.compare(fp_a, fp_b)
        assert 0 <= result.similarity <= 1
        assert result.method == "reef"


class TestREEFProbes:
    def test_probes_not_empty(self):
        assert len(_REEF_PROBES) > 0

    def test_probes_are_strings(self):
        assert all(isinstance(p, str) for p in _REEF_PROBES)
