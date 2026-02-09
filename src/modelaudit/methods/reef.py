"""REEF 白盒指纹方法 — 基于 CKA (Centered Kernel Alignment).

参考 REEF (NeurIPS 2024):
通过比对模型中间层的表示相似性来检测蒸馏关系。
CKA 是一种衡量两组表示是否具有相同结构的方法。

核心思路:
1. 输入一组探测文本到两个模型
2. 提取各层的隐藏状态 (hidden states)
3. 用 CKA 计算逐层表示相似度
4. 汇总为蒸馏判定分数
"""

import hashlib
import logging
from typing import Any

from modelaudit.base import WhiteBoxFingerprinter
from modelaudit.models import ComparisonResult, Fingerprint
from modelaudit.registry import register

logger = logging.getLogger(__name__)

# REEF 探测用的短文本（覆盖多种语义场景）
_REEF_PROBES = [
    "The capital of France is",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the ethical implications of artificial intelligence?",
    "Translate 'hello world' into Chinese.",
    "Summarize the theory of relativity.",
    "1 + 1 =",
    "List three programming languages and their use cases.",
]


def _compute_cka(X: Any, Y: Any) -> float:
    """计算线性 CKA (Centered Kernel Alignment).

    Args:
        X: shape (n, p) — 模型 A 某一层对 n 个样本的表示
        Y: shape (n, q) — 模型 B 某一层对 n 个样本的表示

    Returns:
        CKA 相似度 [0, 1]
    """
    import numpy as np

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    # HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    n = X.shape[0]
    if n < 2:
        return 0.0

    # 中心化
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


def _extract_hidden_states(
    model_name_or_path: str,
    texts: list[str],
    device: str = "cpu",
    num_layers: int | None = None,
) -> list[list[list[float]]]:
    """提取模型的逐层隐藏状态.

    Returns:
        list of layers, each layer is list of per-sample pooled representations
        shape: (num_layers, num_samples, hidden_dim)
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "需要 torch + transformers 库。请运行: pip install knowlyr-modelaudit[whitebox]"
        )

    logger.info("加载模型: %s (device=%s)", model_name_or_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path, trust_remote_code=True, output_hidden_states=True
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]  # (batch, seq_len)

    # 选择要比对的层
    total_layers = len(hidden_states)
    if num_layers and num_layers < total_layers:
        # 均匀采样层
        indices = [int(i * (total_layers - 1) / (num_layers - 1)) for i in range(num_layers)]
    else:
        indices = list(range(total_layers))

    result: list[list[list[float]]] = []
    for idx in indices:
        hs = hidden_states[idx]  # (batch, seq_len, hidden_dim)
        # mean pooling (忽略 padding)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        result.append(pooled.cpu().tolist())

    return result


@register("reef")
class REEFFingerprinter(WhiteBoxFingerprinter):
    """REEF 白盒指纹 — 基于 CKA 中间层表示比对.

    需要访问模型权重（本地或 HuggingFace Hub）。
    参考: REEF (NeurIPS 2024)
    """

    @property
    def name(self) -> str:
        return "reef"

    def __init__(self, device: str = "cpu", num_layers: int = 8):
        self.device = device
        self.num_layers = num_layers
        self._model: str = ""

    def prepare(self, model: str, **kwargs) -> None:
        """设置目标模型路径或 HuggingFace 模型名."""
        self._model = model
        if "device" in kwargs:
            self.device = kwargs["device"]

    def get_fingerprint(self) -> Fingerprint:
        """提取模型白盒指纹 (逐层隐藏状态)."""
        if not self._model:
            raise RuntimeError("请先调用 prepare() 设置目标模型")

        hidden_states = _extract_hidden_states(
            self._model, _REEF_PROBES, device=self.device, num_layers=self.num_layers,
        )

        # 用隐藏状态的统计摘要作为指纹哈希
        import numpy as np

        flat = np.array([
            np.mean(layer, axis=0)[:16] for layer in hidden_states  # 取前16维的均值
        ]).flatten()
        fp_hash = hashlib.md5(flat.tobytes()).hexdigest()[:16]

        return Fingerprint(
            model_id=self._model,
            method="reef",
            fingerprint_type="whitebox",
            data={
                "hidden_states": hidden_states,
                "hash": fp_hash,
                "num_layers": len(hidden_states),
                "num_probes": len(_REEF_PROBES),
                "probe_texts": _REEF_PROBES,
            },
        )

    def compare(self, fp_a: Fingerprint, fp_b: Fingerprint) -> ComparisonResult:
        """用 CKA 比对两个 REEF 指纹."""
        hs_a = fp_a.data.get("hidden_states", [])
        hs_b = fp_b.data.get("hidden_states", [])

        if not hs_a or not hs_b:
            return ComparisonResult(
                model_a=fp_a.model_id,
                model_b=fp_b.model_id,
                method="reef",
                similarity=0.0,
                is_derived=False,
                threshold=0.85,
                confidence=0.0,
                details={"error": "缺少隐藏状态数据"},
            )

        # 逐层计算 CKA
        num_layers = min(len(hs_a), len(hs_b))
        layer_cka: list[float] = []
        for i in range(num_layers):
            cka = _compute_cka(hs_a[i], hs_b[i])
            layer_cka.append(cka)

        avg_cka = sum(layer_cka) / len(layer_cka) if layer_cka else 0.0
        threshold = 0.85

        return ComparisonResult(
            model_a=fp_a.model_id,
            model_b=fp_b.model_id,
            method="reef",
            similarity=round(avg_cka, 6),
            is_derived=avg_cka >= threshold,
            threshold=threshold,
            confidence=min(abs(avg_cka - threshold) / 0.15, 1.0),
            details={
                "layer_cka": [round(c, 6) for c in layer_cka],
                "num_layers_compared": num_layers,
                "hash_a": fp_a.data.get("hash", ""),
                "hash_b": fp_b.data.get("hash", ""),
            },
        )
