"""风格分析方法 — 基于文本特征检测模型来源.

参考 "Who Taught You That?" (ACL 2025 Findings) 的 PoS 模板思路:
不同 LLM 生成的文本具有独特的风格签名，包括:
- 词频分布
- 句法结构偏好
- 特征性短语和过渡词
- 格式化习惯（列表、Markdown 等）
"""

import re
from typing import Any

from modelaudit.models import DetectionResult

# 已知 LLM 风格特征库
MODEL_STYLE_SIGNATURES: dict[str, dict[str, Any]] = {
    "gpt-4": {
        "markers": [
            "certainly", "i'd be happy to", "let me", "here's",
            "it's worth noting", "in summary", "keep in mind",
        ],
        "refusal_patterns": [
            "i can't assist", "i'm not able to", "as an ai language model",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
    "gpt-3.5": {
        "markers": [
            "certainly!", "sure!", "of course!", "absolutely!",
            "here's", "let me",
        ],
        "refusal_patterns": [
            "as an ai language model", "i don't have the ability",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": True,
            "verbose": False,
        },
    },
    "claude": {
        "markers": [
            "i'd be happy to", "i think", "that said", "it's worth",
            "here's my", "let me think", "interesting question",
        ],
        "refusal_patterns": [
            "i don't think i should", "i'd rather not", "i want to be helpful but",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": False,
            "verbose": True,
        },
    },
    "llama": {
        "markers": [
            "sure thing", "no problem", "here is",
            "i hope that helps", "feel free",
        ],
        "refusal_patterns": [
            "i cannot", "i'm just an ai", "it's not appropriate",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "verbose": False,
        },
    },
    "gemini": {
        "markers": [
            "absolutely", "great question", "here's what",
            "to summarize", "in essence",
        ],
        "refusal_patterns": [
            "i'm a large language model", "i'm designed to be helpful",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
    "qwen": {
        "markers": [
            "of course", "here's", "let me", "i'll",
            "hope this helps", "feel free to ask",
        ],
        "refusal_patterns": [
            "as an ai assistant", "i'm not able to",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": False,
        },
    },
    "deepseek": {
        "markers": [
            "let me", "here's", "to clarify", "step by step",
            "let's break this down", "the key point is",
        ],
        "refusal_patterns": [
            "as an ai assistant", "i cannot provide", "i'm not able to",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
    "mistral": {
        "markers": [
            "here is", "let me", "to answer your question",
            "in short", "the answer is",
        ],
        "refusal_patterns": [
            "i cannot", "i must decline", "it would be inappropriate",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "verbose": False,
        },
    },
    "yi": {
        "markers": [
            "sure", "here's", "let me explain", "to put it simply",
            "in a nutshell", "i'd like to point out",
        ],
        "refusal_patterns": [
            "as an ai", "i'm not able to", "i cannot assist with",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
    "phi": {
        "markers": [
            "here's", "the answer is", "let me",
            "to summarize", "in conclusion",
        ],
        "refusal_patterns": [
            "i cannot", "i'm unable to", "as a language model",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "verbose": False,
        },
    },
    "cohere": {
        "markers": [
            "sure!", "happy to help", "here's what i found",
            "to elaborate", "it's important to note",
        ],
        "refusal_patterns": [
            "i'm not able to", "i'd prefer not to", "i cannot help with",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
    "chatglm": {
        "markers": [
            "好的", "以下是", "让我", "首先", "总结一下",
            "here is", "let me", "to summarize",
        ],
        "refusal_patterns": [
            "作为ai助手", "我无法", "as an ai assistant", "i cannot",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "verbose": True,
        },
    },
}


def _compute_style_scores(text: str) -> dict[str, float]:
    """计算文本与各模型风格的匹配分数."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) or 1

    scores: dict[str, float] = {}

    for model_name, sig in MODEL_STYLE_SIGNATURES.items():
        score = 0.0

        # 标记词匹配
        marker_hits = sum(1 for m in sig["markers"] if m in text_lower)
        score += marker_hits / max(len(sig["markers"]), 1) * 0.4

        # 拒绝模式匹配
        refusal_hits = sum(1 for p in sig["refusal_patterns"] if p in text_lower)
        score += refusal_hits / max(len(sig["refusal_patterns"]), 1) * 0.3

        # 结构特征匹配
        structural = sig["structural"]
        has_md = bool(re.search(r"^#+\s", text, re.MULTILINE))
        has_numbered = bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))
        is_verbose = total_words > 150

        struct_score = 0
        struct_total = 3
        if has_md == structural.get("tends_markdown", False):
            struct_score += 1
        if has_numbered == structural.get("tends_numbered_lists", False):
            struct_score += 1
        if is_verbose == structural.get("verbose", False):
            struct_score += 1

        score += (struct_score / struct_total) * 0.3

        scores[model_name] = round(score, 4)

    return scores


def detect_text_source(texts: list[str]) -> list[DetectionResult]:
    """检测文本来源 — 判断文本可能由哪个模型生成.

    Args:
        texts: 待检测的文本列表

    Returns:
        每条文本的检测结果
    """
    results: list[DetectionResult] = []

    for i, text in enumerate(texts):
        scores = _compute_style_scores(text)

        if scores:
            best_model = max(scores, key=lambda k: scores[k])
            best_score = scores[best_model]
        else:
            best_model = "unknown"
            best_score = 0.0

        preview = text[:80] + "..." if len(text) > 80 else text
        preview = preview.replace("\n", " ")

        results.append(
            DetectionResult(
                text_id=i,
                text_preview=preview,
                predicted_model=best_model,
                confidence=best_score,
                scores=scores,
            )
        )

    return results


def compute_style_fingerprint(texts: list[str]) -> dict[str, float]:
    """从一组文本中提取风格指纹向量."""
    all_scores: dict[str, list[float]] = {}

    for text in texts:
        scores = _compute_style_scores(text)
        for model, score in scores.items():
            all_scores.setdefault(model, []).append(score)

    # 聚合为平均分
    return {
        model: round(sum(s) / len(s), 4) for model, s in all_scores.items() if s
    }
