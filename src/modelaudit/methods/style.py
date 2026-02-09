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
# markers: 独特短语 (越独特越好，避免跨模型重叠)
# refusal_patterns: 拒绝时的典型表达
# structural: 格式偏好
# lang: 主要语言 ("en", "zh", "both")
MODEL_STYLE_SIGNATURES: dict[str, dict[str, Any]] = {
    "gpt-4": {
        "markers": [
            "certainly! here's", "comprehensive breakdown",
            "it's important to note", "it's worth noting",
            "let me know if you'd like", "keep in mind",
            "in more detail", "would you like me to",
            "let me walk you through", "active area of research",
        ],
        "refusal_patterns": [
            "i can't assist", "i'm not able to", "as an ai language model",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": True,
            "verbose": True,
        },
        "lang": "en",
    },
    "gpt-3.5": {
        "markers": [
            "certainly!", "sure!", "of course!", "absolutely!",
            "no problem!", "hope that helps!",
        ],
        "refusal_patterns": [
            "as an ai language model", "i don't have the ability",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": True,
            "tends_code_blocks": False,
            "verbose": False,
        },
        "lang": "en",
    },
    "claude": {
        "markers": [
            "i'd be happy to help", "let me think through this",
            "i should note", "i want to be straightforward",
            "nuanced", "i want to be careful",
            "different perspectives", "ethical implications",
            "would you like me to continue",
            "take it in a different direction",
        ],
        "refusal_patterns": [
            "i don't think i should", "i'd rather not",
            "i want to be helpful but",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": False,
            "tends_code_blocks": True,
            "verbose": True,
        },
        "lang": "en",
    },
    "llama": {
        "markers": [
            "sure thing!", "no problem", "pretty cool",
            "check out this", "here you go",
            "let me know if you need anything else",
            "so basically", "think of it like",
        ],
        "refusal_patterns": [
            "i cannot", "i'm just an ai", "it's not appropriate",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "tends_code_blocks": True,
            "verbose": False,
        },
        "lang": "en",
    },
    "gemini": {
        "markers": [
            "great question!", "here's what you need to know",
            "**key applications**", "key applications",
            "it's worth noting that", "progress is accelerating",
            "i can provide a code implementation",
            "systematically", "noisy and error-prone",
        ],
        "refusal_patterns": [
            "i'm a large language model", "i'm designed to be helpful",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": False,
            "verbose": True,
        },
        "lang": "en",
    },
    "qwen": {
        "markers": [
            "好的，我来", "为您详细", "具体来说",
            "以下是一个高效的", "这个实现", "时间复杂度",
            "核心思想", "优势在于",
        ],
        "refusal_patterns": [
            "作为ai助手", "我无法提供",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": True,
            "verbose": False,
        },
        "lang": "zh",
    },
    "deepseek": {
        "markers": [
            "嗯，让我仔细想想", "从多个角度", "本质上是",
            "状态转移方程", "边界条件", "如果需要优化",
            "从技术层面看", "从实际应用角度",
        ],
        "refusal_patterns": [
            "作为ai助手", "我无法提供",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": True,
            "verbose": True,
        },
        "lang": "zh",
    },
    "mistral": {
        "markers": [
            "to answer your question",
            "in short", "the answer is", "straightforward",
        ],
        "refusal_patterns": [
            "i cannot", "i must decline", "it would be inappropriate",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "tends_code_blocks": False,
            "verbose": False,
        },
        "lang": "en",
    },
    "yi": {
        "markers": [
            "to put it simply", "in a nutshell",
            "i'd like to point out",
        ],
        "refusal_patterns": [
            "as an ai", "i'm not able to", "i cannot assist with",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": False,
            "verbose": True,
        },
        "lang": "en",
    },
    "phi": {
        "markers": [
            "in conclusion", "the answer is simply",
        ],
        "refusal_patterns": [
            "i cannot", "i'm unable to", "as a language model",
        ],
        "structural": {
            "tends_markdown": False,
            "tends_numbered_lists": False,
            "tends_code_blocks": False,
            "verbose": False,
        },
        "lang": "en",
    },
    "cohere": {
        "markers": [
            "here's what i found", "to elaborate",
            "happy to help with that",
        ],
        "refusal_patterns": [
            "i'm not able to", "i'd prefer not to", "i cannot help with",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": False,
            "verbose": True,
        },
        "lang": "en",
    },
    "chatglm": {
        "markers": [
            "好的", "以下是", "总结一下",
            "首先我们需要", "希望对您有帮助",
        ],
        "refusal_patterns": [
            "作为ai助手", "我无法",
        ],
        "structural": {
            "tends_markdown": True,
            "tends_numbered_lists": True,
            "tends_code_blocks": False,
            "verbose": True,
        },
        "lang": "zh",
    },
}


def _detect_lang(text: str) -> str:
    """检测文本主要语言: 'zh' 或 'en'."""
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    # 绝对数量兜底: 即使代码多, 10 个汉字也算中文
    if cjk_count >= 10:
        return "zh"
    total = len(text) or 1
    return "zh" if cjk_count / total > 0.15 else "en"


def _compute_style_scores(text: str) -> dict[str, float]:
    """计算文本与各模型风格的匹配分数."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) or 1
    text_lang = _detect_lang(text)

    # 预计算结构特征 (只算一次)
    has_md = bool(re.search(r"^#+\s", text, re.MULTILINE))
    has_numbered = bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))
    has_code_blocks = "```" in text
    is_verbose = total_words > 150

    # 检测是否有拒绝内容 (决定是否启用拒绝分数)
    has_refusal_hint = any(
        kw in text_lower
        for kw in ("i cannot", "i can't", "unable to", "我无法", "作为ai")
    )

    scores: dict[str, float] = {}

    for model_name, sig in MODEL_STYLE_SIGNATURES.items():
        score = 0.0
        model_lang = sig.get("lang", "en")

        # ── 1. 语言匹配 (权重 0.20) ──
        if text_lang == model_lang:
            score += 0.20
        elif model_lang == "both":
            score += 0.10
        # 语言不匹配: +0, 形成天然屏障

        # ── 2. 标记词匹配 (权重 0.50) ──
        # 用固定分母 (3) 归一化, 避免 marker 多的模型吃亏
        marker_hits = sum(1 for m in sig["markers"] if m in text_lower)
        score += min(marker_hits / 3, 1.0) * 0.50

        # ── 3. 结构特征匹配 (权重 0.20) ──
        # 只对文本实际展现的特征计分, 避免 "双否" 虚假匹配
        structural = sig["structural"]
        struct_score = 0.0
        for text_has, key in [
            (has_md, "tends_markdown"),
            (has_numbered, "tends_numbered_lists"),
            (has_code_blocks, "tends_code_blocks"),
            (is_verbose, "verbose"),
        ]:
            model_tends = structural.get(key, False)
            if text_has and model_tends:
                struct_score += 0.05   # 正向匹配
            elif text_has and not model_tends:
                struct_score -= 0.02   # 文本有但模型不倾向 → 轻微惩罚
        score += struct_score

        # ── 4. 拒绝模式 (权重 0.10, 仅文本含拒绝时生效) ──
        if has_refusal_hint:
            refusal_hits = sum(1 for p in sig["refusal_patterns"] if p in text_lower)
            score += refusal_hits / max(len(sig["refusal_patterns"]), 1) * 0.10

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
