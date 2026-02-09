"""审计报告生成.

支持两种模式:
- 简略报告: 只有 ComparisonResult 时生成
- 详细报告: 有完整指纹数据和逐条探测结果时生成（标准能力）
"""

import json
from collections import Counter
from datetime import datetime
from typing import Any

from modelaudit import __version__
from modelaudit.models import AuditResult

# 特征显示配置: (key, label, format)
_DISPLAY_FEATURES: list[tuple[str, str, str]] = [
    ("avg_length_chars", "平均字符数", ".1f"),
    ("avg_length_words", "平均词数", ".1f"),
    ("avg_unique_word_ratio", "词汇多样性", ".3f"),
    ("avg_punctuation_ratio", "标点使用率", ".3f"),
    ("avg_newline_ratio", "换行习惯", ".3f"),
    ("ratio_has_bullet_points", "列表使用率", ".0%"),
    ("ratio_has_code_blocks", "代码块使用率", ".0%"),
    ("ratio_has_numbered_list", "编号列表", ".0%"),
    ("ratio_has_markdown_headers", "Markdown 标题", ".0%"),
    ("style_helpful", "helpful 标记", ".4f"),
    ("style_hedging", "hedging 标记", ".4f"),
    ("style_structured", "structured 标记", ".4f"),
]

# 数值特征的典型范围（用于判定差异程度）
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "avg_length_chars": (50, 3000),
    "avg_length_words": (10, 500),
    "avg_length_sentences": (1, 20),
    "avg_avg_word_length": (3, 8),
    "avg_avg_sentence_length": (5, 40),
}

# 探测维度中文名
_CATEGORY_LABELS: dict[str, str] = {
    "self_awareness": "自我认知",
    "safety_boundary": "安全边界",
    "injection": "注入测试",
    "knowledge": "知识立场",
    "reasoning": "推理测试",
    "style": "风格测试",
    "creative": "创意写作",
    "multilingual": "多语言",
    "format": "格式控制",
    "roleplay": "角色扮演",
    "code": "代码生成",
    "summarization": "摘要能力",
}

# 探测维度说明
_CATEGORY_EXPLANATIONS: dict[str, str] = {
    "self_awareness": "模型身份、创建者",
    "safety_boundary": "拒绝策略、措辞差异",
    "injection": "Prompt injection 响应",
    "knowledge": "知识立场",
    "reasoning": "逻辑推理、伦理判断",
    "style": "风格差异",
    "creative": "叙事风格、类比能力",
    "multilingual": "中文响应、多语翻译",
    "format": "JSON 输出、Markdown 表格",
    "roleplay": "角色一致性",
    "code": "编码风格",
    "summarization": "信息压缩",
}

# Provider 显示名称
_PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "custom": "Custom API",
}

# Provider 默认 API 地址
_PROVIDER_APIS: dict[str, str] = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
}


def generate_report(result: AuditResult, format: str = "markdown") -> str:
    """生成审计报告.

    Args:
        result: 审计结果
        format: 输出格式 (markdown / json)
    """
    if format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2, default=str)

    # 有详细指纹数据 → 生成完整报告；否则生成简略报告
    if result.details.get("fingerprints"):
        return _generate_detailed_report(result)
    return _generate_basic_report(result)


# ---------------------------------------------------------------------------
# 详细报告（标准能力）
# ---------------------------------------------------------------------------


def _generate_detailed_report(result: AuditResult) -> str:
    """生成完整 Markdown 审计报告（6 节结构）."""
    details = result.details
    teacher_info = details.get("teacher_info", {})
    student_info = details.get("student_info", {})
    fingerprints = details.get("fingerprints", {})
    probe_details = details.get("probe_details", [])

    teacher_name = result.model_a
    student_name = result.model_b

    # 提取指纹向量
    teacher_vec = fingerprints.get("teacher", {}).get("data", {}).get("vector", {})
    student_vec = fingerprints.get("student", {}).get("data", {}).get("vector", {})

    # 比对结果
    comparison = result.comparisons[0] if result.comparisons else None
    similarity = comparison.similarity if comparison else 0.0
    threshold = comparison.threshold if comparison else 0.85

    # 判定文本
    verdict_map = {
        "likely_derived": ("⚠️", "可能存在蒸馏关系"),
        "independent": ("✓", "两个模型独立"),
        "inconclusive": ("?", "无法确定"),
    }
    verdict_icon, verdict_text = verdict_map.get(result.verdict, ("", result.verdict))

    # 置信度文字
    if result.confidence > 0.7:
        confidence_text = "高"
    elif result.confidence > 0.4:
        confidence_text = "中"
    else:
        confidence_text = "低"

    now = datetime.now().strftime("%Y-%m-%d")
    total_probes = len(probe_details)
    lines: list[str] = []

    # ── 标题 ──
    lines.append(f"# 模型蒸馏审计报告：{student_name} vs {teacher_name}")
    lines.append("")
    lines.append(f"**审计时间**: {now}")
    lines.append(f"**审计工具**: knowlyr-modelaudit {__version__}")
    lines.append("**审计方法**: LLMmap 黑盒指纹 + DLI 行为签名 + 风格分析")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. 审计对象 ──
    _section_audit_objects(
        lines, teacher_name, student_name, teacher_info, student_info,
    )

    # ── 2. 审计方法 ──
    _section_methodology(lines, probe_details, threshold)

    # ── 3. 审计结果 ──
    consistent_count = _section_results(
        lines, teacher_name, student_name,
        teacher_vec, student_vec,
        probe_details, similarity, threshold,
        verdict_icon, verdict_text, confidence_text,
        result_comparisons=result.comparisons,
    )

    # ── 4. 关键发现 ──
    _section_findings(
        lines, teacher_name, student_name,
        teacher_vec, student_vec,
        probe_details, similarity, threshold,
        consistent_count, total_probes, result.verdict,
    )

    # ── 5. 结论 ──
    _section_conclusion(
        lines, teacher_name, student_name,
        teacher_vec, student_vec,
        similarity, threshold, total_probes, result.verdict,
    )

    # ── 跳过方法提示 ──
    skipped_methods = result.details.get("skipped_methods", [])
    if skipped_methods:
        lines.append("> **注意**: 以下方法被跳过: " + ", ".join(skipped_methods))
        lines.append("")

    # ── 6. 局限性声明 ──
    _section_limitations(lines, total_probes)

    # ── 页脚 ──
    lines.append("---")
    lines.append("")
    lines.append("由 [knowlyr-modelaudit](https://github.com/liuxiaotong/model-audit) 生成")
    lines.append("")

    return "\n".join(lines)


def _section_audit_objects(
    lines: list[str],
    teacher_name: str,
    student_name: str,
    teacher_info: dict[str, Any],
    student_info: dict[str, Any],
) -> None:
    """第 1 节：审计对象."""
    s_provider = student_info.get("provider", "openai")
    t_provider = teacher_info.get("provider", "openai")
    s_api = student_info.get("api_base", "") or _PROVIDER_APIS.get(s_provider, "")
    t_api = teacher_info.get("api_base", "") or _PROVIDER_APIS.get(t_provider, "")
    s_provider_label = _PROVIDER_LABELS.get(s_provider, s_provider)
    t_provider_label = _PROVIDER_LABELS.get(t_provider, t_provider)

    lines.append("## 1. 审计对象")
    lines.append("")
    lines.append("| 角色 | 模型 | 提供方 | API |")
    lines.append("|------|------|--------|-----|")
    lines.append(f"| 被审计模型 | **{student_name}** | {s_provider_label} | {s_api} |")
    lines.append(f"| 参考模型 | **{teacher_name}** | {t_provider_label} | {t_api} |")
    lines.append("")
    lines.append(
        f"**审计目标**: 判断 {student_name} 是否对 {teacher_name} 进行了知识蒸馏。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")


def _section_methodology(
    lines: list[str],
    probe_details: list[dict[str, Any]],
    threshold: float,
) -> None:
    """第 2 节：审计方法."""
    # 统计各维度 Probe 数量
    category_counts: Counter[str] = Counter()
    for pd in probe_details:
        category_counts[pd["category"]] += 1

    num_probes = len(probe_details)
    num_categories = len(category_counts)

    lines.append("## 2. 审计方法")
    lines.append("")
    lines.append("### 2.1 探测设计")
    lines.append("")
    lines.append(
        f"使用 {num_probes} 个精心设计的探测 Prompt，覆盖 {num_categories} 个维度："
    )
    lines.append("")
    lines.append("| 维度 | Probe 数量 | 说明 |")
    lines.append("|------|-----------|------|")

    for cat, count in category_counts.items():
        label = _CATEGORY_LABELS.get(cat, cat)
        explanation = _CATEGORY_EXPLANATIONS.get(cat, "")
        lines.append(f"| {label} | {count} | {explanation} |")

    lines.append("")
    lines.append("### 2.2 指纹提取")
    lines.append("")
    lines.append("对每条响应提取 18 维特征向量：")
    lines.append("")
    lines.append("- **长度特征** (5 维): 字符数、词数、句数、平均词长、平均句长")
    lines.append("- **比率特征** (3 维): 词汇多样性、标点密度、换行密度")
    lines.append("- **结构特征** (5 维): 列表、编号、Markdown 标题、代码块、拒绝开头")
    lines.append("- **风格标记** (5 维): apologetic / helpful / hedging / structured / ai_aware")
    lines.append("")
    lines.append("### 2.3 比对方法")
    lines.append("")
    lines.append("- 特征归一化（消除量纲差异）后计算余弦相似度")
    lines.append(f"- 蒸馏判定阈值: **{threshold}**")
    lines.append("")
    lines.append("### 2.4 DLI 行为签名比对")
    lines.append("")
    lines.append("- 从探测响应中提取行为签名 (bigram 分布 + 多维特征)")
    lines.append("- 用 Jensen-Shannon 散度衡量分布差异")
    lines.append("- 综合 bigram 相似度 (40%) + 特征余弦相似度 (60%)")
    lines.append("- DLI 蒸馏判定阈值: **0.80**")
    lines.append("")
    lines.append("---")
    lines.append("")


def _section_results(
    lines: list[str],
    teacher_name: str,
    student_name: str,
    teacher_vec: dict[str, float],
    student_vec: dict[str, float],
    probe_details: list[dict[str, Any]],
    similarity: float,
    threshold: float,
    verdict_icon: str,
    verdict_text: str,
    confidence_text: str,
    result_comparisons: list | None = None,
) -> int:
    """第 3 节：审计结果. 返回风格一致的 Probe 数量."""
    lines.append("## 3. 审计结果")
    lines.append("")

    # 3.1 总体判定
    lines.append("### 3.1 总体判定")
    lines.append("")
    lines.append("```")
    lines.append("┌──────────────────────────────────────────────┐")
    lines.append("│                                              │")
    lines.append(f"│   {verdict_icon}  {verdict_text}")
    lines.append("│                                              │")
    lines.append(f"│   余弦相似度:  {similarity:.4f}")
    lines.append(f"│   判定阈值:    {threshold}")
    lines.append(f"│   置信度:      {confidence_text}")
    lines.append("│                                              │")
    lines.append("└──────────────────────────────────────────────┘")
    lines.append("```")
    lines.append("")

    # 3.1b 多方法比对结果
    if result_comparisons and len(result_comparisons) > 1:
        lines.append("### 3.1b 多方法投票")
        lines.append("")
        lines.append("| 方法 | 相似度 | 阈值 | 判定 |")
        lines.append("|------|--------|------|------|")
        for c in result_comparisons:
            derived_text = "⚠️ 派生" if c.is_derived else "✓ 独立"
            lines.append(f"| {c.method} | {c.similarity:.4f} | {c.threshold} | {derived_text} |")
        lines.append("")
        derived_count = sum(1 for c in result_comparisons if c.is_derived)
        total_methods = len(result_comparisons)
        lines.append(f"**投票结果**: {derived_count}/{total_methods} 方法判定为派生关系")
        lines.append("")

    # 3.2 指纹相似度详情
    lines.append("### 3.2 指纹相似度详情")
    lines.append("")
    lines.append(f"| 特征维度 | {student_name} | {teacher_name} | 差异 | 判定 |")
    lines.append("|---------|-----------|--------|------|------|")

    for key, label, fmt in _DISPLAY_FEATURES:
        s_val = student_vec.get(key, 0)
        t_val = teacher_vec.get(key, 0)
        diff = abs(s_val - t_val)

        s_str = format(s_val, fmt)
        t_str = format(t_val, fmt)
        diff_str = format(diff, fmt)
        judgment = _judge_difference(key, diff)

        lines.append(f"| {label} | {s_str} | {t_str} | {diff_str} | {judgment} |")

    lines.append("")

    # 3.3 逐条探测结果
    lines.append("### 3.3 逐条探测结果")
    lines.append("")
    lines.append(
        f"| # | 探测维度 | Probe ID | {student_name} 风格匹配 "
        f"| {teacher_name} 风格匹配 | 一致 |"
    )
    lines.append("|---|---------|----------|--------------|----------------|------|")

    consistent_count = 0
    for i, pd in enumerate(probe_details):
        cat_label = _CATEGORY_LABELS.get(pd["category"], pd["category"])
        s_style = pd.get("student_style", "")
        t_style = pd.get("teacher_style", "")
        is_consistent = pd.get("is_consistent", False)

        # 如果 student 被识别为 teacher 风格，加粗显示
        s_display = f"**{s_style}**" if _is_teacher_style(s_style, teacher_name) else s_style
        t_display = f"**{t_style}**" if _is_teacher_style(t_style, teacher_name) else t_style

        if is_consistent:
            consistent_count += 1
        consistent_mark = "✓" if is_consistent else ""

        lines.append(
            f"| {i + 1} | {cat_label} | {pd['probe_id']} "
            f"| {s_display} | {t_display} | {consistent_mark} |"
        )

    total_probes = len(probe_details)
    pct = consistent_count / total_probes * 100 if total_probes else 0
    lines.append("")
    lines.append(f"**风格一致率: {consistent_count}/{total_probes} ({pct:.0f}%)**")
    lines.append("")
    lines.append("---")
    lines.append("")

    return consistent_count


def _section_findings(
    lines: list[str],
    teacher_name: str,
    student_name: str,
    teacher_vec: dict[str, float],
    student_vec: dict[str, float],
    probe_details: list[dict[str, Any]],
    similarity: float,
    threshold: float,
    consistent_count: int,
    total_probes: int,
    verdict: str,
) -> None:
    """第 4 节：关键发现（自动分析）."""
    lines.append("## 4. 关键发现")
    lines.append("")

    # ─── 4.1 支持蒸馏关系的证据 ───
    lines.append("### 4.1 支持蒸馏关系的证据")
    lines.append("")

    evidence_num = 1
    pct = consistent_count / total_probes * 100 if total_probes else 0

    # 证据 1: 高相似度
    if similarity > threshold:
        lines.append(
            f"{evidence_num}. **指纹相似度极高 ({similarity:.4f})**: "
            f"远超 {threshold} 的蒸馏判定阈值，表明两个模型在响应模式上高度一致。"
        )
        lines.append("")
        evidence_num += 1

    # 证据 2: 风格标记分布一致
    style_diffs = []
    for key, _, _ in _DISPLAY_FEATURES:
        if key.startswith("style_"):
            s_val = student_vec.get(key, 0)
            t_val = teacher_vec.get(key, 0)
            style_diffs.append(abs(s_val - t_val))

    if style_diffs and max(style_diffs) < 0.005:
        lines.append(
            f"{evidence_num}. **风格标记分布一致**: "
            "helpful、hedging、structured、ai_aware 等风格维度的数值差异均在 "
            f"{max(style_diffs):.3f} 以内，说明两个模型的「语气」和「表达习惯」几乎相同。"
        )
        lines.append("")
        evidence_num += 1

    # 证据 3: student 在部分场景表现出 teacher 风格
    teacher_style_in_student = [
        pd for pd in probe_details
        if _is_teacher_style(pd.get("student_style", ""), teacher_name)
    ]
    if teacher_style_in_student:
        affected_ids = [pd["probe_id"] for pd in teacher_style_in_student]
        lines.append(
            f"{evidence_num}. **{student_name} 在安全相关场景中表现出 {teacher_name} 风格**: "
            f"在 {', '.join(affected_ids)} 等 {len(teacher_style_in_student)} 个场景中，"
            f"{student_name} 被识别为 {teacher_name} 风格。"
            "安全对齐（alignment）行为是蒸馏中最容易被继承的特征之一。"
        )
        lines.append("")
        evidence_num += 1

    # 证据 4: 词汇多样性、标点习惯一致
    vocab_diff = abs(
        student_vec.get("avg_unique_word_ratio", 0)
        - teacher_vec.get("avg_unique_word_ratio", 0)
    )
    punct_diff = abs(
        student_vec.get("avg_punctuation_ratio", 0)
        - teacher_vec.get("avg_punctuation_ratio", 0)
    )
    if vocab_diff < 0.05 and punct_diff < 0.01:
        lines.append(
            f"{evidence_num}. **词汇多样性、标点习惯几乎完全一致**: "
            "这些是模型底层语言能力的反映，不容易通过表面微调改变。"
        )
        lines.append("")
        evidence_num += 1

    # 证据 5: 风格一致率
    if pct > 50:
        lines.append(
            f"{evidence_num}. **{pct:.0f}% 的探测结果风格一致**: "
            f"超过半数的场景中，{student_name} 和 {teacher_name} "
            "被判定为相同的风格模式。"
        )
        lines.append("")
        evidence_num += 1

    if evidence_num == 1:
        lines.append("未发现明显支持蒸馏关系的证据。")
        lines.append("")

    # ─── 4.2 差异点 ───
    lines.append("### 4.2 差异点")
    lines.append("")

    diff_num = 1
    s_chars = student_vec.get("avg_length_chars", 0)
    t_chars = teacher_vec.get("avg_length_chars", 0)

    if abs(s_chars - t_chars) > 200:
        longer = student_name if s_chars > t_chars else teacher_name
        lines.append(
            f"{diff_num}. **回复长度**: {student_name} 平均 {s_chars:.0f} 字符，"
            f"{teacher_name} 平均 {t_chars:.0f} 字符。"
            f"{longer} 倾向于更长、更详细的回复。"
        )
        lines.append("")
        diff_num += 1

    s_sent = student_vec.get("avg_avg_sentence_length", 0)
    t_sent = teacher_vec.get("avg_avg_sentence_length", 0)
    if abs(s_sent - t_sent) > 3:
        longer = student_name if s_sent > t_sent else teacher_name
        lines.append(
            f"{diff_num}. **句子长度**: {longer} 平均句长更长，"
            "说明偏好更复杂的句式。"
        )
        lines.append("")
        diff_num += 1

    if diff_num == 1:
        lines.append("未发现显著差异。")
        lines.append("")

    # 蒸馏 + 微调假设说明
    if verdict == "likely_derived" and diff_num > 1:
        lines.append(
            "这些差异与「蒸馏后进行风格微调」的假设一致——"
            "底层的知识和安全对齐行为被继承，"
            f"但输出风格（长度、详细程度）被调整为更适合 {student_name} 产品定位的形态。"
        )
        lines.append("")

    # ─── 4.3 风格分布 ───
    lines.append("### 4.3 与其他模型的对比参考")
    lines.append("")
    lines.append(f"{student_name} 在风格检测中被判定为以下模型风格的分布：")
    lines.append("")

    student_style_counts: Counter[str] = Counter(
        pd.get("student_style", "unknown") for pd in probe_details
    )
    lines.append("| 风格 | 出现次数 | 占比 |")
    lines.append("|------|---------|------|")
    for style, count in student_style_counts.most_common():
        style_pct = count / total_probes * 100 if total_probes else 0
        if _is_teacher_style(style, teacher_name):
            lines.append(f"| **{style}** | **{count}** | **{style_pct:.0f}%** |")
        else:
            lines.append(f"| {style} | {count} | {style_pct:.0f}% |")

    lines.append("")

    teacher_style_count = sum(
        1 for pd in probe_details
        if _is_teacher_style(pd.get("student_style", ""), teacher_name)
    )
    teacher_style_pct = teacher_style_count / total_probes * 100 if total_probes else 0
    if teacher_style_pct > 0:
        lines.append(
            f"值得注意的是，{student_name} 在 "
            f"**{teacher_style_pct:.0f}% 的场景中直接被判定为 {teacher_name} 风格**"
            "，而这些场景集中在安全边界和知识推理等核心能力上。"
        )
        lines.append("")

    lines.append("---")
    lines.append("")


def _section_conclusion(
    lines: list[str],
    teacher_name: str,
    student_name: str,
    teacher_vec: dict[str, float],
    student_vec: dict[str, float],
    similarity: float,
    threshold: float,
    total_probes: int,
    verdict: str,
) -> None:
    """第 5 节：结论."""
    lines.append("## 5. 结论")
    lines.append("")

    exceeds = "显著超过" if similarity > threshold else "未超过"
    lines.append(
        f"基于 {total_probes} 个探测 Prompt 的黑盒指纹分析，"
        f"**{student_name} 与 {teacher_name} 的行为指纹相似度为 {similarity:.4f}**，"
        f"{exceeds} {threshold} 的蒸馏判定阈值。"
    )
    lines.append("")

    if verdict == "likely_derived":
        lines.append("两个模型在以下方面高度一致：")
        lines.append("- 词汇选择和多样性")
        lines.append("- 标点和格式习惯")
        lines.append("- 安全对齐行为（拒绝策略、措辞风格）")
        lines.append("- 风格标记分布")
        lines.append("")

        s_chars = student_vec.get("avg_length_chars", 0)
        t_chars = teacher_vec.get("avg_length_chars", 0)
        if abs(s_chars - t_chars) > 200:
            lines.append(
                "差异仅体现在输出长度和句式复杂度上，这些可以通过微调轻易改变。"
            )
            lines.append("")

        lines.append(
            f"**审计判定: {student_name} 可能对 {teacher_name} "
            f"进行了知识蒸馏或使用了 {teacher_name} 的输出数据进行训练。**"
        )
    elif verdict == "independent":
        lines.append(
            f"**审计判定: {student_name} 与 {teacher_name} "
            "的行为模式差异较大，不太可能存在蒸馏关系。**"
        )
    else:
        lines.append(
            f"**审计判定: 基于当前证据，无法确定 {student_name} 与 {teacher_name} "
            "之间是否存在蒸馏关系。建议增加探测样本或使用白盒方法进一步分析。**"
        )

    lines.append("")
    lines.append("---")
    lines.append("")


def _section_limitations(lines: list[str], total_probes: int) -> None:
    """第 6 节：局限性声明."""
    lines.append("## 6. 局限性声明")
    lines.append("")
    lines.append(
        "1. **黑盒方法的固有局限**: "
        "本报告仅基于模型输出的风格分析，无法访问模型权重或训练数据，不能提供确定性证据。"
    )
    lines.append(
        f"2. **样本量**: {total_probes} 个探测 Prompt 的样本量有限，"
        "增加样本可以提高结论的统计可靠性。"
    )
    lines.append(
        "3. **风格签名库覆盖**: 当前支持 12 个模型家族的风格签名，"
        "可能存在未覆盖的模型风格。"
    )
    lines.append(
        "4. **替代解释**: 高相似度也可能源于相似的训练数据来源、"
        "相似的 RLHF 方法论或共同的对齐策略，不一定是直接蒸馏。"
    )
    lines.append("")


# ---------------------------------------------------------------------------
# 简略报告（无详细数据时的 fallback）
# ---------------------------------------------------------------------------


def _generate_basic_report(result: AuditResult) -> str:
    """生成简略 Markdown 报告."""
    verdict_text = {
        "likely_derived": "可能存在蒸馏关系",
        "independent": "两个模型独立",
        "inconclusive": "无法确定",
    }
    verdict_icon = {
        "likely_derived": "⚠️",
        "independent": "✓",
        "inconclusive": "?",
    }

    lines = [
        "# 模型蒸馏审计报告",
        "",
        f"**审计工具**: knowlyr-modelaudit {__version__}",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 审计对象",
        "",
        "| 角色 | 模型 |",
        "|------|------|",
        f"| 教师模型 (Teacher) | {result.model_a} |",
        f"| 学生模型 (Student) | {result.model_b} |",
        "",
        "## 判定结果",
        "",
        f"**{verdict_icon.get(result.verdict, '')} "
        f"{verdict_text.get(result.verdict, result.verdict)}**",
        "",
        f"- 置信度: {result.confidence:.2%}",
        "",
    ]

    if result.comparisons:
        lines.extend([
            "## 指纹比对详情",
            "",
            "| 方法 | 相似度 | 阈值 | 判定 |",
            "|------|--------|------|------|",
        ])
        for c in result.comparisons:
            derived_text = "派生" if c.is_derived else "独立"
            lines.append(
                f"| {c.method} | {c.similarity:.4f} | {c.threshold} | {derived_text} |"
            )
        lines.append("")

    lines.extend([
        "## 说明",
        "",
        "- **相似度 > 0.85**: 两个模型的行为模式高度相似，可能存在蒸馏关系",
        "- **相似度 0.5-0.85**: 部分相似，可能共享训练数据或架构",
        "- **相似度 < 0.5**: 两个模型行为差异较大，可能是独立模型",
        "",
        "---",
        "",
        "由 [knowlyr-modelaudit](https://github.com/liuxiaotong/model-audit) 生成",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _judge_difference(key: str, diff: float) -> str:
    """判定两个特征值的差异程度."""
    # 数值特征 → 按特征范围归一化后判定
    if key in _FEATURE_RANGES:
        lo, hi = _FEATURE_RANGES[key]
        norm_diff = diff / (hi - lo) if hi > lo else 0
        if norm_diff < 0.02:
            return "**完全一致**"
        elif norm_diff < 0.1:
            return "**高度一致**"
        elif norm_diff < 0.2:
            return "接近"
        else:
            return "显著不同"

    # ratio_ 特征（0-1 范围）
    if key.startswith("ratio_"):
        if diff < 0.01:
            return "**完全一致**"
        elif diff < 0.05:
            return "**高度一致**"
        elif diff < 0.1:
            return "接近"
        else:
            return "显著不同"

    # style_ 特征（极小值）
    if key.startswith("style_"):
        if diff < 0.001:
            return "**完全一致**"
        elif diff < 0.003:
            return "**高度一致**"
        elif diff < 0.005:
            return "接近"
        else:
            return "显著不同"

    # 默认 0-1 范围判定
    if diff < 0.01:
        return "**完全一致**"
    elif diff < 0.05:
        return "**高度一致**"
    elif diff < 0.1:
        return "接近"
    else:
        return "显著不同"


def _is_teacher_style(style: str, teacher_name: str) -> bool:
    """检查风格标签是否与教师模型匹配."""
    style_lower = style.lower()
    teacher_lower = teacher_name.lower()
    # 风格名在教师名中，或教师名在风格名中
    return style_lower in teacher_lower or teacher_lower in style_lower
