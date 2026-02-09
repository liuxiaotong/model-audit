"""审计报告生成."""

import json
from datetime import datetime

from modelaudit.models import AuditResult


def generate_report(result: AuditResult, format: str = "markdown") -> str:
    """生成审计报告.

    Args:
        result: 审计结果
        format: 输出格式 (markdown / json)
    """
    if format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2, default=str)
    return _generate_markdown_report(result)


def _generate_markdown_report(result: AuditResult) -> str:
    """生成 Markdown 格式审计报告."""
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
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        f"**{verdict_icon.get(result.verdict, '')} {verdict_text.get(result.verdict, result.verdict)}**",
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
        "*由 knowlyr-modelaudit 生成*",
    ])

    return "\n".join(lines)
