"""ModelAudit MCP Server — Model Context Protocol 服务."""

from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from modelaudit.config import AuditConfig
from modelaudit.engine import AuditEngine
from modelaudit.report import generate_report


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-modelaudit[mcp]")

    server = Server("modelaudit")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="detect_text_source",
                description="检测文本数据来源 — 判断文本可能由哪个 LLM 生成",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "待检测的文本列表",
                        },
                    },
                    "required": ["texts"],
                },
            ),
            Tool(
                name="verify_model",
                description="验证模型身份 — 检查 API 背后是不是声称的模型",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "模型名称 (如 gpt-4o, claude-3-opus)",
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "anthropic", "custom"],
                            "description": "API 提供商 (默认: openai)",
                        },
                    },
                    "required": ["model"],
                },
            ),
            Tool(
                name="compare_models",
                description="比对两个模型的指纹相似度，判断是否存在蒸馏/派生关系",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_a": {"type": "string", "description": "模型 A 名称"},
                        "model_b": {"type": "string", "description": "模型 B 名称"},
                        "provider": {
                            "type": "string",
                            "description": "API 提供商 (默认: openai)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["llmmap", "dli", "style"],
                            "description": "指纹方法 (默认: llmmap)",
                        },
                    },
                    "required": ["model_a", "model_b"],
                },
            ),
            Tool(
                name="compare_models_whitebox",
                description="白盒比对两个本地模型 — 使用 REEF CKA 方法比较模型隐藏状态相似度（需要模型权重）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_a": {
                            "type": "string",
                            "description": "本地模型路径或 HuggingFace 模型名 A",
                        },
                        "model_b": {
                            "type": "string",
                            "description": "本地模型路径或 HuggingFace 模型名 B",
                        },
                        "device": {
                            "type": "string",
                            "description": "计算设备 (默认: cpu)",
                        },
                    },
                    "required": ["model_a", "model_b"],
                },
            ),
            Tool(
                name="audit_distillation",
                description=(
                    "完整蒸馏审计 — 综合指纹比对 + 风格分析，生成详细审计报告。"
                    "支持跨 provider 审计（如 Kimi API + Claude API）。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "teacher": {
                            "type": "string",
                            "description": "教师模型 (疑似被蒸馏的源模型)",
                        },
                        "student": {
                            "type": "string",
                            "description": "学生模型 (疑似蒸馏产物)",
                        },
                        "teacher_provider": {
                            "type": "string",
                            "description": "教师模型 API 提供商 (默认: openai)",
                        },
                        "student_provider": {
                            "type": "string",
                            "description": "学生模型 API 提供商 (默认: openai)",
                        },
                        "teacher_api_base": {
                            "type": "string",
                            "description": "教师模型自定义 API 地址",
                        },
                        "student_api_base": {
                            "type": "string",
                            "description": "学生模型自定义 API 地址",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json"],
                            "description": "报告格式 (默认: markdown)",
                        },
                    },
                    "required": ["teacher", "student"],
                },
            ),
            Tool(
                name="audit_memorization",
                description="检测模型是否记忆了训练数据 — 通过前缀补全和逐字检查评估记忆程度",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text_samples": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "候选训练文本列表",
                        },
                        "model": {
                            "type": "string",
                            "description": "待测模型名称 (默认: gpt-4o)",
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "anthropic"],
                            "description": "API 提供商 (默认: openai)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["prefix_completion", "verbatim_check", "both"],
                            "description": "检测方法 (默认: prefix_completion)",
                        },
                    },
                    "required": ["text_samples"],
                },
            ),
            Tool(
                name="audit_report",
                description="生成完整的模型审计报告 — 汇总所有审计工具的结果",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "object",
                            "description": "各审计工具的结果字典 (tool_name -> result_text)",
                        },
                        "model_name": {
                            "type": "string",
                            "description": "被审计模型名称",
                        },
                        "audit_date": {
                            "type": "string",
                            "description": "审计日期 (默认: 今天)",
                        },
                    },
                    "required": ["results", "model_name"],
                },
            ),
            Tool(
                name="audit_watermark",
                description="检测文本中是否包含 AI 水印（统计特征和模式匹配）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "待检测的文本列表",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["statistical", "pattern", "both"],
                            "description": "检测方法（默认 both）",
                            "default": "both",
                        },
                    },
                    "required": ["texts"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """调用工具."""

        if name == "detect_text_source":
            texts = arguments["texts"]
            engine = AuditEngine()
            results = engine.detect(texts)

            lines = ["## 文本来源检测结果", ""]
            lines.append("| # | 预测模型 | 置信度 | 预览 |")
            lines.append("|---|---------|--------|------|")

            for r in results:
                lines.append(
                    f"| {r.text_id} | {r.predicted_model} | {r.confidence:.2%} | {r.text_preview} |"
                )

            # 统计
            model_counts: dict[str, int] = {}
            for r in results:
                model_counts[r.predicted_model] = model_counts.get(r.predicted_model, 0) + 1

            lines.extend(["", "### 来源分布"])
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
                pct = count / len(results) * 100
                lines.append(f"- {model}: {count} ({pct:.1f}%)")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "verify_model":
            model = arguments["model"]
            provider = arguments.get("provider", "openai")

            config = AuditConfig(provider=provider)
            engine = AuditEngine(config)
            result = engine.verify(model, provider=provider)

            verified = result["verified"]
            icon = "✓" if verified else "✗"

            text = f"""## 模型身份验证

{icon} **{'验证通过' if verified else '验证未通过'}**

- 声称模型: {model}
- 最佳匹配: {result['best_match']} (分数: {result['best_score']})

### 风格匹配分数
"""
            for name_k, score in result.get("all_scores", {}).items():
                text += f"- {name_k}: {score:.4f}\n"

            return [TextContent(type="text", text=text)]

        elif name == "compare_models":
            model_a = arguments["model_a"]
            model_b = arguments["model_b"]
            provider = arguments.get("provider", "openai")
            method = arguments.get("method", "llmmap")

            config = AuditConfig(provider=provider)
            engine = AuditEngine(config)
            result = engine.compare(model_a, model_b, method=method, provider=provider)

            derived_text = "可能存在派生关系" if result.is_derived else "可能是独立模型"
            icon = "⚠️" if result.is_derived else "✓"

            text = f"""## 模型比对结果

{icon} **{derived_text}**

- 模型 A: {model_a}
- 模型 B: {model_b}
- 相似度: {result.similarity:.4f}
- 阈值: {result.threshold}
- 置信度: {result.confidence:.4f}
"""
            return [TextContent(type="text", text=text)]

        elif name == "compare_models_whitebox":
            model_a = arguments["model_a"]
            model_b = arguments["model_b"]
            device = arguments.get("device", "cpu")

            engine = AuditEngine(use_cache=False)
            result = engine.compare(
                model_a, model_b, method="reef", device=device,
            )

            derived_text = "可能存在派生关系" if result.is_derived else "可能是独立模型"
            icon = "⚠️" if result.is_derived else "✓"

            text = f"""## 白盒模型比对结果 (REEF CKA)

{icon} **{derived_text}**

- 模型 A: {model_a}
- 模型 B: {model_b}
- 相似度: {result.similarity:.4f}
- 阈值: {result.threshold}
- 置信度: {result.confidence:.4f}
"""
            if "layer_cka" in result.details:
                text += "\n### 逐层 CKA 相似度\n"
                for layer, cka in result.details["layer_cka"].items():
                    text += f"- {layer}: {cka:.4f}\n"

            return [TextContent(type="text", text=text)]

        elif name == "audit_distillation":
            teacher = arguments["teacher"]
            student = arguments["student"]
            output_format = arguments.get("format", "markdown")

            config = AuditConfig()
            engine = AuditEngine(config)
            result = engine.audit(
                teacher, student,
                teacher_provider=arguments.get("teacher_provider"),
                teacher_api_key=arguments.get("teacher_api_key"),
                teacher_api_base=arguments.get("teacher_api_base"),
                student_provider=arguments.get("student_provider"),
                student_api_key=arguments.get("student_api_key"),
                student_api_base=arguments.get("student_api_base"),
            )

            report = generate_report(result, output_format)
            return [TextContent(type="text", text=report)]

        elif name == "audit_memorization":
            from difflib import SequenceMatcher
            text_samples = arguments["text_samples"]
            model = arguments.get("model", "gpt-4o")
            provider = arguments.get("provider", "openai")
            method = arguments.get("method", "prefix_completion")

            # Try to call model API
            try:
                from modelaudit.methods.llmmap import _call_model_api_once
                has_client = True
            except ImportError:
                has_client = False

            if not has_client:
                lines = ["## 记忆检测 — 交互模式", "", "未找到 LLM 客户端库，以下是供手动执行的提示：", ""]
                for i, sample in enumerate(text_samples):
                    mid = len(sample) // 2
                    lines.append(f"### 样本 {i + 1}")
                    lines.append(f"```\n请续写以下文本：\n\n{sample[:mid]}\n```\n")
                return [TextContent(type="text", text="\n".join(lines))]

            results = []
            for i, sample in enumerate(text_samples):
                mid = len(sample) // 2
                prefix = sample[:mid]
                suffix = sample[mid:]
                if method in ("prefix_completion", "both"):
                    prompt = f"请续写以下文本，直接输出续写部分：\n\n{prefix}"
                    completion = _call_model_api_once(model, prompt, provider=provider)
                    score = SequenceMatcher(None, suffix.strip(), completion.strip()).ratio()
                    results.append({"index": i + 1, "method": "prefix_completion", "score": round(score, 4)})
                if method in ("verbatim_check", "both"):
                    prompt = f"请用你自己的话复述以下文本的关键信息：\n\n{sample[:200]}"
                    response = _call_model_api_once(model, prompt, provider=provider)
                    words = sample.split()
                    phrase_len = min(8, len(words) // 4) or 3
                    matches = total = 0
                    for j in range(0, len(words) - phrase_len + 1, phrase_len):
                        phrase = " ".join(words[j:j + phrase_len])
                        total += 1
                        if phrase in response:
                            matches += 1
                    score = matches / total if total else 0
                    results.append({"index": i + 1, "method": "verbatim_check", "score": round(score, 4)})

            lines = [f"## 训练数据记忆检测结果 ({model})", "", f"- 检测方法: {method}", f"- 样本数量: {len(text_samples)}", "",
                     "| # | 方法 | 记忆分数 | 评级 |", "|---|------|---------|------|"]
            for r in results:
                level = "high" if r["score"] >= 0.7 else ("medium" if r["score"] >= 0.4 else "low")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[level]
                lines.append(f"| {r['index']} | {r['method']} | {r['score']:.4f} | {icon} {level} |")
            if results:
                avg = sum(r["score"] for r in results) / len(results)
                lines.extend(["", f"- 平均记忆分数: {avg:.4f}"])
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "audit_report":
            from datetime import date
            results = arguments["results"]
            model_name = arguments["model_name"]
            audit_date = arguments.get("audit_date", date.today().isoformat())
            sections = {
                "detect_text_source": "文本来源检测",
                "verify_model": "模型身份验证",
                "audit_distillation": "蒸馏分析",
                "compare_models": "模型指纹比对",
                "audit_memorization": "记忆检测",
            }
            lines = [f"# 模型审计报告：{model_name}", "", f"**审计日期**: {audit_date}", ""]
            for key, title in sections.items():
                lines.append(f"## {title}")
                lines.append("")
                if key in results:
                    lines.append(results[key])
                else:
                    lines.append("*未执行此项检查。*")
                lines.append("")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "audit_watermark":
            texts = arguments["texts"]
            method = arguments.get("method", "both")

            results = []
            for i, text in enumerate(texts):
                score = 0.0
                signals = []

                if method in ("statistical", "both"):
                    # Statistical: check token distribution uniformity
                    words = text.split()
                    if len(words) > 20:
                        # Check for unusually uniform word length distribution
                        lengths = [len(w) for w in words]
                        mean_len = sum(lengths) / len(lengths)
                        variance = sum((wl - mean_len) ** 2 for wl in lengths) / len(lengths)
                        # Low variance in word lengths can indicate watermarking
                        if variance < 2.0:
                            score += 0.3
                            signals.append(f"低词长方差 ({variance:.2f})")

                        # Check for repetitive n-gram patterns
                        bigrams = [f"{words[j]} {words[j+1]}" for j in range(len(words)-1)]
                        unique_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1
                        if unique_ratio < 0.5:
                            score += 0.2
                            signals.append(f"低 bigram 唯一率 ({unique_ratio:.2f})")

                if method in ("pattern", "both"):
                    # Pattern: check for known watermark patterns
                    # Unicode zero-width characters
                    zwc_count = sum(1 for c in text if c in "\u200b\u200c\u200d\ufeff")
                    if zwc_count > 0:
                        score += 0.5
                        signals.append(f"零宽字符: {zwc_count} 个")

                    # Unusual whitespace patterns
                    double_space = text.count("  ")
                    if double_space > 3:
                        score += 0.2
                        signals.append(f"双空格: {double_space} 处")

                level = "high" if score >= 0.5 else ("medium" if score >= 0.3 else "low")
                results.append({"index": i + 1, "score": round(score, 2), "level": level, "signals": signals})

            lines = [
                "## AI 水印检测结果", "",
                f"- 检测方法: {method}", f"- 文本数: {len(texts)}", "",
                "| # | 水印分数 | 级别 | 信号 |",
                "|---|---------|------|------|",
            ]
            for r in results:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["level"]]
                sigs = "; ".join(r["signals"]) if r["signals"] else "无"
                lines.append(f"| {r['index']} | {r['score']:.2f} | {icon} {r['level']} | {sigs} |")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-modelaudit[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
