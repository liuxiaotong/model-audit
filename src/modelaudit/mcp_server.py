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
