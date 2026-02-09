"""ModelAudit CLI — 命令行界面."""

import json
import sys
from pathlib import Path

import click

from modelaudit import __version__
from modelaudit.config import AuditConfig
from modelaudit.engine import AuditEngine


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-modelaudit")
def main():
    """ModelAudit — LLM 蒸馏检测与模型指纹审计工具

    检测文本来源、验证模型身份、审计蒸馏关系。
    """
    pass


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="结果输出路径")
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="输出格式",
)
@click.option("-n", "--limit", type=int, default=0, help="限制检测条数 (0=全部)")
def detect(data_path: str, output: str | None, output_format: str, limit: int):
    """检测文本数据来源 — 判断文本是哪个模型生成的

    DATA_PATH: JSONL 文件，每行一个 JSON 对象，需包含 text 字段
    """
    texts = _load_texts(data_path)

    if not texts:
        click.echo("错误: 文件中没有找到文本数据", err=True)
        sys.exit(1)

    if limit > 0:
        texts = texts[:limit]

    click.echo(f"正在分析 {len(texts)} 条文本...")

    engine = AuditEngine()
    results = engine.detect(texts)

    if output_format == "table":
        _print_detection_table(results)
    else:
        result_dicts = [r.model_dump() for r in results]
        output_text = json.dumps(result_dicts, ensure_ascii=False, indent=2)
        click.echo(output_text)

    if output:
        result_dicts = [r.model_dump() for r in results]
        Path(output).write_text(
            json.dumps(result_dicts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        click.echo(f"\n结果已保存: {output}")

    # 统计
    model_counts: dict[str, int] = {}
    for r in results:
        model_counts[r.predicted_model] = model_counts.get(r.predicted_model, 0) + 1

    click.echo("\n来源分布:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        click.echo(f"  {model}: {count} ({pct:.1f}%)")


@main.command()
@click.argument("model", type=str)
@click.option(
    "-p", "--provider",
    type=click.Choice(["openai", "anthropic", "custom"]),
    default="openai",
    help="API 提供商",
)
@click.option("--api-key", type=str, default="", help="API Key (默认使用环境变量)")
@click.option("--api-base", type=str, default="", help="自定义 API 地址")
def verify(model: str, provider: str, api_key: str, api_base: str):
    """验证模型身份 — 检查 API 背后是不是声称的模型

    MODEL: 模型名称 (如 gpt-4o, claude-3-opus)
    """
    click.echo(f"正在验证 {model} (provider: {provider})...")

    config = AuditConfig(provider=provider, api_key=api_key, api_base=api_base)
    engine = AuditEngine(config)

    try:
        result = engine.verify(model, provider=provider, api_key=api_key, api_base=api_base)
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    verified = result["verified"]
    icon = "✓" if verified else "✗"

    click.echo(f"\n{icon} 模型身份验证: {'通过' if verified else '未通过'}")
    click.echo(f"  声称模型: {model}")
    click.echo(f"  最佳匹配: {result['best_match']} (分数: {result['best_score']})")

    if result.get("all_scores"):
        click.echo("\n  风格匹配分数:")
        for name, score in result["all_scores"].items():
            bar = "█" * int(score * 20)
            click.echo(f"    {name:>10}: {bar} {score:.4f}")


@main.command()
@click.argument("model_a", type=str)
@click.argument("model_b", type=str)
@click.option(
    "-p", "--provider",
    type=str,
    default="openai",
    help="API 提供商 (可用逗号分隔指定两个: openai,anthropic)",
)
@click.option("--api-key", type=str, default="", help="API Key")
@click.option("--api-base", type=str, default="", help="自定义 API 地址")
@click.option("--threshold", type=float, default=0.85, help="派生判定阈值")
def compare(
    model_a: str,
    model_b: str,
    provider: str,
    api_key: str,
    api_base: str,
    threshold: float,
):
    """比对两个模型 — 判断是否存在蒸馏关系

    MODEL_A: 第一个模型
    MODEL_B: 第二个模型
    """
    click.echo(f"正在比对 {model_a} vs {model_b}...")

    config = AuditConfig(
        provider=provider.split(",")[0],
        api_key=api_key,
        api_base=api_base,
        similarity_threshold=threshold,
    )
    engine = AuditEngine(config)

    try:
        result = engine.compare(
            model_a, model_b,
            method="llmmap",
            provider=provider.split(",")[0],
            api_key=api_key,
            api_base=api_base,
        )
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    icon = "⚠️" if result.is_derived else "✓"
    derived_text = "可能存在派生关系" if result.is_derived else "可能是独立模型"

    click.echo(f"\n{icon} 比对结果: {derived_text}")
    click.echo(f"  相似度: {result.similarity:.4f}")
    click.echo(f"  阈值: {result.threshold}")
    click.echo(f"  置信度: {result.confidence:.4f}")


@main.command()
@click.option("--teacher", type=str, required=True, help="教师模型 (疑似被蒸馏的源模型)")
@click.option("--student", type=str, required=True, help="学生模型 (疑似蒸馏产物)")
@click.option(
    "-p", "--provider",
    type=str,
    default="openai",
    help="API 提供商",
)
@click.option("--api-key", type=str, default="", help="API Key")
@click.option("--api-base", type=str, default="", help="自定义 API 地址")
@click.option("-o", "--output", type=click.Path(), help="审计报告输出路径")
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="报告格式",
)
def audit(
    teacher: str,
    student: str,
    provider: str,
    api_key: str,
    api_base: str,
    output: str | None,
    output_format: str,
):
    """完整蒸馏审计 — 综合指纹比对 + 风格分析

    生成详细审计报告。
    """
    click.echo(f"正在审计: {teacher} → {student}...")

    config = AuditConfig(provider=provider, api_key=api_key, api_base=api_base)
    engine = AuditEngine(config)

    try:
        result = engine.audit(
            teacher, student,
            provider=provider, api_key=api_key, api_base=api_base,
        )
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    verdict_text = {
        "likely_derived": "⚠️  可能存在蒸馏关系",
        "independent": "✓ 两个模型独立",
        "inconclusive": "? 无法确定",
    }

    click.echo(f"\n判定结果: {verdict_text.get(result.verdict, result.verdict)}")
    click.echo(f"置信度: {result.confidence:.4f}")
    click.echo(f"\n{result.summary}")

    if output:
        from modelaudit.report import generate_report

        report_content = generate_report(result, output_format)
        Path(output).write_text(report_content, encoding="utf-8")
        click.echo(f"\n报告已保存: {output}")


@main.command()
def methods():
    """列出所有可用的检测方法"""
    # 确保方法已注册
    import modelaudit.methods  # noqa: F401
    from modelaudit.registry import list_methods

    available = list_methods()

    click.echo("\n可用指纹方法:")
    click.echo("=" * 40)

    for name, fp_type in available.items():
        type_icon = "🔓" if fp_type == "whitebox" else "🔒"
        type_label = "白盒" if fp_type == "whitebox" else "黑盒"
        click.echo(f"\n  {type_icon} {name} ({type_label})")

    method_details = {
        "llmmap": "基于探测 prompt 响应模式识别模型身份\n      参考: LLMmap (USENIX Security 2025)",
    }

    for name, desc in method_details.items():
        if name in available:
            click.echo(f"      {desc}")

    click.echo("\n" + "=" * 40)
    click.echo("\n其他功能:")
    click.echo("  - detect:  分析文本来源（基于风格分析）")
    click.echo("  - verify:  验证模型身份")
    click.echo("  - compare: 比对两个模型")
    click.echo("  - audit:   完整蒸馏审计")


def _load_texts(data_path: str) -> list[str]:
    """从文件加载文本数据. 支持 JSONL 和纯文本."""
    path = Path(data_path)
    texts: list[str] = []

    if path.suffix in (".jsonl", ".ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 尝试多种字段名
                text = obj.get("text") or obj.get("content") or obj.get("output") or ""
                if isinstance(obj, str):
                    text = obj
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("output") or ""
                    if text:
                        texts.append(text)
    else:
        # 纯文本，按段落分割
        content = path.read_text(encoding="utf-8")
        paragraphs = content.split("\n\n")
        texts = [p.strip() for p in paragraphs if p.strip()]

    return texts


def _print_detection_table(results: list) -> None:
    """打印检测结果表格."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="文本来源检测结果")

        table.add_column("ID", justify="right", style="dim")
        table.add_column("预览", max_width=50)
        table.add_column("预测模型", style="cyan")
        table.add_column("置信度", justify="right", style="green")

        for r in results:
            conf_str = f"{r.confidence:.2%}"
            table.add_row(str(r.text_id), r.text_preview, r.predicted_model, conf_str)

        console.print(table)
    except ImportError:
        # 无 rich 时用纯文本
        click.echo(f"{'ID':>4} | {'预测模型':>10} | {'置信度':>8} | 预览")
        click.echo("-" * 60)
        for r in results:
            click.echo(f"{r.text_id:>4} | {r.predicted_model:>10} | {r.confidence:>7.2%} | {r.text_preview}")


if __name__ == "__main__":
    main()
