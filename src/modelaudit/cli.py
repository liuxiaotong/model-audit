"""ModelAudit CLI — 命令行界面."""

import json
import logging
import sys
from pathlib import Path

import click

from modelaudit import __version__
from modelaudit.config import AuditConfig
from modelaudit.engine import AuditEngine


def _setup_logging(verbose: bool) -> None:
    """配置日志级别."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-modelaudit")
@click.option("-v", "--verbose", is_flag=True, default=False, help="显示详细日志")
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """ModelAudit — LLM 蒸馏检测与模型指纹审计工具

    检测文本来源、验证模型身份、审计蒸馏关系。
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="结果输出路径")
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["json", "csv", "table"]),
    default="table",
    help="输出格式",
)
@click.option("-n", "--limit", type=int, default=0, help="限制检测条数 (0=全部)")
@click.option(
    "--field", type=str, default=None,
    help="JSONL/JSON 中的文本字段名 (默认自动尝试 text/content/output)",
)
def detect(
    data_path: str,
    output: str | None,
    output_format: str,
    limit: int,
    field: str | None,
):
    """检测文本数据来源 — 判断文本是哪个模型生成的

    DATA_PATH: JSONL/JSON/TXT 文件

    \b
    示例:
      knowlyr-modelaudit detect texts.jsonl
      knowlyr-modelaudit detect data.jsonl --field response -f csv -o result.csv
    """
    texts = _load_texts(data_path, field=field)

    if not texts:
        click.echo("错误: 文件中没有找到文本数据", err=True)
        sys.exit(1)

    if limit > 0:
        texts = texts[:limit]

    short_count = sum(1 for t in texts if len(t.split()) < 10)
    if short_count > 0:
        click.echo(f"⚠ {short_count} 条文本少于 10 个词，检测置信度可能较低", err=True)

    engine = AuditEngine()

    # 大批量时显示进度条
    if len(texts) > 10:
        try:
            from rich.progress import Progress

            results = []
            with Progress() as progress:
                task = progress.add_task("分析文本来源...", total=len(texts))
                # 分批处理以更新进度
                batch_size = 50
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    results.extend(engine.detect(batch))
                    progress.update(task, advance=len(batch))
            # 修正 text_id（分批后需要重新编号）
            for idx, r in enumerate(results):
                r.text_id = idx
        except ImportError:
            click.echo(f"正在分析 {len(texts)} 条文本...")
            results = engine.detect(texts)
    else:
        click.echo(f"正在分析 {len(texts)} 条文本...")
        results = engine.detect(texts)

    if output_format == "table":
        _print_detection_table(results)
    elif output_format == "csv":
        _print_detection_csv(results)
    else:
        result_dicts = [r.model_dump() for r in results]
        output_text = json.dumps(result_dicts, ensure_ascii=False, indent=2)
        click.echo(output_text)

    if output:
        if output_format == "csv":
            _save_detection_csv(results, output)
        else:
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
    engine = AuditEngine(config, use_cache=True)

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
    help="默认 API 提供商 (两个模型相同时使用)",
)
@click.option("--api-key", type=str, default="", help="默认 API Key")
@click.option("--api-base", type=str, default="", help="默认 API 地址")
@click.option("--teacher-provider", type=str, default=None, help="教师模型 API 提供商")
@click.option("--teacher-api-key", type=str, default=None, help="教师模型 API Key")
@click.option("--teacher-api-base", type=str, default=None, help="教师模型 API 地址")
@click.option("--student-provider", type=str, default=None, help="学生模型 API 提供商")
@click.option("--student-api-key", type=str, default=None, help="学生模型 API Key")
@click.option("--student-api-base", type=str, default=None, help="学生模型 API 地址")
@click.option("-o", "--output", type=click.Path(), help="审计报告输出路径")
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="报告格式",
)
@click.option("--no-cache", is_flag=True, default=False, help="不使用指纹缓存，强制重新调用 API")
def audit(
    teacher: str,
    student: str,
    provider: str,
    api_key: str,
    api_base: str,
    teacher_provider: str | None,
    teacher_api_key: str | None,
    teacher_api_base: str | None,
    student_provider: str | None,
    student_api_key: str | None,
    student_api_base: str | None,
    output: str | None,
    output_format: str,
    no_cache: bool,
):
    """完整蒸馏审计 — 综合指纹比对 + 风格分析

    生成详细审计报告。支持双模型分别配置不同 API 提供商。

    \b
    示例:
      # 同一 provider
      knowlyr-modelaudit audit --teacher gpt-4o --student gpt-3.5 -p openai

      # 跨 provider 审计
      knowlyr-modelaudit audit \\
        --teacher claude-opus --teacher-provider anthropic \\
        --student kimi-k2.5 --student-provider openai \\
        --student-api-base https://api.moonshot.cn/v1 \\
        -o report.md
    """
    click.echo(f"正在审计: {teacher} → {student}...")

    config = AuditConfig(provider=provider, api_key=api_key, api_base=api_base)
    engine = AuditEngine(config, use_cache=not no_cache)

    try:
        result = engine.audit(
            teacher, student,
            teacher_provider=teacher_provider,
            teacher_api_key=teacher_api_key,
            teacher_api_base=teacher_api_base,
            student_provider=student_provider,
            student_api_key=student_api_key,
            student_api_base=student_api_base,
        )
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    # 终端输出摘要
    verdict_text = {
        "likely_derived": "⚠️  可能存在蒸馏关系",
        "independent": "✓ 两个模型独立",
        "inconclusive": "? 无法确定",
    }

    click.echo(f"\n判定结果: {verdict_text.get(result.verdict, result.verdict)}")
    click.echo(f"置信度: {result.confidence:.4f}")
    click.echo(f"\n{result.summary}")

    # 生成报告
    from modelaudit.report import generate_report

    report_content = generate_report(result, output_format)

    if output:
        Path(output).write_text(report_content, encoding="utf-8")
        click.echo(f"\n报告已保存: {output}")
    else:
        # 无 -o 时自动保存到 reports/ 目录
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        filename = f"{student}-vs-{teacher}-audit.{'md' if output_format == 'markdown' else 'json'}"
        report_path = reports_dir / filename
        report_path.write_text(report_content, encoding="utf-8")
        click.echo(f"\n报告已自动保存: {report_path}")


@main.command()
@click.option("--category", type=click.Choice(["qa", "creative", "code", "reasoning"]), help="Filter by category")
@click.option("--label", help="Filter by model family (gpt-4, claude, llama, etc.)")
@click.pass_context
def benchmark(ctx, category, label):
    """运行内置 benchmark — 评估文本来源检测准确率."""
    from modelaudit.benchmark import evaluate_accuracy, get_benchmark_samples

    verbose = ctx.obj.get("verbose", False) if ctx.obj else False
    _setup_logging(verbose)

    samples = get_benchmark_samples(category=category, label=label)
    if not samples:
        click.echo("没有匹配的 benchmark 样本。")
        return

    click.echo(f"运行 benchmark: {len(samples)} 条样本...")

    engine = AuditEngine(use_cache=False)
    texts = [s.text for s in samples]
    results = engine.detect(texts)

    # 匹配预测和真实标签
    predictions = []
    for result, sample in zip(results, samples, strict=True):
        pred = result.predicted_model
        predictions.append((pred, sample.label))

    eval_result = evaluate_accuracy(predictions)

    # 显示结果
    click.echo(f"\n{'='*50}")
    click.echo(f"总体准确率: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
    click.echo(f"{'='*50}")

    if eval_result["per_class"]:
        click.echo("\n按模型家族:")
        for model, acc in sorted(eval_result["per_class"].items()):
            click.echo(f"  {model:<12} {acc:.1%}")

    # 详细结果
    click.echo(f"\n{'─'*50}")
    click.echo(f"{'#':<4} {'真实':<12} {'预测':<12} {'结果'}")
    click.echo(f"{'─'*50}")

    for i, (pred, true) in enumerate(predictions, 1):
        icon = "✓" if pred == true else "✗"
        click.echo(f"{i:<4} {true:<12} {pred:<12} {icon}")


@main.group()
def cache():
    """管理指纹缓存"""
    pass


@cache.command("list")
@click.option("--cache-dir", type=str, default=".modelaudit_cache", help="缓存目录")
def cache_list(cache_dir: str):
    """列出缓存的指纹"""
    from modelaudit.cache import FingerprintCache

    c = FingerprintCache(cache_dir)
    entries = c.list_entries()

    if not entries:
        click.echo("缓存为空")
        return

    click.echo(f"\n缓存目录: {cache_dir}/")
    click.echo(f"共 {len(entries)} 条指纹:\n")

    for e in entries:
        click.echo(f"  {e['model']:>20} | {e['method']:>8} | {e['type']:>8} | {e['size']:>8} | {e['file']}")


@cache.command("clear")
@click.option("--cache-dir", type=str, default=".modelaudit_cache", help="缓存目录")
@click.confirmation_option(prompt="确认清除所有缓存?")
def cache_clear(cache_dir: str):
    """清除所有缓存"""
    from modelaudit.cache import FingerprintCache

    c = FingerprintCache(cache_dir)
    count = c.clear()
    click.echo(f"已清除 {count} 条缓存")


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
        "reef": "基于 CKA 中间层表示相似度检测蒸馏关系\n      参考: REEF (NeurIPS 2024)",
        "dli": "基于行为签名 + JS 散度的蒸馏血缘推断\n      参考: DLI (ICLR 2026)",
    }

    for name, desc in method_details.items():
        if name in available:
            click.echo(f"      {desc}")

    click.echo("\n" + "=" * 40)
    click.echo("\n其他功能:")
    click.echo("  - detect:  分析文本来源（基于风格分析）")
    click.echo("  - verify:  验证模型身份")
    click.echo("  - compare: 比对两个模型")
    click.echo("  - audit:   完整蒸馏审计（生成详细报告）")


def _load_texts(data_path: str, field: str | None = None) -> list[str]:
    """从文件加载文本数据. 支持 JSONL/JSON/CSV/TXT."""
    path = Path(data_path)
    texts: list[str] = []

    if path.suffix in (".jsonl", ".ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = _extract_text(obj, field)
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
                    text = _extract_text(item, field)
                    if text:
                        texts.append(text)
    elif path.suffix == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                text = _extract_text(row, field)
                if text:
                    texts.append(text)
            # 如果没有找到文本且未指定 field，提示可用列名
            if not texts and not field and rows:
                available = ", ".join(rows[0].keys())
                raise click.UsageError(
                    f"CSV 中未找到 text/content/output 列。"
                    f"可用列: {available}\n"
                    f"请用 --field 指定文本列名"
                )
    else:
        # 纯文本，按段落分割
        content = path.read_text(encoding="utf-8")
        paragraphs = content.split("\n\n")
        texts = [p.strip() for p in paragraphs if p.strip()]

    return texts


def _extract_text(obj: dict | str, field: str | None = None) -> str:
    """从字典中提取文本字段."""
    if isinstance(obj, str):
        return obj
    if field:
        return obj.get(field, "")
    return obj.get("text") or obj.get("content") or obj.get("output") or ""


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


def _print_detection_csv(results: list) -> None:
    """打印 CSV 格式检测结果."""
    click.echo("id,predicted_model,confidence,text_preview")
    for r in results:
        # 转义预览中的逗号和引号
        preview = r.text_preview.replace('"', '""')
        click.echo(f'{r.text_id},"{r.predicted_model}",{r.confidence:.4f},"{preview}"')


def _save_detection_csv(results: list, output_path: str) -> None:
    """保存 CSV 格式检测结果到文件."""
    import csv
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_model", "confidence", "text_preview"])
        for r in results:
            writer.writerow([r.text_id, r.predicted_model, f"{r.confidence:.4f}", r.text_preview])


if __name__ == "__main__":
    main()
