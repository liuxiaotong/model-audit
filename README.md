# model-audit

LLM 蒸馏检测与模型指纹审计工具。

## 功能

- **检测文本来源** — 判断文本数据是哪个模型生成的
- **验证模型身份** — 验证 API 背后是不是声称的模型
- **比对模型指纹** — 判断两个模型是否存在蒸馏/派生关系
- **蒸馏审计** — 综合审计生成报告

## 安装

```bash
# 基础安装
pip install knowlyr-modelaudit

# 黑盒功能（需要 API 调用）
pip install knowlyr-modelaudit[blackbox]

# 全部功能
pip install knowlyr-modelaudit[all]
```

## 使用

```bash
# 检测文本来源
knowlyr-modelaudit detect texts.jsonl

# 验证模型身份
knowlyr-modelaudit verify gpt-4o --provider openai

# 比对两个模型
knowlyr-modelaudit compare gpt-4o claude-sonnet --provider openai

# 完整蒸馏审计
knowlyr-modelaudit audit --teacher gpt-4o --student my-model -o report.md

# 列出可用方法
knowlyr-modelaudit methods
```

## 许可证

MIT
