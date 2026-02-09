<div align="center">

# ModelAudit

**LLM 蒸馏检测与模型指纹审计 — 文本溯源、身份验证、蒸馏关系判定**
**LLM distillation detection & model fingerprinting – text provenance, identity verification, distillation auditing**

[![PyPI](https://img.shields.io/pypi/v/knowlyr-modelaudit?color=blue)](https://pypi.org/project/knowlyr-modelaudit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-4_Tools-purple.svg)](#mcp-server)

[快速开始](#快速开始) · [检测方法](#检测方法) · [MCP Server](#mcp-server) · [Data Pipeline 生态](#data-pipeline-生态)

</div>

---

**GitHub Topics**: `model-fingerprint`, `llm-distillation`, `model-audit`, `cli`, `mcp`, `ai-data-pipeline`

检测文本数据来源、验证 API 模型身份、审计模型蒸馏关系。黑盒优先，标注员友好。

## 核心能力 / Core Capabilities

```
文本/模型 → 探测 Prompt → 响应特征提取 → 指纹比对 → 审计报告
```

### 审计仪表盘预览 / Sample Dashboard

```
┌───────────────────────────────────────────────┐
│  模型蒸馏审计报告                              │
├───────────────┬──────────────┬────────────────┤
│ 教师: gpt-4o  │ 学生: my-llm │ 相似度: 0.9213 │
├───────────────┴──────────────┴────────────────┤
│ ⚠️  判定: 可能存在蒸馏关系                      │
│ 📊 置信度: 87.5%                               │
│ 🔍 风格匹配: helpful 0.82 / hedging 0.79       │
└───────────────────────────────────────────────┘
```

### 功能矩阵 / Features

| 功能 | 说明 |
|------|------|
| 🔍 **文本来源检测** | 判断一批文本是哪个 LLM 生成的 |
| ✅ **模型身份验证** | 验证 API 背后是不是声称的模型 |
| 🔗 **模型指纹比对** | 比对两个模型的行为特征相似度 |
| 📋 **蒸馏审计报告** | 综合分析生成 Markdown / JSON 报告 |

## 安装 / Installation

```bash
pip install knowlyr-modelaudit
```

可选依赖：

```bash
pip install knowlyr-modelaudit[blackbox]   # 黑盒指纹 (openai, anthropic, httpx)
pip install knowlyr-modelaudit[whitebox]   # 白盒指纹 (torch, transformers)
pip install knowlyr-modelaudit[mcp]        # MCP 服务器
pip install knowlyr-modelaudit[all]        # 全部功能
```

## 快速开始 / Quick Start

### 检测文本来源 / CLI

```bash
# 检测文本数据是哪个模型生成的
knowlyr-modelaudit detect texts.jsonl

# 限制条数，输出 JSON
knowlyr-modelaudit detect texts.jsonl -n 50 -f json -o result.json
```

<details>
<summary>输出示例</summary>

```
正在分析 3 条文本...

  ID | 预测模型   |   置信度 | 预览
------------------------------------------------------------
   1 |    chatgpt |  72.50% | Certainly! I'd be happy to...
   2 |    chatgpt |  65.00% | I think that's an interest...
   3 |    chatgpt |  70.00% | Sure thing! No problem at ...

来源分布:
  chatgpt: 3 (100.0%)
```

</details>

### 验证模型身份

```bash
# 验证 API 背后是不是声称的 GPT-4o
knowlyr-modelaudit verify gpt-4o --provider openai

# 自定义 API
knowlyr-modelaudit verify my-model --provider custom --api-base http://localhost:8000
```

### 比对模型指纹

```bash
# 比对两个模型是否存在蒸馏关系
knowlyr-modelaudit compare gpt-4o claude-sonnet --provider openai
```

### 完整蒸馏审计

```bash
# 生成审计报告
knowlyr-modelaudit audit --teacher gpt-4o --student my-model -o report.md
```

<details>
<summary>输出示例</summary>

```
正在审计: gpt-4o → my-model...

判定结果: ⚠️  可能存在蒸馏关系
置信度: 0.8750

教师模型 gpt-4o 与学生模型 my-model 的行为模式高度相似，
可能存在蒸馏关系。置信度: 87.50%

报告已保存: report.md
```

</details>

### 在 Python 中接入 / Python SDK

```python
from modelaudit import AuditEngine

engine = AuditEngine()

# 检测文本来源
results = engine.detect(["Hello! I'd be happy to help..."])
for r in results:
    print(f"{r.predicted_model}: {r.confidence:.2%}")

# 比对模型指纹 (需要 API key)
result = engine.compare("gpt-4o", "my-model", method="llmmap")
print(f"相似度: {result.similarity:.4f}")
print(f"蒸馏关系: {'是' if result.is_derived else '否'}")
```

---

## 检测方法 / Detection Methods

### 已实现

| 方法 | 类型 | 说明 | 参考 |
|------|------|------|------|
| **LLMmap** | 黑盒 | 20 个探测 Prompt，分析响应模式 | USENIX Security 2025 |
| **StyleAnalysis** | 风格分析 | 12 个模型家族的风格签名匹配 | — |

### 支持识别的模型家族

`gpt-4` · `gpt-3.5` · `claude` · `llama` · `gemini` · `qwen` · `deepseek` · `mistral` · `yi` · `phi` · `cohere` · `chatglm`

### 探测维度（20 个 Probe）

| 维度 | 探测内容 |
|------|----------|
| 自我认知 | 模型身份、创建者、训练截止 |
| 安全边界 | 拒绝策略、措辞差异 |
| 注入测试 | Prompt injection 响应差异 |
| 知识与推理 | 知识边界、逻辑推理、伦理判断 |
| 创意写作 | 叙事风格、类比能力 |
| 多语言 | 中文响应、多语翻译 |
| 格式控制 | JSON 输出、Markdown 表格 |
| 角色扮演 | 角色一致性、创意表达 |
| 代码生成 | 编码风格、注释习惯 |
| 摘要能力 | 信息压缩、表达密度 |

### 规划中

| 方法 | 类型 | 说明 | 参考 |
|------|------|------|------|
| **REEF** | 白盒 | CKA 隐层相似度比对 | ICLR 2025 Oral |
| **DLI** | 蒸馏检测 | 影子模型 + 行为签名 | ICLR 2026 |

### 查看可用方法

```bash
knowlyr-modelaudit methods
```

---

## MCP Server

在 Claude Desktop / Claude Code 中直接使用。

### 配置

添加到 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "knowlyr-modelaudit": {
      "command": "uv",
      "args": ["--directory", "/path/to/model-audit", "run", "python", "-m", "modelaudit.mcp_server"]
    }
  }
}
```

### 可用工具

| 工具 | 功能 |
|------|------|
| `detect_text_source` | 检测文本数据来源 |
| `verify_model` | 验证模型身份 |
| `compare_models` | 比对两个模型指纹 |
| `audit_distillation` | 完整蒸馏审计 |

### 使用示例

```
用户: 帮我检测这批文本是哪个模型生成的

Claude: [调用 detect_text_source]

        ## 文本来源检测结果

        | # | 预测模型 | 置信度 | 预览 |
        |---|---------|--------|------|
        | 1 | chatgpt | 72.50% | Certainly! I'd be happy... |

        ### 来源分布
        - chatgpt: 3 (100.0%)
```

---

## Data Pipeline 生态

ModelAudit 是 Data Pipeline 生态的模型质检组件：

```mermaid
graph LR
    Radar["🔍 Radar<br/>情报发现"] --> Recipe["📋 Recipe<br/>逆向分析"]
    Recipe --> Synth["🔄 Synth<br/>数据合成"]
    Recipe --> Label["🏷️ Label<br/>数据标注"]
    Synth --> Check["✅ Check<br/>数据质检"]
    Label --> Check
    Check --> Audit["🔬 Audit<br/>模型审计"]
    Audit --> Hub["🎯 Hub<br/>编排层"]
    Hub --> Sandbox["📦 Sandbox<br/>执行沙箱"]
    Sandbox --> Recorder["📹 Recorder<br/>轨迹录制"]
    Recorder --> Reward["⭐ Reward<br/>过程打分"]
    style Audit fill:#0969da,color:#fff,stroke:#0969da
```

### 生态项目

| 层 | 项目 | 说明 | 仓库 |
|---|---|---|---|
| 情报 | **AI Dataset Radar** | 数据集竞争情报、趋势分析 | [GitHub](https://github.com/liuxiaotong/ai-dataset-radar) |
| 分析 | **DataRecipe** | 逆向分析、Schema 提取、成本估算 | [GitHub](https://github.com/liuxiaotong/data-recipe) |
| 生产 | **DataSynth** | LLM 批量合成、种子数据扩充 | [GitHub](https://github.com/liuxiaotong/data-synth) |
| 生产 | **DataLabel** | 轻量标注工具、多标注员合并 | [GitHub](https://github.com/liuxiaotong/data-label) |
| 质检 | **DataCheck** | 规则验证、重复检测、分布分析 | [GitHub](https://github.com/liuxiaotong/data-check) |
| 质检 | **ModelAudit** | 蒸馏检测、模型指纹、身份验证 | You are here |
| Agent | **AgentSandbox** | Docker 执行沙箱、轨迹重放 | [GitHub](https://github.com/liuxiaotong/agent-sandbox) |
| Agent | **AgentRecorder** | 标准化轨迹录制、多框架适配 | [GitHub](https://github.com/liuxiaotong/agent-recorder) |
| Agent | **AgentReward** | 过程级 Reward、Rubric 多维评估 | [GitHub](https://github.com/liuxiaotong/agent-reward) |
| 编排 | **TrajectoryHub** | Pipeline 编排、数据集导出 | [GitHub](https://github.com/liuxiaotong/agent-trajectory-hub) |

### 端到端工作流

```bash
# 1. DataRecipe: 分析数据集，生成 Schema 和样例
knowlyr-datarecipe deep-analyze tencent/CL-bench -o ./output

# 2. DataSynth: 基于种子数据批量合成
knowlyr-datasynth generate ./output/tencent_CL-bench/ -n 1000

# 3. DataCheck: 数据质量检查
knowlyr-datacheck validate ./output/tencent_CL-bench/

# 4. ModelAudit: 检测合成数据来源，验证模型身份
knowlyr-modelaudit detect ./output/synthetic.jsonl
knowlyr-modelaudit verify gpt-4o --provider openai
```

### 组合 MCP 配置

```json
{
  "mcpServers": {
    "knowlyr-datarecipe": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-recipe", "run", "knowlyr-datarecipe-mcp"]
    },
    "knowlyr-datacheck": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-check", "run", "python", "-m", "datacheck.mcp_server"]
    },
    "knowlyr-modelaudit": {
      "command": "uv",
      "args": ["--directory", "/path/to/model-audit", "run", "python", "-m", "modelaudit.mcp_server"]
    }
  }
}
```

---

## 命令参考

| 命令 | 功能 |
|------|------|
| `knowlyr-modelaudit detect <file>` | 检测文本数据来源 |
| `knowlyr-modelaudit detect <file> -n 50` | 限制检测条数 |
| `knowlyr-modelaudit verify <model>` | 验证模型身份 |
| `knowlyr-modelaudit compare <a> <b>` | 比对两个模型指纹 |
| `knowlyr-modelaudit audit --teacher <a> --student <b>` | 完整蒸馏审计 |
| `knowlyr-modelaudit methods` | 列出可用检测方法 |

---

## API 使用

```python
from modelaudit import AuditEngine, Fingerprint, ComparisonResult

# 创建引擎
engine = AuditEngine()

# 检测文本来源
results = engine.detect(texts)
for r in results:
    print(f"#{r.text_id} {r.predicted_model} ({r.confidence:.2%})")

# 指纹比对 (需要 API key)
result = engine.compare("gpt-4o", "my-model", method="llmmap")
print(f"相似度: {result.similarity:.4f}")

# 完整审计
audit = engine.audit("gpt-4o", "my-model")
print(audit.verdict)       # likely_derived / independent / inconclusive
print(audit.confidence)    # 0.875

# 生成报告
from modelaudit.report import generate_report
report = generate_report(audit, "markdown")
```

---

## 项目架构

```
src/modelaudit/
├── engine.py         # AuditEngine 总入口
├── models.py         # Pydantic 数据模型
├── base.py           # Fingerprinter 抽象基类
├── registry.py       # 方法注册表
├── config.py         # 配置
├── methods/
│   ├── llmmap.py     # LLMmap 黑盒指纹
│   └── style.py      # 风格分析
├── probes/
│   └── prompts.py    # 探测 Prompt 库
├── report.py         # 报告生成
├── cli.py            # CLI 命令行 (5 命令)
└── mcp_server.py     # MCP Server (4 工具)
```

---

## License

[MIT](LICENSE)

---

## AI Data Pipeline 生态

> 10 个工具覆盖 AI 数据工程全流程，均支持 CLI + MCP，可独立使用也可组合成流水线。

| Tool | Description | Link |
|------|-------------|------|
| **AI Dataset Radar** | Competitive intelligence for AI training datasets | [GitHub](https://github.com/liuxiaotong/ai-dataset-radar) |
| **DataRecipe** | Reverse-engineer datasets into annotation specs & cost models | [GitHub](https://github.com/liuxiaotong/data-recipe) |
| **DataSynth** | Seed-to-scale synthetic data generation | [GitHub](https://github.com/liuxiaotong/data-synth) |
| **DataLabel** | Lightweight, serverless HTML labeling tool | [GitHub](https://github.com/liuxiaotong/data-label) |
| **DataCheck** | Automated quality checks & anomaly detection | [GitHub](https://github.com/liuxiaotong/data-check) |
| **ModelAudit** | LLM distillation detection & model fingerprinting | You are here |
| **AgentSandbox** | Reproducible Docker sandbox for Code Agent execution | [GitHub](https://github.com/liuxiaotong/agent-sandbox) |
| **AgentRecorder** | Standardized trajectory recording for Code Agents | [GitHub](https://github.com/liuxiaotong/agent-recorder) |
| **AgentReward** | Process-level rubric-based reward engine | [GitHub](https://github.com/liuxiaotong/agent-reward) |
| **TrajectoryHub** | Pipeline orchestrator for Agent trajectory data | [GitHub](https://github.com/liuxiaotong/agent-trajectory-hub) |

```mermaid
graph LR
    A[Radar] --> B[Recipe] --> C[Synth] --> E[Check] --> F[Audit] --> G[Hub]
    B --> D[Label] --> E
    G --> H[Sandbox] --> I[Recorder] --> J[Reward]
```

---

<div align="center">
<sub>为数据团队提供模型质量保障与蒸馏审计能力</sub>
</div>
