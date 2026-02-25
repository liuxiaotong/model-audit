<div align="center">

<h1>ModelAudit</h1>

<h3>LLM 蒸馏检测与模型指纹审计<br/>LLM Distillation Detection and Model Fingerprinting<br/>via Statistical Forensics</h3>

<p>
<strong>通过行为探测、风格指纹和表示相似度分析<br/>检测未经授权的模型蒸馏行为</strong>
</p>

<p><em>统计取证 · 行为签名 · 跨模型血缘推断</em></p>

</div>

## 问题背景

大语言模型的蒸馏行为 (knowledge distillation) 已成为模型知识产权保护的核心威胁。学生模型通过模仿教师模型的输出分布，可以在未经授权的情况下复制其能力。现有检测方法要么依赖白盒权重访问（实际场景中通常不可得），要么仅分析表面文本特征（易被规避）。

## 解决方案

ModelAudit 是基于**统计取证**的多方法蒸馏检测框架。通过**行为探测**提取模型指纹，基于**假设检验**判定蒸馏关系，融合四种互补方法构成从黑盒到白盒的完整审计链。

### 四种互补检测方法

| 方法 | 类型 | 原理 |
|:---|:---|:---|
| **LLMmap** | 黑盒 | 20 个行为探测，Pearson 相关比对响应模式 |
| **DLI** | 黑盒 | 行为签名 + Jensen-Shannon 散度血缘推断 |
| **REEF** | 白盒 | CKA 逐层隐藏状态相似度 |
| **StyleAnalysis** | 风格分析 | 12 个模型家族风格签名 + 语言检测 |

### 10 维行为探测

超越简单的文本统计特征，ModelAudit 从 10 个认知维度进行结构化探测——自我认知、安全边界、注入测试、知识与推理、创意写作、多语言、格式控制、角色扮演、代码生成、摘要能力——捕获在 RLHF 对齐后仍保留的深层行为差异。

### 跨 Provider 审计链

无缝支持跨 Provider 蒸馏审计，教师和学生模型可来自不同 API：

```bash
knowlyr-modelaudit audit \
  --teacher claude-opus --teacher-provider anthropic \
  --student kimi-k2.5 --student-provider openai \
  --student-api-base https://api.moonshot.cn/v1 \
  -o report.md
```

## 快速开始

```bash
pip install knowlyr-modelaudit

# 检测文本来源
knowlyr-modelaudit detect texts.jsonl

# 验证模型身份
knowlyr-modelaudit verify gpt-4o --provider openai

# 完整蒸馏审计
knowlyr-modelaudit audit --teacher gpt-4o --student my-model -o report.md
```

```python
from modelaudit import AuditEngine

engine = AuditEngine()
audit = engine.audit("claude-opus", "suspect-model")
print(f"{audit.verdict} (confidence: {audit.confidence:.3f})")
```

## MCP 集成

ModelAudit 内置 8 个 MCP 工具，无缝融入 AI 工作流：

`detect_text_source` · `verify_model` · `compare_models` · `compare_models_whitebox` · `audit_distillation` · `audit_memorization` · `audit_report` · `audit_watermark`

## 内置基准测试

在 6 个模型家族（14 个样本）上达到 100% 检测准确率。支持识别 12 个模型家族：GPT-4 · GPT-3.5 · Claude · LLaMA · Gemini · Qwen · DeepSeek · Mistral · Yi · Phi · Cohere · ChatGLM。

<div align="center">
<br/>
<p><a href="https://github.com/liuxiaotong/model-audit">GitHub</a> · <a href="https://pypi.org/project/knowlyr-modelaudit/">PyPI</a></p>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> — LLM 蒸馏检测与模型指纹审计 · 统计取证</sub>
</div>
