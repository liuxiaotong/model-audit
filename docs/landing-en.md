<div align="center">

<h1>ModelAudit</h1>

<h3>LLM Distillation Detection and Model Fingerprinting<br/>via Statistical Forensics</h3>

<p>
<strong>Detect unauthorized model distillation through behavioral probing,<br/>stylistic fingerprinting, and representation similarity analysis.</strong>
</p>

<p><em>Statistical Forensics · Behavioral Signatures · Cross-Model Lineage Inference</em></p>

</div>

## The Problem

Large language model distillation has become a core threat to model IP protection. Student models can replicate teacher model capabilities by mimicking output distributions -- without authorization. Existing detection methods either require white-box weight access (often unavailable) or only analyze surface-level text features (easily evaded).

## The Solution

ModelAudit is a multi-method distillation detection framework based on **statistical forensics**. It extracts model fingerprints through **behavioral probing**, applies **hypothesis testing** to determine distillation relationships, and combines four complementary methods to form a complete black-box to white-box audit chain.

### Four Complementary Detection Methods

| Method | Type | Mechanism |
|:---|:---|:---|
| **LLMmap** | Black-box | 20 behavioral probes, Pearson correlation on response patterns |
| **DLI** | Black-box | Behavioral signatures + Jensen-Shannon divergence lineage inference |
| **REEF** | White-box | CKA layer-wise hidden state similarity |
| **StyleAnalysis** | Stylistic | 12 model family style signatures + language detection |

### 10-Dimensional Behavioral Probing

Go beyond simple text statistics. ModelAudit probes 10 cognitive dimensions -- self-awareness, safety boundaries, injection testing, reasoning, creative writing, multilingual, format control, role-playing, code generation, and summarization -- capturing deep behavioral differences that persist even after RLHF alignment.

### Cross-Provider Audit Chain

Audit across providers seamlessly. Teacher and student models can come from different APIs:

```bash
knowlyr-modelaudit audit \
  --teacher claude-opus --teacher-provider anthropic \
  --student kimi-k2.5 --student-provider openai \
  --student-api-base https://api.moonshot.cn/v1 \
  -o report.md
```

## Get Started

```bash
pip install knowlyr-modelaudit

# Detect text source
knowlyr-modelaudit detect texts.jsonl

# Verify model identity
knowlyr-modelaudit verify gpt-4o --provider openai

# Full distillation audit
knowlyr-modelaudit audit --teacher gpt-4o --student my-model -o report.md
```

```python
from modelaudit import AuditEngine

engine = AuditEngine()
audit = engine.audit("claude-opus", "suspect-model")
print(f"{audit.verdict} (confidence: {audit.confidence:.3f})")
```

## MCP Integration

ModelAudit ships with 8 MCP tools for seamless integration into AI workflows:

`detect_text_source` · `verify_model` · `compare_models` · `compare_models_whitebox` · `audit_distillation` · `audit_memorization` · `audit_report` · `audit_watermark`

## Built-in Benchmark

100% detection accuracy across 6 model families (14 samples). Supports 12 model families: GPT-4 · GPT-3.5 · Claude · LLaMA · Gemini · Qwen · DeepSeek · Mistral · Yi · Phi · Cohere · ChatGLM.

<div align="center">
<br/>
<p><a href="https://github.com/liuxiaotong/model-audit">GitHub</a> · <a href="https://pypi.org/project/knowlyr-modelaudit/">PyPI</a></p>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> — LLM distillation detection and model fingerprinting via statistical forensics</sub>
</div>
