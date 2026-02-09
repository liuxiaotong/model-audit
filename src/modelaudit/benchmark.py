"""内置 benchmark 数据集 — 用于评估 detect 准确率.

包含已知来源的 LLM 生成文本样本，用于评估风格检测的准确率。
"""

from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    """单条 benchmark 样本."""

    text: str
    label: str  # 正确模型家族 (gpt-4, claude, llama 等)
    category: str  # 文本类别 (qa, creative, code, reasoning)


# ── 内置 benchmark 样本 ──
# 模拟各 LLM 家族典型风格的文本 (用于离线评估)
BENCHMARK_SAMPLES: list[BenchmarkSample] = [
    # GPT-4 风格
    BenchmarkSample(
        text=(
            "Certainly! Here's a comprehensive breakdown of the topic. "
            "First, let's consider the key factors at play. The primary "
            "consideration is that machine learning models learn patterns "
            "from data, and these patterns can sometimes reflect biases "
            "present in the training data. It's important to note that "
            "this is an active area of research with ongoing developments."
        ),
        label="gpt-4",
        category="qa",
    ),
    BenchmarkSample(
        text=(
            "Here's a Python implementation that addresses your requirements:\n\n"
            "```python\ndef calculate_fibonacci(n: int) -> list[int]:\n"
            '    """Calculate the first n Fibonacci numbers."""\n'
            "    if n <= 0:\n        return []\n"
            "    if n == 1:\n        return [0]\n"
            "    fib = [0, 1]\n"
            "    for _ in range(2, n):\n"
            "        fib.append(fib[-1] + fib[-2])\n"
            "    return fib\n```\n\n"
            "This solution uses an iterative approach with O(n) time complexity "
            "and O(n) space complexity. Let me know if you'd like me to explain "
            "any part in more detail!"
        ),
        label="gpt-4",
        category="code",
    ),
    BenchmarkSample(
        text=(
            "That's a great question! Let me walk you through the reasoning "
            "step by step. The key insight here is that we need to consider "
            "both the short-term and long-term implications. In the short term, "
            "the approach seems viable, but when we look at scalability, "
            "there are several factors worth considering. Would you like me "
            "to elaborate on any specific aspect?"
        ),
        label="gpt-4",
        category="reasoning",
    ),
    # Claude 风格
    BenchmarkSample(
        text=(
            "I'd be happy to help with that. Let me think through this carefully. "
            "The question touches on some nuanced aspects of the topic. "
            "I should note that there are different perspectives on this, and "
            "I want to be straightforward about the uncertainty involved. "
            "Based on my understanding, the most balanced view would be that "
            "while there are clear benefits, we should also consider potential "
            "drawbacks and ethical implications."
        ),
        label="claude",
        category="qa",
    ),
    BenchmarkSample(
        text=(
            "I appreciate you sharing that creative prompt. Here's my attempt:\n\n"
            "The morning light filtered through the old library's windows, "
            "casting long shadows across the dusty shelves. Sarah traced her "
            "fingers along the spines of forgotten books, each one holding "
            "stories that had waited decades to be read again. She paused "
            "at a leather-bound volume with no title — just a small, "
            "embossed symbol that seemed to shimmer in the light.\n\n"
            "I tried to capture a sense of mystery and atmosphere. "
            "Would you like me to continue or take it in a different direction?"
        ),
        label="claude",
        category="creative",
    ),
    BenchmarkSample(
        text=(
            "Let me reason through this step by step. First, we need to "
            "identify the core constraints. The problem states that we have "
            "a limited budget and need to maximize coverage. This is essentially "
            "a variant of the knapsack problem. I want to be careful here — "
            "the optimal solution depends on whether we're looking at the "
            "fractional or 0/1 variant. For the 0/1 case, dynamic programming "
            "gives us the best approach."
        ),
        label="claude",
        category="reasoning",
    ),
    # Llama 风格
    BenchmarkSample(
        text=(
            "Sure thing! So basically, machine learning is all about teaching "
            "computers to learn from data. Think of it like training a dog — "
            "you show it examples of what you want, and it learns to do it. "
            "The main types are supervised learning (where you give labeled data), "
            "unsupervised learning (where the model finds patterns on its own), "
            "and reinforcement learning (where it learns by trial and error). "
            "Pretty cool stuff!"
        ),
        label="llama",
        category="qa",
    ),
    BenchmarkSample(
        text=(
            "Here you go! Check out this code:\n\n"
            "```\ndef merge_sort(arr):\n"
            "    if len(arr) <= 1:\n        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    result.extend(left[i:])\n"
            "    result.extend(right[j:])\n"
            "    return result\n```\n\n"
            "This is a classic merge sort! It's O(n log n) time. "
            "Let me know if you need anything else!"
        ),
        label="llama",
        category="code",
    ),
    # Gemini 风格
    BenchmarkSample(
        text=(
            "Great question! Here's what you need to know about quantum computing. "
            "Quantum computers leverage quantum mechanical phenomena like "
            "superposition and entanglement to process information. Unlike "
            "classical bits (0 or 1), qubits can exist in multiple states "
            "simultaneously. **Key applications** include:\n\n"
            "* Cryptography and security\n"
            "* Drug discovery and molecular simulation\n"
            "* Optimization problems\n"
            "* Machine learning acceleration\n\n"
            "It's worth noting that we're still in the early stages of "
            "practical quantum computing. Current systems are noisy and "
            "error-prone, but progress is accelerating rapidly."
        ),
        label="gemini",
        category="qa",
    ),
    BenchmarkSample(
        text=(
            "Let me break this problem down systematically. We're looking at "
            "a graph traversal problem, and I think BFS would be the most "
            "appropriate approach here. Here's why:\n\n"
            "1. We need the shortest path\n"
            "2. All edges have equal weight\n"
            "3. The graph is unweighted\n\n"
            "The time complexity would be O(V + E) where V is vertices and "
            "E is edges. This is optimal for this type of problem. "
            "I can provide a code implementation if that would be helpful!"
        ),
        label="gemini",
        category="reasoning",
    ),
    # Qwen 风格
    BenchmarkSample(
        text=(
            "好的，我来为您详细解答这个问题。关于深度学习中的注意力机制，"
            "它的核心思想是让模型能够动态地关注输入序列中最相关的部分。"
            "具体来说，Transformer 架构中的自注意力机制通过 Query、Key、"
            "Value 三个矩阵来计算注意力权重。这种机制的优势在于能够捕捉"
            "长距离依赖关系，同时支持并行计算。"
        ),
        label="qwen",
        category="qa",
    ),
    BenchmarkSample(
        text=(
            "以下是一个高效的解决方案：\n\n"
            "```python\nclass LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n"
            "            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)\n```\n\n"
            "这个实现使用 OrderedDict 来维护访问顺序，"
            "get 和 put 操作均为 O(1) 时间复杂度。"
        ),
        label="qwen",
        category="code",
    ),
    # DeepSeek 风格
    BenchmarkSample(
        text=(
            "嗯，让我仔细想想这个问题。这是一个关于动态规划的经典问题。"
            "我们可以定义状态 dp[i] 表示到达第 i 个位置的最优解。"
            "状态转移方程为 dp[i] = max(dp[j] + value[i]) 其中 j < i "
            "且满足约束条件。边界条件是 dp[0] = value[0]。"
            "时间复杂度 O(n²)，空间复杂度 O(n)。"
            "如果需要优化，可以考虑单调栈或线段树来降低复杂度。"
        ),
        label="deepseek",
        category="reasoning",
    ),
    BenchmarkSample(
        text=(
            "这个问题可以从多个角度来分析。首先从技术层面看，"
            "大语言模型的训练过程本质上是在海量文本数据上进行"
            "概率分布的学习。模型通过最小化交叉熵损失来优化参数，"
            "使得生成的文本概率分布尽可能接近训练数据的分布。"
            "从实际应用角度来看，这意味着模型会倾向于生成"
            "训练数据中常见的模式和表达方式。"
        ),
        label="deepseek",
        category="qa",
    ),
]


def get_benchmark_samples(
    category: str | None = None,
    label: str | None = None,
) -> list[BenchmarkSample]:
    """获取 benchmark 样本.

    Args:
        category: 按类别过滤 (qa/creative/code/reasoning)
        label: 按模型家族过滤 (gpt-4/claude/llama/gemini/qwen/deepseek)
    """
    samples = BENCHMARK_SAMPLES
    if category:
        samples = [s for s in samples if s.category == category]
    if label:
        samples = [s for s in samples if s.label == label]
    return samples


def evaluate_accuracy(
    predictions: list[tuple[str, str]],
) -> dict:
    """评估检测准确率.

    Args:
        predictions: [(predicted_label, true_label), ...] 列表

    Returns:
        包含 accuracy, per_class_accuracy, confusion 的字典
    """
    if not predictions:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "per_class": {}}

    total = len(predictions)
    correct = sum(1 for pred, true in predictions if pred == true)

    # 按类别统计
    per_class: dict[str, dict[str, int]] = {}
    for pred, true in predictions:
        if true not in per_class:
            per_class[true] = {"total": 0, "correct": 0}
        per_class[true]["total"] += 1
        if pred == true:
            per_class[true]["correct"] += 1

    per_class_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in per_class.items()
    }

    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "per_class": per_class_acc,
    }
