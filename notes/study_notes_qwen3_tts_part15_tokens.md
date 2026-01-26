# Qwen3-TTS 深度笔记 (Part 15)：吐出来的到底是什么？

> **Q: Qwen3-TTS 的 LLM 吐出的是离散的 Codec Tokens 还是 embedding?**

**结论：是离散的 Codec Tokens (ID)。**

我们刚去“解剖室”看了源码 (`modeling_qwen3_tts.py`)，证据确凿。

---

## 1. 证据 (The Evidence)

在代码的第 1579 行，有一个关键的层：
```python
self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

以及在推理循环里：
*   **输入**: Embedding Vector (2048维)。
*   **处理**: 经过 28 层 Transformer。
*   **输出**: 依然是 Vector (2048维)。
*   **最后一步**: **`codec_head` (Linear Projection)**。
    *   把 2048 维向量 -> 投影成 **3072 维** (Logits)。
    *   这 3072 代表了每一个可能的 Token ID 的“概率”。

## 2. 采样 (Sampling)

拿到这 3072 个概率值后，我们会做一个**采样动作** (argmax 或 multiv-nomial sampling)。
*   比如：第 204 号位置的概率最大。
*   **结果**: 输出整数 `204`。
*   这就是 **Codec Token (Discrete ID)**。

---

## 3. 为什么要变回 ID？(The Cycle)

这就回到了我们的“循环”：
1.  **Input**: 拿 ID 查表变成 Vector。
2.  **Process**: 在 Vector 空间里计算。
3.  **Output**: 变回 ID。
4.  **Next Input**: 拿着这个新生成的 ID，**再查表变成 Vector**，喂给下一步。

> **Q: 如果直接吐出 embedding 不行吗？**
*   **不行**。因为 Embedding 空间是连续的，模型直接吐出的 float 向量往往带有噪音，或者是一个“四不像”的向量（既不像 Code 1 也不像 Code 2）。
*   **量子化 (Quantization)** 的本质就是：强制把混乱的中间状态，坍缩成一个确定的、干净的整数 ID。这能有效**防止误差积累**。

**总结**：
LLM 从数学上吐出的是概率分布 (Logits)，经过采样后变成离散整数 (ID)。
正是这个“变回整数”的过程，让生成的音频可以无限长而不糊掉。
