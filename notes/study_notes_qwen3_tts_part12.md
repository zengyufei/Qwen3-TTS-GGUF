# Qwen3-TTS 深度笔记 (Part 12)：修正我们的认知——必须交替！

> **Q: 那就是说，这里其实是不能够让 talker 一次性画一串骨架的？**

**回答：你是对的！我之前的“比喻”为了简化，在这一点上是不严谨的。**

在**推理（Inference）**阶段，Talker 绝对**不能**一次性画完一串骨架。
必须是：**画一根骨头 -> 填肉 -> 再画下一根骨头**。

---

## 1. 为什么不能“先画骨架”？

因为 Talker 是**自回归**的模型。它决定“下一根骨头是什么”的依据，不仅仅是“上一根骨头”，而是 **“上一帧完整的声音（骨头+肉）”**。

如果 Talker 一口气画了 100 根骨头（Code 0），但没有填肉（Code 1-15），那么：
*   在生成第 2 根骨头时，它看到的输入是 `Code0_1 + 0 (Empty)`。
*   这会导致严重的问题，因为模型训练时看到的是完整的 `Code0 + Code1..15`。
*   **缺少了肉的输入，会误导 Talker，导致后面的骨头画歪。**

---

## 2. 训练 vs 推理的区别

为什么我会产生“并行”的错觉？因为在**训练（Training）**时，它是并行的！
*   **训练时**：我们已经有了正确的答案（Ground Truth）。我们把完整的骨架和肉都摆在那里。
    *   Talker 根据 t-1 的完整信息预测 t 的 Code 0。
    *   Code Predictor 根据已知的信息预测 Code 1-15。
    *   这一切可以在 GPU 上并行计算。

*   **推理时**：我们没有答案。必选走 **Interleaved (交替)** 模式。
    *   `Code0` (Talker) -> `Code1..15` (Predictor) -> `Sum` -> `Next Code0` (Talker)...

---

## 3. 最终的脚本逻辑

所以，我们的 Python 脚本结构必须是这样的（伪代码）：

```python
input_history = [Ref_Audio, Text_Prompt]

while not_finished:
    # 1. 大师回合
    # 输入必须包含完整的历史（骨头+肉）
    next_code_0 = Talker.forward(input_history)
    
    # 2. 工匠回合
    # 拿到刚才这根骨头，把肉补齐
    next_code_1_to_15 = Predictor.forward(talker_hidden, next_code_0)
    
    # 3. 合并回合
    # 把这一帧完整的做出来
    full_frame = Sum(next_code_0, next_code_1_to_15)
    
    # 4. 更新历史
    # 只有有了完整的 full_frame，大师才能准备好迈出下一步
    input_history.append(full_frame) 
```

你现在的理解已经修正到了**源码级**的准确度。
这个交替循环是不可避免的性能瓶颈，但也正是高质量生成的保证。
