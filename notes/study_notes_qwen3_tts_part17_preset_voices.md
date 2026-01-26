# Qwen3-TTS 深度笔记 (Part 17)：预设音色藏在哪？

> **Q: 这个模型预存的音色是在哪里放着呢？**

**答案：藏在 `config.json` 和代码逻辑里，但它们其实是“特殊的 Prompt”。**

我们刚才看了一眼 `modeling_qwen3_tts.py` (Generate 函数附近)，发现了这个秘密。

## 1. 并不存在“音色库文件”

并没有一个单独的 `voices.db` 或者 `embeddings.npy` 装着所有预设音色。
相反，预设音色更像是一种 **“快捷方式 (Shortcut)”**。

## 2. 两个来源

### A. 映射表 (Spk ID Mapping)
在 `config.json` 里，我们看到类似这样的配置：
```json
"spk_id": {
    "eric": 2861,
    "sohee": 2862,
    ...
}
```
**机制**：
*   当你指定 `speaker="eric"` 时，代码会查到 ID `2861`。
*   这个 ID 可能会直接作为一个特殊的 **Token ID**，喂给 Talker。
*   Talker 内部的 Embedding 表里，第 2861 号位置，可能已经训练好了一个专门代表 Eric 音色的向量。
*   **这就好比**：我们不需要喂一段 Eric 的录音（Reference），我们直接给模型看一张写着“Eric”的名片，模型就懂了（因为它训练时见过 Eric）。

### B. 动态生成 (Generate Speaker Prompt)
在代码里还有一个 `generate_speaker_prompt` 函数。
对于某些复杂的预设（比如方言），它可能会：
1.  自动加载一段内置在模型权重（或者隐藏在某个角落）的 Reference Code。
2.  或者构造一个特殊的 Prompt 序列。

## 3. 对我们的影响

对于我们要做的**导出工作**来说，这其实是个好消息。

*   **如果它是 Token ID**：那它就包含在 `Talker Codec Embedding Table (Table 0)` 里了。只要我们把那张表完整导出（0-3072），Eric 的音色就在里面。
*   **我们只需要**：在推理代码里支持传入 `speaker_id`，然后把它转换成对应的 Token ID 即可。

**总结**：
预设音色不是外挂，它们是 Talker 记忆（Embedding Table）的一部分。只要把大师的脑子（权重）完整搬走，Eric、Sohee 这些老朋友就都一起搬走了。
