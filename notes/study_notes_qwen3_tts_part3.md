# Qwen3-TTS 深度笔记 (Part 3)：音频和文本到底是怎么“排排坐”的？

这是一个非常关键的问题！如果位置搞错了，就会出现你担心的“把音频加到 `<|im_start|>` 上”的乌龙。我们来看看真实的输入队列长什么样。

---

## 1. 真实的输入长龙：并不是简单的“加在一起”

你的直觉是对的：**音频不是无脑叠加在 System Prompt 上的。**

实际上，输入序列分成了 **三个阶段**：

### 阶段一：纯文本阶段 (Preface / Instruction)
这一段是纯文本，音频部分是空的（或者是 Padding）。
*   **内容**：`<|im_start|>system...<|im_start|>user...语音转写：`
*   **Audio Embedding**：全是 **0**（或者 Pad Embedding）。
*   **计算**：`Text_Embed + 0`。
*   **目的**：让 LLM 理解任务指令。

### 阶段二：混合阶段 (The Mixing Zone) —— 这里才是戏肉！
当指令结束，真正开始处理**要读的文本**和**生成的音频**时，真正的叠加才开始。
但这里有个**反直觉**的设计：在 TTS 任务中，我们其实是在做 **Causal Language Modeling (接龙)**。

假设我们要让它读“你好”。
序列结构是这样的：

| 位置 | 文本 Token | 音频 Code 0 | 实际输入给 LLM 的 |
| :--- | :--- | :--- | :--- |
| T=100 | `<|audio_start|>` | (空/Pad) | `<|audio_start|>` 的 Embedding |
| T=101 | `你` | **Code0_Frame1** | `Embed(你) + Embed(Code0_Frame1)` |
| T=102 | `好` | **Code0_Frame2** | `Embed(好) + Embed(Code0_Frame2)` |
| T=103 | `<tts_pad>` | **Code0_Frame3** | `Embed(Pad) + Embed(Code0_Frame3)` |
| ... | ... | ... | ... |

**回答你的问题：**
> **Q: 第一帧音频的 embedding 不就会加到 <|im_start|> 所代表的 embedding 上面了吗？**
*   **不会！** `<|im_start|>` 等系统指令在 **阶段一** 就已经跑完了。
*   音频帧是**从阶段二开始**，对齐到**真正要朗读的文本内容**上的。

> **Q: 也就是说，一个文字 token 的维度是1024... 两秒的音频对应25个token...**
*   这里有个**长度不匹配**的问题，对吧？“你好”只有 2 个字，但音频可能有 25 帧。
*   **对齐策略**：看上面的表格。
    *   前 2 帧：`你` 和 `好` 分别加到了第 1、2 帧音频上。
    *   第 3-25 帧：文本已经没了！这时候模型会用一个特殊的 `<tts_pad>` Token 来填补文本的位置。
    *   **所以**：后面的 23 帧，实际上是 `Embed(<tts_pad>) + Embed(Audio_Frame_i)`。

---

## 2. “尾巴”不是尾巴 (Tail vs. Parallel)

> **Q: 所以，音频不是 tail 在 prefix prompt 后面？**

这是最容易混淆的地方：
1.  **对于 Instruction (指令)**：是 Tail。
    *   先有 System Prompt，**后面** 才有 Audio/Text 混合区。
    *   `[System Prompt] -> [Audio Start] -> [混合区]`

2.  **对于 Content (正文)**：是 Parallel (并行叠加)。
    *   在混合区内部，正文文本（"你好"）和音频帧（Code 0, 1, 2...）是 **上下叠在一起** 喂进去的。

---

## 3. 为什么只有 Frame-by-Frame？

你之前问的“逐帧相加”，现在更清晰了吗？
它不是把一段话加到一帧里，而是：
*   **时刻 T**：Input = `Text[T] + Audio[T]`。
*   如果 `Text` 比 `Audio` 短（通常都是这样，因为几秒钟音频就有几十帧），`Text` 用完后就用 `Pad` 顶替。

**总结一下形状流：**
1.  **Input IDs (Text)**: `[System... User... "你", "好", Pad, Pad...]` (长度 L)
2.  **Input Codes (Audio)**: `[Pad... Pad... Code_1, Code_2, Code_3, Code_4...]` (长度 L)
    *   *注意：音频在 System Prompt 期间也是 Pad。*
3.  **Embedding 后**: 两个都是 `[L, 2048]`。
4.  **相加**: `Final_Input = Embed(Text) + Embed(Audio)` -> `[L, 2048]`。

这下放心了吧？音频绝对不会污染你的 System Prompt！
