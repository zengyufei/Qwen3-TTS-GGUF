# Qwen3-TTS 深度笔记 (Part 20)：GGUF 里的 Text Embedding 放哪？

> **Q: 那 gguf 里用不用存 text_embedding 呢？llama.cpp 有没有要求？**

**答案：必须存，而且它就是 GGUF 的“门面”。**

## 1. llama.cpp 的硬性要求
GGUF 格式规范里，`token_embd.weight` 是核心组件。
*   如果你不存它，`llama.cpp` 加载时会报错（找不到词表或维度不匹配）。
*   它定义了模型的 `vocab_size` 和 `hidden_size`。

## 2. 我们的策略：存 GGUF，借出来用
我们要完全照抄 `fun_asr_gguf` 的作业：

1.  **打包时 (Export)**：
    *   将 `text_embedding` (151936张表) 作为标准的 `token_embd.weight` 写入 GGUF。
    *   (可选) 将剩下的 16 张 Codec 表也作为自定义 Tensor 写入 GGUF，或者单独存 `.npy`。

2.  **推理时 (Inference)**：
    *   **步骤 A (加载)**：`llama_model_load_from_file` 会把 GGUF 加载进内存/显存。
    *   **步骤 B (借用)**：我们用 `gguf` 库的 API (或者模仿 `fun_asr_gguf` 手动解析) 把 `token_embd.weight` 的**数值**读出来，变成一个 Numpy Array。
    *   **步骤 C (查表)**：在 Python 里，用这个 Numpy Array 把 `Input IDs` 变成 `Vectors`。
    *   **步骤 D (喂食)**：把计算好的 `Vectors` (包含 Audio Sum) 喂给 `llama_decode`。

## 3. 为什么不删掉省空间？
虽然我们在步骤 D 喂的是 Vectors（绕过了模型内部的 Embedding 层），但：
1.  **完整性**：没有 Embedding 的模型是不完整的，无法通过标准工具（如 `llama-quantize`）进行处理。
2.  **方便性**：直接把 text embedding 打包在 GGUF 里，我们就不用单独维护一个 `text_embd.npy` 文件了。GGUF 变成了一个“自带数据库”的单一文件。

**结论**：
**GGUF 里必须有 `token_embd.weight`。**
我们会在导出脚本中，把它作为标准权重写入。
