# Qwen3-TTS 深度笔记 (Part 16)：数数有几张表？

> **Q: 每一个 codec 是一个表吗？总共是16个表吗？**

**答案：是的，总共 16 张表。**

我们刚刚在代码 (`modeling_qwen3_tts.py`) 第 1983-1989 行亲眼见证了这个过程：

```python
codec_embed = []
for i in range(self.talker.config.num_code_groups): # 这里 num_code_groups = 16
    if i == 0:
        # 第 0 张表属于 Talker 自己
        codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
    else:
        # 第 1-15 张表属于 Code Predictor (工匠)
        # 注意这里是 [i-1]，因为工匠存的是剩下的 15 张表
        codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i-1](ref_code[:, i:i+1]))

# 最后求和
codec_embed = torch.cat(codec_embed, dim=1).sum(1)
```

## 物理分布
这 16 张表并不是放在一起的，而是“分家”了：

1.  **Table 0 (Code 0)**:
    *   **位置**: `talker.model.codec_embedding`
    *   **大小**: `[3072, 2048]` (包含 Code + 特殊 Token)
    *   **主人**: 大师 (Talker)

2.  **Table 1 - 15 (Code 1..15)**:
    *   **位置**: `talker.code_predictor.model.codec_embedding` (这是一个 `nn.ModuleList`)
    *   **大小**: 15 个 `[2048, 2048]`
    *   **主人**: 工匠 (Code Predictor)

## 为什么这么设计？
因为“工匠”在干活（补全 Code 1-15）的时候，它需要自己独立的 Embedding 来理解它正在补全的每一层含义。
而“大师”在做全局把控的时候，它通过求和，把这 16 层含义都“借”了过来。

**结论**：
我们在导出 Vector 的时候，必须去这两个不同的地方（Talker 和 Code Predictor）把这 16 张表全部抓出来。少一张都不行。
