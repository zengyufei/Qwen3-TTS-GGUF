# 修改内容

增加脚本、增加http服务、优化听书接口复用输出流、增加m4a格式音色兼容、增加m4a格式音色导出为json。

## 1、提供bat脚本快速安装、启动

### 一键http.bat

启动后访问 http://127.0.0.1:8210/

提供网页版测试

### 一键stream http.bat

专供开源阅读app听书接口 

http://192.168.1.222:8210/api/stream,
{
    "method": "POST",
    "timeout": "120000",
    "body": {
        "text": "{{ speakText }}",
        "seed": 1.0
    }
}
ContextType: audio/wav
请求头 Header: application/json

## 2、模型

modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base

## 3、安装

    1、 一键创建虚拟环境并安装依赖.bat
    2、模型下载好后，执行以下命令
    
```shell
阶段一：导出小组件
python 11-Export-Codec-Encoder.py    # 编码器，用于克隆
python 12-Export-Speaker-Encoder.py  # 说话人特征提取器
python 13-Export-Decoder.py          # 解码器，核心渲染器
python 14-Export-Embeddings.py       # Embedding 权重
python 15-Copy-Tokenizer.py          # 文本分词器
python 16-Quantize-ONNX-Models.py    # 重要：将 ONNX 转为 FP16 以供 DML 加速
阶段二：导出大师（Talker）
大师是 1.42B 的 LLM backbone，负责理解文本、生成语音骨架：

python 21-Extract-Talker-Weights.py    # 拆分并初始化权重
python 22-Prepare-Talker-Tokenizer.py  # 构造 GGUF 所需的迷你词表
python 23-Convert-Talker-GGUF.py       # 转换为 F16 的 GGUF
python 24-Quantize-Talker-GGUF.py      # 量化为 q5_k，这是推理引擎默认加载的版本
阶段三：导出工匠（Predictor）
工匠是 142M 的小模型，负责给骨架补充细节：

python 31-Extract-Predictor-Weights.py
python 32-Prepare-Predictor-Tokenizer.py
python 33-Convert-Predictor-GGUF.py
python 34-Quantize-Predictor-GGUF.py    # 量化为 q8_0，这是推理引擎默认加载的版本
```
    
    3、全部导出完毕，执行 一键test.bat
    
    4、不用下载  llama.cpp，我已经集成了
    

## 4、性能

### RTF: 0.23~0.26 左右

#### 以下是运行日志示例：

```
INFO:     Started server process [24612]
INFO:     Waiting for application startup.
📦 [Engine] 资产与词表加载完成 (耗时: 0.44s)
🎤 [Engine] 编码器加载完成 (耗时: 0.60s)
⏳ [Engine] 正在拉起子进程解码器...
🧠 [Engine] GGUF 推理后端就绪 (耗时: 1.24s)
🔊 [DecoderWorker] 已就绪 (Provider: CUDAExecutionProvider)
✅ [Engine] 解码器就绪: Decoder True | Speaker True (总并行初始化耗时: 5.00s)
🚀 [Engine] 引擎全链路初始化完成! 总耗时: 6.04s
[web] startup workers=1 startup_warmup=True default_voice=Vivian output_gain=1.5 chunk_size=24 poll_s=0.005 ort_intra=0 ort_inter=0
[web] default voice ready name=Vivian
[web] startup warmup start text='hi' voice=Vivian
----------------------------------------
性能分析报告 (音频长度: 0.72s)
  1. Prompt:    0.17s
  2. Prefill:   0.11s
  3. Generate:  0.41s (Talker: 0.06s, Predictor: 0.35s)
  4. Decode:    0.11s
  5. Latency:   0.63s (Generate: 0.52s, Decode: 0.11s)
----------------------------------------
核心总耗时: 0.69s | RTF (Core): 0.95
[web] startup warmup done total_ms=820.9
[web] worker bound streams ready count=1 voice=Vivian
[web] worker ready index=0
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8210 (Press CTRL+C to quit)


[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000目送着刘令萍和于文走远，张肃脸上的坏笑逐渐收敛，拿出对讲机联络吴大强，让他监督好哨岗的执勤人员，这件事万万不能马虎，必须严肃对待，亲自突击检查也是有必要的！', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='目送着刘令萍和于文走远，张肃脸上的坏笑逐渐收敛，拿出对讲机联络吴大强，让他监督好哨岗的执勤人员，这件事万万不能马虎，必须严肃对待，亲自突击检查也是有必要的！' queue_size=1 total_ms=0.1
[web] task start worker=0 voice=Vivian wait_ms=0.1
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.1
[web] clone start worker=0 task_id=task_0 sample_rate=24000
INFO:     127.0.0.1:64618 - "POST /api/tts HTTP/1.1" 200 OK
[web] clone finished(non-stream) elapsed_ms=4564.5
----------------------------------------
性能分析报告 (音频长度: 15.84s)
  1. Prompt:    0.02s
  2. Prefill:   0.14s
  3. Generate:  4.03s (Talker: 0.54s, Predictor: 3.49s)
  4. Decode:    0.30s
  5. Latency:   4.49s (Generate: 4.19s, Decode: 0.30s)
----------------------------------------
核心总耗时: 4.19s | RTF (Core): 0.26
[web] task finished worker=0 task_id=task_0 run_ms=4568.4 total_ms=4568.6
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000安排完哨岗的事宜后，张肃朝小幸运走去，心中琢磨着军团未来发展的一些规划。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='安排完哨岗的事宜后，张肃朝小幸运走去，心中琢磨着军团未来发展的一些规划。' queue_size=1 total_ms=0.0
[web] task start worker=0 voice=Vivian wait_ms=0.1
INFO:     127.0.0.1:64618 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.2
[web] clone start worker=0 task_id=task_1 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=1744.0
----------------------------------------
性能分析报告 (音频长度: 7.12s)
  1. Prompt:    0.01s
  2. Prefill:   0.01s
  3. Generate:  1.61s (Talker: 0.25s, Predictor: 1.36s)
  4. Decode:    0.10s
  5. Latency:   1.72s (Generate: 1.62s, Decode: 0.10s)
----------------------------------------
核心总耗时: 1.62s | RTF (Core): 0.23
[web] task finished worker=0 task_id=task_1 run_ms=1746.2 total_ms=1746.3
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000从目前的情况来看，需要尽快让更多人接触到橘氏武道的训练……', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='从目前的情况来看，需要尽快让更多人接触到橘氏武道的训练……' queue_size=1 total_ms=0.0
[web] task start worker=0 voice=Vivian wait_ms=0.1
INFO:     127.0.0.1:64618 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.5
[web] clone start worker=0 task_id=task_2 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=1367.9
----------------------------------------
性能分析报告 (音频长度: 5.36s)
  1. Prompt:    0.00s
  2. Prefill:   0.01s
  3. Generate:  1.20s (Talker: 0.18s, Predictor: 1.02s)
  4. Decode:    0.14s
  5. Latency:   1.35s (Generate: 1.21s, Decode: 0.14s)
----------------------------------------
核心总耗时: 1.21s | RTF (Core): 0.23
[web] task finished worker=0 task_id=task_2 run_ms=1369.8 total_ms=1369.9
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000当初没有往精英军团推广的原因是考虑到大部分人的身体素质不够，不仅无法得到加强，还有受伤的风险。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='当初没有往精英军团推广的原因是考虑到大部分人的身体素质不够，不仅无法得到加强，还有受伤的风险。' queue_size=1 total_ms=0.1
[web] task start worker=0 voice=Vivian wait_ms=0.3
INFO:     127.0.0.1:64618 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.4
[web] clone start worker=0 task_id=task_3 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=2008.3
----------------------------------------
性能分析报告 (音频长度: 8.16s)
  1. Prompt:    0.01s
  2. Prefill:   0.01s
  3. Generate:  1.84s (Talker: 0.28s, Predictor: 1.56s)
  4. Decode:    0.13s
  5. Latency:   1.99s (Generate: 1.85s, Decode: 0.13s)
----------------------------------------
核心总耗时: 1.85s | RTF (Core): 0.23
[web] task finished worker=0 task_id=task_3 run_ms=2013.2 total_ms=2013.5
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000今非昔比，所有人的身体都在悄无声息之中得到强化……哪怕是不锻炼也在变强，相信现如今大部分人都能够适应那种强度的训练。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='今非昔比，所有人的身体都在悄无声息之中得到强化……哪怕是不锻炼也在变强，相信现如今大部分人都能够适应那种强度的训练。' queue_size=1 total_ms=0.1
INFO:     127.0.0.1:64618 - "POST /api/tts HTTP/1.1" 200 OK
[web] task start worker=0 voice=Vivian wait_ms=0.2
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.6
[web] clone start worker=0 task_id=task_4 sample_rate=24000
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000当初段五湖一行人带回来的洪老四，他或许称不上武学大师……但棍法确实有些名堂，四两拨千斤的巧妙如果融入到战斗当中，可以起到奇效。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='当初段五湖一行人带回来的洪老四，他或许称不上武学大师……但棍法确实有些名堂，四两拨千斤的巧妙如果融入到战斗当中，可以起到奇效。' queue_size=1 total_ms=0.0
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] clone finished(non-stream) elapsed_ms=3120.9
----------------------------------------
性能分析报告 (音频长度: 12.72s)
  1. Prompt:    0.01s
  2. Prefill:   0.01s
  3. Generate:  2.83s (Talker: 0.44s, Predictor: 2.39s)
  4. Decode:    0.25s
  5. Latency:   3.09s (Generate: 2.84s, Decode: 0.25s)
----------------------------------------
核心总耗时: 2.84s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_4 run_ms=3134.0 total_ms=3134.2
[web] task start worker=0 voice=Vivian wait_ms=1015.7
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=1.3
[web] clone start worker=0 task_id=task_5 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=3483.4
----------------------------------------
性能分析报告 (音频长度: 13.84s)
  1. Prompt:    0.01s
  2. Prefill:   0.01s
  3. Generate:  3.20s (Talker: 0.48s, Predictor: 2.72s)
  4. Decode:    0.23s
  5. Latency:   3.45s (Generate: 3.23s, Decode: 0.23s)
----------------------------------------
核心总耗时: 3.22s | RTF (Core): 0.23
[web] task finished worker=0 task_id=task_5 run_ms=3489.1 total_ms=4504.9
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000当两个人的身体素质相仿，那么比拼的就是意识和技巧！', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='当两个人的身体素质相仿，那么比拼的就是意识和技巧！' queue_size=1 total_ms=0.0
[web] task start worker=0 voice=Vivian wait_ms=0.1
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.2
[web] clone start worker=0 task_id=task_6 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=1157.5
----------------------------------------
性能分析报告 (音频长度: 4.64s)
  1. Prompt:    0.00s
  2. Prefill:   0.01s
  3. Generate:  1.01s (Talker: 0.16s, Predictor: 0.85s)
  4. Decode:    0.11s
  5. Latency:   1.14s (Generate: 1.02s, Decode: 0.11s)
----------------------------------------
核心总耗时: 1.03s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_6 run_ms=1160.3 total_ms=1160.3
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000之前每周一次的课程，教授众人练习棍法，并没有太好的效果，如今看来有必要增加课时！', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='之前每周一次的课程，教授众人练习棍法，并没有太好的效果，如今看来有必要增加课时！' queue_size=1 total_ms=0.1
[web] task start worker=0 voice=Vivian wait_ms=0.1
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.3
[web] clone start worker=0 task_id=task_7 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=1983.2
----------------------------------------
性能分析报告 (音频长度: 8.40s)
  1. Prompt:    0.00s
  2. Prefill:   0.01s
  3. Generate:  1.82s (Talker: 0.27s, Predictor: 1.55s)
  4. Decode:    0.12s
  5. Latency:   1.96s (Generate: 1.84s, Decode: 0.12s)
----------------------------------------
核心总耗时: 1.84s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_7 run_ms=1985.7 total_ms=1985.8
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000只是如果让洪老四专注于棍法的传授，他自己就没时间训练，多少有点不公平。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='只是如果让洪老四专注于棍法的传授，他自己就没时间训练，多少有点不公平。' queue_size=1 total_ms=0.2
[web] task start worker=0 voice=Vivian wait_ms=0.3
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.1
[web] clone start worker=0 task_id=task_8 sample_rate=24000
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] clone finished(non-stream) elapsed_ms=1834.2
----------------------------------------
性能分析报告 (音频长度: 7.44s)
  1. Prompt:    0.00s
  2. Prefill:   0.01s
  3. Generate:  1.62s (Talker: 0.25s, Predictor: 1.37s)
  4. Decode:    0.19s
  5. Latency:   1.81s (Generate: 1.63s, Decode: 0.19s)
----------------------------------------
核心总耗时: 1.63s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_8 run_ms=1836.3 total_ms=1836.7
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000“反正我学得快，干脆我先找他学明白，阎罗军团就由我来教，让他专注教授精英和预备两个军团，然后上课的同时也跟着练……”', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='“反正我学得快，干脆我先找他学明白，阎罗军团就由我来教，让他专注教授精英和预备两个军团，然后上课的同时也跟着练……”' queue_size=1 total_ms=0.1
[web] task start worker=0 voice=Vivian wait_ms=0.2
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.2
[web] clone start worker=0 task_id=task_9 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=2735.0
----------------------------------------
性能分析报告 (音频长度: 11.52s)
  1. Prompt:    0.01s
  2. Prefill:   0.01s
  3. Generate:  2.52s (Talker: 0.40s, Predictor: 2.12s)
  4. Decode:    0.17s
  5. Latency:   2.70s (Generate: 2.54s, Decode: 0.17s)
----------------------------------------
核心总耗时: 2.53s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_9 run_ms=2741.9 total_ms=2742.1
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000思忖着，张肃回到住所。', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='思忖着，张肃回到住所。' queue_size=1 total_ms=0.0
[web] task start worker=0 voice=Vivian wait_ms=0.1
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] worker stream reuse index=0 voice=Vivian
[web] request stream ready elapsed_ms=0.2
[web] clone start worker=0 task_id=task_10 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=750.6
----------------------------------------
性能分析报告 (音频长度: 2.88s)
  1. Prompt:    0.00s
  2. Prefill:   0.01s
  3. Generate:  0.63s (Talker: 0.10s, Predictor: 0.54s)
  4. Decode:    0.09s
  5. Latency:   0.73s (Generate: 0.65s, Decode: 0.09s)
----------------------------------------
核心总耗时: 0.64s | RTF (Core): 0.22
[web] task finished worker=0 task_id=task_10 run_ms=752.6 total_ms=752.7
[web] /api/tts request method=POST content_type='application/json' query={} form={} json={'text': '\u3000\u3000“回来啦？喏，桌上，你的礼物。”', 'token': 'mytoken', 'seed': 1} fallback_form={}
[web] enqueue voice=Vivian text='“回来啦？喏，桌上，你的礼物。”' queue_size=1 total_ms=0.1
[web] task start worker=0 voice=Vivian wait_ms=0.1
[web] worker stream reuse index=0 voice=Vivian
INFO:     127.0.0.1:60205 - "POST /api/tts HTTP/1.1" 200 OK
[web] request stream ready elapsed_ms=0.1
[web] clone start worker=0 task_id=task_11 sample_rate=24000
[web] clone finished(non-stream) elapsed_ms=1067.1
----------------------------------------
性能分析报告 (音频长度: 3.84s)
  1. Prompt:    0.00s
  2. Prefill:   0.02s
  3. Generate:  0.97s (Talker: 0.16s, Predictor: 0.80s)
  4. Decode:    0.05s
  5. Latency:   1.05s (Generate: 0.99s, Decode: 0.05s)
----------------------------------------
核心总耗时: 0.99s | RTF (Core): 0.26
[web] task finished worker=0 task_id=task_11 run_ms=1069.5 total_ms=1069.6
```


---

以下是源项目内容

# Qwen3-TTS GGUF

用 llama.cpp 跑的 Qwen3-TTS，支持流式合成、声音克隆。

## 模型类型

本项目支持三种官方模型，对应三种场景：

| 源模型 | 场景 |
| :--- | :--- |
| Qwen3-TTS-12Hz-1.7B-Base | 声音克隆 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 内置音色 + 风格指令 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 用自然语言设计音色 |
| Qwen3-TTS-12Hz-0.6B-Base | 声音克隆 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 内置音色 + 风格指令 |

你想用哪个，就导出哪个，但是需要先下载官方模型才能导出。

## 性能表现

在我的 RTX 5050 上的实测数据：

- **RTX 5050 (独显)**: RTF 0.35 (实时率，1秒音频只需0.35秒生成)
- **CPU**: RTF 1.3
- **集显**: RTF 1.3

RTF < 1 就表示生成速度比实时播放还快。没有独显的电脑，很难做到 RTF < 1，因此不太可能流畅地流式播放。

显存占用：

- **Encoder** 用于从音频提取特征克隆，无需显卡加速，可节省显存
- **Talker 1.7B**，用 Q5_k 量化，载入 955MB，上下文224MB，计算50MB，共1229MB
- **Predictor 0.1B**，用 Q8_0 量化，载入 144MB，上下文5MB，计算7MB，共156MB
- **Decoder** 用 fp16 量化，DML 加速，模型 237MB，推理204MB

使用 1.7B 总共需 1.8G 显存。

用 0.6B 版可以再省 500MB 显存，但对速度的提升不大，因为**计算瓶径在于 Predictor**，每一秒的音频需要自回归 12.5*15=187.5 次，0.6B 和 1.7B 的差异仅在于 Talker。


## 项目特性

- **流式合成**：大幅缩短流式实际首音延迟，最低可至 300ms 内。
- **加速推理**：对 1.7B 模型，RTX5050 可以做到 RTF0.35，AMD 显卡也可以用 Vulkan 加速
- **确定性控制**：支持独立设置 Talker 和 Predictor 的随机种子，确保输出可复现。

## Clone 原理

声音克隆的本质是 **接续说话** (In-Context Learning)。

想象一下，你正在读一段话，读到一半时，我让你接着往下读，你自然会用同样的语气和声音。Qwen3-TTS 的克隆原理也是如此：

1. **文本拼接**：把「参考文本」和「目标文本」连接起来。
2. **注入记忆**：把「参考音频」转为 spk_emb 和 codes，注入记忆，让模型以为前面的音频是它自己刚说的：
   - **嗓子（spk_emb）**：整体音色特征，告诉模型我是用什么样的嗓子在说话。
   - **音节（codes）**：具体的发音码（12.5Hz），让模型认为「我刚才确实亲口说出了这些音节」。
3. **顺势说完**：有了之前的「嗓子」和音节记忆，模型就会顺理成章地继续保持这个声音，把剩下的文字接着读完。

## Custom Voice 原理

与克隆类似，只是模型内置了一些说话人音色（spk_embd），模型的记忆中只会有嗓子，没有音节。它要根据指令和目标文本，让自己处于某种情感状态（入戏），然后用嗓子音色把目标文本读出来。

有点像是读台词：

- Clone是读了一半（参考文本），然后接着读（目标文本）。
- Custom Voice 是还没有读，酝酿一下情感，从头开始读。

因此带来了不同的特点：

- Clone 因为已经读了一半，情感基调已定，音色就更稳定可控。
- Custom Voice 因为要酝酿情感，不同的随机种子，会酝酿出不同的基调，就会有些抽卡。

因此最佳的配音实践是：用 Custom Voice 对同一段文字抽卡出最好的效果，再用 Base 模型基于这个音频克隆阅读其它文本。

## 快速开始

#### 下载模型

- [Qwen3-TTS-12Hz-1.7B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

- [Qwen3-TTS-12Hz-0.6B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

```
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

#### 依赖环境

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制，将 DLL 放入 `qwen_asr_gguf/bin/`：

| 平台 | 下载文件 |
|------|----------|
| **Windows** | `llama-bXXXX-bin-win-vulkan-x64.zip` |

另外还需安装 FFmpeg，用于读取音频文件。

``` 
pip install -r requirements.txt
```

#### 配置路径

打开 `export_config.py`，在里面配置好模型的导出路径：


#### 阶段一：导出小组件

```bash
python 11-Export-Codec-Encoder.py    # 编码器，用于克隆
python 12-Export-Speaker-Encoder.py  # 说话人特征提取器
python 13-Export-Decoder.py          # 解码器，核心渲染器
python 14-Export-Embeddings.py       # Embedding 权重
python 15-Copy-Tokenizer.py          # 文本分词器
python 16-Quantize-ONNX-Models.py    # 重要：将 ONNX 转为 FP16 以供 DML 加速
```

#### 阶段二：导出大师（Talker）

大师是 1.42B 的 LLM backbone，负责理解文本、生成语音骨架：

```bash
python 21-Extract-Talker-Weights.py    # 拆分并初始化权重
python 22-Prepare-Talker-Tokenizer.py  # 构造 GGUF 所需的迷你词表
python 23-Convert-Talker-GGUF.py       # 转换为 F16 的 GGUF
python 24-Quantize-Talker-GGUF.py      # 量化为 q5_k，这是推理引擎默认加载的版本
```

#### 阶段三：导出工匠（Predictor）

工匠是 142M 的小模型，负责给骨架补充细节：

```bash
python 31-Extract-Predictor-Weights.py
python 32-Prepare-Predictor-Tokenizer.py
python 33-Convert-Predictor-GGUF.py
python 34-Quantize-Predictor-GGUF.py    # 量化为 q8_0，这是推理引擎默认加载的版本
```

导完之后，`EXPORT_DIR` 里就有你需要的所有文件了。

## 推理

### 脚本模式

三个示例脚本，对应三种模型：

```bash
python 41-Inference-Custom.py  # 精品音色
python 42-Inference-Design.py  # 音色设计
python 43-Inference-Base.py    # 声音克隆
```

### 交互模式（推荐）

```bash
python 51-Interactive-Clone.py
```

启动后直接打字，边推边播。

## 代码调用

```python
from qwen3_tts_gguf import TTSEngine, TTSConfig

# 初始化引擎（后台自动并行加载模型）
engine = TTSEngine(model_dir="model-base")
stream = engine.create_stream()

# 设置音色（支持 .wav 路径、.json 路径或 TTSResult 对象）
stream.set_voice("output/elaborate/sample.json")

# 配置推理参数
config = TTSConfig(
    temperature=0.8,      # 核心温度，控制随机性
    sub_temperature=0.8,  # 细节温度，控制随机性
    seed=42,           # 核心种子
    sub_seed=45,       # 细节种子
    streaming=True,    # 开启流式
)

# 流式合成
result = stream.clone("你好，世界！", config=config)
stream.join()  # 等待播完

# 保存结果
result.save("output/output.wav")
result.save("output/output.json")  # 保存 codes，下次可无损加载
```

## 可用说话人

CustomVoice 模型内置 9 个音色：

| ID | 说明 |
| :--- | :--- |
| vivian | 年轻女声，明亮利落 |
| serena | 温暖女声，柔和亲切 |
| uncle_fu | 成熟男声，沉稳低沉 |
| dylan | 北京男声，自然清晰 |
| eric | 成都男声，略带沙哑 |
| ryan | 活力男声，节奏感强 |
| aiden | 阳光美男，中频清澈 |
| ono_anna | 日语女声，俏皮轻快 |
| sohee | 韩语女声，温润动情 |

## 支持的语言

- chinese, english, japanese, korean
- german, spanish, french, russian, italian, portuguese
- beijing_dialect, sichuan_dialect

## 架构简述

用人体器官来理解：

1. **耳朵**（Encoder）: 听参考音频，提取音色特征
2. **大脑**（Talker）: 生成语音骨架 (28层, 1.42B)
3. **双手**（Predictor）: 补充细节 (8层, 142M)
4. **嘴巴**（Decoder）: 把 codes 解码成声音

推理引擎：
- Talker / Predictor: llama.cpp (GGUF格式，Vulkan/Cuda 加速)
- Encoder / Decoder: ONNX Runtime (ONNX格式，DirectML/Cuda 加速)

## 常见问题

**Q: 为什么不用官方 PyTorch？**

官方实现要大显存。llama.cpp 省资源，还能用 Vulkan/DML 加速。

**Q: 流式和离线有什么区别？**

流式边推边播，首包延迟低（~300ms）。

**Q: 怎么调质量？**

`TTSConfig(temperature=0.8, sub_temperature=0.8, seed=42, sub_seed=45)` 温度控制随机性，用种子控制稳定复现。

## 相关链接

- [Qwen3-TTS 技术报告](./Qwen3-TTS%20Technical%20Report.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
