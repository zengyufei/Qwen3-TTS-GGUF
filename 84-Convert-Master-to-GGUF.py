"""
[83] Convert Master to GGUF with Monkey Patching
使用 Python Monkey Patch 技术绕过 convert_hf_to_gguf.py 的哈希校验。
强制将当前迷你 Tokenizer 识别为 "qwen2" 类型。
"""
import sys
import os
import shutil
import logging
from unittest.mock import patch
import json
from pathlib import Path

# 1. 确保能导入 qwen3_tts_gguf 目录下的模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf")

if CONVERT_LIB_DIR not in sys.path:
    # 插入到最前面以确保优先加载
    sys.path.insert(0, CONVERT_LIB_DIR)

# 2. 导入目标模块
try:
    import convert_hf_to_gguf
    # get_vocab_base_pre 是 TextModel 的方法
    from convert_hf_to_gguf import TextModel
    import gguf
except ImportError as e:
    print(f"❌ Error importing convert_hf_to_gguf: {e}")
    sys.exit(1)


# 3. 定义补丁函数
def patched_get_vocab_base_pre(self, tokenizer) -> str:
    """
    Monkey Patch 替代函数。
    不进行任何哈希计算，直接返回 'qwen2'。
    """
    print(f"💉 [补丁] 拦截到 get_vocab_base_pre 调用。")
    print(f"💉 [补丁] 绕过哈希检查，强制返回 'qwen2'。")
    return "qwen2"

def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    """
    Monkey Patch 替代函数。
    强制从 config.json 加载参数，绕过 AutoConfig 的潜在远程代码加载和解析干扰。
    """
    print(f"💉 [补丁] 拦截到 load_hparams 调用。")
    print(f"💉 [补丁] 强制从 {dir_model / 'config.json'} 加载配置。")
    
    if is_mistral_format:
        with open(dir_model / "params.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    try:
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        raise

    if "llm_config" in config:
        config["text_config"] = config["llm_config"]
    if "lm_config" in config:
        config["text_config"] = config["lm_config"]
    if "thinker_config" in config:
        config["text_config"] = config["thinker_config"]["text_config"]
    if "lfm" in config:
        config["text_config"] = config["lfm"]
        
    return config

# 4. 转换主逻辑
def main():
    MASTER_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "hf")
    GGUF_OUT = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_codec_only.gguf")
    TEMP_DIR = os.path.join(PROJECT_ROOT, "model", "temp_hf")

    print(f"--- 正在执行 GGUF 转换 (零污染流程) ---")
    print(f"源目录: {MASTER_MODEL_DIR}")
    print(f"输出文件: {GGUF_OUT}")

    # [步骤 A] 准备临时转换目录 (为了给 Key 增加 'model.' 前缀)
    # GGUF 转换器在 qwen2 模式下期望 backbone 带有 model. 前缀
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    print("[1/3] 正在准备临时转换权重 (添加 model. 前缀)...")
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    src_weights = os.path.join(MASTER_MODEL_DIR, "model.safetensors")
    processed_weights = {}
    
    with safe_open(src_weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # lm_head 不需要前缀
            if key.startswith("lm_head"):
                processed_weights[key] = tensor
            else:
                # 给骨干网络加上 model. 前缀
                new_key = key if key.startswith("model.") else f"model.{key}"
                processed_weights[new_key] = tensor
                
    save_file(processed_weights, os.path.join(TEMP_DIR, "model.safetensors"))
    
    # 拷贝必要的配置文件
    for f in ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]:
        src = os.path.join(MASTER_MODEL_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, TEMP_DIR)

    # [步骤 B] 应用 Monkey Patch
    print("[2/3] 正在向 convert_hf_to_gguf.TextModel 和 ModelBase 应用补丁...")
    
    # Patch TextModel 类
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    # Patch ModelBase 类
    from convert_hf_to_gguf import ModelBase
    ModelBase.load_hparams = staticmethod(patched_load_hparams)
    
    # [步骤 C] 调用转换器主函数
    print(f"[3/3] 正在执行 convert_hf_to_gguf.main()...")
    
    # 模拟命令行参数给 convert_hf_to_gguf
    sys.argv = [
        "convert_hf_to_gguf.py",
        TEMP_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    try:
        convert_hf_to_gguf.main()
        print(f"\n✅ GGUF 转换成功!")
        print(f"输出路径: {GGUF_OUT}")
    except Exception as e:
        print(f"\n❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(TEMP_DIR):
            print(f"[清理] 正在删除临时目录...")
            shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
