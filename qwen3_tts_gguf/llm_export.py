import os
import shutil
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM
from . import logger

import traceback
def extract_and_save_llm(source_model_path, config_path, output_hf_dir, tokenizer_output_dir):
    try:
        return _extract_and_save_llm_impl(source_model_path, config_path, output_hf_dir, tokenizer_output_dir)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        return False

def _extract_and_save_llm_impl(source_model_path, config_path, output_hf_dir, tokenizer_output_dir):
    """
    提取混合模型中的 LLM 主干，并保存为标准的 Hugging Face 格式。
    
    Args:
        source_model_path (str): 原始 PyTorch (.pt) 模型路径
        config_path (str): 原始模型配置文件路径
        output_hf_dir (str): 输出 HF 模型 (safetensors) 的目录
        tokenizer_output_dir (str): 输出 Tokenizer 配置的目录
    """
    logger.info(f"[LLM Export] Loading full model from {source_model_path} ...")
    
    llm_weights = {}
    
    if source_model_path.endswith(".safetensors"):
        import struct
        import numpy as np
        
        print("   [LLM Export] Using Streaming Pure-Python Safetensors Parser...")
        
        # 1. 读取源文件 Header
        with open(source_model_path, "rb") as f_src:
            h_size_bytes = f_src.read(8)
            h_size = struct.unpack("<Q", h_size_bytes)[0]
            header = json.loads(f_src.read(h_size).decode("utf-8"))
            
            # 2. 识别前缀
            full_keys = header.keys()
            prefix = None
            if "talker.model.text_embedding.weight" in full_keys:
                prefix = "talker."
            elif "model.text_embedding.weight" in full_keys:
                prefix = ""
            else:
                for key in full_keys:
                    if "model.text_embedding.weight" in key:
                        prefix = key.replace("model.text_embedding.weight", "")
                        break
            
            if prefix is None:
                print("   [Error] Could not identify LLM prefix.")
                return False
            print(f"   [LLM Export] Identified prefix: '{prefix}'")
            
            # 3. 规划映射任务
            tasks = []
            for key in sorted(full_keys):
                if not key.startswith(prefix):
                    continue
                
                new_key = key[len(prefix):]
                final_key = None
                if new_key == "model.text_embedding.weight":
                    final_key = "model.embed_tokens.weight"
                elif new_key == "codec_head.weight":
                    final_key = "lm_head.weight"
                elif new_key.startswith("model.layers.") or new_key == "model.norm.weight":
                    final_key = new_key
                
                if final_key:
                    tasks.append((key, final_key, header[key]))
            
            # 4. 加载并保存配置
            print(f"   [LLM Export] Saving config and tokenizer to {output_hf_dir} ...")
            os.makedirs(output_hf_dir, exist_ok=True)
            with open(config_path, 'r', encoding='utf-8') as conf_f:
                full_config = json.load(conf_f)
            
            llm_config_dict = full_config.get("talker_config", full_config)
            llm_config_dict["architectures"] = ["Qwen3ForCausalLM"]
            llm_config_dict["model_type"] = "qwen3"
            
            # 强制对齐物理权重维度 (151936)
            llm_config_dict["vocab_size"] = 151936
            print(f"   [LLM Export] Forced physical vocab_size: {llm_config_dict['vocab_size']}")

            with open(os.path.join(output_hf_dir, "config.json"), 'w', encoding='utf-8') as out_conf:
                json.dump(llm_config_dict, out_conf, indent=2)
            
            # 复制 Tokenizer 文件
            src_dir = os.path.dirname(config_path)
            for file in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']:
                s = os.path.join(src_dir, file)
                d = os.path.join(output_hf_dir, file)
                if os.path.exists(s):
                    shutil.copy(s, d)
            
            # 5. 流式写入权重
            out_file = os.path.join(output_hf_dir, "model.safetensors")
            print(f"   [LLM Export] Streaming {len(tasks)} tensors to {out_file} (F32 converted)...")
            
            # 5.1 预计算目标 Header
            target_header = {"__metadata__": {"format": "pt"}}
            offset = 0
            for src_key, final_key, info in tasks:
                num_elements = 1
                for s in info["shape"]: num_elements *= s
                size = num_elements * 4 # 全部转为 F32 (4 bytes)
                target_header[final_key] = {
                    "dtype": "F32",
                    "shape": info["shape"],
                    "data_offsets": [offset, offset + size]
                }
                offset += size
            
            header_json = json.dumps(target_header).encode("utf-8")
            header_padding = (8 - (len(header_json) % 8)) % 8
            header_json += b' ' * header_padding
            
            # 5.2 循环：读一、转一、写一
            with open(out_file, "wb") as f_dst:
                f_dst.write(struct.pack("<Q", len(header_json)))
                f_dst.write(header_json)
                
                for src_key, final_key, info in tasks:
                    print(f"      Writing {final_key} ({info['dtype']}) ...", end="\r")
                    start_src = 8 + h_size + info["data_offsets"][0]
                    end_src = 8 + h_size + info["data_offsets"][1]
                    
                    f_src.seek(start_src)
                    raw_data = f_src.read(end_src - start_src)
                    
                    dt = info["dtype"]
                    if dt == "BF16":
                        arr_u16 = np.frombuffer(raw_data, dtype=np.uint16)
                        # 转换并写入流
                        f32_data = (arr_u16.astype(np.uint32) << 16).view(np.float32)
                        f_dst.write(f32_data.tobytes())
                    elif dt == "F16":
                        f32_data = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
                        f_dst.write(f32_data.tobytes())
                    else:
                        f_dst.write(raw_data)
            
            print(f"\n   [LLM Export] Successfully exported {len(tasks)} tensors.")
            return True
            
    else:
        # 旧的 .pt 加载逻辑 (如果需要也应优化，但目前主要是 safetensors)
        full_model = torch.load(source_model_path, map_location='cpu')
        # ... 原有逻辑 (为了简洁此处略，但实际代码中应保留或重写)
        # TODO: 如果后续遇到大 .pt 文件也需类似优化
        print("   [LLM Export] .pt format extraction not optimized for memory yet.")
        # 简单模拟一下原有提取并保存
        prefix = None
        for key in full_model.keys():
            if key.startswith("talker.model."):
                prefix = "talker."
                break
            if key.startswith("model.layers."): # 直接是 LLM
                prefix = ""
                break
                
        if prefix is None:
            # 尝试暴力匹配，找 model.embed_tokens.weight
            for key in full_model.keys():
                if "model.embed_tokens.weight" in key:
                    # e.g. "talker.model.embed_tokens.weight" -> prefix="talker."
                    prefix = key.replace("model.embed_tokens.weight", "")
                    break
        
        if prefix is None:
            print("   [Error] Could not identify LLM prefix in state dict.")
            return False
            
        print(f"   [LLM Export] Identified prefix: '{prefix}'")

        for key, value in full_model.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                llm_weights[new_key] = value

        print(f"   [LLM Export] Extracted {len(llm_weights)} keys.")
        del full_model

        # 2. 加载并转换配置
        print(f"   [LLM Export] Loading config from {config_path} ...")
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            
        # 提取 talker_config 作为主配置
        if "talker_config" in full_config:
            llm_config_dict = full_config["talker_config"]
        else:
            llm_config_dict = full_config # 假设直接是 LLM 配置
            
        # 确保架构名称正确
        llm_config_dict["architectures"] = ["Qwen3ForCausalLM"]
        llm_config_dict["model_type"] = "qwen3"

        # Detect actual vocab size
        src_dir = os.path.dirname(config_path)
        vocab_json_path = os.path.join(src_dir, "vocab.json")
        if os.path.exists(vocab_json_path):
            with open(vocab_json_path, 'r', encoding='utf-8') as f_v:
                vocab_data = json.load(f_v)
            actual_max_id = max(vocab_data.values())
            llm_config_dict["vocab_size"] = actual_max_id + 1
            print(f"   [LLM Export] Detected real vocab size: {llm_config_dict['vocab_size']}")

        # config = AutoConfig.for_model("qwen2", **llm_config_dict)
        # Note: Since transformers might not know 'qwen3' yet, we skip AutoConfig 
        # and manually create the config object or just pass it to from_config if we use AutoModel
        # But for now, we just save the config.json and rely on the convert script.
        
        # 3. 初始化并保存
        print("   [LLM Export] Initializing Qwen2ForCausalLM ...")
        # 使用 AutoModel 避免硬编码 import
        qwen_model = AutoModelForCausalLM.from_config(config)
        
        print("   [LLM Export] Loading state dict ...")
        # 使用 strict=False 允许一些非关键键缺失（如 code_predictor 相关，如果它不在标准 CausalLM 中）
        # 标准 CausalLM 只需要 model.* 和 lm_head.*
        # 我们的 llm_weights 可能包含 code_predictor.* (如果它在 talker 下)，这在标准 Qwen2 中是不需要的。
        # 过滤掉不需要的键，或者让它不匹配。
        # 为了保险，过滤:
        clean_weights = {k: v for k, v in llm_weights.items() if k.startswith("model.") or k.startswith("lm_head.")}
        
        missing, unexpected = qwen_model.load_state_dict(clean_weights, strict=False)
        print(f"   [LLM Export] Loaded. Missing: {len(missing)}, Unexpected keys ignored: {len(unexpected)}")

        os.makedirs(output_hf_dir, exist_ok=True)
        print(f"   [LLM Export] Saving to {output_hf_dir} ...")
        qwen_model.save_pretrained(output_hf_dir, safe_serialization=True)
    
    # 4. 处理 Tokenizer (复制相关文件)
    print(f"   [LLM Export] Copying tokenizer files to {tokenizer_output_dir} ...")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    
    # 假设源目录有一些 tokenizer 文件，或者我们需要从包含 tokenizer 的目录复制
    # 通常 Qwen3-TTS 目录结构中 tokenizer 文件在根目录
    src_dir = os.path.dirname(config_path)
    files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
    
    for file in files_to_copy:
        src = os.path.join(src_dir, file)
        dst = os.path.join(tokenizer_output_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"      Copied {file}")
        else:
            print(f"      Warning: {file} not found in source.")
            
    # 特别注意：Qwen2 Tokenizer 需要特制的 tokenizer_config.json 或 tokenizer.json
    # 如果原始的是 Qwen3TTS 特有的，可能导致 convert_hf_to_gguf 失败。
    # 但通常 Qwen 使用的是 BPE，格式兼容。
    
    return True
