import os
import sys
import subprocess

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen3_tts_gguf.llm_export import extract_and_save_llm
from qwen3_tts_gguf import logger

def main():
    # 1. 路径配置
    SOURCE_MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    OUTPUT_HF_DIR = r'./model/hf_temp' # 临时目录
    OUTPUT_GGUF_DIR = r'./model'
    
    # 模型文件名为 model.safetensors 或 pytorch_model.bin 或 model.pt
    # Qwen3-TTS 通常是 model.safetensors
    # 但之前的 task 发现它是 model.pt 吗？ 
    # check list_dir output from earlier steps -> Qwen3-TTS-12Hz... has 13 children.
    # 假设它是标准的 HF 目录结构供 from_pretrained 使用，或者包含 model.pt
    # 我们的辅助函数 extract_and_save_llm 假设是 model.pt。
    # 让我们检查一下该目录的内容，或者更健壮地处理。
    # 这里先假设包含 model.pt (像 01 脚本那样)。
    # 如果是 safetensors，可以直接用 convert 脚本（如果是标准 Qwen 结构）。
    # 但 Qwen3TTS 是复合模型，所以必须提取。
    
    # 检测 model.safetensors, model.pt 或 pytorch_model.bin
    model_path = None
    if os.path.exists(os.path.join(SOURCE_MODEL_DIR, "model.safetensors")):
        model_path = os.path.join(SOURCE_MODEL_DIR, "model.safetensors")
    elif os.path.exists(os.path.join(SOURCE_MODEL_DIR, "model.pt")):
        model_path = os.path.join(SOURCE_MODEL_DIR, "model.pt")
    elif os.path.exists(os.path.join(SOURCE_MODEL_DIR, "pytorch_model.bin")):
        model_path = os.path.join(SOURCE_MODEL_DIR, "pytorch_model.bin")
    
    if not model_path:
        logger.error(f"在 {SOURCE_MODEL_DIR} 中找不到权重文件 (model.safetensors/model.pt/pytorch_model.bin)")
        return

    config_path = os.path.join(SOURCE_MODEL_DIR, "config.json")
    
    logger.info("Step 1: 提取 LLM 并转换为 HF 格式")
    # 如果已经存在 HF 临时模型，可以选择跳过（这里为了保险每次都重新生成）
    # 清理旧的临时目录
    if os.path.exists(OUTPUT_HF_DIR):
        import shutil
        shutil.rmtree(OUTPUT_HF_DIR)
        
    success = extract_and_save_llm(
        source_model_path=model_path,
        config_path=config_path,
        output_hf_dir=OUTPUT_HF_DIR,
        tokenizer_output_dir=OUTPUT_HF_DIR # Tokenizer 也放在一起
    )
    
    if not success:
        logger.error("提取 LLM 失败")
        return
    
    # 2. 调用 GGUF 转换脚本
    CONVERT_SCRIPT = r'./fun_asr_gguf/convert_hf_to_gguf.py'
    if not os.path.exists(CONVERT_SCRIPT):
        logger.error(f"找不到转换脚本: {CONVERT_SCRIPT}")
        return

    OUTPUT_GGUF = os.path.join(OUTPUT_GGUF_DIR, "Qwen3-LLM-1.7B-F16.gguf")
    
    logger.info("Step 2: 转换为 GGUF")
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        OUTPUT_HF_DIR,
        "--outfile", OUTPUT_GGUF,
        "--outtype", "f16"
    ]
    
    logger.info(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("GGUF 转换完成！")

if __name__ == "__main__":
    main()
