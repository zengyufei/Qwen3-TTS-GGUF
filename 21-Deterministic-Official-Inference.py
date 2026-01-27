import os
import sys
import torch
import soundfile as sf
# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

from qwen_tts import Qwen3TTSModel

def main():
    # 检测设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            device_map=device, 
            dtype=dtype
        )
        
        # 确定性参数：仿照 20 号脚本固定随机性
        # 设置 do_sample=False 会使用 greedy search，从而固定 Master 输出
        # subtalker_dosample=False 会固定 Code Predictor (工匠) 的输出
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        text = "今天天气好"
        speaker = "Vivian"
        output_file = "21_deterministic_inference_output.wav"
        
        print(f"正在以固定采样参数推理: 「{text}」...")
        # 调用官方推理接口
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language="Chinese",
            speaker=speaker,
            instruct="",
            **deterministic_kwargs
        )
        
        # 保存为 wav 文件
        sf.write(output_file, wavs[0], sr)
        print(f"✅ 音频已保存至: {output_file}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
