import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 全局变量用于捕获
captured_codes_list = []

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print("Loading official model...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # --- 拦截逻辑 ---
    original_decode = tts.model.speech_tokenizer.decode
    
    def intercepted_decode(list_of_dict, **kwargs):
        # 捕获生成的码
        codes = list_of_dict[0]["audio_codes"]
        captured_codes_list.append(codes.detach().cpu())
        return original_decode(list_of_dict, **kwargs)
    
    # Monkey patch
    tts.model.speech_tokenizer.decode = intercepted_decode
    
    # 确定性参数
    deterministic_kwargs = {
        "do_sample": False,
        "subtalker_dosample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    }
    
    print("\n--- Starting Run 1 ---")
    wavs1, sr = tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        language="Chinese",
        **deterministic_kwargs
    )
    
    print("\n--- Starting Run 2 ---")
    wavs2, sr = tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        language="Chinese",
        **deterministic_kwargs
    )
    
    # --- 验证 ---
    if len(captured_codes_list) < 2:
        print("\n❌ Error: Failed to capture two sets of codes.")
        return

    codes1 = captured_codes_list[0]
    codes2 = captured_codes_list[1]
    
    print(f"\nComparing codes...")
    print(f"Run 1 codes shape: {codes1.shape}")
    print(f"Run 2 codes shape: {codes2.shape}")
    
    if codes1.shape != codes2.shape:
        print("❌ FAILED: Shapes mismatch!")
    else:
        mismatch_count = torch.sum(codes1 != codes2).item()
        if mismatch_count == 0:
            print("✅ SUCCESS: Both runs produced IDENTICAL codes! (Bit-exact)")
        else:
            print(f"❌ FAILED: {mismatch_count} tokens mismatch out of {codes1.numel()}.")

    # 同时也对比一下 Waveform
    wav1 = wavs1[0]
    wav2 = wavs2[0]
    min_len = min(len(wav1), len(wav2))
    wav_diff = np.max(np.abs(wav1[:min_len] - wav2[:min_len]))
    print(f"Waveform max difference: {wav_diff}")

if __name__ == "__main__":
    main()
