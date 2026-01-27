import os
import sys
import torch
import soundfile as sf
import random
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

SAVE_DIR = "captured_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"Loading Official Model from {MODEL_PATH}...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        
        # Test Case 1: Preset Speaker (Vivian)
        text_1 = "这是一个测试，我们要验证不同说话人的生成效果。"
        speaker_1 = "Vivian"
        print(f"\n--- Case 1: {speaker_1} ---")
        print(f"Text: {text_1}")
        
        wavs_1, sr = tts.generate_custom_voice(
            text=text_1,
            speaker=speaker_1,
            language="Chinese",
            do_sample=True, # Randomness for variety
            temperature=0.7
        )
        
        out_path_1 = os.path.join(SAVE_DIR, "official_dynamic_vivian.wav")
        sf.write(out_path_1, wavs_1[0], sr)
        print(f"Saved to {out_path_1}")

        # Test Case 2: Preset Speaker (Uncle Fu - Male)
        text_2 = "今天天气真不错，适合出去钓鱼。"
        speaker_2 = "Uncle_Fu" # Check valid ID from config if failing, or "uncle_fu"
        print(f"\n--- Case 2: {speaker_2} ---")
        print(f"Text: {text_2}")
        
        wavs_2, sr = tts.generate_custom_voice(
            text=text_2,
            speaker=speaker_2,
            language="Chinese", 
             do_sample=True,
            temperature=0.7
        )
        
        out_path_2 = os.path.join(SAVE_DIR, "official_dynamic_uncle_fu.wav")
        sf.write(out_path_2, wavs_2[0], sr)
        print(f"Saved to {out_path_2}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
