
import torch
from safetensors.torch import load_file
import os

model_path = r'c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice\model.safetensors'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
else:
    print(f"Loading weights from {model_path}...")
    state_dict = load_file(model_path)
    
    print("\n[Code Predictor Embeddings (Tables 1-15)]")
    # Note: table 0 is in talker.model.codec_embedding.weight
    for k, v in state_dict.items():
        if "talker.code_predictor.model.codec_embedding" in k:
            print(f"{k}: {v.shape} ({v.dtype})")
            
    print("\n[Code Predictor Output Heads]")
    # Predictor generates code 1-15
    for k, v in state_dict.items():
        if "talker.code_predictor.lm_head" in k:
            print(f"{k}: {v.shape} ({v.dtype})")
