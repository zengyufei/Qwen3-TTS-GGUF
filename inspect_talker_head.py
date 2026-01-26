
import torch
from safetensors.torch import load_file
import os

model_path = r'c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice\model.safetensors'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
else:
    print(f"Loading weights from {model_path}...")
    # We only generally need to know the shape, but safetensors.torch.load_file loads everything or we can filter.
    # safe_open is better for partial loading but let's stick to the user's pattern or robust method.
    from safetensors import safe_open
    
    found = False
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        for k in keys:
            if "talker.codec_head" in k:
                tensor = f.get_tensor(k)
                print(f"Original Tensor '{k}' shape: {tensor.shape}")
                found = True
            elif "talker.lm_head" in k: # fallback check
                tensor = f.get_tensor(k)
                print(f"Original Tensor '{k}' shape: {tensor.shape}")
                found = True
        
    if not found:
        print("Could not find any tensor matching 'talker.codec_head' or 'talker.lm_head'")
