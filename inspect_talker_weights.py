
import torch
from safetensors.torch import load_file
import os

model_path = r'c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice\model.safetensors'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
else:
    print(f"Loading weights from {model_path}...")
    # Load only necessary tensors to save memory
    state_dict = load_file(model_path)
    
    print("\n[Talker Text Projection Weights]")
    for k, v in state_dict.items():
        if "talker.text_projection" in k:
            print(f"{k}: {v.shape} ({v.dtype})")
            
    print("\n[Talker Embeddings]")
    for k, v in state_dict.items():
        if "talker.model.text_embedding" in k or "talker.model.codec_embedding" in k:
            print(f"{k}: {v.shape} ({v.dtype})")

    print("\n[Talker Attention Norms (QK-Norm Check)]")
    for k, v in state_dict.items():
        if "talker.model.layers.0.self_attn.q_norm" in k or "talker.model.layers.0.self_attn.k_norm" in k:
            print(f"{k}: {v.shape} ({v.dtype})")
