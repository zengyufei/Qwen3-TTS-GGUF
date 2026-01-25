import json
from safetensors import safe_open
import sys

model_path = r'c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice\model.safetensors'

def log(msg):
    print(msg)
    sys.stdout.flush()

try:
    log(f"Opening {model_path}...")
    with safe_open(model_path, framework='pt') as f:
        keys = f.keys()
        info = {}
        for k in keys:
            if 'embedding' in k or 'embed_tokens' in k:
                tensor = f.get_tensor(k)
                info[k] = list(tensor.shape)
                log(f"Found {k}: {info[k]}")
        
        log("Full relevant layer report:")
        log(json.dumps(info, indent=2))
except Exception as e:
    log(f"Error: {e}")
    import traceback
    traceback.print_exc()
