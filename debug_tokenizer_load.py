import os
from transformers import AutoTokenizer
import json

def debug_load():
    model_dir = "Standalone-Bare-Master"
    print(f"Loading tokenizer from {model_dir}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Vocab size (len): {len(tokenizer.vocab)}")
    
    if len(tokenizer.vocab) == 0:
        print("WARN: Vocab is empty!")
        return

    vals = tokenizer.vocab.values()
    max_id = max(vals)
    min_id = min(vals)
    print(f"Max ID: {max_id}")
    print(f"Min ID: {min_id}")
    
    if max_id >= 3072:
        print("ERROR: Max ID >= 3072")
        # Find tokens with high IDs
        high_tokens = {k: v for k, v in tokenizer.vocab.items() if v >= 3072}
        print(f"Found {len(high_tokens)} tokens with ID >= 3072")
        print(f"First 10 high ID tokens: {list(high_tokens.items())[:10]}")
    else:
        print("SUCCESS: Max ID < 3072")

    # Check config vocab size
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        conf = json.load(f)
    print(f"Config vocab_size: {conf.get('vocab_size')}")

    # Check actual vocab.json content
    with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
        v = json.load(f)
    print(f"Raw vocab.json len: {len(v)}")
    print(f"Raw vocab.json max val: {max(v.values())}")

if __name__ == "__main__":
    debug_load()
