import json
import os

def debug_vocab():
    MASTER_DIR = "Standalone-Bare-Master"
    vocab_path = os.path.join(MASTER_DIR, "vocab.json")
    config_path = os.path.join(MASTER_DIR, "config.json")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    v_max = max(vocab.values())
    v_min = min(vocab.values())
    v_size = len(vocab)
    c_vocab_size = config.get("vocab_size")
    
    print(f"Vocab Len: {v_size}")
    print(f"Vocab Min Index: {v_min}")
    print(f"Vocab Max Index: {v_max}")
    print(f"Config vocab_size: {c_vocab_size}")
    
    if v_max >= c_vocab_size:
        print("ERROR: max(vocab) >= vocab_size")
    else:
        print("SUCCESS: max(vocab) < vocab_size")

if __name__ == "__main__":
    debug_vocab()
