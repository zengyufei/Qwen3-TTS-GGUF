import ctypes
import os

def check_symbol(dll_path, symbol_name):
    try:
        dll = ctypes.CDLL(dll_path)
        func = getattr(dll, symbol_name)
        print(f"✅ Symbol '{symbol_name}' found.")
        return True
    except AttributeError:
        print(f"❌ Symbol '{symbol_name}' NOT found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    dll_path = r"d:\qwen3-tts\qwen3_tts_gguf\bin\llama.dll"
    check_symbol(dll_path, "llama_kv_cache_clear")
    check_symbol(dll_path, "llama_batch_init")
    check_symbol(dll_path, "llama_memory_clear")
    check_symbol(dll_path, "llama_get_memory")
