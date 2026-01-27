"""
[84] Verify GGUF Loading & Numerical Correctness (80 Series)
仿照 94-Verify-GGUF-with-llama-cpp.py，验证 3072 词表的 GGUF 模型。

验证内容：
1. 加载 master-codec-only-3072-f16.gguf
2. 注入 40_first_step_embeds.npy (模拟第一步推理)
3. 检查 Logits 输出形状是否为 3072
4. 验证预测结果 (Expected: 1995)
"""
import os
import sys
import numpy as np
import ctypes

os.environ["VK_ICD_FILENAMES"] = "none" # 禁用 Vulkan 以保证数值一致性测试稳健
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 确保能导入 qwen3_tts_gguf
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import qwen3_tts_gguf.nano_llama as nano_llama
except ImportError:
    print("❌ Failed to import nano_llama. Please ensure qwen3_tts_gguf is in path.")
    sys.exit(1)

def run_inference(ctx, inputs_embeds, vocab_size, expected_token_id):
    """执行单次推理"""
    print(f"\n--- Running Inference ---")
    
    # 1. 准备 Batch Setup (仿照 94 的 IMRoPE 兼容逻辑)
    # 输入 shape: [1, n_tokens, n_embd]
    n_input_tokens = inputs_embeds.shape[1]
    dim = inputs_embeds.shape[2]
    
    full_embd = inputs_embeds[0].astype(np.float32)
    if not full_embd.flags['C_CONTIGUOUS']:
        full_embd = np.ascontiguousarray(full_embd)

    # 分配 4倍 token 容量以容纳 M-RoPE 的多重 pos
    batch = nano_llama.llama_batch_init(n_input_tokens * 4, dim, 1)
    batch.n_tokens = n_input_tokens # 逻辑上包含 n_input_tokens 个 token
    
    # 注入 embedding 数据
    batch.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token)) # 指明这是纯 embedding 输入
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    # 设置 Pos (IMRoPE 模式: 3个 pos + 1个 padding)
    for k in range(n_input_tokens):
        batch.pos[k] = k
        batch.pos[n_input_tokens + k] = k
        batch.pos[2 * n_input_tokens + k] = k
        batch.pos[3 * n_input_tokens + k] = 0
        
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        # 只计算最后一个 token 的 logits
        batch.logits[k] = 1 if k == n_input_tokens - 1 else 0

    # 2. 推理
    ret = nano_llama.llama_decode(ctx, batch)
    nano_llama.llama_batch_free(batch)

    if ret != 0:
        print(f"❌ llama_decode failed with error code {ret}")
        return False

    # 3. 获取 Logits
    logits_ptr = nano_llama.llama_get_logits(ctx)
    # 转换为 numpy 数组
    logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
    
    max_logit_idx = int(np.argmax(logits_arr))
    max_logit_val = np.max(logits_arr)
    
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Predicted Token ID: {max_logit_idx}")
    print(f"  Max Logit Value:    {max_logit_val:.4f}")
    
    if max_logit_idx == expected_token_id:
        print("  ✅ MATCH Expected Token ID")
    else:
        print(f"  ⚠️ MISMATCH! Expected {expected_token_id}, got {max_logit_idx}")
    
    # 额外的数值检查
    if vocab_size != 3072:
        print(f"  ⚠️ Warning: GGUF vocab size is {vocab_size}, expected 3072 for Codec-Only model.")
    
    return True

def main():
    GGUF_PATH = os.path.join(PROJECT_ROOT, "master-codec-only-3072-f16.gguf")
    INPUT_NPY = os.path.join(PROJECT_ROOT, "40_first_step_embeds.npy")

    if not os.path.exists(GGUF_PATH):
        print(f"❌ GGUF model not found: {GGUF_PATH}")
        return
    
    if not os.path.exists(INPUT_NPY):
        print(f"❌ Input embeddings not found: {INPUT_NPY}")
        return

    print(f"Loading model: {GGUF_PATH}")
    # 加载模型 (CPU mode for verification usually suffices and is deterministic)
    model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    if not model:
        print("❌ Failed to load model")
        return

    vocab = nano_llama.llama_model_get_vocab(model)
    vocab_size = nano_llama.llama_vocab_n_tokens(vocab)
    print(f"Model loaded. Vocab size: {vocab_size}")

    # 创建 Context
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.n_threads = 4
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    
    if not ctx:
        print("❌ Failed to create context")
        nano_llama.llama_model_free(model)
        return

    # 加载输入
    embeds = np.load(INPUT_NPY)
    print(f"Loaded inputs from {INPUT_NPY}, shape: {embeds.shape}")

    # 运行验证 (Expected ID 1995 comes from previous 94 script knowledge)
    run_inference(ctx, embeds, vocab_size, expected_token_id=1995)

    # 清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    print("\nDone.")

if __name__ == "__main__":
    main()
