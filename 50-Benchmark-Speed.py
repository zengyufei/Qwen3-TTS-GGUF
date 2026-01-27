import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import ctypes
import torch
import numpy as np
import time
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CRAFTSMAN_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")
GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")

def benchmark_master():
    print("\n--- Benchmarking Master (GGUF) ---")
    
    # 1. Load Resources
    gguf_model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 1024
    ctx_params.embeddings = True
    gguf_ctx = nano_llama.llama_init_from_model(gguf_model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(gguf_model)
    
    master_head_weight = np.load(MASTER_HEAD_PATH)
    
    # Load just one codec table for lookup simulation
    codec_table = np.load(os.path.join(MODEL_DIR, "codec_embedding_0.npy"))
    
    # 2. Setup Dummy Context
    n_steps = 50
    batch = nano_llama.llama_batch_init(512, n_embd, 1)
    
    # Dummy Prefill (1 token)
    dummy_embed = np.random.randn(1, n_embd).astype(np.float32)
    batch.n_tokens = 1
    batch.pos[0] = 0
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1
    ctypes.memmove(batch.embd, np.ascontiguousarray(dummy_embed).ctypes.data, dummy_embed.nbytes)
    
    nano_llama.llama_decode(gguf_ctx, batch) # Warmup / Init
    
    # 3. Benchmark Loop
    print(f"Running {n_steps} steps...")
    start_time = time.time()
    
    current_pos = 1
    
    # Simulate: Decode -> Head -> Argmax -> Lookup -> Decode
    
    # Fake hidden state from prefill
    out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
    current_hidden = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd)).copy()
    
    for i in range(n_steps):
        # 1. Head Project
        logits = current_hidden @ master_head_weight.T
        token_id = np.argmax(logits)
        
        # 2. Lookup Embed
        next_embed = codec_table[token_id % 2048].reshape(1, n_embd) # Safe mod
        
        # 3. Decode
        batch.n_tokens = 1
        batch.pos[0] = current_pos
        current_pos += 1
        ctypes.memmove(batch.embd, np.ascontiguousarray(next_embed).ctypes.data, next_embed.nbytes)
        
        if nano_llama.llama_decode(gguf_ctx, batch) != 0:
            print("Decode Error")
            break
            
        out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
        current_hidden = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd)) # No copy for speed?
        
    end_time = time.time()
    duration = end_time - start_time
    tps = n_steps / duration
    
    print(f"Master Speed: {tps:.2f} tokens/sec (Total: {duration:.2f}s for {n_steps} steps)")
    
    nano_llama.llama_free(gguf_ctx)
    nano_llama.llama_model_free(gguf_model)

def benchmark_craftsman():
    print("\n--- Benchmarking Craftsman (ONNX) ---")
    
    # 1. Load Session
    sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(CRAFTSMAN_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    n_embd = 2048
    n_steps = 200 # More steps for Craftsman
    
    print(f"Running {n_steps} steps...")
    start_time = time.time()
    
    # Re-use pasts to simulate state
    current_pasts = {f"past_{k}": np.zeros((1, 8, 0, 128), dtype=np.float32) for k in range(10)}

    for i in range(n_steps):
        # Random Input [1, 2, 2048]
        inputs_embeds = np.random.randn(1, 2, 2048).astype(np.float32)
        
        inputs = {'inputs_embeds': inputs_embeds}
        inputs.update(current_pasts)
        
        outputs = sess.run(None, inputs)
        
        # Update Pasts (Simulate autoregressive state update cost)
        pasts = outputs[1:]
        for k in range(5):
            current_pasts[f"past_{2*k}"] = pasts[2*k]
            current_pasts[f"past_{2*k+1}"] = pasts[2*k+1]
            
    end_time = time.time()
    duration = end_time - start_time
    tps = n_steps / duration
    
    print(f"Craftsman Speed: {tps:.2f} steps/sec (Total: {duration:.2f}s for {n_steps} steps)")
    print(f"Note: 1 Master Step triggers 15 Craftsman Steps.")
    print(f"Effective Master TPS limit by Craftsman: {tps/15:.2f} master-tokens/sec")

if __name__ == "__main__":
    benchmark_master()
    benchmark_craftsman()
