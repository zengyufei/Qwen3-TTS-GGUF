import os
import sys
import ctypes
import torch
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
from transformers import AutoTokenizer
import qwen3_tts_gguf.nano_llama as nano_llama

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Assets Paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
EAR_PATH = os.path.join(MODEL_DIR, "qwen3_tts_encoder.onnx")
CRAFTSMAN_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")
MOUTH_PATH = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")
GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
TOKENIZER_DIR = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
HEADS_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")

SAVE_DIR = "captured_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants
SAMPLE_RATE = 24000
STEPS = 250
EOS_TOKEN_ID = 2150
TTS_PAD_ID = 151671
TTS_EOS_ID = 151673
CODEC_PAD_ID = 2148
CODEC_BOS_ID = 2149

def load_tables():
    print(f"[Brain] Loading Embedding Tables...")
    text_table = np.load(os.path.join(MODEL_DIR, "text_embedding_projected.npy"))
    # Load all 16 codec tables
    codec_tables = []
    for i in range(16):
        codec_tables.append(np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")))
    return text_table, codec_tables

def construct_preset_prompt(target_text, spk_id, tokenizer, tables):
    print(f"[Brain] Constructing Preset Prompt (SpkID: {spk_id})...")
    text_table, codec_tables = tables
    codec_table_0 = codec_tables[0]
    
    embed_list = []
    
    # H1: Role Prefix
    embed_list.append(text_table[151644]) # <|im_start|>
    embed_list.append(text_table[77091])  # assistant
    embed_list.append(text_table[198])    # \n
    
    # H2: Control Block (5 tokens)
    embed_list.append(text_table[151671] + codec_table_0[2154]) # Think
    embed_list.append(text_table[151671] + codec_table_0[2156]) # TBOS
    
    # Lang ID Mapping
    # simple fix: use Chinese (2055) for now, or look up
    lang_id = 2055 
    embed_list.append(text_table[151671] + codec_table_0[lang_id])
    
    embed_list.append(text_table[151671] + codec_table_0[2157]) # TEOS
    embed_list.append(text_table[151671] + codec_table_0[spk_id]) # Spk
    
    # H3: Audio Start
    embed_list.append(text_table[151672] + codec_table_0[2148]) # TTS_BOS + C_PAD
    
    # Content
    content_ids = tokenizer.encode(target_text, add_special_tokens=False)
    for tid in content_ids:
        embed_list.append(text_table[tid] + codec_table_0[2148])
        
    # Trailer
    embed_list.append(text_table[TTS_EOS_ID] + codec_table_0[2148]) # TTS_EOS
    embed_list.append(text_table[151671] + codec_table_0[CODEC_BOS_ID]) # TTS_PAD + C_BOS
    
    return np.stack(embed_list)[np.newaxis, ...].astype(np.float32)

def construct_clone_prompt(ref_audio_path, ref_text, target_text, tokenizer, tables, sess_encoder):
    print(f"[Brain] Constructing Clone Prompt (ICL)...")
    text_table, codec_tables = tables
    
    # 1. Process Reference Audio (Ear)
    print(f"  Thinking with Ear: {ref_audio_path}")
    audio, sr = librosa.load(ref_audio_path, sr=SAMPLE_RATE)
    # Pad or trim? Just use as is.
    inputs = {'input_values': audio[np.newaxis, :]}
    # Run Encoder
    # Expected output: audio_codes [1, 16, T] (or similar, verifying)
    # Check if export script used dynamic axes. Yes.
    ref_codes = sess_encoder.run(None, inputs)[0] # [1, 16, T]
    
    print(f"    Got Ref Codes: {ref_codes.shape}")
    T = ref_codes.shape[2]
    
    # 2. Process Reference Text
    ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
    L = len(ref_ids)
    print(f"  Ref Text Len: {L}, Codec Len: {T}")
    
    # 3. Construct ICL Embedding
    # Logic: Sum(Codes) + Text(padded)
    
    # A. Sum Codec Embeddings
    # [1, 16, T] -> [T, 16] (transpose for iteration)
    codes_T = ref_codes[0].T # [T, 16]
    
    codec_embed_sum = np.zeros((T, 2048), dtype=np.float32)
    for t in range(T):
        for layer in range(16):
            code = codes_T[t, layer]
            codec_embed_sum[t] += codec_tables[layer][code]
            
    # B. Text Embeddings with Padding
    text_embeds = []
    # Ref Text Prefix? Usually just IDs?
    # generate_icl_prompt joins ref_id + text_ids?
    # Let's assume just ref_ids here.
    
    # Pad text to length T
    tts_pad_vec = text_table[TTS_PAD_ID]
    
    for i in range(T):
        if i < L:
            tid = ref_ids[i]
            text_embeds.append(text_table[tid])
        else:
            text_embeds.append(tts_pad_vec)
            
    text_embed_arr = np.stack(text_embeds) # [T, 2048]
    
    # C. Fusion
    icl_embed = text_embed_arr + codec_embed_sum # [T, 2048]
    
    # 4. Assemble Full Prompt
    embed_list = []
    
    # H1: Role Prefix
    embed_list.append(text_table[151644])
    embed_list.append(text_table[77091])
    embed_list.append(text_table[198])
    
    # ICL Block
    for i in range(T):
        embed_list.append(icl_embed[i])
        
    # Divider? TTS_EOS?
    # In generate_icl_prompt, it ends with TTS_EOS.
    embed_list.append(text_table[TTS_EOS_ID] + codec_tables[0][CODEC_PAD_ID])
    
    # Start Target
    # Target Text
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    for tid in target_ids:
        embed_list.append(text_table[tid] + codec_tables[0][CODEC_PAD_ID])
        
    # Trailer
    embed_list.append(text_table[TTS_EOS_ID] + codec_tables[0][CODEC_PAD_ID])
    embed_list.append(text_table[TTS_PAD_ID] + codec_tables[0][CODEC_BOS_ID])
    
    final_embed = np.stack(embed_list)[np.newaxis, ...].astype(np.float32)
    return final_embed

def run_pipeline(prompt_embeds, output_name, tables):
    print(f"\n--- Running Full Pipeline for {output_name} ---")
    
    # 0. Load Aux
    master_head_weight = np.load(MASTER_HEAD_PATH)
    predictor_heads = np.load(HEADS_PATH)
    
    sess_options = ort.SessionOptions()
    sess_craftsman = ort.InferenceSession(CRAFTSMAN_PATH, sess_options, providers=['CPUExecutionProvider'])
    sess_mouth = ort.InferenceSession(MOUTH_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 1. Init Master
    gguf_model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 8192 # Increase context for cloning
    ctx_params.embeddings = True
    gguf_ctx = nano_llama.llama_init_from_model(gguf_model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(gguf_model)
    
    # 2. Prefill
    print(f"[Master] Prefilling {prompt_embeds.shape[1]} tokens...")
    n_tokens = prompt_embeds.shape[1]
    batch = nano_llama.llama_batch_init(8192, n_embd, 1)
    
    batch.n_tokens = n_tokens
    full_embd = np.ascontiguousarray(prompt_embeds[0])
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    for k in range(n_tokens):
        batch.pos[k] = k
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        batch.logits[k] = 1 if k == n_tokens - 1 else 0
        
    ret = nano_llama.llama_decode(gguf_ctx, batch)
    if ret != 0:
        print(f"❌ Prefill Failed: {ret}")
        return
        
    out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
    result_embd = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
    current_hidden = result_embd[-1].copy().reshape(1, 1, n_embd)
    
    current_pos = n_tokens
    
    # 3. Generation
    print("[Master] Generating...")
    generated_codes = []
    
    # Retrieve TTS_PAD from table directly
    tts_pad = tables[0][TTS_PAD_ID].reshape(1, 1, 2048)
    
    try:
        for i in range(STEPS):
            # A. Master Generate
            master_logits = current_hidden[0] @ master_head_weight.T
            last_id = np.argmax(master_logits)
            
            if last_id == EOS_TOKEN_ID:
                print(f"\n🛑 EOS at step {i}")
                break
                
            print(f"Step {i}: Master Token {last_id}")
            
            # Get Embedding for Last ID (Codec 0)
            last_id_embed = tables[1][0][last_id].reshape(1, 1, n_embd) # tables[1] is codec_tables list
            
            # B. Craftsman Generate (15 Codes)
            craftsman_input = np.concatenate([current_hidden, last_id_embed], axis=1) # [1, 2, 2048]
            
            # Reset Craftsman Pasts for every step?
            # Check `46`: It defined `current_pasts` INSIDE the step loop.
            # So yes, it seems Craftsman (Flow) is stateless or resets per frame?
            # Architecture: "Flow Matching". Usually context independent per frame given condition?
            # Yes, `46` reset it.
            current_pasts = {f"past_{k}": np.zeros((1, 8, 0, 128), dtype=np.float32) for k in range(10)}
            
            step_codes = [last_id]
            step_embeds = [last_id_embed]
            
            for c_step in range(15):
                 # ONNX Inference
                inputs = {'inputs_embeds': craftsman_input}
                inputs.update(current_pasts)
                outputs = sess_craftsman.run(None, inputs)
                
                onnx_hidden = outputs[0]
                pasts = outputs[1:] # Update pasts if needed (but loop resets?)
                
                 # Update Pasts (Maybe needed for inner autoregression of 15 codes?)
                 # Yes, `past_key_values` are updated.
                for k in range(5):
                    current_pasts[f"past_{2*k}"] = pasts[2*k]
                    current_pasts[f"past_{2*k+1}"] = pasts[2*k+1]
                
                head_w = predictor_heads[c_step]
                logits = onnx_hidden[0, -1] @ head_w.T
                code = np.argmax(logits)
                
                step_codes.append(code)
                
                # Prepare next input
                code_embed = tables[1][c_step + 1][code].reshape(1, 1, n_embd)
                step_embeds.append(code_embed)
                craftsman_input = code_embed # Next input is just this code embed
                
            generated_codes.append(step_codes)
            
            # C. Glue for Next Master Step
            summed = np.sum(step_embeds, axis=0) # Sum 16 embeddings
            summed += tts_pad # Add Pad
            
            # D. Master Inference
            batch.n_tokens = 1
            full_embd = np.ascontiguousarray(summed[0])
            ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
            
            batch.pos[0] = current_pos
            current_pos += 1
            
            if nano_llama.llama_decode(gguf_ctx, batch) != 0:
                print("❌ Master Decode Failed")
                break
                
            out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
            result_embd = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd))
            current_hidden = result_embd[0].reshape(1, 1, n_embd)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by User at Step {i}. Saving partial result...")
        
    # 4. Mouth
    print(f"[Mouth] Decoding {len(generated_codes)} frames...")
    if len(generated_codes) > 0:
        codes_array = np.array(generated_codes).T[np.newaxis, ...] # [1, 16, T]
        mouth_input = codes_array.transpose(0, 2, 1).astype(np.int64) # [1, T, 16]
        
        mouth_out = sess_mouth.run(None, {'audio_codes': mouth_input})[0]
        audio = mouth_out.squeeze()
        
        out_path = os.path.join(SAVE_DIR, f"{output_name}.wav")
        sf.write(out_path, audio, 24000)
        print(f"✅ Saved to {out_path}")

def main():
    # --- Configuration ---
    # MODE: "PRESET" (Use Speaker ID) or "CLONE" (Use Ref Audio)
    MODE = "PRESET" 
    
    # Target Text to Synthesize
    TEXT = "今天天气好"
    
    # [Preset Mode Config]
    # 3065: Vivian, 3010: Uncle Fu, etc.
    SPEAKER_ID = 3065 
    
    # [Clone Mode Config]
    REF_AUDIO = "captured_audio/official_dynamic_uncle_fu.wav" 
    REF_TEXT = "今天天气真不错，适合出去钓鱼。"
    
    OUTPUT_NAME = "dynamic_hardcoded_test"
    # ---------------------

    print(f"=== Qwen3-TTS Dynamic Pipeline ===")
    print(f"Mode: {MODE}")
    print(f"Text: {TEXT}")

    tables = load_tables()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    
    prompt_embeds = None
    
    if MODE == "CLONE":
        print(f"Reference Audio: {REF_AUDIO}")
        print(f"Reference Text: {REF_TEXT}")
        
        if not os.path.exists(REF_AUDIO):
            print(f"❌ Error: Ref audio not found: {REF_AUDIO}")
            return

        sess_encoder = ort.InferenceSession(EAR_PATH, providers=['CPUExecutionProvider'])
        prompt_embeds = construct_clone_prompt(REF_AUDIO, REF_TEXT, TEXT, tokenizer, tables, sess_encoder)
        
    else: # PRESET
        print(f"Speaker ID: {SPEAKER_ID}")
        prompt_embeds = construct_preset_prompt(TEXT, SPEAKER_ID, tokenizer, tables)
        
    run_pipeline(prompt_embeds, OUTPUT_NAME, tables)

if __name__ == "__main__":
    main()
