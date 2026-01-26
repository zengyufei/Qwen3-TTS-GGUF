import os
import sys
import torch
import numpy as np
import soundfile as sf
import onnxruntime as ort
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 全局变量用于捕获
captured_codes = None

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ONNX_DECODER_PATH = os.path.abspath("model/Qwen3-Codec-Decoder.onnx")
    
    if not os.path.exists(ONNX_DECODER_PATH):
        print(f"Error: ONNX Decoder not found at {ONNX_DECODER_PATH}")
        return

    print("Loading official model...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # --- 拦截逻辑 ---
    original_decode = tts.model.speech_tokenizer.decode
    
    def intercepted_decode(list_of_dict, **kwargs):
        global captured_codes
        # list_of_dict looks like [{"audio_codes": tensor}, ...]
        captured_codes = list_of_dict[0]["audio_codes"]
        print(f"\n[INTERCEPT] Captured audio codes with shape: {captured_codes.shape}")
        return original_decode(list_of_dict, **kwargs)
    
    # Monkey patch
    tts.model.speech_tokenizer.decode = intercepted_decode
    
    # --- 运行推理生成 Codes ---
    print("Running official generation for '今天天气好'...")
    official_wavs, sr = tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        language="Chinese"
    )
    official_audio = official_wavs[0]
    sf.write("35_official_audio.wav", official_audio, sr)
    print("Official audio saved.")
    
    # --- 使用 ONNX Decoder 解码 ---
    print(f"Loading ONNX Decoder from {ONNX_DECODER_PATH}...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(ONNX_DECODER_PATH, providers=providers)
    
    # 准备输入: ONNX 需要的形状是 [B, T, Q], 而我们的 captured_codes 通常是 [T, Q]
    # 注意：我们的导出脚本定义 dummy_input 是 (1, 100, num_quantizers)
    input_codes = captured_codes.detach().cpu().numpy()
    if input_codes.ndim == 2:
        input_codes = np.expand_dims(input_codes, axis=0) # [1, T, Q]
    
    # 强制转为 int64，因为 ONNX input_ids 通常是 long
    input_codes = input_codes.astype(np.int64)
    
    print("Decoding using ONNX...")
    # 运行 ONNX 推理
    # input_names=['audio_codes'], output_names=['audio_values']
    onnx_out = sess.run(None, {'audio_codes': input_codes})[0]
    
    # ONNX 输出形状通常是 [1, T_wave]
    onnx_audio = onnx_out.flatten()
    sf.write("35_onnx_decoded_audio.wav", onnx_audio, sr)
    print("ONNX decoded audio saved.")
    
    # --- 比较 ---
    # 两个音频长度可能不完全一致，取最小长度对齐
    min_len = min(len(official_audio), len(onnx_audio))
    official_cut = official_audio[:min_len]
    onnx_cut = onnx_audio[:min_len]
    
    diff = np.abs(official_cut - onnx_cut)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    # 计算 信噪比 SNR (Signal-to-Noise Ratio)
    signal_power = np.mean(official_cut**2)
    noise_power = np.mean(diff**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
    
    print(f"\n--- Verification Results ---")
    print(f"Max Absolute Difference: {max_diff:.8f}")
    print(f"Mean Absolute Difference: {avg_diff:.8f}")
    print(f"SNR (Signal-to-Noise Ratio): {snr:.2f} dB")
    
    print("\nSample Values (Official vs ONNX):")
    for i in range(min(10, min_len)):
        print(f"[{i}]: Off={official_cut[i]:.6f}, ONNX={onnx_cut[i]:.6f}, Diff={diff[i]:.6f}")
    
    if snr > 40:
        print("\n✅ SUCCESS: The ONNX Decoder is highly accurate (SNR > 40dB)!")
    elif snr > 20:
        print("\n✅ SUCCESS: The ONNX Decoder is decent (SNR > 20dB), likely precision/padding diffs.")
    else:
        print("\n⚠️ WARNING: Significant difference detected.")

if __name__ == "__main__":
    main()
