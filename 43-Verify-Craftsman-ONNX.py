import os
import sys
import numpy as np
import onnxruntime as ort

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")
HEADS_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")

def compare(official, onnx_val, name, step):
    official = official.flatten()
    onnx_val = onnx_val.flatten()
    mae = np.mean(np.abs(official - onnx_val))
    cos_sim = np.dot(official, onnx_val) / (np.linalg.norm(official) * np.linalg.norm(onnx_val) + 1e-9)
    pass_mark = "✅" if mae < 1e-4 and cos_sim > 0.9999 else "⚠️"
    print(f"  {pass_mark} [Step {step}] {name: <15} MAE: {mae:.6f}, CosSim: {cos_sim:.6f}")
    return mae, cos_sim

def run_verification():
    print(f"Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    print(f"Loading Predictor Heads: {HEADS_PATH}")
    predictor_heads = np.load(HEADS_PATH) # (15, 3072, 2048)
    
    # 准备初始状态
    # 第 0 步是 Prefill，输入是 2048 维的投影向量
    # 后续步骤是增量输入
    
    # 获取步数数据
    num_steps = 15 
    
    # 模拟 KV Cache 传递
    # 初始 KV 为空 (1, 8, 0, 128)
    current_pasts = {f"past_{i}": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(10)}
    
    print(f"\n开始工匠 (Craftsman) ONNX 对齐验证 (共 {num_steps} 步)...\n")
    
    for step in range(num_steps):
        # 1. 准备输入
        # 使用捕获的 2048 维原始输入 (进入投影层之前的)
        input_2048 = np.load(os.path.join(CAPTURED_DIR, f"step_{step}_input_2048.npy"))
        
        inputs = {"inputs_embeds": input_2048}
        inputs.update(current_pasts)
        
        # 2. 运行 ONNX
        outputs = sess.run(None, inputs)
        onnx_hidden = outputs[0]
        onnx_presents = outputs[1:]
        
        # 3. 验证 Hidden State
        official_hidden = np.load(os.path.join(CAPTURED_DIR, f"step_{step}_output_hidden.npy"))
        compare(official_hidden, onnx_hidden, "HiddenState", step)
        
        # 4. 验证 KV Cache (Present)
        for i in range(5): # 5 层 Transformer
            official_k = np.load(os.path.join(CAPTURED_DIR, f"step_{step}_layer_{i}_k_present.npy"))
            official_v = np.load(os.path.join(CAPTURED_DIR, f"step_{step}_layer_{i}_v_present.npy"))
            
            compare(official_k, onnx_presents[i*2], f"L{i}_K_Present", step)
            compare(official_v, onnx_presents[i*2+1], f"L{i}_V_Present", step)
            
            # 更新过去的状态用于下一步
            current_pasts[f"past_{i*2}"] = onnx_presents[i*2]
            current_pasts[f"past_{i*2+1}"] = onnx_presents[i*2+1]
            
        # 5. 验证该步的最终 Token 预测 (Logits)
        # 官方代码是: self.lm_head[i](hidden_states)
        # 这里 step 对应的是预测第几个 code group
        head_weight = predictor_heads[step] # (3072, 2048)
        # hidden 为 (1, 1, 2048)
        onnx_logits = onnx_hidden[0, -1] @ head_weight.T
        
        # 注意：官方 23 号脚本没直接存 Logits，我们可以从 hidden_state 计算
        # 此处主要验证 head 权重加载和矩阵乘法的正确性
        print(f"  ℹ️ [Step {step}] Token Logits calculated (Dim: {onnx_logits.shape})")
        
        print("-" * 60)

    print("\n工匠 (Craftsman) 代码预测器验证完成。")

if __name__ == "__main__":
    run_verification()
