import os
import sys
import torch
from qwen3_tts_gguf import logger
from qwen3_tts_gguf.codec_export import CodecEncoderExportWrapper
# 确保导入的是本地的 model definition
from qwen3_tts_gguf.modeling_tokenizer import Qwen3TTSTokenizerV2Model
# 【关键修复】Monkey Patch transformers 库的实现
from transformers.models.mimi.modeling_mimi import MimiCausalConvNet
import math

# 这是一个 Dynamo 友好的 padding 计算函数，替换库中的 math.ceil 实现
def _dynamo_friendly_get_extra_padding(self, hidden_state):
    length = hidden_state.shape[-1]
    # n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
    # ceil(n_frames) implementation:
    
    num = length - self.kernel_size + self.padding
    ceil_div = (num + self.stride - 1) // self.stride
    n_frames_ceil = ceil_div + 1
    
    ideal_length = (n_frames_ceil - 1) * self.stride + (self.kernel_size - self.padding)
    return ideal_length - length

print("Applying Monkey Patch to MimiCausalConvNet._get_extra_padding_for_conv1d...")
MimiCausalConvNet._get_extra_padding_for_conv1d = _dynamo_friendly_get_extra_padding

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    OUTPUT_DIR = r'./model'
    ONNX_PATH = os.path.join(OUTPUT_DIR, 'Qwen3-Codec-Encoder.onnx')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 路径处理
    logger.info(f"正在加载模型: {MODEL_DIR}")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    
    # 2. 加载 PyTorch 模型
    try:
        model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
        model.eval()
        
        # 【关键补丁】全局禁用 return_dict (虽然现在不再通过 encode 调用，但加上更保险)
        logger.info("Applying global config patch: return_dict=False")
        model.config.return_dict = False
        if hasattr(model, 'encoder'):
            model.encoder.config.return_dict = False
            
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 3. 准备导出包装器
    logger.info("准备 ONNX 导出包装器 (Encoder)...")
    encoder_wrapper = CodecEncoderExportWrapper(model).eval()
    
    # 4. Dummy Input
    dummy_input = torch.randn(1, 24000 * 1) 
    
    # 5. Export
    logger.info(f"开始导出 Encoder ONNX 到: {ONNX_PATH}")
    
    # 【Direct Export Mode】
    # 我们不仅修复了 ModelOutput 问题 (V3 Wrapper)，还修复了 Dynamo Control Flow 问题 (modeling patch).
    # 现在可以直接使用 torch.onnx.export (Direct/Dynamo mode).
    USE_JIT_TRACE = False 
    
    try:
        if USE_JIT_TRACE:
            logger.info("Strategy: Using JIT Trace...")
            export_module = torch.jit.trace(encoder_wrapper, (dummy_input,), check_trace=False, strict=False)
        else:
            logger.info("Strategy: Direct Export (Dynamo)...")
            export_module = encoder_wrapper

        torch.onnx.export(
            export_module,
            (dummy_input,),
            ONNX_PATH,
            input_names=['input_values'],
            output_names=['audio_codes'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'audio_codes': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14, # 14 is safe. 
            do_constant_folding=True
        )
        logger.info("✅ Encoder ONNX 导出成功！")
        
    except Exception as e:
        logger.error(f"❌ Encoder ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 验证
    logger.info("验证导出结果...")
    try:
        import onnxruntime
        import numpy as np
        
        sess = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        
        with torch.no_grad():
            torch_out = encoder_wrapper(dummy_input).numpy()
            
        onnx_out = sess.run(None, {'input_values': dummy_input.numpy()})[0]
        
        logger.info(f"Shapes: PyTorch {torch_out.shape} vs ONNX {onnx_out.shape}")
        
        # 允许一定误差，因为 Dynamo 可能会做优化
        if np.allclose(torch_out, onnx_out, atol=1e-3):
             logger.info("数值验证通过！")
        else:
            diff = np.max(np.abs(torch_out - onnx_out))
            logger.warning(f"数值验证存在差异，最大误差: {diff:.6f}")
            
    except ImportError:
        logger.warning("未安装 onnxruntime，跳过验证。")

if __name__ == "__main__":
    main()
