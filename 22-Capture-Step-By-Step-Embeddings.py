import os
import sys
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_steps")
os.makedirs(SAVE_DIR, exist_ok=True)

class StepCounter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1

counter = StepCounter()

def talker_pre_hook(module, args, kwargs):
    """
    捕获 Transformer Backbone 的输入 (inputs_embeds)
    """
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None:
        step_idx = counter.count
        file_name = os.path.join(SAVE_DIR, f"step_{step_idx}_input_embeds.npy")
        # 转换为 float32 统一保存
        np.save(file_name, inputs_embeds.detach().cpu().to(torch.float32).numpy())
        print(f"[CAPTURE] Step {step_idx}: Saved input_embeds (Shape: {inputs_embeds.shape})")
    return None

def talker_post_hook(module, input, output):
    """
    捕获 Transformer Backbone 的输出 (last_hidden_state)
    """
    # output 是 BaseModelOutputWithPast 类型，第 0 个是 hidden_states
    hidden_states = output[0] if isinstance(output, (list, tuple)) else output.last_hidden_state
    
    step_idx = counter.count
    file_name = os.path.join(SAVE_DIR, f"step_{step_idx}_output_hidden.npy")
    np.save(file_name, hidden_states.detach().cpu().to(torch.float32).numpy())
    print(f"[CAPTURE] Step {step_idx}: Saved output_hidden (Shape: {hidden_states.shape})")
    
    # 该 Hook 执行完意味着一步推理结束
    counter.increment()
    return None

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行插桩 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            device_map=device, 
            dtype=dtype
        )
        
        # 注册 Hook 到 Talker 的 Backbone (model.talker.model)
        # Qwen3TTSTalkerModel 的 forward 结果即为 Transformer 顶层输出
        backbone = tts.model.talker.model
        handle_in = backbone.register_forward_pre_hook(talker_pre_hook, with_kwargs=True)
        handle_out = backbone.register_forward_hook(talker_post_hook)
        
        # 固定随机性
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        text = "今天天气好"
        speaker = "Vivian"
        
        print(f"开始推理并捕获: 「{text}」...")
        # 推理过程中，generate 内部会多次调用 backbone.forward
        with torch.no_grad():
            tts.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker=speaker,
                instruct="",
                **deterministic_kwargs
            )
        
        print(f"\n✅ 数据捕获完成！")
        print(f"文件保存在: {SAVE_DIR}")
        print(f"总计推理步数: {counter.count}")
        
    except Exception as e:
        print(f"❌ 捕获失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 移除 Hook
        if 'handle_in' in locals(): handle_in.remove()
        if 'handle_out' in locals(): handle_out.remove()

if __name__ == "__main__":
    main()
