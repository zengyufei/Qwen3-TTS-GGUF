import torch
import torch.nn as nn
from . import logger

# 引入本地模型定义
try:
    from .modeling_tokenizer import Qwen3TTSTokenizerV2Model
except ImportError:
    Qwen3TTSTokenizerV2Model = None
    logger.warning("Could not import Qwen3TTSTokenizerV2Model from .modeling_tokenizer")

class CodecEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频编码器导出包装类。
    
    【核心修复 V3】
    针对 JIT Trace 报 `unordered_map` 的问题，这通常是因为 Transformers 的子模块
    （MimiEncoder 或 Quantizer）内部返回了 ModelOutput 对象（类似字典）。
    
    本类采取以下策略：
    1. 强制覆盖 config.return_dict = False。
    2. 对于 Quantizer 这种可能不遵守 config 的组件，尝试通过属性访问手动剥离对象。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.config = model.config
        
        # 1. 强力覆写配置，告诉所有子模块：只要 Tuples，不要 Dict/Object！
        self.config.return_dict = False
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'config'):
            model.encoder.config.return_dict = False
        
        # 2. 提取核心模块
        self.mimi_model = model.encoder
        self.cnn_encoder = self.mimi_model.encoder 
        self.quantizer = self.mimi_model.quantizer
        
        # 3. 锁定为评估模式
        self.cnn_encoder.eval()
        self.quantizer.eval()

    def forward(self, input_values):
        """
        Args:
            input_values (torch.FloatTensor): Shape [Batch, Time]
        Returns:
            audio_codes (torch.LongTensor): Shape [Batch, Time, Q]
        """
        # 1. Input: [B, T] -> [B, 1, T]
        x = input_values.unsqueeze(1)
        
        # 2. Run CNN Encoder
        # 即使 config.return_dict=False，MimiEncoder 有时还是会返回 tuple(BaseModelOutput) 或类似的怪东西
        # 我们这里假设它返回 tuple
        cnn_out = self.cnn_encoder(x)
        
        # 极为防御性的解包逻辑
        if isinstance(cnn_out, tuple):
            hidden_states = cnn_out[0]
        elif hasattr(cnn_out, 'last_hidden_state'):
            hidden_states = cnn_out.last_hidden_state
        else:
            hidden_states = cnn_out

        # 3. Run Quantizer
        # MimiVectorQuantizer.encode(hidden_states)
        # 它的返回值通常是 VectorQuantizerOutput(audio_codes=..., audio_values=...)
        # 或者是 tuple (audio_codes, audio_values)
        try:
            # 尝试强制不返回字典（部分版本支持此参数）
            q_out = self.quantizer.encode(hidden_states, return_dict=False)
        except TypeError:
            # 如果不支持该参数，直接调用
            q_out = self.quantizer.encode(hidden_states)
        
        # 再次防御性解包
        if isinstance(q_out, tuple):
            codes = q_out[0]
        elif hasattr(q_out, 'audio_codes'):
            codes = q_out.audio_codes
        else:
            # 假设只有一个返回
            codes = q_out

        # 4. Qwen3 Slice Logic
        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:, :valid_q, :]
        
        # 5. Transpose: [B, Q, T] -> [B, T, Q]
        codes = codes.transpose(1, 2)
        
        return codes

class CodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器导出包装类 (保持不变)。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        # Input: [B, T, Q] -> [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        wav = self.decoder(codes)
        # Output: [B, 1, T] -> [B, T]
        return wav.squeeze(1)
