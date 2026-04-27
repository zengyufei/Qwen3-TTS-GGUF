"""
talker.py - Talker inference wrapper.
"""
import numpy as np


class TalkerPredictor:
    """
    Encapsulates Talker model behavior.
    """

    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets

        self.n_ctx = 4096
        self.cur_pos = 0

        # Pre-allocated buffers for step decode.
        self.pos_step_buffer = np.zeros(4, dtype=np.int32)
        self.fused_step_buffer = np.empty(self.assets.text_table.shape[1], dtype=np.float32)
        self.fused_step_view = self.fused_step_buffer.reshape(1, -1)

        self.trailing_text_pool = None
        self.step_idx = 0

    def clear_memory(self):
        """Clear full KV cache."""
        self.ctx.clear_kv_cache()
        self.cur_pos = 0
        self.trailing_text_pool = None
        self.step_idx = 0

    def prefill(self, pdata, seq_id: int = 0) -> np.ndarray:
        """
        Feed full initial prompt.
        """
        prompt_embeds = pdata.embd
        n_p = prompt_embeds.shape[1]

        if pdata.trailing_text_embd is not None:
            self.trailing_text_pool = pdata.trailing_text_embd[0]
        else:
            self.trailing_text_pool = None
        self.step_idx = 0

        pos_base = np.arange(self.cur_pos, self.cur_pos + n_p, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_p, dtype=np.int32)])

        self.batch.set_embd(prompt_embeds[0], pos=pos_arr, seq_id=seq_id)

        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Talker Prefill Decode failed with status {llama_status} at pos {self.cur_pos}")

        hidden_dim = self.assets.text_table.shape[1]
        hidden_ptr = self.ctx.get_embeddings()
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(n_p, hidden_dim))[-1].copy()

        self.cur_pos += n_p
        return hidden

    def decode_step(self, audio_embed: np.ndarray, seq_id: int = 0) -> np.ndarray:
        """
        One-step decode: fuse [audio + trailing text] and feed Talker.
        """
        if self.cur_pos >= self.n_ctx - 1:
            raise IndexError(f"Talker context overflow: {self.cur_pos} >= {self.n_ctx}")

        if self.trailing_text_pool is not None and self.step_idx < len(self.trailing_text_pool):
            text_vec = self.trailing_text_pool[self.step_idx]
        else:
            text_vec = self.assets.tts_pad

        np.add(audio_embed, text_vec, out=self.fused_step_buffer, casting="unsafe")
        self.step_idx += 1

        self.pos_step_buffer[0:3] = self.cur_pos
        self.batch.set_embd(self.fused_step_view, pos=self.pos_step_buffer, seq_id=seq_id)

        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Talker Step Decode failed at pos {self.cur_pos}")

        hidden_dim = self.assets.text_table.shape[1]
        hidden_ptr = self.ctx.get_embeddings()
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(1, hidden_dim))[0].copy()

        self.cur_pos += 1
        return hidden
