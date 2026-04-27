"""
predictor.py - Predictor inference wrapper.
"""
import numpy as np
from . import llama


class Predictor:
    """
    Encapsulates Predictor model behavior.
    """

    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets

    def predict_frame(
        self,
        master_hidden: np.ndarray,
        code_0: int,
        sampler=None,
        do_sample=True,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
    ) -> tuple:
        """
        For one master hidden state + first code, generate a full 16-code frame.
        Returns:
            step_codes: np.ndarray shape [16]
            summed_embed: np.ndarray shape [hidden]
        """
        proj = self.assets.proj
        if proj is not None:
            # 1.7B path: project 2048 -> 1024 for predictor input.
            m_h_1024 = master_hidden @ proj["weight"].T + proj["bias"]
        else:
            # 0.6B path: already 1024.
            m_h_1024 = master_hidden

        step_codes = np.empty(16, dtype=np.int64)
        step_codes[0] = code_0
        summed_embed = self.assets.get_codec_embedding(0, code_0).copy()

        c_in = np.empty((2, m_h_1024.shape[0]), dtype=m_h_1024.dtype)
        c_in[0] = m_h_1024
        c_in[1] = self.assets.get_codec_embedding_1024(0, code_0)

        self.ctx.clear_kv_cache()
        self.batch.set_embd(c_in, pos=0, seq_id=0)
        self.ctx.decode(self.batch)

        local_sampler = False
        if sampler is None:
            local_sampler = True
            if do_sample:
                sampler = llama.LlamaSampler(temperature=temperature, top_p=top_p, top_k=top_k)
            else:
                sampler = llama.LlamaSampler(temperature=0)

        try:
            for cs in range(1, 16):
                start_offset = (cs - 1) * 2048
                end_offset = cs * 2048
                token_id = sampler.sample(self.ctx, limit_start=start_offset, limit_end=end_offset)
                c = token_id - start_offset

                step_codes[cs] = c
                summed_embed += self.assets.get_codec_embedding(cs, c)

                if cs < 15:
                    emb_1024 = self.assets.get_codec_embedding_1024(cs, c).reshape(1, -1)
                    self.batch.set_embd(emb_1024, pos=cs + 1, seq_id=0)
                    self.ctx.decode(self.batch)
        finally:
            if local_sampler:
                sampler.free()

        return step_codes, summed_embed
