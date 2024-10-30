import torch


class MaskGenerator:

    def __init__(self, initial_attn_mask, is_cfg, is_pag):
        self.initial_attn_mask = initial_attn_mask
        self.is_pag = is_pag
        self.is_cfg = is_cfg

    def __call__(self, generated_ids):
        assert generated_ids.ndim == 2, f"generated_ids should have 2 dimensions, but got {generated_ids.ndim}."
        batch_len, seq_len = generated_ids.shape
        
        initial_seq_len = self.initial_attn_mask.shape[1]
        to_add = torch.ones((batch_len, seq_len - initial_seq_len), dtype=torch.long)
        if self.is_pag and self.is_cfg:
            pag_batch_inds = 2 * batch_len // 3
            to_add[pag_batch_inds:, :-1] = 0
        elif self.is_pag and not self.is_cfg:
            pag_batch_inds = batch_len // 2
            to_add[pag_batch_inds:, :-1] = 0
        
        return torch.cat([self.initial_attn_mask, to_add], dim=1)


def pag_attn_mask_generator(generated_ids):
    assert generated_ids.ndim == 2, f"generated_ids should have 2 dimensions, but got {generated_ids.ndim}."
    assert generated_ids.shape[0] % 3 == 0, f"generated_ids should have a multiple of 3 rows, but got {generated_ids.shape[0]}."

