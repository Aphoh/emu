import torch


class MaskPosGenerator:

    def __init__(self, initial_attn_mask, is_cfg, is_pag, pag_with_position=True):
        self.initial_attn_mask = initial_attn_mask
        self.is_pag = is_pag
        self.is_cfg = is_cfg
        self.pag_with_position = pag_with_position
        self.initial_pos_ids = torch.cumsum(initial_attn_mask, dim=1) - 1

    def __call__(self, generated_ids: torch.LongTensor):
        assert (
            generated_ids.ndim == 2
        ), f"generated_ids should have 2 dimensions, but got {generated_ids.ndim}."
        batch_len, seq_len = generated_ids.shape

        initial_seq_len = self.initial_attn_mask.shape[1]
        to_add = torch.ones(
            (batch_len, seq_len - initial_seq_len),
            dtype=torch.long,
            device=generated_ids.device,
        )
        if self.is_pag and self.is_cfg:
            chunk_size = batch_len // 3
            to_add[chunk_size : 2 * chunk_size, :-1] = 0
        elif self.is_pag and not self.is_cfg:
            pag_batch_inds = batch_len // 2
            to_add[pag_batch_inds:, :-1] = 0

        attn = torch.cat([self.initial_attn_mask, to_add], dim=1)

        if self.pag_with_position:
            pos_deltas = torch.arange(1, seq_len + 1 - initial_seq_len, device=generated_ids.device).repeat(batch_len, 1)
            pos_to_add = self.initial_pos_ids[:, -1].unsqueeze(1) + pos_deltas
        else:
            pos_to_add = torch.cumsum(to_add, dim=1) + self.initial_pos_ids[:, -1].unsqueeze(1)
        pos = torch.cat([self.initial_pos_ids, pos_to_add], dim=1)

        return attn, pos


def pag_attn_mask_generator(generated_ids):
    assert (
        generated_ids.ndim == 2
    ), f"generated_ids should have 2 dimensions, but got {generated_ids.ndim}."
    assert (
        generated_ids.shape[0] % 3 == 0
    ), f"generated_ids should have a multiple of 3 rows, but got {generated_ids.shape[0]}."


def test():
    batch_size = 3
    seq_len = 5
    initial_attn_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    initial_attn_mask[1, 2:] = 0
    initial_attn_mask[2, 3:] = 0
    new_seq_len = seq_len + 4
    generated = torch.arange(batch_size * new_seq_len).reshape(batch_size, new_seq_len)

    
    mask, pos = MaskPosGenerator(initial_attn_mask, is_cfg=True, is_pag=True)(generated)
    print(f"PAG + CFG Mask:\n{mask}")
    print(f"PAG + CFG Pos:\n{pos}")
    mask, pos = MaskPosGenerator(initial_attn_mask, is_cfg=True, is_pag=True, pag_with_position=False)(generated)
    print(f"PAG nopos + CFG Mask:\n{mask}")
    print(f"PAG nopos + CFG Pos:\n{pos}")
    mask, pos = MaskPosGenerator(initial_attn_mask[:2], is_cfg=False, is_pag=True)(generated[:2])
    print(f"PAG only Mask:\n{mask}")
    print(f"PAG only Pos:\n{pos}")
    mask, pos = MaskPosGenerator(initial_attn_mask[:2], is_cfg=True, is_pag=False)(generated[:2])
    print(f"CFG only Mask:\n{mask}")
    print(f"CFG only Pos:\n{pos}")


if __name__ == "__main__":
    test()  
