from typing import Optional
import torch

def prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    min_value: Optional[float] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
            `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache,
            to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class MaskPosGenerator:

    def __init__(self, initial_attn_mask, is_cfg, is_pag, pag_with_position=True, null_identity_map=False):
        self.initial_attn_mask = initial_attn_mask
        self.is_pag = is_pag
        self.is_cfg = is_cfg
        self.pag_with_position = pag_with_position
        self.null_identity_map = null_identity_map
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

        if self.null_identity_map and self.is_cfg:
            start_idx = 2 * batch_len // 3 if self.is_pag else batch_len // 2
            attn[start_idx:, :-1] = 0
            attn[start_idx:, -1] = 1

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
