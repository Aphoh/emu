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
    context_bias: float = 0.0,
    context_offset: int = 0,
    min_value: float = None,
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
        min_dtype = torch.finfo(dtype).min if min_value is None else min_value
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        if context_offset > 0 and context_bias != 0.0:
            ones = torch.ones(sequence_length, context_offset, device=device)
            context_mask = (
                torch.arange(context_offset, target_length, device=device)
                > cache_position.reshape(-1, 1) - 1
            )
            context_mask = 1 - torch.cat((ones, context_mask), dim=-1)
            context_mask = context_mask * (-context_bias)
            causal_mask += context_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)

    return causal_mask


class MaskPosGenerator:

    def __init__(
        self,
        initial_attn_mask,
        is_cfg,
        is_pag,
        pag_with_position=True,
        null_identity_map=False,
        condition_bias=None,
        dtype=torch.bfloat16,
    ):
        self.initial_attn_mask = initial_attn_mask
        self.is_pag = is_pag
        self.is_cfg = is_cfg
        self.pag_with_position = pag_with_position
        self.null_identity_map = null_identity_map
        self.condition_bias = condition_bias
        self.initial_pos_ids = torch.cumsum(initial_attn_mask, dim=1) - 1
        self.dtype = dtype

    def __call__(
        self,
        generated_ids: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        kv_size: int = None,
    ):
        assert (
            generated_ids.ndim == 2
        ), f"generated_ids should have 2 dimensions, but got {generated_ids.ndim}."
        batch_len, target_length = generated_ids.shape

        initial_seq_len = self.initial_attn_mask.shape[1]
        to_add = torch.ones(
            (batch_len, target_length - initial_seq_len),
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

        cache_position = torch.arange(attn.shape[1], device=input_ids.device)[
            -input_ids.shape[1] :
        ]

        if kv_size is not None and kv_size > target_length:
            zero_pad = torch.zeros(
                batch_len,
                kv_size - target_length,
                dtype=attn.dtype,
                device=attn.device,
            )
            attn = torch.cat(
                [attn, zero_pad],
                dim=1,
            )

        if self.null_identity_map and self.is_cfg:
            start_idx = 2 * batch_len // 3 if self.is_pag else batch_len // 2
            attn[start_idx:, :-1] = 0
            attn[start_idx:, -1] = 1

        if self.pag_with_position:
            pos_deltas = torch.arange(
                1, target_length + 1 - initial_seq_len, device=generated_ids.device
            ).repeat(batch_len, 1)
            pos_to_add = self.initial_pos_ids[:, -1].unsqueeze(1) + pos_deltas
        else:
            pos_to_add = torch.cumsum(to_add, dim=1) + self.initial_pos_ids[
                :, -1
            ].unsqueeze(1)
        pos = torch.cat([self.initial_pos_ids, pos_to_add], dim=1)[
            :, -input_ids.shape[1] :
        ]


        if self.condition_bias is not None:
            sequence_length = (
                generated_ids.shape[1] if input_ids is None else input_ids.shape[1]
            )
            attn = prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask=attn,
                sequence_length=sequence_length,
                target_length=attn.shape[1],
                dtype=self.dtype,
                device=generated_ids.device,
                cache_position=cache_position,
                batch_size=batch_len,
                context_bias=self.condition_bias,
                context_offset=self.initial_pos_ids.shape[1],
            )

        return attn, pos, cache_position


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
    kv_size = 12
    initial_attn_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    initial_attn_mask[1, 2:] = 0
    initial_attn_mask[2, 3:] = 0
    new_seq_len = seq_len + 4
    generated = torch.arange(batch_size * new_seq_len).reshape(batch_size, new_seq_len)
    input_ids = generated[:, -1].unsqueeze(1)
    print("input ids shape")
    print(input_ids.shape)

    mask, pos, _ = MaskPosGenerator(initial_attn_mask, is_cfg=True, is_pag=True)(
        generated, input_ids, kv_size
    )
    print(f"PAG + CFG Mask:\n{mask}")
    print(f"PAG + CFG Pos:\n{pos}")
    mask, pos, _ = MaskPosGenerator(
        initial_attn_mask, is_cfg=True, is_pag=True, pag_with_position=False
    )(generated, input_ids, kv_size)
    print(f"PAG nopos + CFG Mask:\n{mask}")
    print(f"PAG nopos + CFG Pos:\n{pos}")
    mask, pos, _ = MaskPosGenerator(initial_attn_mask[:2], is_cfg=False, is_pag=True)(
        generated[:2], input_ids, kv_size
    )
    print(f"PAG only Mask:\n{mask}")
    print(f"PAG only Pos:\n{pos}")
    mask, pos, _ = MaskPosGenerator(initial_attn_mask[:2], is_cfg=True, is_pag=False)(
        generated[:2], input_ids, kv_size
    )
    print(f"CFG only Mask:\n{mask}")
    print(f"CFG only Pos:\n{pos}")
    mask, pos, _ = MaskPosGenerator(
        initial_attn_mask[:2], is_cfg=True, is_pag=False, condition_bias=2
    )(generated[:2], input_ids, kv_size)
    print(f"CFG cond bias Mask:\n{mask.clip(-4, 0)}")
    print(f"CFG cond bias Pos:\n{pos}")


if __name__ == "__main__":
    test()
