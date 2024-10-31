
import math
from typing import Callable, List, Tuple
import torch
from transformers import LogitsProcessor


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice and Bob", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=5)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob are friends

    >>> # We can contrain it with `prefix_allowed_tokens_fn` to force a certain behavior based on a prefix.
    >>> # For instance, we can force an entire entity to be generated when its beginning is detected.
    >>> entity = tokenizer(" Bob Marley", return_tensors="pt").input_ids[0]  # 3 tokens
    >>> def prefix_allowed_tokens_fn(batch_id, input_ids):
    ...     '''
    ...     Attempts to generate 'Bob Marley' when 'Bob' is detected.
    ...     In this case, `batch_id` is not used, but you can set rules for each batch member.
    ...     '''
    ...     if input_ids[-1] == entity[0]:
    ...         return [entity[1].item()]
    ...     elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
    ...         return [entity[2].item()]
    ...     return list(range(tokenizer.vocab_size))  # If no match, allow all tokens

    >>> outputs = model.generate(**inputs, max_new_tokens=5, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob Marley
    ```
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]]):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert input_ids.shape[0] == scores.shape[0], f"input_ids {input_ids.shape} and scores {scores.shape} should have the same batch size."
        assert input_ids.ndim == 2, f"input_ids should have 2 dimensions, but got {input_ids.ndim}."
        assert scores.ndim == 2, f"scores should have 2 dimensions, but got {scores.ndim}."
        mask = torch.full_like(scores, -math.inf)
        for batch_id, sent in enumerate(input_ids):
            prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
            if len(prefix_allowed_tokens) == 0:
                raise ValueError(
                    f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                    f"This means that the constraint is unsatisfiable. Please check your implementation"
                    f"of `prefix_allowed_tokens_fn` "
                )
            mask[batch_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed

class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

    </Tip>

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.

    Examples:

    ```python
    >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

    >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    >>> inputs = processor(
    ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    ...     padding=True,
    ...     return_tensors="pt",
    ... )
    >>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    ```
    """

    def __init__(self, cfg_scale, pag_scale):
        assert cfg_scale >= 0, f"`cfg_scale` should be greater than or equal to 0, but got {cfg_scale}."
        assert pag_scale >= 0, f"`pag_scale` should be greater than or equal to 0, but got {pag_scale}."
        self.cfg_scale = cfg_scale
        self.pag_scale = pag_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> Tuple[torch.FloatTensor, int]:
        assert scores.shape[0] == input_ids.shape[0], f"input_ids {input_ids.shape} and scores {scores.shape} should have the same batch size."

        if self.cfg_scale > 0 and self.pag_scale > 0:
            unguided_bsz = scores.shape[0] // 3
            cond_logits, pag_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
            scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale + (pag_logits - cond_logits) * self.pag_scale
            return scores_processed, 3
        elif self.cfg_scale > 0 and self.pag_scale == 0:
            unguided_bsz = scores.shape[0] // 2
            cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
            scores_processed = cond_logits + (cond_logits - uncond_logits) * self.cfg_scale
            return scores_processed, 2
        elif self.cfg_scale == 0 and self.pag_scale > 0:
            unguided_bsz = scores.shape[0] // 2
            cond_logits, pag_logits = scores.split(unguided_bsz, dim=0)
            scores_processed = cond_logits + (pag_logits - cond_logits) * self.pag_scale
            return scores_processed, 2
        else:
            return scores_processed, 1

