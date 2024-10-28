
import math
from typing import Callable, List
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

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

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

    def __init__(self, guidance_scale):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # simple check to make sure we have compatible batch sizes between our
        # logits scores (cond + uncond) and input ids (cond only)
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores_processed

