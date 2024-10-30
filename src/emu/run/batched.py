import argparse
from collections import defaultdict
from typing import Optional
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, StaticCache
from accelerate import init_empty_weights, load_checkpoint_in_model
from emu.mllm.configuration_emu3 import Emu3Config
from ..mllm.modeling_emu3 import Emu3ForCausalLM
from ..mllm.processor import (
    PrefixConstrainedLogitsProcessor,
    ClassifierFreeGuidanceLogitsProcessor,
)
from pathlib import Path
from huggingface_hub import snapshot_download
import time
from tqdm import tqdm
from emu.mllm.processing_emu3 import Emu3Processor
import wandb

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


def sample_from_logits(
    logits,
    sample: bool = True,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    return_probs: bool = False,
):
    shp = logits.shape[:-1]

    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[
            ..., :-1
        ]

        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float("inf")

    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )

    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token


def generate_step(
    *,
    model_fn,
    generated,
    attention_mask,
    temperature,
    top_p,
    top_k,
    next_tokens=None,
    past_key_values=None,
    cfg_proc=None,
    prefix_proc=None,
    extras=None,
):
    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    outputs = model_fn(
        input_ids=next_tokens if next_tokens else generated,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    # Process logits
    if cfg_proc is not None:
        logits, expand_factor = cfg_proc(generated, logits)
    if prefix_proc is not None:
        logits = prefix_proc(generated[: logits.shape[0]], logits)

    # Sample next token
    next_tokens = sample_from_logits(
        logits, temperature=temperature, top_p=top_p, top_k=top_k
    )
    if expand_factor > 1:
        next_tokens = next_tokens.repeat(expand_factor)
    new_generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
    # Extend attention mask for new token
    attention_mask_extension = torch.ones(
        (generated.size(0), 1), device=generated.device
    )
    new_mask = torch.cat([attention_mask, attention_mask_extension], dim=1)
    if extras is None:
        extras = defaultdict(list)
    if extras:
        topk_logits = logits.topk(100, dim=-1)
        extras["logit_hist_vals"].append(topk_logits.values)
        extras["logit_hist_inds"].append(topk_logits.indices)
    return new_generated, new_mask, past_key_values, extras


def manual_generate(
    model,
    initial_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    prefix_proc: PrefixConstrainedLogitsProcessor,
    cfg_proc: ClassifierFreeGuidanceLogitsProcessor,
    temperature=1.0,
    top_p=None,
    top_k=None,
    callback=None,
):
    """
    Manually generate tokens using past_key_values for efficiency
    Handles different lengths between positive and negative prompts
    """

    # Track generated sequence
    model.forward = torch.compile(model.forward, mode="reduce-overhead")
    past_key_values = None
    # past_key_values = StaticCache(
    #    config=model.config,
    #    batch_size=initial_input_ids.size(0),
    #    max_cache_len=initial_input_ids.size(1) + 9000,
    #    device=initial_input_ids.device,
    #    dtype=model.dtype,
    # )
    generated, generated_mask, past_key_values, extras = generate_step(
        model_fn=model,
        generated=initial_input_ids,
        attention_mask=attention_mask,
        cfg_proc=cfg_proc,
        prefix_proc=prefix_proc,
        past_key_values=past_key_values,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    if callback is not None:
        callback(0, 1, generated, generated[:, -1])

    start_t = time.time()
    i = 0
    pbar = tqdm()
    while True:
        step_start_t = time.time()
        pbar.update(1)
        # Forward pass with only the new token and past_key_values
        generated, generated_mask, past_key_values, extras = generate_step(
            model_fn=model,
            generated=generated,
            next_tokens=generated[:, -1].unsqueeze(-1),
            attention_mask=generated_mask,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            past_key_values=past_key_values,
            cfg_proc=cfg_proc,
            prefix_proc=prefix_proc,
            extras=extras,
        )
        # Optional callback
        if callback is not None:
            pbar.set_description_str(
                callback(
                    i,
                    time.time() - step_start_t,
                    generated,
                    generated[:, -1].unsqueeze(-1),
                    past_key_values,
                )
            )

        # Check if we've hit the EOS token for all sequences
        if (generated[:, -1] == tokenizer.eos_token_id).all():
            break
        i += 1

    pbar.close()
    print(f"Generation completed in {time.time() - start_t:.2f}s")

    return generated, extras


def parse_args():
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument(
        "-p", "--prompt", type=str, default="a shiba inu", help="input text"
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="number of images to generate"
    )
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--pag_scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_p", type=float, help="temperature")
    parser.add_argument("--top_k", type=float, default=2048, help="temperature")
    return parser.parse_args()


def generation_callback(step, step_time, generated, next_tokens, past_key_values=None):
    """
    Example callback function with access to past_key_values
    """
    num_tokens = next_tokens.numel()
    tps = num_tokens / step_time
    wandb.log({"tps": tps})
    return f"Step {step}: curr_len: {generated.size(1)}, tps: {tps:.2f}"


def main():
    args = parse_args()

    wandb.init(project="cfg-stuff", config=vars(args))
    prompt: str = args.prompt
    num_images: int = args.num_images
    cfg_scale: float = args.cfg_scale
    pag_scale: float = args.pag_scale
    temperature: float = args.temp
    top_p: float = args.top_p
    top_k: int = args.top_k

    # Model initialization with data parallelism
    model_dl = snapshot_download(EMU_HUB)
    config = Emu3Config.from_json_file(f"{model_dl}/config.json")
    config._attn_implementation = "sdpa"

    # Initialize model
    with init_empty_weights():
        model = Emu3ForCausalLM(config)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load state dict
    print("Loading state dict")
    start_t = time.time()
    load_checkpoint_in_model(
        model, model_dl, device_map={"model": 0, "lm_head": 0}, dtype=torch.bfloat16
    )
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()  # Set to evaluation mode
    end_t = time.time()
    print(f"State dict loaded in {end_t - start_t:.2f}s")

    print("Preparing")
    # Wrap model with accelerator for data parallelism

    # Initialize processors and tokenizers on the same device as the model
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = (
        AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True).to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        EMU_HUB, trust_remote_code=True, padding_side="right"
    )
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    # Prepare input
    pos_prompt = prompt
    neg_prompt = ""

    text_inputs = ([pos_prompt] * num_images) + ([neg_prompt] * num_images)
    inputs = processor(
        text=([pos_prompt] * num_images) + ([neg_prompt] * num_images),
        mode="G",
        ratio=["1:1"] * len(text_inputs),
        image_area=config.image_area,
        return_tensors="pt",
        padding="longest",
    )

    h = inputs.image_size[:, 0]
    w = inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    prefix_proc = PrefixConstrainedLogitsProcessor(constrained_fn)
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(cfg_scale, pag_scale)

    # Manual generation
    with torch.inference_mode():
        generated_tokens, extras = manual_generate(
            model=model,
            initial_input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            tokenizer=tokenizer,
            prefix_proc=prefix_proc,
            cfg_proc=cfg_proc,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            callback=generation_callback,
        )

    columns = ["type", "image", "text"]
    rows = []
    for i, tokens in enumerate(generated_tokens):
        mm_list = processor.decode(tokens)
        for idx_j, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            img_type = "pos" if i < num_images else "neg"
            rows.append([img_type, wandb.Image(im), text_inputs[i]])

    wandb.log({"images": wandb.Table(data=rows, columns=columns)})
    # Log extras and generated_tokens to wandb
    out_dir = Path(f"./outputs/{str(wandb.run.id)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(extras, out_dir / "extras.pt")
    torch.save(generated_tokens, out_dir / "generated_tokens.pt")
    wandb.save(out_dir, policy="now")


if __name__ == "__main__":
    main()
