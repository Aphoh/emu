import argparse
from collections import defaultdict
import os
from typing import List, Optional, Tuple
from PIL import Image
import torch
import torch.distributed
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    LlamaConfig,
    StaticCache,
    DynamicCache,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from accelerate import init_empty_weights, load_checkpoint_in_model
from emu.mllm.configuration_emu3 import Emu3Config
from emu.mllm.kvcache import ChunkedDynamicCache
from emu.run.utils import MaskPosGenerator
from ..mllm.modeling_llama import LlamaForCausalLM
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
from torch.distributed.device_mesh import init_device_mesh
from ..mllm.parallel import get_tensor_sharded_model

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
    mask_pos_generator: MaskPosGenerator,
    temperature,
    top_p,
    top_k,
    next_tokens=None,
    past_key_values: ChunkedDynamicCache = None,
    cfg_proc=None,
    prefix_proc=None,
    extras=None,
):
    input_ids = next_tokens if next_tokens is not None else generated
    kv_size = past_key_values.get_seq_length() if past_key_values is not None else None
    attention_mask, position_ids, cache_position = mask_pos_generator(generated, input_ids, kv_size)

    outputs = model_fn(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    base_logits = outputs.logits[:, -1, :].to(torch.float32)
    base_logprobs = F.log_softmax(base_logits, dim=-1)
    past_key_values = outputs.past_key_values

    if extras is None:
        extras = defaultdict(list)
    topk_logprobs = base_logprobs.topk(100, dim=-1)
    extras["logprob_hist_vals"].append(topk_logprobs.values)
    extras["logprob_hist_inds"].append(topk_logprobs.indices)

    # Process logits
    new_logits = base_logits.clone()
    if cfg_proc is not None:
        new_logits, expand_factor = cfg_proc(generated, new_logits)
    if prefix_proc is not None:
        new_logits = prefix_proc(generated[: new_logits.shape[0]], new_logits)

    # Sample next token
    next_tokens = sample_from_logits(
        new_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )
    if expand_factor > 1:
        next_tokens = next_tokens.repeat(expand_factor)

    samples_logprobs = base_logprobs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
    extras["sampled_logprobs"].append(samples_logprobs)
    if "cumprob" not in extras:
        extras["cumprob"].append(samples_logprobs)
    else:
        extras["cumprob"].append(samples_logprobs + extras["cumprob"][-1])

    new_generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
    # Extend attention mask for new token
    return new_generated, past_key_values, extras


def manual_generate(
    model,
    initial_input_ids: torch.Tensor,
    mask_pos_generator: MaskPosGenerator,
    tokenizer,
    prefix_proc: PrefixConstrainedLogitsProcessor,
    cfg_proc: ClassifierFreeGuidanceLogitsProcessor,
    temperature=1.0,
    top_p=None,
    top_k=None,
    callback=None,
    config=None,
    tp_rank=0,
    callback_kwargs={},
):
    """
    Manually generate tokens using past_key_values for efficiency
    Handles different lengths between positive and negative prompts
    """

    kv_cache_len = initial_input_ids.shape[1] + 9000
    # make sure the cache is a multiple of 128
    kv_cache_len = (kv_cache_len // 128 + 1) * 128
    past_key_values = ChunkedDynamicCache()
    generated, past_key_values, extras = generate_step(
        model_fn=model,
        generated=initial_input_ids,
        mask_pos_generator=mask_pos_generator,
        cfg_proc=cfg_proc,
        prefix_proc=prefix_proc,
        past_key_values=past_key_values,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Bug in torch dynamo with scaled_dot_product_attention
    # torch._dynamo.disallow_in_graph(torch.nn.functional.scaled_dot_product_attention)
    # model.forward = torch.compile(model.forward, mode="reduce-overhead")

    if callback is not None:
        callback(0, 1, generated, generated[:, -1], extras, tp_rank=tp_rank, **callback_kwargs)

    start_t = time.time()
    i = 0
    pbar = tqdm(
        total=8191, disable=bool(os.environ.get("TQDM_DISABLE", False) or tp_rank != 0)
    )
    while True:
        step_start_t = time.time()
        pbar.update(1)
        # Forward pass with only the new token and past_key_values
        generated, past_key_values, extras = generate_step(
            model_fn=model,
            generated=generated,
            next_tokens=generated[:, -1].unsqueeze(-1),
            mask_pos_generator=mask_pos_generator,
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
                    extras,
                    tp_rank=tp_rank,
                    **callback_kwargs
                )
            )

        # Check if we've hit the EOS token for a sequence
        if (generated[:, -1] == tokenizer.eos_token_id).all():
            break
        elif (generated[:, -1] == tokenizer.eos_token_id).any():
            print("WOAH THIS IS A BUG")
            print("expected all to be", tokenizer.eos_token_id)
            print("got", generated[:, -1])
            break
        elif i >= 8195:
            print("Breaking due to max length")
            break
        i += 1

    pbar.close()
    print(f"Generation completed in {time.time() - start_t:.2f}s")

    return generated, extras


def parse_args():
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument(
        "-p",
        action="append",
        help="input prompts",
    )
    parser.add_argument(
        "-o",
        action="append",
        help="output directory",
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="number of images to generate"
    )
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--pag_scale", type=float, default=0.0, help="PAG scale")
    parser.add_argument("--cd_beta", type=float, default=0.0, help="PAG scale")
    parser.add_argument("--cd_alpha", type=float, default=0.5, help="PAG scale")
    parser.add_argument("--pag_no_pos", action="store_true", help="PAG don't use pos")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_p", type=float, help="top p")
    parser.add_argument("--top_k", type=int, help="top k")
    parser.add_argument(
        "--null_ident", action="store_true", help="null identity matrix in cfg"
    )
    parser.add_argument(
        "--cfg_clip_quantile", type=float, default=1.0, help="CFG clip quantile"
    )
    parser.add_argument(
        "--cfg_logsm", action="store_true", help="CFG delta use log softmax"
    )
    parser.add_argument(
        "--extras", type=str, default="extras", help="directory to store extras"
    )
    parser.add_argument(
        "--condition_bias", type=float, default=0.0, help="use context bias in generation"
    )
    parser.add_argument(
        "-t", '--tag', action='append', help="tags for wandb"
    )
    return parser.parse_args()


def generation_callback(
    step,
    step_time,
    generated,
    next_tokens,
    extras,
    tp_rank=0,
    is_cfg=False,
    is_pag=False,
):
    """
    Example callback function with access to past_key_values
    """
    num_tokens = next_tokens.numel()
    tps = num_tokens / step_time

    if tp_rank == 0:
        to_log = {"tps": tps}
        cumprob = extras["cumprob"][-1]
        for i in range(cumprob.size(0)):
            to_log[f"cumprob_{i}"] = cumprob[i].item()
        sampled_logprobs = extras["sampled_logprobs"][-1]
        for i in range(sampled_logprobs.size(0)):
            to_log[f"sampled_logprob_{i}"] = sampled_logprobs[i].item()
        if is_cfg and not is_pag:
            n_samp = cumprob.size(0) // 2
            cumprob_delta = (cumprob[:n_samp] - cumprob[n_samp:])
            logprob_delta = (sampled_logprobs[:n_samp] - sampled_logprobs[n_samp:])
            for i in range(cumprob_delta.size(0)):
                to_log[f"cumprob_delta_{i}"] = cumprob_delta[i].item()
                to_log[f"logprob_delta_{i}"] = logprob_delta[i].item()
        wandb.log(to_log)

    return f"Step {step}: curr_len: {generated.size(1)}, tps: {tps:.2f}"


def initialize_model(
    tp_rank, device_mesh, device
) -> Tuple[LlamaForCausalLM, Emu3Config, LlamaConfig]:
    # Model initialization with data parallelism
    model_dl = None
    if tp_rank == 0:
        model_dl = snapshot_download(EMU_HUB)
    if device_mesh is not None:
        torch.distributed.barrier()
    if model_dl is None:
        model_dl = snapshot_download(EMU_HUB)
    config: Emu3Config = Emu3Config.from_json_file(f"{model_dl}/config.json")
    llama_config: LlamaConfig = config.to_llama()

    # Initialize model
    with init_empty_weights():
        model = LlamaForCausalLM(llama_config)

    # Load state dict
    print("Loading state dict")
    start_t = time.time()
    dev_str = f"cuda:{tp_rank}"
    load_checkpoint_in_model(
        model,
        model_dl,
        device_map={"model": dev_str, "lm_head": dev_str},
        dtype=torch.bfloat16,
    )
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()  # Set to evaluation mode
    end_t = time.time()
    print(f"State dict loaded in {end_t - start_t:.2f}s")
    return model, config, llama_config


def main():
    args = parse_args()

    tp_rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)
    print(f"TP RANK: {tp_rank}, cuda_devs: {torch.cuda.device_count()}")
    is_tp = "RANK" in os.environ

    if tp_rank == 0:
        print("Initializing wandb")
        wandb.init(project="cfg-stuff", config=vars(args), tags=args.tag)

    device_mesh = None
    if is_tp:
        device_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))

    prompts: List[str] = args.p
    if len(prompts) == 0:
        prompts = ["a shiba inu"]
    output_dirs: List[str] = args.o
    if output_dirs:
        assert len(prompts) == len(
            output_dirs
        ), "Number of prompts and output directories must match"
    num_images: int = args.num_images
    pag_pos: bool = not args.pag_no_pos
    cfg_scale: float = args.cfg_scale
    cfg_clip_quantile: float = args.cfg_clip_quantile
    pag_scale: float = args.pag_scale
    cd_beta: float = args.cd_beta
    cd_alpha: float = args.cd_alpha
    temperature: float = args.temp
    top_p: float = args.top_p
    top_k: int = args.top_k
    extras_dir: str = args.extras
    null_ident: bool = args.null_ident
    cfg_logsm: bool = args.cfg_logsm
    condition_bias: float = args.condition_bias
    del args

    if top_k is None and top_p is None:
        print("Using default topk=2048 sampling")
        top_k = 2048

    if condition_bias != 0.0:
        assert cfg_scale == 0, "Context bias requires no CFG"
        assert pag_scale == 0, "Context bias requires no PAG"

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    # Experimental features to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True
    # torch._functorch.config.enable_autograd_cache = True

    # Initialize model
    model, config, llama_config = initialize_model(tp_rank, device_mesh, device)

    print("Preparing")
    # Wrap model with accelerator for data parallelism

    # Initialize processors and tokenizers on the same device as the model
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = (
        AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True).to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        EMU_HUB,
        trust_remote_code=True,
        padding_side="left",
    )
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    # Prepare input
    is_cfg = cfg_scale > 0
    is_pag = pag_scale > 0 or cd_beta > 0

    pos_prompts = []
    for prompt in prompts:
        pos_prompts.extend([prompt] * num_images)
    neg_prompts = [""] * len(pos_prompts)

    num_pos = 1 if not is_pag else 2  # 2x for PAG
    num_neg = 0 if not is_cfg else 1

    text_inputs = (pos_prompts * num_pos) + (neg_prompts * num_neg)
    inputs = processor(
        text=text_inputs,
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
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(
        cfg_scale, pag_scale, cd_beta=cd_beta, cfg_clip_quantile=cfg_clip_quantile, cfg_logsm=cfg_logsm, cd_alpha=cd_alpha
    )
    mask_pos_generator = MaskPosGenerator(
        inputs.attention_mask.to(device),
        is_cfg,
        is_pag,
        pag_with_position=pag_pos,
        null_identity_map=null_ident,
        condition_bias=condition_bias
    )

    if is_tp:
        model = get_tensor_sharded_model(model, device_mesh)
        print("Rank ", tp_rank, "sharded model between", device_mesh.size(), "devices")

    # Manual generation
    print("Initial input size", inputs.input_ids.size())
    print("is_cfg:", is_cfg)
    print("is_pag:", is_pag)
    print("got num prompts", len(prompts))
    print("num images", num_images)
    print("num pos prompts", len(neg_prompts))
    print("num neg prompts", len(neg_prompts))
    with torch.no_grad():
        generated_tokens, extras = manual_generate(
            model=model,
            initial_input_ids=inputs.input_ids.to(device),
            mask_pos_generator=mask_pos_generator,
            tokenizer=tokenizer,
            prefix_proc=prefix_proc,
            cfg_proc=cfg_proc,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            callback=generation_callback,
            config=llama_config,
            tp_rank=tp_rank,
            callback_kwargs=dict(is_cfg=is_cfg, is_pag=is_pag),
        )

    if tp_rank == 0:
        images = []
        for i, tokens in enumerate(generated_tokens[: num_images * len(prompts)]):
            mm_list = processor.decode(tokens)
            for idx_j, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                if output_dirs:
                    dir_idx = i // num_images  # first num_images are the first dir, etc
                    img_idx = i % num_images  # image index within the dir
                    if dir_idx >= len(output_dirs):
                        print("This shouldn't happen, saving to last dir")
                        dir_idx = len(output_dirs) - 1
                    out_dir = Path(output_dirs[dir_idx])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{img_idx:04}.png"
                    im.save(out_path)
                images.append(wandb.Image(im, file_type="jpg"))

        wandb.log({"images": images})
        # Log extras and generated_tokens to wandb
        out_dir = Path(extras_dir) / str(wandb.run.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(extras, out_dir / "extras.pt")
        torch.save(generated_tokens, out_dir / "generated_tokens.pt")
        wandb.save(str(out_dir) + "/*", policy="end")

    if is_tp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
