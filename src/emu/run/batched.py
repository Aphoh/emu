import argparse
from typing import Optional
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from torch.profiler import profile, record_function, ProfilerActivity
from accelerate import init_empty_weights, load_checkpoint_in_model
from emu.mllm.configuration_emu3 import Emu3Config
from ..mllm.modeling_emu3 import Emu3ForCausalLM
from ..mllm.processor import (
    PrefixConstrainedLogitsProcessor,
    ClassifierFreeGuidanceLogitsProcessor,
)
from huggingface_hub import snapshot_download
import time
from tqdm import tqdm
from emu.mllm.processing_emu3 import Emu3Processor
import wandb

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


def sample_from_logits(logits, temperature=1.0, top_p=0.9):
    """
    Sample next token from logits with temperature and nucleus sampling
    """
    if temperature > 0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    # Nucleus sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_keep = cumulative_probs <= top_p
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1

        indices_to_keep = sorted_indices_to_keep.scatter(
            1, sorted_indices, sorted_indices_to_keep
        )
        probs = probs * indices_to_keep
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def manual_generate(
    model,
    initial_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    prefix_proc: PrefixConstrainedLogitsProcessor,
    cfg_proc: ClassifierFreeGuidanceLogitsProcessor,
    temperature=1.0,
    top_p=0.9,
    callback=None,
):
    """
    Manually generate tokens using past_key_values for efficiency
    Handles different lengths between positive and negative prompts
    """
    device = initial_input_ids.device

    # Track generated sequence
    generated = initial_input_ids.clone()
    generated_mask = attention_mask.clone()

    # First forward pass with full input sequence
    outputs = model(
        input_ids=generated,
        attention_mask=generated_mask,
        use_cache=True,
        return_dict=True,
    )

    # Get initial logits and past key values
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    # Process initial logits
    if cfg_proc is not None:
        logits, expand_factor = cfg_proc(generated, logits)  # Pass positive samples for CFG
    if prefix_proc is not None:
        logits = prefix_proc(generated, logits)

    # Sample first new token
    next_tokens = sample_from_logits(logits, temperature, top_p)
    if expand_factor > 1:
        assert next_tokens.ndim == 2, f"Expected 2D next_tokens tensor, got {next_tokens.ndim}"
        next_tokens = next_tokens.repeat(expand_factor, 1)
    generated = torch.cat([generated, next_tokens], dim=1)
    # Extend attention mask for new token
    attention_mask_extension = torch.ones((generated.size(0), 1), device=device)
    generated_mask = torch.cat([generated_mask, attention_mask_extension], dim=1)

    # Optional callback
    if callback is not None:
        callback(0, 1, generated, next_tokens, logits)

    # Generation loop using past_key_values
    @torch.compile
    def next_logits_fn(model, input_ids, attn_mask, past_key_values):
        with torch.inference_mode():
            return model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

    start_t = time.time()
    i = 0
    pbar = tqdm()
    while True:
        step_start_t = time.time()
        pbar.update(1)
        # Forward pass with only the new token and past_key_values
        if i > 5:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("model_inference"):
                    outputs = next_logits_fn(
                        model, generated, generated_mask, past_key_values
                    )
            prof.export_chrome_trace("trace.json")
            return
        outputs = next_logits_fn(model, next_tokens, generated_mask, past_key_values)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Process logits
        # if cfg_proc is not None:
        #    logits = cfg_proc(generated[:batch_size], logits)  # Pass positive samples for CFG
        if prefix_proc is not None:
            logits = prefix_proc(generated, logits)

        # Sample next token
        next_tokens = sample_from_logits(logits, temperature, top_p)
        generated = torch.cat([generated, next_tokens], dim=1)
        # Extend attention mask for new token
        generated_mask = torch.cat([generated_mask, attention_mask_extension], dim=1)

        # Optional callback
        if callback is not None:
            pbar.set_description_str(
                callback(
                    i,
                    time.time() - step_start_t,
                    generated,
                    next_tokens,
                    logits,
                    past_key_values,
                )
            )

        # Check if we've hit the EOS token for all sequences
        if (next_tokens == tokenizer.eos_token_id).all():
            break
        i += 1

    pbar.close()
    print(f"Generation completed in {time.time() - start_t:.2f}s")

    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument("-p", "--prompt", type=str, default="a shiba inu", help="input text")
    parser.add_argument(
        "--num_images", type=int, default=1, help="number of images to generate"
    )
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    parser.add_argument("--topp", type=float, default=0.9, help="temperature")
    parser.add_argument("-m", "--method", type=str, default="nocfg", choices=["nocfg", "cfg", "jjeoni"])
    return parser.parse_args()


def generation_callback(
    step, step_time, generated, next_tokens, logits, past_key_values=None
):
    """
    Example callback function with access to past_key_values
    """
    num_tokens = next_tokens.numel()
    tps = num_tokens / step_time
    wandb.log({"tps": tps})
    return f"Step {step}: curr_len: {generated.size(1)}, tps: {tps:.2f}"


def main():
    args = parse_args()

    wandb.init(project="cfg-stuff", config=dict(args))
    prompt: str = args.prompt
    num_images: int = args.num_images
    cfg_scale: float = args.cfg_scale
    temperature: float = args.temp
    top_p: float = args.topp
    method: str = args.method


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
        EMU_HUB, trust_remote_code=True, padding_side="left"
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
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(cfg_scale)

    # Manual generation
    with torch.inference_mode():
        generated_tokens = manual_generate(
            model=model,
            initial_input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            tokenizer=tokenizer,
            prefix_proc=prefix_proc,
            cfg_proc=cfg_proc,
            temperature=temperature,
            top_p=top_p,
            callback=generation_callback,
        )

    columns = ["type", "image", "text"]
    rows = []
    for i, tokens in enumerate(generated_tokens):
        mm_list = processor.decode(tokens)
        for idx_j, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            img_type = "pos" if  i < num_images else "neg"
            rows.append([img_type, wandb.Image(im), text_inputs[i]])

    wandb.log({"images": wandb.Table(data=rows, columns=columns)})
    

if __name__ == "__main__":
    main()
