import argparse
from typing import Optional
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
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

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


def pad_to_length(
    tensor: torch.Tensor, target_length: int, pad_value: int
) -> torch.Tensor:
    """Pad tensor to target length along dimension 1"""
    pad_length = target_length - tensor.size(1)
    if pad_length <= 0:
        return tensor
    padding = torch.full(
        (tensor.size(0), pad_length),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create attention mask for padded sequence"""
    return (input_ids != pad_token_id).long()


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
    initial_input_ids,
    tokenizer,
    prefix_proc,
    cfg_proc,
    temperature=1.0,
    top_p=0.9,
    callback=None,
):
    """
    Manually generate tokens using past_key_values for efficiency
    Handles different lengths between positive and negative prompts
    """
    device = initial_input_ids.device

    # Pad sequences to same length
    input_length = initial_input_ids.size(1)
    input_ids = pad_to_length(initial_input_ids, input_length, tokenizer.pad_token_id)
    attention_mask = create_attention_mask(input_ids, tokenizer.pad_token_id)

    # Track generated sequence
    generated = input_ids.clone()
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
    # if cfg_proc is not None:
    #    logits = cfg_proc(generated[:batch_size], logits)  # Pass positive samples for CFG
    if prefix_proc is not None:
        logits = prefix_proc(generated, logits)

    # Sample first new token
    next_tokens = sample_from_logits(logits, temperature, top_p)
    generated = torch.cat([generated, next_tokens], dim=1)
    # Extend attention mask for new token
    attention_mask_extension = torch.ones((generated.size(0), 1), device=device)
    generated_mask = torch.cat([generated_mask, attention_mask_extension], dim=1)

    # Optional callback
    if callback is not None:
        callback(0, generated, next_tokens, logits)

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
        pbar.update(1)
        # Forward pass with only the new token and past_key_values
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
                callback(i, generated, next_tokens, logits, past_key_values)
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
    parser.add_argument("--input", type=str, default="a shiba inu", help="input text")
    parser.add_argument(
        "--num_images", type=int, default=1, help="number of images to generate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_tokens.pt",
        help="output file for tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="nucleus sampling threshold"
    )
    return parser.parse_args()


def generation_callback(step, generated, next_tokens, logits, past_key_values=None):
    """
    Example callback function with access to past_key_values
    """
    return f"Step {step}: curr_len: {generated.size(1)}, next_tokens: {next_tokens.tolist()}"


def main():
    args = parse_args()
    input_text: Optional[str] = args.input
    num_images: int = args.num_images

    # Model initialization with data parallelism
    model_dl = snapshot_download(EMU_HUB)
    config = Emu3Config.from_json_file(f"{model_dl}/config.json")

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
    pos_prompt = input_text
    neg_prompt = ""
    classifier_free_guidance = 3.0

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
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(classifier_free_guidance)

    # Manual generation
    with torch.inference_mode():
        generated_tokens = manual_generate(
            model=model,
            initial_input_ids=inputs.input_ids.to(device),
            tokenizer=tokenizer,
            prefix_proc=prefix_proc,
            cfg_proc=cfg_proc,
            temperature=args.temperature,
            top_p=args.top_p,
            callback=generation_callback,
        )

    for i, tokens in enumerate(generated_tokens):
        mm_list = processor.decode(tokens)
        for idx_j, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            print(f"Saving generated image {i}_{idx_j}")
            im.save(f"generated_{i}_{idx_j}.png")

    return generated_tokens


if __name__ == "__main__":
    main()
