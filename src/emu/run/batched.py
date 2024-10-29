import argparse
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from accelerate import init_empty_weights, load_checkpoint_in_model
from emu.mllm.configuration_emu3 import Emu3Config
from ..mllm.modeling_emu3 import Emu3ForCausalLM
from ..mllm.processor import PrefixConstrainedLogitsProcessor, ClassifierFreeGuidanceLogitsProcessor
from huggingface_hub import snapshot_download 
import safetensors.torch as safett
import safetensors
import glob
import time
from emu.mllm.processing_emu3 import Emu3Processor

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

def pad_to_length(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    """Pad tensor to target length along dimension 1"""
    pad_length = target_length - tensor.size(1)
    if pad_length <= 0:
        return tensor
    padding = torch.full((tensor.size(0), pad_length), pad_value, dtype=tensor.dtype, device=tensor.device)
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
        
        indices_to_keep = sorted_indices_to_keep.scatter(1, sorted_indices, sorted_indices_to_keep)
        probs = probs * indices_to_keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def manual_generate(
    model, 
    pos_input_ids,
    neg_input_ids,
    tokenizer, 
    prefix_proc, 
    cfg_proc, 
    temperature=1.0, 
    top_p=0.9, 
    callback=None
):
    """
    Manually generate tokens using past_key_values for efficiency
    Handles different lengths between positive and negative prompts
    """
    device = pos_input_ids.device
    
    # Pad sequences to same length
    max_input_length = max(pos_input_ids.size(1), neg_input_ids.size(1))
    pos_input_ids_padded = pad_to_length(pos_input_ids, max_input_length, tokenizer.pad_token_id)
    neg_input_ids_padded = pad_to_length(neg_input_ids, max_input_length, tokenizer.pad_token_id)
    
    # Create attention masks
    pos_attention_mask = create_attention_mask(pos_input_ids_padded, tokenizer.pad_token_id)
    neg_attention_mask = create_attention_mask(neg_input_ids_padded, tokenizer.pad_token_id)
    
    # Combine inputs and attention masks
    input_ids = torch.cat([pos_input_ids_padded, neg_input_ids_padded], dim=0)
    attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)
    
    # Track generated sequence
    generated = input_ids.clone()
    generated_mask = attention_mask.clone()
    
    # First forward pass with full input sequence
    outputs = model(
        input_ids=generated,
        attention_mask=generated_mask,
        use_cache=True,
        return_dict=True
    )
    
    # Get initial logits and past key values
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    
    # Process initial logits
    #if cfg_proc is not None:
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
    i = 0
    while True:
        # Forward pass with only the new token and past_key_values
        outputs = model(
            input_ids=next_tokens,
            attention_mask=generated_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        # Process logits
        #if cfg_proc is not None:
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
            callback(i, generated, next_tokens, logits, past_key_values)
        
        # Check if we've hit the EOS token for all sequences
        if (next_tokens == tokenizer.eos_token_id).all():
            break
        i += 1
            
    return generated

def parse_args():
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument("--input", type=str, default="a shiba inu", help="input text")
    parser.add_argument("--num_images", type=int, default=1, help="number of images to generate")
    parser.add_argument("--output_file", type=str, default="generated_tokens.pt", help="output file for tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling threshold")
    return parser.parse_args()

def generation_callback(step, generated, next_tokens, logits, past_key_values=None):
    """
    Example callback function with access to past_key_values
    """
    print(f"Step {step}:")
    print(f"Next tokens: {next_tokens.tolist()}")
    print(f"Current sequence length: {generated.size(1)}")
    
    if past_key_values is not None:
        num_layers = len(past_key_values)
        kv_size = past_key_values[0][0].size()
        print(f"Past KV state: {num_layers} layers, size: {kv_size}")

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
    load_checkpoint_in_model(model, model_dl, device_map={"model": 0, "lm_head": 0}, dtype=torch.bfloat16)
    #state_dict = {}
    #for filepath in glob.glob(f"{model_dl}/*.safetensors"):
    #    state_dict.update(safett.load_file(filepath, device="cuda:0"))
    
    #print("Loading state dict to model")
    ## Load model on single GPU
    #model.load_state_dict(state_dict, assign=True)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()  # Set to evaluation mode
    end_t = time.time()
    print(f"State dict loaded in {end_t - start_t:.2f}s")
    
    print("Preparing")
    # Wrap model with accelerator for data parallelism
    
    # Initialize processors and tokenizers on the same device as the model
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    
    # Prepare input
    pos_prompt = input_text
    neg_prompt = ""
    classifier_free_guidance = 3.0
    
    kwargs = dict(
        mode='G',
        ratio=["1:1"],
        image_area=config.image_area,
        return_tensors="pt",
        padding="longest",
    )
    
    pos_inputs = processor(text=[pos_prompt] * num_images, **kwargs)
    neg_inputs = processor(text=[neg_prompt] * num_images, **kwargs)
    
    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    prefix_proc = PrefixConstrainedLogitsProcessor(constrained_fn)
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(classifier_free_guidance)
    
    # Prepare input tensors - now keeping pos and neg separate
    pos_input_ids = pos_inputs.input_ids.to(device).repeat(num_images, 1)
    neg_input_ids = neg_inputs.input_ids.to(device).repeat(num_images, 1)
    
    # Manual generation
    with torch.no_grad():
        generated_tokens = manual_generate(
            model=model,
            pos_input_ids=pos_input_ids,
            neg_input_ids=neg_input_ids,
            tokenizer=tokenizer,
            prefix_proc=prefix_proc,
            cfg_proc=cfg_proc,
            temperature=args.temperature,
            top_p=args.top_p,
            callback=generation_callback
        )
    
    generated_tokens = generated_tokens[:num_images]
    breakpoint()
    torch.save(generated_tokens, args.output_file)
    print(f"Generated tokens saved to {args.output_file}")
    print(f"Token shape: {generated_tokens.shape}")
    print(f"Sample tokens: {generated_tokens[0][:10]}")
    
    return generated_tokens

if __name__ == "__main__":
    main()