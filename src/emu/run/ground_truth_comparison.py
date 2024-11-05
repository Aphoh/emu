
import argparse
import wandb
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
)

from emu.mllm.processing_emu3 import Emu3Processor
from emu.run.batched import initialize_model
from emu.tokenizer.image_processing_emu3visionvq import Emu3VisionVQImageProcessor
from emu.tokenizer.modeling_emu3visionvq import Emu3VisionVQModel
import torch.nn.functional as F

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Process an image file.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    device = torch.device("mps")

    # Initialize wandb
    wandb.init(project='ground_truth_comparison')

    # Load the image
    image = Image.open(args.image_path)

    # Log the image to wandb
    wandb.log({"input_image": wandb.Image(image)})

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        EMU_HUB,
        trust_remote_code=True,
        padding_side="left",
    )
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    image = Image.open(args.image_path).convert("RGB")
    assert image.size[0] == image.size[1], "Image must be square"
    image = image.resize((720, 720))

    image = image_processor(image, return_tensors="pt")["pixel_values"]
    pos_condition = "Wonderful historic villa with direct access to the lake, private wharf and boat house - 2"
    null_condition = ""
    tokens = image_tokenizer.encode(image.to(device))
    imgstr = processor.to_imgstr(tokens[0])
    h, w = processor.calculate_generate_size("1:1", 518400, image_tokenizer.spatial_scale_factor)
    prompts = [(
        tokenizer.bos_token +
        text +
        tokenizer.boi_token +
        processor.prefix_template.format(H=h, W=w) +
        tokenizer.img_token +
        imgstr +
        tokenizer.eol_token +
        tokenizer.eoi_token +
        tokenizer.eos_token

    ) for text in [null_condition, pos_condition]]
    tokenized = tokenizer(prompts, return_tensors='pt', padding_side="left")
    

    # Logits are shape [2, seq_len, vocab_size]
    img_token_idx = torch.nonzero(tokenized.input_ids[0] == tokenizer.img_token_id, as_tuple=True)[0, 0].item()
    assert (tokenized.input_ids[:, img_token_idx] == tokenizer.img_token_id).all()

    model, _, _ = initialize_model(EMU_HUB, None, device)

    # run it through the model
    output = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask, return_dict=True)
    # Should be shape [2, img_len, vocab_size]
    logprobs = F.log_softmax(output.logits, dim=-1)    
    target_logprobs = logprobs[:, :-1, :].gather(2, tokenized.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    topk_logprobs, topk_indices = torch.topk(target_logprobs, 100, dim=-1)

    output = {
        "topk_logprobs": topk_logprobs,
        "topk_indices": topk_indices,
        "target_logprobs": target_logprobs,
    }

    torch.save(output, "/tmp/output.pt")
    wandb.save("/tmp/output.pt", policy="now")


    




    


    



if __name__ == "__main__":
    main()