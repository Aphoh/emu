import argparse
from typing import Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.generation.configuration_utils import GenerationConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from emu.mllm.configuration_emu3 import Emu3Config
from ..mllm.modeling_emu3 import Emu3ForCausalLM
from ..mllm.processor import PrefixConstrainedLogitsProcessor, ClassifierFreeGuidanceLogitsProcessor
from huggingface_hub import snapshot_download 
import safetensors.torch as safett
import glob

from emu.mllm.processing_emu3 import Emu3Processor

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

def parse_args():
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument("--input", type=str, help="input text")
    parser.add_argument("--num_images", type=int, default=1, help="number of images to generate")
    return parser.parse_args()

def main():
    args = parse_args()

    input_text: Optional[str] = args.input
    num_images: int = args.num_images

    model_dl = snapshot_download(EMU_HUB)
    config = Emu3Config.from_json_file(f"{model_dl}/config.json")
    torch.cuda.set_device(0)
    with init_empty_weights():
        model = Emu3ForCausalLM(config)
    load_checkpoint_and_dispatch(model, model_dl, device_map="auto")

    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, device_map="cuda:0")
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True, device_map="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)


    # prepare input
    pos_prompt = input_text or "a shiba inu"
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
    prefix_proc = PrefixConstrainedLogitsProcessor(constrained_fn, 1)
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(classifier_free_guidance)


    pos_input_ids: torch.Tensor = pos_inputs.input_ids.to("cuda:0").repeat(num_images, 1)
    neg_input_ids: torch.Tensor = neg_inputs.input_ids.to("cuda:0").repeat(num_images, 1)

    input_batch = torch.cat([pos_input_ids, neg_input_ids], dim=0)

    print(pos_inputs)






if __name__ == "__main__":
    main()