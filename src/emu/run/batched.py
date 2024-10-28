# -*- coding: utf-8 -*-
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.generation.configuration_utils import GenerationConfig

from emu.mllm.configuration_emu3 import Emu3Config
from ..mllm.modeling_emu3 import Emu3ForCausalLM
from ..mllm.processor import PrefixConstrainedLogitsProcessor, ClassifierFreeGuidanceLogitsProcessor
from huggingface_hub import hf_hub_download
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

    model_dl = hf_hub_download(EMU_HUB)
    config = Emu3Config.from_json_file(f"{model_dl}/config.json")
    model = Emu3ForCausalLM(config)
    for safetensors_file in glob.glob(f"{model_dl}/*.safetensors"):
        model.load_state_dict(safett.load(safetensors_file), strict=False)

    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)


    # prepare input
    pos_prompt = args.input
    neg_prompt = ""

    classifier_free_guidance = 3.0
    prompt = ["a portrait of young girl.", "a shiba inu"]
    prompt = [p + pos_prompt for p in prompt]

    kwargs = dict(
        mode='G',
        ratio=["1:1", "16:9"],
        image_area=config.image_area,
        return_tensors="pt",
        padding="longest",
    )
    pos_inputs = processor(text=prompt, **kwargs)
    neg_inputs = processor(text=[neg_prompt] * len(prompt), **kwargs)


    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    prefix_proc = PrefixConstrainedLogitsProcessor(constrained_fn)
    cfg_proc = ClassifierFreeGuidanceLogitsProcessor(classifier_free_guidance)


    input_ids: torch.Tensor = pos_inputs.input_ids.to("cuda:0")

    print(input_ids.shape)
