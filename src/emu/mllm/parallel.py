

import torch
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Replicate
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from emu.mllm.modeling_llama import LlamaAttention, LlamaForCausalLM

def get_tensor_sharded_model(model: LlamaForCausalLM, mesh: DeviceMesh):
    NUM_DEVICES = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    mesh = init_device_mesh(device, (NUM_DEVICES,))

    layer_tp_plan = {
        "mlp.up_proj": ColwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel()
    }

    for layer in model.model.layers:
        attn: LlamaAttention = layer.self_attn
        attn.num_heads // mesh.size()
        attn.num_key_value_heads // mesh.size()
        parallelize_module(layer, mesh, layer_tp_plan)

    # Parallelize the embedding submodules.
    
    parallelize_module(model.model.embed_tokens, mesh, RowwiseParallel(output_layouts=Replicate()))
    parallelize_module(model.lm_head, mesh, ColwiseParallel(output_layouts=Replicate()))

    return model