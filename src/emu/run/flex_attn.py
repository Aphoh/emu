from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def custom_mask(b, h, q_idx, kv_idx):
    

def noop(score, b, h, q_idx, kv_idx):
    return score