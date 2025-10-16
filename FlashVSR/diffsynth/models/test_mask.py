# import torch
# from einops import rearrange

# def generate_causal_block_mask(batch_size, nheads, seqlen, local_num, window_size, device='cuda'):
#     # i: 当前 query 的位置，j: key 的位置
#     i = torch.arange(seqlen, device=device).view(-1, 1)  # [seqlen, 1]
#     j = torch.arange(seqlen, device=device).view(1, -1)  # [1, seqlen]

#     # 条件：j <= i 且 j >= i - local_num + 1
#     causal_mask = (j <= i) & (j >= i - local_num + 1)  # [seqlen, seqlen]

#     causal_mask = causal_mask.unsqueeze(1).unsqueeze(-1).repeat(1, window_size, 1, window_size)
#     causal_mask = rearrange(causal_mask, 'a n1 b n2 -> (a n1) (b n2)')
#     # 扩展 batch 维度
#     causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1)
#     return causal_mask

# if __name__ == '__main__':
#     causal_mask = generate_causal_block_mask(1, 12, 8, 3, 2)
#     # print(int(causal_mask.float().item())) 
#     print(causal_mask.float().int())
#     print(causal_mask.shape)
import torch
from block_sparse_attn import block_sparse_attn_func

seqlen = 10240
head_num = 12
head_dim = 128
batch_size = 1
device = torch.device('cuda')
dtype = torch.bfloat16

q_unpad = torch.randn(batch_size*seqlen, head_num, head_dim, device=device, dtype=dtype)
k_unpad = torch.randn(batch_size*seqlen, head_num, head_dim, device=device, dtype=dtype)
v_unpad = torch.randn(batch_size*seqlen, head_num, head_dim, device=device, dtype=dtype)
cu_seqlens_q = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
head_mask_type = torch.tensor([1]*head_num, device=device, dtype=torch.int32)
streaming_info = None
base_blockmask = torch.ones(batch_size, head_num, seqlen//128, seqlen//128, device=device, dtype=torch.bool)
max_seqlen_q_ = seqlen
max_seqlen_k_ = seqlen
p_dropout = 0.0

output = block_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
).unsqueeze(0)
print(output.shape)

import flash_attn

x = flash_attn.flash_attn_func(q_unpad.unsqueeze(0), k_unpad.unsqueeze(0), v_unpad.unsqueeze(0))
print(x.shape)

# 计算差异
diff = torch.sum(torch.abs(output - x))/torch.sum(torch.abs(x))
print(diff)

print(output[0, 0, 0, :10])
print(x[0, 0, 0, :10])
# print(x)