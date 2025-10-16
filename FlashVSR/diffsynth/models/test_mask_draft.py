import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple, Optional, List
from einops import rearrange

from block_sparse_attn import block_sparse_attn_func

out_path = '/cpfs01/shared/xpixel/zhuangjunhao/swin_base/DiffSynth-Studio/diffsynth/models/out_mask/'
class WindowPartition3D:
    """Partition / reverse‑partition helpers for 5‑D tensors (B,F,H,W,C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."

        x = x.view(B,
                   F // wf, wf,
                   H // wh, wh,
                   W // ww, ww,
                   C)
        # (B, nf, nh, nw, wf, wh, ww, C) -> (B*nf*nh*nw, wf*wh*ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor,
                win: Tuple[int, int, int],
                orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)
    

def generate_causal_draft_block_mask(batch_size, nheads, seqlen, local_num, window_size, q_w, k_w, topk=10, device='cuda'):
    assert batch_size == 1, "batch_size must be 1"
    # 生成 causal_mask
    causal_mask = generate_causal_block_mask(batch_size, nheads, seqlen, local_num, window_size, device).squeeze(0)

    # q_w 和 k_w 的形状为 (nf*nh*nw, wf*wh*ww, D)
    # avgpool_q 和 avgpool_k 的形状为 (nf*nh*nw, D) avgpool wf*wh*ww
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    # avgpool_q = torch.randn(avgpool_q.shape, device=avgpool_q.device, dtype=avgpool_q.dtype)
    # avgpool_k = torch.randn(avgpool_k.shape, device=avgpool_k.device, dtype=avgpool_k.dtype)
    # print(avgpool_q.shape, avgpool_k.shape)

    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads) # (nf*nh*nw, h, D)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads) # (nf*nh*nw, h, D)


    q_heads = avgpool_q.permute(1, 0, 2) # (h, nf*nh*nw, D)
    k_heads = avgpool_k.permute(1, 0, 2) # (h, nf*nh*nw, D)

    D = avgpool_q.shape[-1]

    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D) # (h, nf*nh*nw, nf*nh*nw)

    # causal_mask 中为 False 的区域设置为 -inf True 的区域设置为 0
    causal_mask = causal_mask.to(torch.float32)
    causal_mask = causal_mask.masked_fill(causal_mask == False, -float('inf'))
    causal_mask = causal_mask.masked_fill(causal_mask == True, 0)

    scores = scores + causal_mask

    attn_map = torch.softmax(scores, dim=-1)
    # print(attn_map)
    assert attn_map.shape == causal_mask.shape, "attn_map.shape must be equal to causal_mask.shape"


    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)

    loop_num = attn_map.shape[0]

    mask = torch.zeros_like(attn_map, dtype=torch.bool)

    # process each head independently
    for h in range(loop_num):
        head_scores = attn_map[h]                # [S1, S2]
        # print(head_scores)
        flat = head_scores.flatten()             # [S1*S2]
        n = flat.numel()
        k = topk                   # number of smallest to exclude

        if k == 0:
            mask[h] = True
            continue
        if k >= n:
            # nothing to keep
            continue

        # threshold = max of the k smallest scores
        # print(flat, k)
        # 计算非0元素个数
        # non_zero_num = (flat > 0).sum()
        # print('non_zero_num', non_zero_num, k)
        # threshold = torch.topk(flat, k, largest=False).values.max()
        # print(torch.sort(flat, descending=True)[0])
        threshold = torch.sort(flat, descending=True)[0][k]

        # build head mask
        # print(threshold)
        mask[h] = head_scores > threshold
        # print(mask[h])

    mask = rearrange(mask, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)

    # save mask 为图像
    import matplotlib.pyplot as plt
    for i in range(mask.shape[0]):
        plt.imshow(mask[i].float().int().cpu().numpy())
        plt.savefig(out_path + f'mask_{i}.png')

    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return mask


def generate_causal_block_mask(batch_size, nheads, seqlen, local_num, window_size, device='cuda'):
    # i: 当前 query 的位置，j: key 的位置
    i = torch.arange(seqlen, device=device).view(-1, 1)  # [seqlen, 1]
    j = torch.arange(seqlen, device=device).view(1, -1)  # [1, seqlen]

    # 条件：j <= i 且 j >= i - local_num + 1
    causal_mask = (j <= i) & (j >= i - local_num + 1)  # [seqlen, seqlen]

    # print(causal_mask.float().int())

    causal_mask = causal_mask.unsqueeze(1).unsqueeze(-1).repeat(1, window_size, 1, window_size)
    causal_mask = rearrange(causal_mask, 'a n1 b n2 -> (a n1) (b n2)')
    # save causal_mask 为图像
    # import matplotlib.pyplot as plt
    # plt.imshow(causal_mask.float().int())
    # plt.savefig('causal_mask.png')
    # 扩展 batch 维度
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1)
    # causal_mask[:] = True
    return causal_mask


win = (2, 8, 8)
B = 1
nheads = 12
f = 16
h = 32
w = 56
dim = 128
device = 'cuda'
dtype = torch.bfloat16

seqlen = f//win[0]

token_num = f*h*w

local_num = random.randint(2, seqlen)
# local_num = 4
print(local_num)
# print(seqlen)

window_size = win[0]*h*w//128

q = torch.randn(B, token_num, nheads*dim, device=device, dtype=dtype)
k = torch.randn(B, token_num, nheads*dim, device=device, dtype=dtype)
k = q+torch.randn(B, token_num, nheads*dim, device=device, dtype=dtype)
v = torch.randn(B, token_num, nheads*dim, device=device, dtype=dtype)

q = q.view(B, f, h, w, nheads*dim)
k = k.view(B, f, h, w, nheads*dim)
v = v.view(B, f, h, w, nheads*dim)

q_w = WindowPartition3D.partition(q, win) # (B*nf*nh*nw, wf*wh*ww, D)
k_w = WindowPartition3D.partition(k, win) # (B*nf*nh*nw, wf*wh*ww, D)
v_w = WindowPartition3D.partition(v, win) # (B*nf*nh*nw, wf*wh*ww, D)

square_num = window_size*window_size
topk_ratio = (random.uniform(0., 1.0)**2)*(local_num-1.2)+1.2
topk = int(square_num*topk_ratio)
print(topk/square_num)
# topk = int(window_size*window_size*1.5)
attention_mask = generate_causal_draft_block_mask(B, nheads, seqlen, local_num, window_size, q_w, k_w, topk=topk, device=q.device)

print(attention_mask.shape)
print(attention_mask.float().mean())















