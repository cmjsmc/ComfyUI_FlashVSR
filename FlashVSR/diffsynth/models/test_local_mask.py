import torch

@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
    """
    向量化产生局部块稀疏注意力 mask（bool），边界处通过“窗口整体移位”保持固定 win_h x win_w 可见范围。
    - 输入:
        block_h, block_w: 网格块数 (H, W)
        win_h, win_w: 视野窗口大小（默认 6x6）
        include_self: 是否包含自注意（默认 True）
        device: torch 设备
    - 输出:
        mask: (B, B) 的 bool 张量，B = H * W；True 表示可注意
    备注：偶数窗口采用“左偏置居中”，例如 6 -> 行/列偏移范围 [-3, +2]。
    """
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    B = H * W

    # --- 所有块的 (r, c) 坐标 ---
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)   # (B,)
    c_all = XX.reshape(-1)   # (B,)

    # --- 对每个查询块的窗口起止行/列（向内移位以保证完整窗口）---
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)   # (B,)
    end_r   = start_r + win_h - 1                         # (B,)
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)   # (B,)
    end_c   = start_c + win_w - 1                         # (B,)

    # --- 广播判断键块是否落在各自查询块的窗口内 ---
    # r_all[None, :] 形状 (1, B)，与 start_r[:, None] (B, 1) 广播为 (B, B)
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col                               # (B, B) bool

    if not include_self:
        mask.fill_diagonal_(False)

    return mask


# =========================
#           Tests
# =========================

def _rc_to_idx(r, c, W):
    return r * W + c

def _expected_window(h, w, r, c, win_h=6, win_w=6):
    """返回 (start_r, end_r, start_c, end_c)，用于断言边界位置。"""
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = int(torch.clamp(torch.tensor(r - r_half), 0, h - win_h))
    end_r   = start_r + win_h - 1
    start_c = int(torch.clamp(torch.tensor(c - c_half), 0, w - win_w))
    end_c   = start_c + win_w - 1
    return start_r, end_r, start_c, end_c

def run_all_tests():
    H, W = 12, 20
    win_h, win_w = 6, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 基本形状与类型 ---
    mask = build_local_block_mask_shifted_vec(H, W, win_h, win_w, include_self=True, device=device)
    assert mask.shape == (H*W, H*W) and mask.dtype == torch.bool, "形状/类型不正确"

    # --- 行和恒定为 win_h*win_w ---
    row_sums = mask.sum(dim=1)
    assert row_sums.min().item() == win_h * win_w and row_sums.max().item() == win_h * win_w, \
        "每个查询块可见数量应恒为 win_h*win_w"

    # --- include_self=False 时行和减少 1 且对角为 False ---
    mask_no_self = build_local_block_mask_shifted_vec(H, W, win_h, win_w, include_self=False, device=device)
    row_sums2 = mask_no_self.sum(dim=1)
    assert row_sums2.min().item() == win_h*win_w - 1 and row_sums2.max().item() == win_h*win_w - 1, \
        "关闭自注意时每行可见应为 win_h*win_w - 1"
    assert (mask_no_self.diagonal() == 0).all().item(), "关闭自注意时对角必须为 False"

    # --- 若需要：抽查多个典型位置，验证窗口坐标 ---
    def check_rc(r, c):
        idx = _rc_to_idx(r, c, W)
        vis = mask[idx].reshape(H, W)
        sr, er, sc, ec = _expected_window(H, W, r, c, win_h, win_w)
        expected = torch.zeros((H, W), dtype=torch.bool, device=mask.device)
        expected[sr:er+1, sc:ec+1] = True
        assert torch.equal(vis, expected), f"窗口不匹配: rc=({r},{c}), 应为 [{sr}:{er}], [{sc}:{ec}]"

    samples = [(5, 10), (0, 0), (0, W-1), (H-1, 0), (H-1, W-1), (0, 10), (3, 0), (11, 10), (5, 19)]
    for rc in samples:
        check_rc(*rc)

    # --- 打印一些信息 ---
    B = H * W
    print(f"[OK] mask.shape={mask.shape}, device={device}, B={B}")
    print(f"每行 True 数恒为 {win_h*win_w}（include_self=True）；关闭自注意则为 {win_h*win_w - 1}")
    # 示例：展示一个位置的窗口范围
    r, c = 0, W-1  # 右上角
    sr, er, sc, ec = _expected_window(H, W, r, c, win_h, win_w)
    print(f"示例窗口 rc=({r},{c}) -> rows[{sr}:{er}], cols[{sc}:{ec}] (均含右端点)")

if __name__ == "__main__":
    # run_all_tests()

    # 如需直接生成加性 bias（-inf 掩码），可这样用：
    H, W = 10, 17
    mask = build_local_block_mask_shifted_vec(H, W, 100, 100, include_self=True)
    print('mask', mask.shape)
    # print('mask', mask)
    # save mask to numpy png

    import numpy as np
    import cv2
    mask = mask.cpu().numpy()*255.0
    mask = mask.astype(np.uint8)
    # mask = cv2.resize(mask, (W, H))
    cv2.imwrite('mask.png', mask)

    # B = H * W
    # attn_bias = torch.zeros((B, B), dtype=torch.float32)
    # attn_bias[~mask] = float("-inf")
