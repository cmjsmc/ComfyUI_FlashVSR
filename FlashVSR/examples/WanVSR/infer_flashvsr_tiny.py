#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import folder_paths
import os, re, time
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange
from .infer_flashvsr_full import tensor_upscale
from ...diffsynth import ModelManager, FlashVSRTinyPipeline
from .utils.utils import Buffer_LQ4x_Proj
from .utils.TCDecoder import build_tcdecoder


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    try:
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    except:
        try:
            frames=frames.cpu()
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
            return frames
        except:
            batch_size = min(32, frames.shape[0]) 
            total_frames = frames.shape[0]
            frame_list = []
            for i in range(0, total_frames, batch_size):
                batch_frames = frames[i:min(i + batch_size, total_frames)]
                batch_frames = ((batch_frames.float() + 1) * 127.5).clip(0, 255)
                batch_frames_np = batch_frames.cpu().numpy().astype(np.uint8)
                for frame in batch_frames_np:
                    frame_list.append(Image.fromarray(frame))
            return frame_list

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def dup_first_frame_1cthw_simple(video_tensor):
    return torch.cat([video_tensor[:, :, :1], video_tensor], dim=2)

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype=dtype)

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def prepare_input_tensor(path: str, scale: float = 4,fps=30, dtype=torch.bfloat16, device='cuda'):

    if isinstance(path,torch.Tensor):
        total,h0,w0,_ = path.shape
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        pil_list=tensor2pillist(path)

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        idx = idx[:F]
        frames = []
        pil_list = [pil_list[i] for i in idx]
        for i in idx:
            img = pil_list[i].convert('RGB')
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        frames = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W  
        torch.cuda.empty_cache()
        return frames, tH, tW, F, fps
        
    elif os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        fps = 30
        return vid, tH, tW, F, fps

    elif is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try: meta = rdr.get_meta_data()
        except Exception: pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf,int) and nf>0: return nf
            except Exception: pass
            try: return r.count_frames()
            except Exception:
                n=0
                try:
                    while True: r.get_data(n); n+=1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total-1]*4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try: rdr.close()
            except Exception: pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, fps
    else:
        raise ValueError(f"Unsupported input: {path}")
   

def init_pipeline_tiny(prompt_path,LQ_proj_in_path = "./FlashVSR/LQ_proj_in.ckpt",ckpt_path="./FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors",TCDecoder_path="./FlashVSR/TCDecoder.ckpt",device="cuda"):

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([ckpt_path,])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu",weights_only=False,), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    
    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load(TCDecoder_path,weights_only=False,), strict=False)
    print(mis)

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path); pipe.load_models_to_device(["dit","vae"])
    return pipe

def run_inference_tiny(pipe,input,seed,scale,kv_ratio=3.0,local_range=9,step=1,cfg_scale=1.0,sparse_ratio=2.0,color_fix=True,fix_method="wavelet",split_num=81,dtype=torch.bfloat16,device="cuda", save_vodeo_=False,):
    pipe.to('cuda')

    pad_first_frame = True  if "wavelet"== fix_method and color_fix else False

    torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    LQ, th, tw, F, fps = prepare_input_tensor(input, scale=scale,dtype=dtype, device=device)
    frames,LQ_cur_idx = pipe(
        prompt="", negative_prompt="", cfg_scale=cfg_scale, num_inference_steps=step, seed=seed,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio*768*1280/(th*tw), 
        kv_ratio=kv_ratio,
        local_range=local_range,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
        color_fix = color_fix,
    )
    pipe.dit.to('cpu')  
    torch.cuda.empty_cache()   
    #print(LQ.shape,frames.shape,LQ_cur_idx)#torch.Size([1, 3, 81, 384, 640]) torch.Size([1, 16, 20, 48, 80]) 77
    try:
        frames = pipe.TCDecoder.decode_video(frames.transpose(1, 2),parallel=False, show_progress_bar=False, cond=LQ[:,:,:LQ_cur_idx,:,:]).transpose(1, 2).mul_(2).sub_(1)
    except:
        print("TCDecoder decode_video OOM.")
        pipe.TCDecoder.to('cpu')
        torch.cuda.empty_cache()   
        pipe.TCDecoder.to('cuda')
        frames = frames.transpose(1, 2)  # 转换为 [B, T, C, H, W] 格式
        cond_frames = LQ[:,:,:LQ_cur_idx,:,:]
        segment_size = (split_num-1) * 2  # 160
        latent_segment_size = segment_size // 4 # 40
        decoded_frames_list = []
        total_latent_frames = frames.shape[1]
        for start_idx in range(0, total_latent_frames, latent_segment_size):
            end_idx = min(start_idx + latent_segment_size, total_latent_frames)
            
            # 潜在空间的切片
            frames_segment = frames[:, start_idx:end_idx, :, :, :]
            
            # 对应的条件帧（在原始空间中）
            start_cond_idx = start_idx * 4
            end_cond_idx = min(end_idx * 4 , LQ_cur_idx)
            cond_segment = cond_frames[:, :, start_cond_idx:end_cond_idx, :, :]
            
            decoded_segment = pipe.TCDecoder.decode_video(
                frames_segment,
                parallel=False,
                show_progress_bar=False,
                cond=cond_segment
            )
            decoded_frames_list.append(decoded_segment)
        
        decoded_frames = torch.cat(decoded_frames_list, dim=1)  # 
        frames = decoded_frames.transpose(1, 2).mul_(2).sub_(1)
        
    # 颜色校正（wavelet）
    # shape: 1,16, 20, 64, 96
    try:
        if color_fix:
            if pad_first_frame:
                frames = dup_first_frame_1cthw_simple(frames)
                LQ=dup_first_frame_1cthw_simple(LQ)
            frames = pipe.ColorCorrector(
                frames.to(device=LQ.device),
                LQ[:, :, :frames.shape[2], :, :],
                clip_range=(-1, 1),
                chunk_size=16,
                method=fix_method
                )
            if pad_first_frame:
                frames = frames[:, :, 1:, :, :] # remove first frame
    except:
        pass
    print("Done.")
    pipe.TCDecoder.to('cpu')
    del LQ
    torch.cuda.empty_cache()   
    frames = tensor2video(frames[0]) 
    if save_vodeo_:
        save_video(frames, os.path.join(folder_paths.get_output_directory(),f"FlashVSR_Full_seed{seed}.mp4"), fps=fps, quality=6)
    return frames
