 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  tensor_upscale
from .FlashVSR.examples.WanVSR.infer_flashvsr_full import init_pipeline,run_inference
from .FlashVSR.examples.WanVSR.infer_flashvsr_tiny import   init_pipeline_tiny,run_inference_tiny
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes


MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_FlashVSR_current_path = os.path.join(folder_paths.models_dir, "FlashVSR")
if not os.path.exists(weigths_FlashVSR_current_path):
    os.makedirs(weigths_FlashVSR_current_path)

folder_paths.add_model_folder_path("FlashVSR", weigths_FlashVSR_current_path) #  FlashVSR dir


class FlashVSR_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="FlashVSR_SM_Model",
            display_name="FlashVSR_SM_Model",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "dmd" in i.lower()]),
                io.Combo.Input("proj_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "proj" in i.lower()]),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("tcd_encoder",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "tcd" in i.lower()] ),
            ],
            outputs=[
                io.Custom("FlashVSR_SM_Model").Output(),
                ],
            )
    @classmethod
    def execute(cls, dit,proj_pt,vae,tcd_encoder) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("FlashVSR", dit) if dit != "none" else None
        proj_pt_path=folder_paths.get_full_path("FlashVSR", proj_pt) if proj_pt != "none" else None
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        tcd_encoder_path=folder_paths.get_full_path("FlashVSR", tcd_encoder) if tcd_encoder != "none" else None
        assert dit_path is not None and proj_pt is not None , "Please select the Sdit,proj_pt,checkpoint file"
        assert vae_path is not None or tcd_encoder_path is not None , "Please select the Sdit,proj_pt,checkpoint file"
        if vae_path is None and tcd_encoder_path is not None:
            model=init_pipeline_tiny(proj_pt_path,dit_path, tcd_encoder_path, device="cuda")
        else:
            model=init_pipeline(proj_pt_path,dit_path, vae_path, device="cuda")
        return io.NodeOutput(model)
    

class FlashVSR_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_KSampler",
            display_name="FlashVSR_SM_KSampler",
            category="FlashVSR",
            inputs=[
                io.Custom("FlashVSR_SM_Model").Input("model"),
                io.Image.Input("image"),
                io.Combo.Input("emb_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "prompt" in i.lower()]),
                io.Int.Input("width", default=1280, min=128, max=nodes.MAX_RESOLUTION,step=128,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=768, min=128, max=nodes.MAX_RESOLUTION,step=128,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("scale", default=4, min=1, max=4),
                io.Float.Input("kv_ratio", default=3.5, min=0.0, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("local_range", default=11, min=1,step=1, max=50),
                io.Int.Input("steps", default=1, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("sparse_ratio", default=2.0, min=0.0, max=10.0, step=0.1,display_mode=io.NumberDisplay.slider),
                io.Conditioning.Input("Conditioning",optional=True),
                
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls, model,image,emb_pt,width,height,seed,scale,kv_ratio,local_range, steps, cfg,sparse_ratio  ,Conditioning=None) -> io.NodeOutput:
        image=tensor_upscale(image,width, height)
        prompt_path=folder_paths.get_full_path("FlashVSR", emb_pt) if emb_pt != "none" else None
        context_tensor=Conditioning[0][0] if Conditioning is not  None else None
        assert prompt_path is not None or context_tensor is not None , "Please select the emb,or link a conditioning tensor"
        if hasattr(model,"TCDecoder") :
            images=run_inference_tiny(model,prompt_path,context_tensor,image,seed,scale,kv_ratio,local_range,steps,cfg,sparse_ratio )
        else:
            images=run_inference(model,prompt_path,context_tensor,image,seed,scale,kv_ratio,local_range,steps,cfg,sparse_ratio )
     
        return io.NodeOutput(images.float())

from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/FlashVSR_SM_Extension")
async def get_hello(request):
    return web.json_response("FlashVSR_SM_Extension")

class FlashVSR_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlashVSR_SM_Model,
            FlashVSR_SM_KSampler,
        ]
async def comfy_entrypoint() -> FlashVSR_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FlashVSR_SM_Extension()



