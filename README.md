# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI

# Upadte
*  Block-Sparse-Attention 正确安装且能调用才是方法的完全体，当前的函数实现会更容易OOM,但是Block-Sparse-Attention轮子实在不好找，目前只有CU128 toch2.7的,
*  方法是基于现有prompt.pt训练的，所以外置cond没有必要已经去掉，新增tile 和 color fix 选项，tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。修复图片索引数不足的错误。  
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的；  
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 打星； if you Like it ， star the official project [link](https://github.com/OpenImagingLab/FlashVSR)

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlashVSR

```

2.requirements  
----

```
pip install -r requirements.txt
```
要复现官方效果，必须安装Block-Sparse-Attention 
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention 
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

3.checkpoints 
----

* 3.1 [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1  
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
```
  

# Example
* full
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example18.png)
* tiny 27s
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_t.png)

# Acknowledgements
[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)  
[taehv](https://github.com/madebyollin/taehv)  

# Citation
```
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}

``

``
