# JigMark: Enhanced Robust Image Watermark against Diffusion Models via Contrastive Learning


<img width="100%" src="https://github.com/pmzzs/JigMark/assets/77789132/ad6f3bbf-e87b-4ee2-a13d-e6c2a54aac52">


# Features
1. **Human Aligned Variation (HAV) Score**: Quantifies human perception of image variations post-diffusion model transformations.这个原作者也没有
2. **Contrastive Learning**: Boosts watermark adaptability and robustness through contrastive learning.
3. **Jigsaw Puzzle Embedding**: A novel, flexible watermarking technique utilizing a 'Jigsaw' puzzle approach.
4. **High Robustness and Adaptability**: Demonstrates exceptional performance against sophisticated image perturbations and various transformations.

# Requirements
```
pip install requirements.txt -r
```

## Training
### Dataset Preparation
Download the ImageNet-1k dataset and organize it in the datasets folder as follows:


```
├── datasets
│   ├── test
│   │   ├── test
│   │       ├── xxx.JPEG
│   │       │
│   │       ├── ...
│   ├── val
│   │   ├── n01440764
│   │   │  ├── xxx.JPEG
│   │   │  │
│   │   │  ├── ...
│   │   ├── ...
```


### Setup Accelerate
Our code utilizes 'accelerate' for multi-GPU training. Set up the accelerate configuration with:

```
accelerate config
```


### Train
Initiate training with:

```
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file /home/jiasun/lun/JigMark/accelerate_config.yaml train.py --train_path /home/jiasun/lun/JigMark/minidataset_15731_20731 --instr_path /home/jiasun/lun/JigMark/dataset/edit_caption_mini_15731_20731.txt --total_epochs 40
```
Trained models will be saved in "./checkpoints/".

## Evaluate

### Download Pretrained Models
Download the following pretrained models:
- 这个我没有上传

After downloading, run `eval.py` for model evaluation.

# Acknowledgement
This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Zero 1-to-3](https://github.com/cvlab-columbia/zero123), and [Diffusers](https://huggingface.co/docs/diffusers/index). We express our gratitude to the authors of these projects for making their code publicly available.
