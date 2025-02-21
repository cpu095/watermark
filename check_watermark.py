#!/usr/bin/env python
# coding: utf-8

# 本程序用于水印检测任务，利用预训练的水印嵌入和解码网络对图像进行水印判定。
# 主要流程包括：
# 1. 加载和预处理数据集（通过 ImageFolder 加载图像数据）。
# 2. 加载预训练模型（包含水印嵌入网络和解码网络）。
# 3. 对图像进行预处理并输入解码网络，提取水印特征。//todo
# 4. 根据设定的阈值判断每张图片是否含有水印。
# 5. 将检测结果（文件路径、得分和判定结果）保存到文件中。

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import random
from utils import *  # 请确保 utils.py 中有这些函数
from networks import *  # 请确保 networks.py 中定义了这些模型

# 设置设备
torch.cuda.set_device(0)
device = "cuda"

# 输出结果保存目录
save_dir = '/home/jiasun/lun/JigMark/output_check'
os.makedirs(save_dir, exist_ok=True)

# 参数设置
img_size = 256


# 数据加载（直接用 ImageFolder，不再使用 ImageEditDataset 以及取子集）
data_path = '/home/jiasun/lun/JigMark/datasets_watermark_detection/'
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size), antialias=True),
    transforms.ToTensor(),
])
dataset = ImageFolder(root=data_path, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 加载模型：水印嵌入网络（encoder）以及水印解码网络（decoder）
encoder = ConvNextU()
decoder = WMDecoder()
replace_batchnorm(decoder)

# 加载预训练 checkpoint（要求 checkpoint 中包含 "encoder" 与 "decoder"）
checkpoint = torch.load("/home/jiasun/lun/JigMark/checkpoints/jiasun1true5000_2GPUs_models.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# 初始化图像打乱工具（decoder 可能需要对应的预处理）
num_of_splits = 4
shuffle_indices = random.sample(range(num_of_splits**2), num_of_splits**2)
shuffler = ImageShuffler(splits=num_of_splits, shuffle_indices=shuffle_indices)

# 定义水印检测的阈值（阈值需要根据实际情况调参）
watermark_threshold = -2.7731  # 这里仅作为示例

results = []  # 用于保存 (文件路径, 得分, 判定结果)

# 遍历整个数据集，检测每一张图片是否含有水印
# 注意：ImageFolder 的 samples 属性中保存了 (文件路径, label) 信息
for batch_idx, (imgs, _) in enumerate(dataloader):
    imgs = imgs.to(device)
    
    # 对输入图像进行预处理：调用 shuffler 对图像进行 shuffle，符合 decoder 的输入要求
    shuffled_imgs = shuffler.shuffle(imgs,update_shuffle_indices=True)
    
    # 利用 decoder 提取水印特征（这里假设 decoder 的输出可以反映水印信息）
    watermark_features = decoder(shuffled_imgs)
    
    
    # 根据阈值判断是否加了水印
    decisions = watermark_features > watermark_threshold
    
    # 计算当前 batch 中每个样本在数据集中的索引，用于从 dataset.samples 中获取文件路径
    batch_size = imgs.size(0)
    start_idx = batch_idx * dataloader.batch_size
    for i in range(batch_size):
        filepath, _ = dataset.samples[start_idx + i]
        score = watermark_features[i].item()
        decision = "Watermarked" if decisions[i].item() else "Not Watermarked"
        results.append((filepath, score, decision))
        print(f"Image: {filepath}, Score: {score:.4f}, Decision: {decision}")

# 将结果保存到文件中
results_file = os.path.join(save_dir, "watermark_detection_results.txt")
with open(results_file, "w") as f:
    for filepath, score, decision in results:
        f.write(f"{filepath}\t{score:.4f}\t{decision}\n")

print("水印检测完成，结果保存在", results_file)
