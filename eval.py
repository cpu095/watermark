#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import numpy as np
import torchvision
from tqdm import tqdm
import torch
from utils import *
from noisy import *
import random
import math
from torch.utils.data import Dataset
import torchshow as ts
from networks import *

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

torch.cuda.set_device(0)
device = "cuda:2"

edit_captions = process_txt_file("/home/jiasun/lun/JigMark/dataset/edit_caption_mini_15731_20731.txt")

save_dir = '/home/jiasun/lun/JigMark/output'  # 你想保存图像的路径
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建它
# In[ ]:


img_size = 256
set_seeds(0)

train_path = '/home/jiasun/lun/JigMark/datasets/'
train_transform = transforms.Compose([
			transforms.Resize((img_size,img_size), antialias=True),
			transforms.ToTensor(),
		])
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
imagenet_val_edit = ImageEditDataset(trainset, "/home/jiasun/lun/JigMark/dataset/imagenet_val_edit.json")
train_subset = torch.utils.data.Subset(imagenet_val_edit, [i for i in range(100)])
dataloader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)


encoder = ConvNextU()
decoder = WMDecoder()
replace_batchnorm(decoder)


# In[8]:


inference_steps = 50

checkpoint = torch.load("/home/jiasun/lun/JigMark/checkpoints/jiasun1true5000_2GPUs_models.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])


# In[ ]:


# siamese = SiameseNetwork(transforms=True)
# siamese.load_state_dict(torch.load("./checkpoints/siamese_resnet50_withaug.pth"))
# siamese.eval()
# siamese = siamese.to(device)


# In[10]:


num_of_splits = 4
shuffle_indices = random.sample(range(num_of_splits**2), num_of_splits**2)
shuffler = ImageShuffler(splits=num_of_splits, shuffle_indices=shuffle_indices)


# ## SDEdit

# In[11]:


from diffusers import StableDiffusionImg2ImgPipeline
model_id_or_path = "/home/jiasun/lun/JigMark/model/miniSD-diffusers"
sdedit = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, requires_safety_checker = False,use_safetensors=False)
sdedit.safety_checker = None
sdedit.feature_extractor = None
sdedit.scheduler = DPMSolverMultistepScheduler.from_config(sdedit.scheduler.config)
sdedit.set_progress_bar_config(disable=True)
sdedit = sdedit.to(device)


# In[ ]:


ex_values, dex_values, dx_values, x_values = [], [], [], []
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()
with torch.no_grad():
    for iters, (img, instruct) in enumerate(dataloader):
        factor = 1
        h_flipper = RandomHFlip()
        v_flipper = RandomVFlip()
        contrast_adjuster = ContrastAdjustment(factor_range=(0.8*factor, 1.5*factor))
        brightness_adjuster = BrightnessAdjustment(value_range=(-0.25*factor, 0.25*factor))
        jpeg_compressor = RandomJPEGCompression(quality_range=(int(10*factor), int(90*factor)))
        noiser = GaussianNoise(mean=0)
        blurer = GaussianBlur(kernel_size=7)
        Cropper = RandomCropAndResize(crop_size=((256-int(128*factor)),(256-int(32*factor))))
        img2img_editer = Img2imgTransforms(pipe = sdedit, strength_range=(0.1+0.2*factor, 0.2+0.4*factor), guidance_scale_range=(5, 20), inference_steps_range = (35,50),num_images_per_prompt=1)
        random_transformer = RandomImageTransforms(
            transforms=[h_flipper, v_flipper, noiser, blurer, contrast_adjuster, brightness_adjuster, jpeg_compressor, Cropper],
            img2img_transforms = img2img_editer
        )

        x = img.to(device)
        # Change image into watermakr version
        ex = shuffler.unshuffle(encoder(shuffler.shuffle(x, update_shuffle_indices=True)))
        
        #Diffusion model
        batch_size = x.shape[0]
                        
        #Diffusion model
        prompts = random.sample(edit_captions, x.shape[0])
        dex = random_transformer.transform(ex, prompts)
        
        
        dx = random_transformer.transform(x, prompts)
        
        
        # # HAV between 0.3 and 0.5
        # hav_scores = siamese(ex, dex.cuda())
        # hav_index = torch.where((hav_scores > 0.3) & (hav_scores < 0.5))[0].cpu().numpy()
        
        # Decoder step
        x_value = decoder(shuffler.shuffle(x))
        dx_value = decoder(shuffler.shuffle(dx).to(device))
        ex_value = decoder(shuffler.shuffle(ex))
        dex_value = decoder(shuffler.shuffle(dex).to(device))
        ex_values.append(ex_value)
        dex_values.append(dex_value)
        dx_values.append(dx_value)
        x_values.append(x_value)
        print("Finish Batch" + str(iters))

    # for i in range(len(ex_values)):
    #     # 将图像转换为PIL格式前，确保它是3维或2维的张量
    #     ex_pil = transforms.ToPILImage()(ex[i])
        
        
    #     # 保存图像到服务器路径
    #     ex_pil.save(os.path.join(save_dir, f'ex_image_{iters}_{i}.png'))
    # for i in range(len(dex_values)):
    #     # 将图像转换为PIL格式前，确保它是3维或2维的张量
    #     dex_pil = transforms.ToPILImage()(dex[i].cpu())
        
        
    #     # 保存图像到服务器路径
    #     dex_pil.save(os.path.join(save_dir, f'dex_image_{iters}_{i}.png'))

       
        
save_txt_path = "values_output.txt"

with open(save_txt_path, "w") as f:
    # 保存 x_values
    f.write("x_values:\n")
    for idx, tensor in enumerate(x_values):
        # 将 tensor 转到 CPU
        tensor_cpu = tensor.detach().cpu()
        # 转为 numpy 数组
        arr = tensor_cpu.numpy()
        # 这里为了方便查看，将其降到 1D 或者保持原形状也行
        # 举例，将其 reshape(-1) 以在一行内写出
        arr_1d = arr.reshape(-1)
        # 转成字符串，再写入文件
        arr_str = " ".join(map(str, arr_1d))
        # 每个批次写一行，或者可自由选择换行格式
        f.write(f"  Batch {idx}: {arr_str}\n")

    # 保存 ex_values
    f.write("\nex_values:\n")
    for idx, tensor in enumerate(ex_values):
        tensor_cpu = tensor.detach().cpu()
        arr = tensor_cpu.numpy()
        arr_1d = arr.reshape(-1)
        arr_str = " ".join(map(str, arr_1d))
        f.write(f"  Batch {idx}: {arr_str}\n")
        
    # 保存 dx_values
    f.write("\ndx_values:\n")
    for idx, tensor in enumerate(dx_values):
        tensor_cpu = tensor.detach().cpu()
        arr = tensor_cpu.numpy()
        arr_1d = arr.reshape(-1)
        arr_str = " ".join(map(str, arr_1d))
        f.write(f"  Batch {idx}: {arr_str}\n")
        
    # 保存 dex_values
    f.write("\ndex_values:\n")
    for idx, tensor in enumerate(dex_values):
        tensor_cpu = tensor.detach().cpu()
        arr = tensor_cpu.numpy()
        arr_1d = arr.reshape(-1)
        arr_str = " ".join(map(str, arr_1d))
        f.write(f"  Batch {idx}: {arr_str}\n")

print(f"所有张量信息已保存到 {save_txt_path}")    


# In[ ]:


save_path='/home/jiasun/lun/JigMark/output/evaluation/roc_curve2.png'
eval_res(ex_values, dex_values, dx_values, x_values,save_path)


