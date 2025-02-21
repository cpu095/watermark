# 本程序用于从原始数据集中创建一个小型数据集，包含指定范围内的图像和相应的描述文件。
# 输入参数：
# - start：指定从哪一张图片和描述开始（索引，从 0 开始）。
# - end：指定到哪一张图片和描述结束（索引，确保不超过原始数据的总数）。
#
# 程序会执行以下操作：
# 1. 从指定的描述文件中读取所有描述。
# 2. 根据 start 和 end 索引选择对应范围的描述，并保存为新的描述文件。
# 3. 从原始图像文件夹中复制选定范围内的图像到新目录。
# 4. 最终生成包含指定图片和描述的小数据集。
#
# 输入路径：
# - 原始图像存放在 `/home/jiasun/lun/JigMark/datasets/test` 目录中。
# - 原始描述文件为 `/home/jiasun/lun/JigMark/dataset/edit_caption.txt`。
#
# 输出路径：
# - 新的数据集图像存放在 `/home/jiasun/lun/JigMark/minidataset_{start}_{end}/test` 目录中。
# - 新的描述文件存放在 `/home/jiasun/lun/JigMark/dataset/edit_caption_mini_{start}_{end}.txt` 目录中。


import os
import shutil

def create_mini_dataset(start, end):
    # 原始图片和描述文件的路径
    image_dir = '/home/jiasun/lun/JigMark/datasets/test'
    caption_file = '/home/jiasun/lun/JigMark/dataset/edit_caption.txt'

    # 新的图片存放路径和描述文件路径
    mini_image_dir = f'/home/jiasun/lun/JigMark/minidataset_{start}_{end}/test'
    mini_caption_file = f'/home/jiasun/lun/JigMark/dataset/edit_caption_mini_{start}_{end}.txt'

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(mini_image_dir):
        os.makedirs(mini_image_dir)

    # 读取原始描述文件
    with open(caption_file, 'r') as f:
        captions = f.readlines()

    # 确保选择的 end 不超过总数量
    end = min(end, len(captions))

    # 获取从 start 到 end 范围的描述
    selected_captions = captions[start:end]

    # 生成新的描述文件
    with open(mini_caption_file, 'w') as f:
        f.writelines(selected_captions)

    # 获取从 start 到 end 范围的图片路径
    for i in range(start, end):
        image_name = f'ILSVRC2012_test_{i+1:08d}.JPEG'
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, mini_image_dir)
        else:
            print(f"Warning: {image_path} does not exist.")

    print(f"Mini dataset created with {end - start} images and captions.")

# 例如，创建包含第 1000 到第 2000 张图片和描述的小数据集
start = 15731
end = 20731
create_mini_dataset(start, end)
