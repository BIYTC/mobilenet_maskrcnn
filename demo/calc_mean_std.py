from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch

file_root = "/root/mobilenet_maskrcnn/datasets/coco/train2014_o/"
file_list = os.listdir(file_root)
transformer = transforms.Compose([transforms.ToTensor()])
img_mean_adder = (0, 0, 0)
img_R_std = None
img_G_std = None
img_B_std = None
for idx,img_name in enumerate(tqdm(file_list)):
    img_path = file_root + img_name
    img = Image.open(img_path).convert('RGB')
    img = transformer(img)
    img_R = img[0]
    img_G = img[1]
    img_B = img[2]

    single_img_mean = (img_R.mean(), img_G.mean(), img_B.mean())
    # img_G_mean = img_G.mean()
    # img_B_mean = img_B.mean()
    # single_img_std = (img_R.std(), img_G.std(), img_B.std())
    # img_G_std = img_G.std()
    # img_B_std = img_B.std()
    img_mean_adder = (img_mean_adder[0] + single_img_mean[0], img_mean_adder[1] + single_img_mean[1],
                      img_mean_adder[2] + single_img_mean[2])
    if idx==0:
        img_R_std=img_R
    else:
        img_R_std = torch.cat((img_R_std, img_R))
all_img_mean = (
    img_mean_adder[0] / len(file_list), img_mean_adder[1] / len(file_list), img_mean_adder[2] / len(file_list))

all_img_std = (img_R_std.std(), img_R_std.std(), img_R_std.std())

print("The dataset's mean is {}, std is {}".format(all_img_mean[0], all_img_std[0]))
