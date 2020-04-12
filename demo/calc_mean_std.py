from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import random

file_root = "/root/mobilenet_maskrcnn/datasets/coco/train2014_o/"
file_list = os.listdir(file_root)
random.shuffle(file_list)
transformer = transforms.Compose([transforms.ToTensor()])
img_mean_adder = (0, 0, 0)
img_R_std = None
img_G_std = None
img_B_std = None
sample=1000
for idx, img_name in enumerate(tqdm(file_list[:sample])):
    img_path = file_root + img_name
    img = Image.open(img_path).convert('RGB')
    img = transformer(img)
    img_R = img[0]*255
    img_G = img[1]*255
    img_B = img[2]*255

    single_img_mean = (img_R.mean(), img_G.mean(), img_B.mean())
    img_mean_adder = (img_mean_adder[0] + single_img_mean[0], img_mean_adder[1] + single_img_mean[1],
                      img_mean_adder[2] + single_img_mean[2])
    if idx == 0:
        img_R_std = img_R
    else:
        img_R_std = torch.cat((img_R_std, img_R))
all_img_mean = (
    img_mean_adder[0] / len(file_list[:sample]), img_mean_adder[1] / len(file_list[:sample]), img_mean_adder[2] / len(file_list[:sample]))

all_img_std = (img_R_std.std(), img_R_std.std(), img_R_std.std())

print("The dataset's mean is {}, std is {}".format(all_img_mean[0], all_img_std[0]))
