#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12



from maskrcnn_benchmark.config import cfg
from predictor import COCODemo




#config_file = "../configs/e2e_faster_rcnn_R_50_C4_1x.yaml"
config_file = "../configs/yzmsection.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
cfg.merge_from_list(["MODEL.WEIGHT", "/home/heqing/maskrcnn-benchmark/model_0002500.pth"])

#cfg.merge_from_list(["MODEL.MASK_ON", False])

# Now we create the `COCODemo` object. It contains a few extra options for conveniency, such as the confidence threshold for detections to be shown.

# In[17]:


coco_demo = COCODemo(
    cfg,
    min_image_size=200,
    confidence_threshold=0.7,
)

def load(url):
    #response = requests.get(url)
    #pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    #pil_image= Image.open("/home/heqing/maskrcnn-benchmark/demo/inclusion_3.jpg","r").convert("RGB")
    pil_image = Image.open("/home/heqing/maskrcnn-benchmark/demo/yzmtest/00095_5_2017-07-10.jpg", "r").convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


# from http://cocodataset.org/#explore?id=345434
image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
imshow(image)


# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
imshow(predictions)


