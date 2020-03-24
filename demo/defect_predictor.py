from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import time
import cv2
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from COCOdemo import draw_rectangle
import json
from tqdm import tqdm

config_file = "/root/mobilenet_maskrcnn/configs/mobilenet.yaml"
annFile = '/root/mobilenet_maskrcnn/datasets/coco/annotations/instances_val2014.json'
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=70,
    confidence_threshold=0.5,
)

# file_root = "/root/mobilenet_maskrcnn/datasets/coco/val2014/"
file_root = "/root/mobilenet_maskrcnn/demo/yzmtest/"
file_list = os.listdir(file_root)

now = time.strftime("%Y-%m-%d-%H:%M:%s", time.localtime(time.time()))

# fileimg = "/home/heqing/maskrcnn-benchmark/demo/" + now
fileimg = "/root/mobilenet_maskrcnn/demo/" + now
os.makedirs(fileimg)
save_out = fileimg + "/"
for img_name in file_list:
    img_path = file_root + img_name
    image = cv2.imread(img_path)
    predictions = coco_demo.run_on_opencv_image(image)  # demo时将POSTPROCESS_MASKS关掉

    save_path = save_out + img_name
    cv2.imwrite(save_path, predictions)
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", predictions)
    # cv2.waitKey(1)


# coco = COCO(annFile)
# catNms = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# catIds_1 = coco.getCatIds(catNms=catNms)
# with open(annFile, 'r', encoding='UTF-8') as f:
#     dataset = json.load(f)
#     for img in tqdm(dataset['images']):
#         img_name = img['file_name']
#         img_path = file_root + img_name
#         anns_all = []
#         image = cv2.imread(img_path)
#         predictions = coco_demo.run_on_opencv_image(image)
#         for i in range(len(catIds_1)):  # 获取每一张图片每个类的标注的id
#             catIds = i + 1
#             annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catIds], iscrowd=None)  # 标注的Id
#             anns = coco.loadAnns(annIds)
#             anns_all.extend(anns)
#             coordinates = []
#             for j in range(len(anns_all)):
#                 # 获取标注的坐标和类别信息
#                 # anns分别为[左上角x，左上角y，x方向长，y方向宽]，坐标原点在左上，向右向下延伸
#                 coordinate = []
#                 coordinate.append(anns_all[j]['bbox'][0])  # 左下角x坐标
#                 coordinate.append(anns_all[j]['bbox'][1] + anns_all[j]['bbox'][3])  # 左下角y坐标
#                 coordinate.append(anns_all[j]['bbox'][0] + anns_all[j]['bbox'][2])  # 右上角x坐标
#                 coordinate.append(anns_all[j]['bbox'][1])  # 右上角y坐标
#                 coordinate.append(anns_all[j]['category_id'])  # 检测框类别
#                 coordinates.append(coordinate)
#         draw_rectangle(coordinates, predictions, img_name, catNms, save_out)
