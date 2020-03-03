import matplotlib.pyplot as plt
from pycocotools.coco import COCO
# import skimage.io as io

import pylab
import cv2
import os
# from skimage.io import imsave
import numpy as np
from tqdm import tqdm

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# img_path = 'D:/desktop/Mask_self/datasets/coco/val2014/'
img_path = '/root/maskrcnn-benchmark/demo/anno_image_train/'
# annFile = 'D:/desktop/Mask_self/datasets/coco/annotations/instances_val2014.json'
annFile = '/root/maskrcnn-benchmark/demo/anno_image_train/instances_train2014_aug.json'
# img_path = 'D:/desktop/Mask_self/datasets/cococode/train2014/'
# annFile = 'D:/desktop/Mask_self/datasets/cococode/annotations/instances_train2014.json'

# if not os.path.exists('anno_image_coco/'):
#     os.makedirs('anno_image_coco/')

# saveDir = 'anno_image_coco/'
saveDir = '/root/maskrcnn-benchmark/demo/anno_image_train_demo/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)


def draw_rectangle(coordinates, image, image_name, catNms, saveDir, ann_show=True):
    if not ann_show:
        cv2.imwrite(saveDir + image_name, image)
        # pass
    else:
        for coordinate in coordinates:
            left = np.rint(coordinate[0])
            right = np.rint(coordinate[1])
            top = np.rint(coordinate[2])
            bottom = np.rint(coordinate[3])
            category = str(catNms[coordinate[4] - 1])
            # 左下角坐标, 右上角坐标
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
            image = cv2.rectangle(image,
                                  (int(left), int(right)),
                                  (int(top), int(bottom)),
                                  (0, 0, 0),
                                  1)
            # cv2.imshow("img", image_1)
            # cv2.waitKey(0)
            image = cv2.putText(image, category, (int(left), int(bottom + 5)), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                                (0, 0, 0), 1)
            cv2.imwrite(saveDir + image_name, image)
            # cv2.imshow("img", image_1)
            # cv2.waitKey(0)


def main():
    # 初始化标注数据的 COCO api
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # catNms = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    catNms = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # catIds_1 = coco.getCatIds(catNms=['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'])
    catIds_1 = coco.getCatIds(catNms=catNms)
    # img_list = os.listdir(img_path)
    # for i in range(len(catIds_1)):
    #     # for i in range(7):
    #     catIds = i + 1
    #     imgIds_1 = coco.getImgIds(catIds=catIds)  # 返回包含这些类别的图片id
    #     # img = coco.loadImgs(imgIds_1[np.random.randint(0, len(imgIds_1))])[0]
    #     # img = (coco.loadImgs(imgIds_1[im_idx])[0] for im_idx in range(len(imgIds_1)))
    #     for im_idx in range(len(imgIds_1)):
    #         img = coco.loadImgs(imgIds_1[im_idx])[0]
    #         image_name = img['file_name']
    #         print(img)
    #         # img = coco.loadImgs(imgIds_1[np.random.randint(0, len(imgIds_1))])[0]
    #         # image_name = img['file_name']
    #
    #         # 加载并显示图片
    #         # I = io.imread('%s/%s' % (img_path, img['file_name']))
    #         # plt.axis('off')
    #         # plt.imshow(I)
    #         # plt.show()
    #
    #         # catIds=[] 说明展示所有类别的box，也可以指定类别
    #         annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catIds], iscrowd=None)
    #         anns = coco.loadAnns(annIds)
    #
    #         # coco.showAnns(anns)
    #
    #         # print(anns)
    #         coordinates = []
    #         img_raw = cv2.imread(os.path.join(img_path, image_name))
    #         for j in range(len(anns)):
    #             coordinate = []
    #             coordinate.append(anns[j]['bbox'][0])
    #             coordinate.append(anns[j]['bbox'][1] + anns[j]['bbox'][3])
    #             coordinate.append(anns[j]['bbox'][0] + anns[j]['bbox'][2])
    #             coordinate.append(anns[j]['bbox'][1])
    #             # print(coordinate)
    #             coordinates.append(coordinate)
    #         # print(coordinates)
    #         draw_rectangle(coordinates, img_raw, image_name)
    pass
    # TODO:获取所有图片的id：
    img_list = os.listdir(img_path)
    imgIds_all = set()
    for i in range(len(catIds_1)):
        catIds = i + 1
        imgIds_1 = coco.getImgIds(catIds=catIds)  # 返回包含这些类别的图片id
        imgIds_all.update(imgIds_1)

    for im_idx in tqdm(range(len(imgIds_all))):  # 获取每一张图片
        imgIds_all = list(imgIds_all)
        img = coco.loadImgs(imgIds_all[im_idx])[0]
        image_name = img['file_name']
        # print(img)
        anns_all = []
        for i in range(len(catIds_1)):  # 获取每一张图片每个类的标注的id
            catIds = i + 1
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catIds], iscrowd=None)
            anns = coco.loadAnns(annIds)
            anns_all.extend(anns)
            # coco.showAnns(anns)

            # print(anns)
            coordinates = []
            img_raw = cv2.imread(os.path.join(img_path, image_name))
            for j in range(len(anns_all)):
                coordinate = []
                coordinate.append(anns_all[j]['bbox'][0])
                coordinate.append(anns_all[j]['bbox'][1] + anns_all[j]['bbox'][3])
                coordinate.append(anns_all[j]['bbox'][0] + anns_all[j]['bbox'][2])
                coordinate.append(anns_all[j]['bbox'][1])
                coordinate.append(anns_all[j]['category_id'])
                # print(coordinate)
                coordinates.append(coordinate)
            # print(coordinates)
            draw_rectangle(coordinates, img_raw, image_name, catNms, saveDir)
    print("finished")


if __name__ == '__main__':
    main()
