import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import pylab
import cv2
import os
from skimage.io import imsave
import numpy as np
import time
from COCOdemo import draw_rectangle
import json
from shutil import copyfile
from tqdm import tqdm


def main():
    #img_path = 'D:/desktop/Mask_self/datasets/coco/val2014/'
    img_path='/root/maskrcnn-benchmark/datasets/coco/train2014/'
    #annFile = 'D:/desktop/Mask_self/datasets/coco/annotations/instances_val2014.json'
    #annFile = '/root/maskrcnn-benchmark/datasets/coco/annotations/instances_train2014.json'
    annFile = '/root/maskrcnn-benchmark/demo/anno_image_train/instances_train2014_aug.json'
    saveDir = '/root/maskrcnn-benchmark/demo/anno_image_train/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    coco = COCO(annFile)
    catNms = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    catIds_1 = coco.getCatIds(catNms=catNms)

    # TODO:选择增强方式，以后改成argparse形式
    img_bbox = False # 选择对图片整体处理还是对bbox做处理，True为对整体处理，False仅对bbox内部做处理
    mirroring = True   # 选择是否y轴镜像
    rotating = False  # 选择是否旋转180度,180度旋转为x，y全都镜像一次
    annwrite = True 
    ann_show = False
    # TODO:获取所有图片的id：
    img_list = os.listdir(img_path)
    imgIds_all = set()
    for i in range(len(catIds_1)):
        catIds = i + 1
        imgIds_1 = coco.getImgIds(catIds=catIds)  # 返回包含这些类别的图片id
        imgIds_all.update(imgIds_1)
    imgIds_all = list(imgIds_all)
    imgIds_max = max(imgIds_all)  # 获取最大的图片Id，防止Id重复
    offset = 1
    # annIds_offset = 1
    # annIds_all = set()
    annId_max = annId_getter(coco, imgIds_all, catIds_1, )  # 获取最大的标注Id
    for im_idx in tqdm(range(len(imgIds_all))):  # 获取每一张图片
        img = coco.loadImgs(imgIds_all[im_idx])[0]
        image_name = img['file_name']
        # img_show(img_path, image_name)
        if os.path.exists(os.path.join(img_path, image_name)):
            img_raw = cv2.imread(os.path.join(img_path, image_name))
        else:
            continue
        (H, W, C) = img_raw.shape  # 获取图片的长宽
        imgIds_new = imgIds_max + offset
        offset += 1
        anns_all = []
        for i in range(len(catIds_1)):  # 获取每一张图片每个类的标注的id
            catIds = i + 1
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catIds], iscrowd=None)  # 标注的Id
            # annIds_all.update(annIds)  # 获得所有的annIds
            anns = coco.loadAnns(annIds)
            anns_all.extend(anns)  # 一张图片中包含的所有标注

            coordinates = []

            for j in range(len(anns_all)):
                # 获取标注的坐标和类别信息
                # anns分别为[左上角x，左上角y，x方向长，y方向宽]，坐标原点在左上，向右向下延伸
                coordinate = []
                coordinate.append(anns_all[j]['bbox'][0])  # 左下角x坐标
                coordinate.append(anns_all[j]['bbox'][1] + anns_all[j]['bbox'][3])  # 左下角y坐标
                coordinate.append(anns_all[j]['bbox'][0] + anns_all[j]['bbox'][2])  # 右上角x坐标
                coordinate.append(anns_all[j]['bbox'][1])  # 右上角y坐标
                coordinate.append(anns_all[j]['category_id'])  # 检测框类别
                # print(coordinate)
                coordinates.append(coordinate)
                # annIds_offset += 1
                # print(coordinates)
        if img_bbox:
            img_raw = img_cropper(img_raw, coordinates, mirroring, rotating)  # 返回bbox内部处理后的图片
            pass
        else:
            img_raw = img_avert(img_raw, mirroring, rotating)  # 对图像做处理
            # cv2.imshow("img", img_raw)
            # cv2.waitKey(0)
            coordinates = ann_avert(coordinates, mirroring, rotating, H, W)  # 对标注做处理
        image_name = img_name(image_name, img_bbox, mirroring, rotating)
        if annwrite:  # 是否将转换结果写入标注文件
            # annId_max = max(list(annIds_all))
            ann_writer(annFile, image_name, saveDir, imgIds_new, coordinates, img, anns_all, annId_max)
        draw_rectangle(coordinates, img_raw, image_name, catNms, saveDir, ann_show)
    print("finished transform at" + " " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


# TODO:进行图像增强
def img_avert(img, mirroring, rotating):
    if mirroring & (~rotating):
        return cv2.flip(img, 1)  # 以y轴为对称轴镜像
    elif mirroring & rotating:
        return cv2.flip(img, 0)  # 先y轴镜像，再旋转，相当于只以x对称轴轴镜像
    elif ~mirroring & rotating:
        return cv2.flip(img, -1)  # 只旋转
    else:
        return img  # 不做任何处理


# TODO:进行标注增强
def ann_avert(coordinates, mirroring, rotating, H, W):
    """
    :param coordinates: 标注坐标列表
    :param mirroring: 是否镜像
    :param rotating: 是否旋转
    :param H: 图像高度
    :param W: 图像宽度
    :return: 处理后的坐标D1,D2,D3,D4
    0-------------------------------------------------->X
     |    C1 ....................C2
     |    .........................
     |    .........................
     |    .........................
     |    C3.....................C4
     |
     v  Y
    """

    for coordinate in coordinates:
        C3_x = np.rint(coordinate[0])
        C3_y = np.rint(coordinate[1])
        C2_x = np.rint(coordinate[2])
        C2_y = np.rint(coordinate[3])
        if mirroring & (~rotating):  # 以y轴为对称轴镜像
            D3_x = W - C2_x
            D3_y = C3_y
            D2_x = W - C3_x
            D2_y = C2_y

            coordinate[0] = D3_x
            coordinate[1] = D3_y
            coordinate[2] = D2_x
            coordinate[3] = D2_y
            continue
        elif mirroring & rotating:  # 以x轴为对称轴镜像
            D3_x = C3_x
            D3_y = H - C2_y
            D2_x = C2_x
            D2_y = H - C3_y

            coordinate[0] = D3_x
            coordinate[1] = D3_y
            coordinate[2] = D2_x
            coordinate[3] = D2_y
            continue
        elif ~mirroring & rotating:  # 只旋转
            D3_x = W - C2_x
            D3_y = H - C2_y
            D2_x = W - C3_x
            D2_y = H - C3_y

            coordinate[0] = D3_x
            coordinate[1] = D3_y
            coordinate[2] = D2_x
            coordinate[3] = D2_y
            continue
        else:
            pass
    return coordinates  # 不做任何处理


# TODO:将转换后的标签写入标注文件,可以参考labelme的文件
def ann_writer(annFile, imgName, savepath, imgId, coordinates, img_inf, ann_inf, annId_max):
    outfile = savepath + 'instances_train2014_aug.json'
    if not os.path.exists(outfile):  # 将原标注文件复制到目标路径
        copyfile(annFile, outfile)
    with open(outfile, 'r', encoding='UTF-8') as f:  # 打开原标注文件
        dataset = json.load(f)
        # TODO:对dataset['images']做处理
        dataset['images'].append(dict(
            file_name=imgName,
            height=img_inf['height'],
            width=img_inf['width'],
            id=imgId))
        # TODO：对dataset['annotations']做处理
        for i in range(len(ann_inf)):
            # TODO:将坐标转化为COCO数据集格式
            bbox = coor2COCO(coordinates[i])
            pass
            dataset['annotations'].append(dict(
                id=ann_inf[i]['id'] + annId_max,
                image_id=imgId,
                category_id=ann_inf[i]['category_id'],
                area=ann_inf[i]['area'],
                bbox=bbox,
                iscrowd=0,
            ))
        with open(outfile, 'w', encoding='UTF-8') as f_out:
            json.dump(dataset, f_out)


# TODO：显示图片
def img_show(path, img):
    I = io.imread('%s/%s' % (path, img))
    plt.axis('off')
    plt.imshow(I)
    plt.show()


# TODO:裁剪图片的bbox并将处理后的bbox插回原图片
def img_cropper(img, coordinates, mirroring, rotating):
    bboxes = []  # 检测框内图像
    RoIs = []  # RoI坐标
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    for coordinate in coordinates:
        C3_x = np.rint(coordinate[0])
        C3_y = np.rint(coordinate[1])
        C2_x = np.rint(coordinate[2])
        C2_y = np.rint(coordinate[3])
        RoIs.append([C3_x, C3_y, C2_x, C2_y])
        cropped = img[int(C2_y):int(C3_y), int(C3_x):int(C2_x)]
        bboxes.append(cropped)
    for idx, bbox in enumerate(bboxes):
        bboxes[idx] = img_avert(bbox, mirroring, rotating)  # 对bbox图像做转化
    for i in range(len(bboxes)):
        img[int(RoIs[i][3]):int(RoIs[i][1]), int(RoIs[i][0]):int(RoIs[i][2])] = bboxes[i]  # bbox插回原图
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img


# TODO:解决图片命名规则
def img_name(name, img_bbox, mirroring, rotating):
    name_front = name.split('.')[0]
    if img_bbox:
        name_front = name_front + '_b'
    if mirroring:
        name_front = name_front + '_m'
    if rotating:
        name_front = name_front + '_r'
    assert name.split('.')[1] == 'jpg', "结尾不是.jpg，检查命名规则"
    name = name_front + '.jpg'
    return name


# TODO:将坐标转化为COCO数据集格式
def coor2COCO(coordinate):
    # anns分别为[左上角x，左上角y，x方向长，y方向宽]，坐标原点在左上，向右向下延伸
    bbox = []
    bbox.append(coordinate[0])
    bbox.append(coordinate[3])
    bbox.append(abs(coordinate[2] - coordinate[0]))
    bbox.append(abs(coordinate[3] - coordinate[1]))
    return bbox


# TODO：获取最大的标注Id最大值
def annId_getter(coco, imgIds_all, catIds_1, ):
    annIds_all = set()
    for im_idx in range(len(imgIds_all)):
        img = coco.loadImgs(imgIds_all[im_idx])[0]
        for i in range(len(catIds_1)):
            catIds = i + 1
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=[catIds], iscrowd=None)  # 标注的Id
            annIds_all.update(annIds)
    annId_max = max(list(annIds_all))
    return annId_max


if __name__ == '__main__':
    main()
