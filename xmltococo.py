# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
import os, sys


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(self.xml):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d\n' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            #       print(self.json_file)
            tmp_file = self.json_file.split('/')[2].split('.')[0]
            #   print(tmp_file)


            self.num = num
            path = os.path.dirname(self.json_file)
            #    print(path)
            path = os.path.dirname(path)
            #     print(path)

            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            #            obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            obj_path = glob.glob(os.path.join(path, 'val_img', '*.jpg'))
            #   print(obj_path)

            with open(json_file, 'r') as fp:
                for p in fp:
                    #           print('come in!!!!!!!')
                    #           print(p)
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    #             if 'filename' in p:
                    #    print('filename')
                    #                 self.filen_ame = p.split('>')[1].split('<')[0]
                    #                        print(self.filen_ame)
                    self.filen_ame = tmp_file + '.jpg'

                    self.path = os.path.join(path, 'val_img', self.filen_ame.split('.')[0] + '.jpg')
                    #           print(self.path)
                    if self.path not in obj_path:
                        print('noooooooo!')
                        break

                    if 'width' in p:
                        #                        print('width')
                        self.width = int(p.split('>')[1].split('<')[0])
                    # print(self.width)
                    if 'height' in p:
                        #                        print('height')
                        self.height = int(p.split('>')[1].split('<')[0])
                        #                        print(self.height)

                        self.images.append(self.image())
                    # print(self.images)

                    if '<object>' in p:
                        #       print('object')
                        # 类别
                        #               d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                        #               print(d) # ['person', 'Unspecified', '1', '0', '\n', '1', '96', '191', '361']
                        tmp = [next(fp).split('>')[1].split('<')[0] for _ in range(7)]
                        #### add 'Unspecified' '1' ############
                        d = []
                        d.append(tmp[0])
                        d.append('Unspecified')
                        d.append('1')
                        d.append(tmp[1])
                        d.append(tmp[2])
                        d.append(tmp[3])
                        d.append(tmp[4])
                        d.append(tmp[5])
                        d.append(tmp[6])

                        #                        print(d)

                        #       print(d) # ['person', '0', '\n', '1', '96', '191', '361']
                        self.supercategory = d[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(d[-4]);
                        y1 = int(d[-3]);
                        x2 = int(d[-2]);
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                        self.annotations.append(self.annotation())
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    @staticmethod
    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.int32)

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID

        # 计算轮廓面积
        contour = PascalVOC2coco.change_format(annotation['segmentation'][0])
        #        print(contour)
        # 轮廓为空
        if len(contour) == 0:
            annotation['area'] = 1
        else:
            annotation['area'] = abs(cv2.contourArea(contour, True))
            #        print(annotation['area'])
            if annotation['area'] == 0:
                print('oooooooooo')
                annotation['area'] = 1

        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i;
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i;
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        '''从mask提取边界点'''
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


xml_file = glob.glob('crazing_1.xml')
# xml_file=['./Annotations/000032.xml']
# print(xml_file)

PascalVOC2coco(xml_file, './new.json')

