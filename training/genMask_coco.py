# -*- coding: utf-8 -*-
import json
import numpy as np
import cv2
import os
import sys
import numpy as np
import os
import tqdm
import time
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

MASK_DIR = "mask2014"
COCO_DIR = "/home/lqy/E/coco2014/"
data_types = ["val2014", "train2014"]



def isRayIntersectsSegment(poi,s_poi,e_poi):
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    # if s_poi[0]<poi[0] and e_poi[0]<poi[0]: #线段在射线左边
    #     return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/float(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False

    return True  #排除上述情况之后

def isPointInPolygon(pts, poly):
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    sinsc=0 #交点个数
    # for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    for i in range(len(poly)): #[0,len-1]
        # do not forget to compute the last line
        if i == len(poly) - 1:
            s_poi=poly[i]
            e_poi=poly[0]
        else:
            s_poi=poly[i]
            e_poi=poly[i+1]
        # print(s_poi, e_poi)
        if isRayIntersectsSegment(pts, s_poi, e_poi):
            sinsc+=1 #有交点就加1
    return sinsc%2==1


# def mask_generate(coco, img_info, annos, data_type):
#
#     img_path = os.path.join(COCO_DIR, os.path.join(data_type, "COCO_" + data_type + "_{:0>12d}.jpg".format(img_info["id"])))
#     img_mask_all = os.path.join(os.path.join(COCO_DIR, MASK_DIR), data_type + "_mask_all_{:0>12d}.png".format(img_info["id"]))
#     img_mask_miss = os.path.join(os.path.join(COCO_DIR, MASK_DIR), data_type + "_mask_miss_{:0>12d}.png".format(img_info["id"]))
#
#     w = img_info["width"]
#     h = img_info["height"]
#
#     mask_all = np.zeros((h, w), dtype=np.uint8)
#     mask_miss = np.zeros((h, w), dtype=np.uint8)
#     mask = np.zeros((h, w), dtype=np.uint8)
#
#     image = cv2.imread(img_path)
#     for i in range(len(annos)):
#         anno = annos[i]
#
#         if anno["iscrowd"] == 0:
#             seg = anno["segmentation"][0]
#             seg_x = [seg[i] for i in range(len(seg)) if i % 2 == 0]
#             seg_y = [seg[i] for i in range(len(seg)) if i % 2 == 1]
#             # print("------------------")
#             # print(seg)
#             # print("------------------")
#             # print(seg_x)
#             # print("------------------")
#             # print(seg_y)
#             seg = [[int(seg_x[i] + 0.5), int(seg_y[i] + 0.5)] for i in range(len(seg_x))]
#             # print(seg)
#         else:
#             print("RLE")
#             seg = coco.annToMask(anno)
#
#         mask[:,:] = 0
#         for m in range(h):
#             for n in range(w):
#                 if isPointInPolygon((n,m), seg):
#                     mask[m,n] = 1
#
#         mask_all = np.bitwise_or(mask_all, mask)
#         if anno["num_keypoints"] == 0:
#             print(anno["num_keypoints"])
#             mask_miss = np.bitwise_or(mask_miss, mask)
#         cv2.polylines(image, np.array([seg]), 1, (0, 0, 255), 2)
#
#     # for m in range(h):
#     #     for n in range(w):
#     #         print("{} ".format(mask_miss[m, n])),
#     #     print("\n")
#     mask_miss[np.where(mask_miss[:,:] == 1)] = 255
#     # each byte will be inverted, eg: 0==>255 1==>254
#     mask_miss = np.bitwise_not(mask_miss)
#     mask_miss[np.where(mask_miss[:,:] == 255)] = 1
#     # for m in range(h):
#     #     for n in range(w):
#     #         print("{} ".format(mask_miss[m, n])),
#     #     print("\n")
#
#     if 1:
#         mask_all[np.where(mask_all[:,:] == 1)] = 255
#         mask_miss[np.where(mask_miss[:,:] == 1)] = 255
#         cv2.imshow("mask_all", mask_all)
#         cv2.imshow("mask_miss", mask_miss)
#         cv2.imshow("origin", image)
#         cv2.waitKey(0)

def mask_generate(coco, img_info, annos, data_type):

    img_path = os.path.join(COCO_DIR, os.path.join(data_type, "COCO_" + data_type + "_{:0>12d}.jpg".format(img_info["id"])))
    img_mask_all_path = os.path.join(os.path.join(COCO_DIR, MASK_DIR), data_type + "_mask_all_{:0>12d}.png".format(img_info["id"]))
    img_mask_miss_path = os.path.join(os.path.join(COCO_DIR, MASK_DIR), data_type + "_mask_miss_{:0>12d}.png".format(img_info["id"]))

    w = img_info["width"]
    h = img_info["height"]

    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # image = cv2.imread(img_path)
    for i in range(len(annos)):
        anno = annos[i]
        # print(anno["num_keypoints"])
        if anno["iscrowd"] == 0:
            mask = coco.annToMask(anno)
            # # only use first segmentation
            # seg = anno["segmentation"][0]
            # seg_x = [seg[i] for i in range(len(seg)) if i % 2 == 0]
            # seg_y = [seg[i] for i in range(len(seg)) if i % 2 == 1]
            # seg = [[int(seg_x[i] + 0.5), int(seg_y[i] + 0.5)] for i in range(len(seg_x))]
        else:
            # print("RLE")
            if anno["num_keypoints"] != 0:
                print("!!!!!!!!!!!!ERROR!!!!!!!!!!!!")
            mask_rle = coco.annToMask(anno)
            mask_rle[np.where(mask_rle[:,:] == 1)] = 255
            cv2.imshow("mask_rle", mask_rle)
            continue

        # mask[:,:] = 0
        # for m in range(h):
        #     for n in range(w):
        #         if isPointInPolygon((n,m), seg):
        #             mask[m,n] = 1

        mask_all = np.bitwise_or(mask_all, mask)
        if anno["num_keypoints"] < 5:
            mask_miss = np.bitwise_or(mask_miss, mask)


        # if 1:
        #     if anno["iscrowd"] == 0:
        #         for s in range(len(anno["segmentation"])):
        #             seg = anno["segmentation"][s]
        #             seg_x = [seg[i] for i in range(len(seg)) if i % 2 == 0]
        #             seg_y = [seg[i] for i in range(len(seg)) if i % 2 == 1]
        #             seg = [[int(seg_x[i] + 0.5), int(seg_y[i] + 0.5)] for i in range(len(seg_x))]
        #             # print(seg)
        #             cv2.polylines(image, np.array([seg]), 1, (0, 0, 255), 2)
        #
        #         if anno["num_keypoints"] > 0:
        #             for k in range(17):
        #                 x = anno["keypoints"][3*k]
        #                 y = anno["keypoints"][3*k + 1]
        #                 v = anno["keypoints"][3*k + 2]
        #                 # print(x,y,v)
        #                 if v != 0:
        #                     cv2.circle(image, (x, y), 2, (0, 255, 0), 2)


    mask_miss[np.where(mask_miss[:,:] == 1)] = 255
    # each byte will be inverted, eg: 0==>255 1==>254
    mask_miss = np.bitwise_not(mask_miss)
    mask_miss[np.where(mask_miss[:,:] == 255)] = 1

    # if 1:
    #     mask_all[np.where(mask_all[:,:] == 1)] = 255
    #     mask_miss[np.where(mask_miss[:,:] == 1)] = 255
    #     cv2.imshow("mask_all", mask_all)
    #     cv2.imshow("mask_miss", mask_miss)
    #     cv2.imshow("origin", image)
    #     cv2.waitKey(0)
    cv2.imwrite(img_mask_all_path, mask_all)
    cv2.imwrite(img_mask_miss_path, mask_miss)

# plt.figure(1)

if not os.path.exists(COCO_DIR + MASK_DIR):
    os.mkdir(COCO_DIR + MASK_DIR)

for sel in range(len(data_types)):
    data_type = data_types[sel]
    ann_file = os.path.join(COCO_DIR, 'annotations/person_keypoints_{}.json'.format(data_type))
    coco = COCO(ann_file)
    catIds = coco.getCatIds(catNms=['person'])
    # print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)

    print("--------------------------------------------------------------------")
    print("@@@ proc {} @@@".format(ann_file))
    print("1. parse keypoints anno")
    for i in tqdm.tqdm(range(len(imgIds))):
        img_info = coco.loadImgs(imgIds[i])[0]
        # print(img_info)
        annoIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)

        # img_path = os.path.join(COCO_DIR,
        #                         os.path.join(data_type, "COCO_" + data_type + "_{:0>12d}.jpg".format(img_info["id"])))
        # image = cv2.imread(img_path)
        # plt.imshow(image[:,:,::-1])
        # print(annoIds)
        annos = coco.loadAnns(annoIds)
        # print(annos)
        # coco.showAnns(annos)
        # plt.show()

        mask_generate(coco, img_info, annos, data_type)