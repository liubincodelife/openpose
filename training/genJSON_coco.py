import json
import numpy as np
import cv2
import os
import sys
import numpy as np
import os
import tqdm
import time
from pycocotools.coco import COCO

COCO_DIR = "/home/lqy/E/coco2014/"

data_types = ["val2014", "train2014"]

# COCO:(  1- 'nose'    2-'left_eye' 3-'right_eye' 4-'left_ear' 5-'right_ear'
#         6-'left_shoulder' 7-'right_shoulder'    8-'left_elbow' 9-'right_elbow' 10-'left_wrist'
#         11-'right_wrist'    12-'left_hip' 13-'right_hip' 14-'left_knee'    15-'right_knee'
#         16-'left_ankle' 17-'right_ankle')

index_val = 0.0
index_train = 0.0

def joints_generate(img_info, annos, data_type, joint_annos):

    global index_val
    global index_train
    prev_center_list = []
    for i in range(len(annos)):
        anno = annos[i]

        # skip small person and person marked too few keypoints
        if anno["num_keypoints"] < 5 or anno["area"] < 32*32:
            continue

        x,y,w,h = anno["bbox"]
        center = [float(x + w/2), float(y + h/2)]

        flag = 0
        for j in range(len(prev_center_list)):
            dist = [prev_center_list[j][0] - center[0], prev_center_list[j][1] - center[1]]
            if dist[0]*dist[0] + dist[1]*dist[1] < prev_center_list[j][2]*0.3:
                flag = 1
                break

        if flag == 1:
            continue

        prev_center_list.append([center[0], center[1], max(w, h)])

        if data_type == "val2014" and index_val < 2600:
            validation = 1
            index_val += 1.0
        else:
            validation = 0
            index_train += 1.0

        if data_type == "val2014":
            dataset = "COCO_val"
        else:
            dataset = "COCO"

        # generate self info
        joint_one = {}
        joint_one["dataset"] = dataset
        joint_one["img_paths"] = os.path.join(data_type, "COCO_" + data_type + "_{:0>12d}.jpg".format(img_info["id"]))
        joint_one["img_width"] = float(img_info["width"])
        joint_one["img_height"] = float(img_info["height"])
        joint_one["isValidation"] = validation
        joint_one["objpos"] = center
        joint_one["image_id"] = img_info["id"]
        joint_one["bbox"] = anno["bbox"]
        joint_one["segment_area"] = anno["area"]
        joint_one["num_keypoints"] = anno["num_keypoints"]
        # joint_one["joint_self"] = np.zeros((3,17), dtype=np.int32)
        joint_one["people_index"] = i
        if validation == 1:
            joint_one["annolist_index"] = index_val - 1.0
        else:
            joint_one["annolist_index"] = index_train - 1.0

        joint_self = np.zeros((3,17), dtype=np.int32)

        for k in range(17):
            joint_self[0][k] = anno["keypoints"][k*3]
            joint_self[1][k] = anno["keypoints"][k*3 + 1]
            if anno["keypoints"][k*3 + 2] == 2:
                joint_self[2][k] = 1
            elif anno["keypoints"][k*3 + 2] == 0:
                joint_self[2][k] = 0
            else:
                joint_self[2][k] = 2

        joint_one["joint_self"] = joint_self.tolist()
        joint_one["scale_provided"] = joint_one["bbox"][3]/368.0

        joint_others_list = []
        scale_privided_others_list = []
        objpos_others_list = []
        # generate others info
        for m in range(len(annos)):
            if m == i or annos[m]["num_keypoints"] == 0:
                continue

            anno_other = annos[m]

            x, y, w, h = map(int, anno_other["bbox"])
            scale_privided_other    = h / 368.0
            objpos_other            = [float(x + w/2), float(y + h/2)]

            joint_other             = np.zeros((3, 17), dtype=np.int32)
            for k in range(17):
                joint_other[0][k] = anno_other["keypoints"][k * 3]
                joint_other[1][k] = anno_other["keypoints"][k * 3 + 1]
                if anno_other["keypoints"][k * 3 + 2] == 2:
                    joint_other[2][k] = 1
                elif anno_other["keypoints"][k * 3 + 2] == 0:
                    joint_other[2][k] = 0
                else:
                    joint_other[2][k] = 2

            scale_privided_others_list.append(scale_privided_other)
            objpos_others_list.append(objpos_other)
            joint_others_list.append(joint_other.tolist())

        joint_one["joint_others"] = joint_others_list
        joint_one["objpos_other"] = objpos_others_list
        joint_one["scale_provided_other"] = scale_privided_others_list
        joint_one["numOtherPeople"] = len(objpos_others_list)

        # print(joint_one)

        joint_annos.append(joint_one)


        # image = cv2.imread(os.path.join(COCO_DIR, joint_one["img_paths"]))
        #
        # for k in range(17):
        #     x = joint_one["joint_self"][0][k]
        #     y = joint_one["joint_self"][1][k]
        #     v = joint_one["joint_self"][2][k]
        #     # print(x,y,v)
        #     if v != 0:
        #         cv2.circle(image, (x,y), 2, (0,0,255), 2)
        #
        # for i in range(joint_one["numOtherPeople"]):
        #     joint = joint_one["joint_others"][i]
        #
        #     for j in range(17):
        #         x = joint[0][j]
        #         y = joint[1][j]
        #         v = joint[2][j]
        #         # print(x,y,v)
        #         if v != 0:
        #             cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
        #
        # cv2.imshow("Person{}_keypoints".format(joint_one["people_index"]), image)
        # cv2.waitKey(0)

    # return joints_list



joint_annos = []
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
        # print(annoIds)
        annos = coco.loadAnns(annoIds)
        # print(annos)
        # coco.showAnns(annos)

        joints_generate(img_info, annos, data_type, joint_annos)



print("2. convert to jason string")
joints_jason = json.dumps(joint_annos, indent = 4)
print("3. save jason")
json_file = open(os.path.join(COCO_DIR, "coco2014.json"), "w")
json_file.writelines(joints_jason)
json_file.close()
print("train anno = {}".format(index_train))
print("val anno = {}".format(index_val))
print("--------------------------------------------------------------------")


