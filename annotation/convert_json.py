import json
import os
import cv2
import numpy as np
from tqdm import tqdm
# 原始标签路径
originLabelsDir = r'E:/whr/yolo/Label_1w6/data_all_6/labels/train'
# 转换后的文件保存路径
saveDir = r'E:/whr/yolo/Label_1w6/data_all_6/annos.txt'

# 原始标签对应的图片路径
originImagesDir = r'E:/whr/yolo/Label_1w6/jpg_all'

txtFileList = os.listdir(originLabelsDir)
with open(saveDir, 'w') as fw:
    for txtFile in tqdm(txtFileList):
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                imagePath = os.path.join(originImagesDir,
                                         txtFile.replace('txt', 'jpg'))
                # print(imagePath)
                # image = cv2.imread(imagePath)
                image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)
                H, W, _ = image.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 为了与coco标签方式对，标签序号从1开始计算
                fw.write(txtFile.replace('txt', 'jpg') + ' {} {} {} {} {}\n'.format(int(label[0]) + 1, x1, y1, x2, y2))

# ------------用os提取images文件夹中的图片名称，并且将BBox都读进去------------
# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),
# 以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = r'E:/whr/yolo/Label_1w6/data_all_6'
# 用于创建训练集或验证集
phase = 'train'  # 需要修正

# dataset用于保存所有数据的图片信息和标注信息
dataset = {'categories': [], 'annotations': [], 'images': []}

# 打开类别标签
with open(r'E:\whr\yolo\Label_1w6\my.name') as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
indexes = os.listdir(os.path.join(root_path, 'labels/val/'))

# 统计处理图片的数量
global count
count = 0

# 读取Bbox信息
with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

    # ---------------接着将，以上数据转换为COCO所需要的格式---------------
    for k, index in enumerate(tqdm(indexes)):
        count += 1
        # 用opencv读取图片，得到图像的宽和高
        img = os.path.join(r'E:/whr/yolo/Label_1w6/jpg_all', index.replace("txt","jpg"))
        im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
        height, width, _ = im.shape

        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index.replace("txt","jpg"),
                                  'id': k,
                                  'width': width,
                                  'height': height})

        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

            # 如果图像的名称和标记的名称对上，则添加标记
            if parts[0] == index:
                # 类别
                cls_id = parts[1]
                # x_min
                x1 = float(parts[2])
                # y_min
                y1 = float(parts[3])
                # x_max
                x2 = float(parts[4])
                # y_max
                y2 = float(parts[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': i,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

        print('{} images handled'.format(count))

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
