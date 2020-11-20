# -*- coding: utf-8 -*-

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    testset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=testset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # print("length of dataloder{}".format(len(dataloader)))
    total_loss = 0
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # img: (batch_size, channel, height, width)
        # target: (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
        # print("lengh of imgs:{0}, targets:{1}".format(len(imgs), len(targets)))
        new_target = []
        for i in range(len(targets)):
            boxes = targets[i, :]
            boxes = boxes.view(-1, 6)
            boxes[:, 0] = i
            boxes = boxes[boxes[:, 1:].sum(dim=1) != 0] #remove all 0 item
            new_target.append(boxes)
        targets = torch.cat(new_target, 0)

        # Extract labels
        labels += targets[:, 1].tolist()  # num 个 cls
        # Rescale target
        # (num, center_x, center_y, widht, height) 把x,y,w,h格式转换为x1,y1,x2,y2的格式，注意，转换后该格式依旧为归一化的值
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        # 将归一化的x1,y1,x2,y2转为真实坐标
        targets[:, 2:] *= img_size

        # imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = Variable(imgs.type(Tensor))
        # change
        targets = Variable(targets.type(Tensor))

        with torch.no_grad():
            # (batch_size, num_anchors*grid_size*grid_size*3, 85)
            # outputs = model(imgs)
            outputs = model(imgs)
            # print("non_max_suppression")
            # print("output shape",outputs.size())
            # (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # outputs: (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
        # target:  (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
        # print("get batch statistics")
        # print("length of targets:{}".format(len(targets)))
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    # 将所有图片的预测信息进行concatenate，每张图片包含了true_positives, pred_scores, pred_labels 这三个信息
    # true_positives：预测框的正确与否，正确设置为1，错误设置为0
    # pred_scores：预测框的x,y,w,h
    # pred_labels：预测框的类别标签
    if sample_metrics:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        precision, recall, AP, f1, ap_class = [0, 0, 0, 0, 0]
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model = torch.nn.DataParallel(model)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    # model.cuda()
    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, total_loss = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP: {}".format(c, class_names[c], AP[i]))

    print("mAP: {}".format(AP.mean()))

