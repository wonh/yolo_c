from __future__ import division

from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

lossMin = 1.0
mapMax = 0.0
lossweight = ''
mapweight = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output", exist_ok=True)
    # os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    model = torch.nn.DataParallel(model)
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # 添加drop_last=True，遗弃最后不够batch_size的数据
    #=============================================================
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (img_pth, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            # img: (batch_size, channel, height, width)
            # target: (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
            imgs = Variable(imgs.to(device))
            targets = targets.to(device)

            loss, outputs = model(imgs, targets)
            # loss = loss.mean()# 添加
            loss.mean().backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            log_str = "---- [Epoch %d/%d, Batch %d/%d] ----" % (epoch, opt.epochs, batch_i, len(dataloader))
            # metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.module.yolo_layers))]]]
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.module.yolo_layers]
                # metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.module.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [("{}_{}".format(name, j+1), metric)]
                tensorboard_log += [("loss", loss.mean())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            log_str += " Total loss {}".format(loss.mean())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += " ---- ETA {}".format(time_left)
            print(log_str,end='\r')
            model.module.seen += imgs.size(0)

            # if batch_i > 10:
            #     break
        if lossMin >= loss.mean().item():
            if lossweight != '':
                os.remove(lossweight)
            lossMin = loss.mean().item()
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_loss_%.4f.pth" % lossMin)
            lossweight = os.path.join('/mnt/whr/yolo/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master',"checkpoints/yolov3_ckpt_loss_%.4f.pth" % lossMin)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            # Print class APs and mAP
            if isinstance(f1,int):
                print("---- mAP 0")
            else:
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                # logger.list_of_scalars_summary(evaluation_metrics, epoch)
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                # print(AsciiTable(ap_table).table)
                print("---- mAP {}".format(AP.mean()))
                if mapMax <= AP.mean():
                    if mapweight != '':
                        os.remove(mapweight)
                    mapMax = AP.mean()
                    torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_map_%.4f.pth" % mapMax)
                    mapweight = os.path.join('/mnt/whr/yolo/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master',"checkpoints/yolov3_ckpt_map_%.4f.pth" % mapMax)
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_%d.pth" % epoch)
