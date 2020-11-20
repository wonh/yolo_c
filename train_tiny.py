from __future__ import division

from models import *
# from utils.logger import Logger as logger
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate, evaluate2

from terminaltables import AsciiTable
#from apex.parallel import DistributedDataParallel
# import horovod.torch as hvd
from tensorboardX import SummaryWriter

import os
import sys
import time
import datetime
import argparse

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.ADAM import Adam_GC

lossMin = 1.0
mapMax = 0.0
lossweight = ''
mapweight = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    #=========================multi processing training====================================
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--start_epoch',default=0,type=int)
    parser.add_argument('--lr',help="learning_rate", default=0.001, type=float)
    parser.add_argument('--model_name', '-mn', default='yolov3_ckpt_')
    parser.add_argument('--flip', '-f', default=False)
    opt = parser.parse_args()
    print(opt)

    #多线程初始化
    dist.init_process_group(init_method=opt.init_method, backend="nccl", world_size=opt.world_size, rank=opt.rank,
                            group_name="pytorch_yolo")
    # os.makedirs("output", exist_ok=True)
    # os.makedirs("checkpoints", exist_ok=True)
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)
    # Initiate model
    model = Darknet(opt.model_def)
    print('model created')
    model.apply(weights_init_normal)
    print('模型初始化完成')
    # sync_initial_weights(model, opt.rank, opt.world_size)
    model = model.cuda()
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    print('开始分发模型')
    # 分发模型
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=False)
    # model = DistributedDataParallel(model)
    print('模型分发完成',model)

    # Get dataloader
    dataset = ListDataset(train_path, augment=opt.flip, multiscale=opt.multiscale_training)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        sampler=train_sampler,
        drop_last=True,
    )

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = opt.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = Adam_GC(model.parameters(), lr=opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True, factor=0.3, patience=3,min_lr=1e-3)
    writer = SummaryWriter('./tensorboard/'+opt.model_name)
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
        epoch += opt.start_epoch
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        epoch_loss = 0
        # adjust_learning_rate(optimizer,epoch)
        for batch_i, ( imgs, targets) in enumerate(tqdm(dataloader)):
            batches_done = len(dataloader) * epoch + batch_i  # len(dataloader) = 1 epoch
            start_time = time.time()
            imgs = Variable(imgs.cuda(), requires_grad=False)
            loss, outputs = model(imgs, targets)
            forward_time = time.time()
            loss = loss.mean()
            loss.backward()
            backward_time = time.time()
            # if batches_done % opt.gradient_accumulations:
            optimizer.step()
            optimizer.zero_grad()
            update_time = time.time()
            epoch_loss += loss.item()
            log_str = "---- [epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            log_str += f"Total loss {loss.item()}\n"

            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += " ---- ETA {}".format(time_left)
            print(log_str, end='\r')
            model.module.seen += imgs.size(0)
            stop_time = time.time()

        writer.add_scalar('epoch_loss', epoch_loss / len(dataloader), epoch)
        if lossMin >= loss.item():
            if lossweight != '':
                os.remove(lossweight)
            lossMin = loss.item()
            # torch.save(model.module.state_dict(), "checkpoints/{}_loss_{:.4f}.pth".format(opt.model_name, lossMin))
            torch.save(model.module.state_dict(), "checkpoints/{}_min_loss.pth".format(opt.model_name))
            lossweight = "checkpoints/{0}_loss_{1}.pth".format(opt.model_name, lossMin)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # caculate traing loss

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

            precision2, recall2, AP2, f12, ap_class2 = evaluate(
                model,
                train_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )


            # Print class APs and mAP
            if isinstance(f1, int):
                print("---- mAP 0")
            else:
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                writer.add_scalars("val",{"test_precision": precision.mean(),
                                            "test_recall": recall.mean(),
                                            "test_mAP": AP.mean(),
                                            "test_f1": f1.mean(),
                                            "train_mAP":AP2.mean(),
                                          "train_recall":recall2.mean(),
                                          "train_precision":precision2.mean(),
                                          "train_f1":f12.mean(),
                                          }, epoch)
            # Evaluate the model on the train set
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                # print(AsciiTable(ap_table).table)
                print("---- mAP {}".format(AP.mean()))
                if mapMax <= AP.mean():
                    # if mapweight != '':
                    #     os.remove(mapweight)
                    mapMax = AP.mean()
                    # torch.save(model.module.state_dict(),"checkpoints/{}_map_{:.4f}.pth".format(opt.model_name, mapMax))
                    torch.save(model.module.state_dict(),"checkpoints/{}_max_map_.pth".format(opt.model_name))
                    # mapweight = os.path.join('/mnt/whr/yolo/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master',
                    #                          "checkpoints/yolov3_ckpt_map_%.4f.pth" % mapMax)
                    mapweight = "checkpoints/{}_{:.4f}.pth".format(opt.model_name, lossMin)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.module.state_dict(), "checkpoints/{0}_{1}.pth".format(opt.model_name, epoch))


