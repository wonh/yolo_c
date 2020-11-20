from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.distributed as dist

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from dataset_bak import NetJam_dataset
from tqdm import tqdm
import time
import shutil
from torchvision import transforms as T


if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(init_method='tcp://127.0.0.1:23567', backend="nccl", world_size=1, rank=0,
                            group_name="pytorch_test")

    # os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    model.eval()  # Set in evaluation mode
    normalizer = transforms.Normalize(mean=[0.265, 0.279, 0.423],
                                      std=[0.209, 0.217, 0.297])
    transformer = T.Compose([
        # transforms.ColorJitter(0.1, 0.1, 0.1),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.ToTensor(),
        normalizer,
    ])
    dataset = ImageFolder2(opt.image_folder,transform=transformer)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        sampler=sampler,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("\nPerforming object detection:")
    print('total image:', len(dataset))
    detect_start = time.time()
    prev_time = time.time()

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    polyp_num = 0
    data_path = ''
    #================查重==================
    with open(r'./data/train_3w6r+pinp1+cds.txt','r') as f:
        lines = f.readlines()
    train_files = [
        path.split('/')[-1].split('\n')[0] for path in lines
    ]

    with open(r'./data/val_1470.txt','r') as f:
        val_lines = f.readlines()
    val_files = [
        path.split('/')[-1].split('\n')[0] for path in val_lines
    ]
    print(val_files)

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            img_detections = model(input_imgs)
            img_detections = non_max_suppression2(img_detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        for (path, image, detections) in zip(img_paths, input_imgs, img_detections):
            polyp_flag = 0
            tmp = ''
            # img = np.asarray(transforms.ToPILImage()(image.cpu()))
            # img = transforms.ToPILImage()(image.cpu())
            basename = os.path.basename(path)
            if basename in train_files or basename in val_files:
                print(basename)
                continue
            id1 = path.split('/')[-2]
            if 'polyp' in id1:
                print(id1)
                continue
            img = np.array(Image.open(path))
            if detections is not None:
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                # h,w = np.asarray(img).shape[1:]
                # print(h,w)
                h=w=416
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1,y1,x2,y2 = (x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy())
                    # if int(cls_pred) not in [0,1,2,5,6]:
                    if int(cls_pred) not in [0]:
                        continue
                    else:
                        if cls_conf < 0.8:
                            continue
                        else:
                            polyp_flag = 1
                            n_data = [str(round((x1 + x2) / 2.0 / w, 6)), str(round((y1 + y2) / 2.0 / h, 6)),
                                      str(round((x2 - x1) / w, 6)), str(round((y2 - y1) / h, 6))]
                            tmp += str(int(cls_pred)) + ' ' + (' ').join(n_data) + '\n'

            filename = path.split("/")[-1]
            id = path.split("/")[-4]
            if polyp_flag:
                polyp_num += 1
                dst_path = r'./data/fake-4571/images/{}'.format(filename)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                Image.fromarray(img).save(dst_path)
                # shutil.copy(path, dst_path)
                if tmp != '':
                    data_path += r'./data/fake-4571/images/{}'.format(filename) + '\n'
                    with open(r'./data/fake-4571/labels/{}'.format(filename.replace('.jpg','.txt').replace('.Jpeg','.txt')), 'w') as f:
                        f.write(tmp)
        #==================================================================================================
    with open('data/sus/train.txt', 'w') as f:
        f.write(data_path)
    end_time = time.time()
    print('total time', end_time - start_time)
    print('prepare time', detect_start - start_time)
    print('detect time', end_time - detect_start)
    print(polyp_num)

