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
from dataset_bak import Netdataset3
from tqdm import tqdm
import time
import shutil
from torchvision import transforms


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
    dist.init_process_group(init_method='tcp://127.0.0.1:23456', backend="nccl", world_size=1, rank=0,
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
    dataset = Netdataset3(opt.image_folder)
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

    # imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    print('nums',len(dataloader))
    detect_start = time.time()
    prev_time = time.time()

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    polyp_num = 0

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            img_detections = model(input_imgs)
            img_detections = non_max_suppression(img_detections, 0.4, 0.4)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        for (path, image, detections) in zip(img_paths, input_imgs, img_detections):
            polyp_flag = 0
            plt.figure()
            fig, ax = plt.subplots(1)
            # img = np.array(Image.open(path))
            # print(image.shape)
            img = np.asarray(transforms.ToPILImage()(image.cpu()))
            ax.imshow(img)
            if detections is not None:
                # detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # if cls_pred != 8 and cls_pred != 9:
                    if int(cls_pred) != 0:
                        continue
                    else:
                        polyp_flag = 1
                        if cls_conf < 0.9:
                            continue
                        else:

                            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                            box_w = x2 - x1
                            box_h = y2 - y1
                            print('x1: {}, y1: {}, x2: {}, y2: {}'.format(x1, y1, x2, y2))
                            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            # Create a Rectangle patch
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color,
                                                     facecolor="none")
                            # Add the bbox to the plot
                            ax.add_patch(bbox)
                            # Add label
                            plt.text(
                                x1,
                                y1 - 30,
                                s=classes[int(cls_pred)],
                                color="white",
                                verticalalignment="top",
                                bbox={"color": color, "pad": 0},
                            )
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            id = path.split("/")[-4]
            if polyp_flag:
                dst_path = os.path.join(opt.image_folder, id, 'yolo4', "p", "{}.jpg".format(filename))
                polyp_num += 1
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                plt.savefig(dst_path, bbox_inches="tight", pad_inches=0.0)
                plt.close()
            else:
                dst_path = os.path.join(opt.image_folder, id, 'yolo4', "f", "{}.jpg".format(filename))
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(path,dst_path)

        #==================================================================================================
    end_time = time.time()
    print('total time', end_time - start_time)
    print('prepare time', detect_start - start_time)
    print('detect time', end_time - detect_start)
    print(polyp_num)
    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    #
    # print("\nSaving images:")
    # # Iterate through images and save plot of detections
    # polyp_num = 0
    # fake_polyp_num = 0
    # draw_start = time.time()
    # for img_i, (path, detections) in tqdm(enumerate(zip(imgs, img_detections))):
    #
    #     print("(%d) Image: '%s'" % (img_i, path))
    #
    #     # Create plot
    #     img = np.array(Image.open(path))
    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)
    #
    #     polyp_flag = 0
    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
    #         detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
    #         unique_labels = detections[:, -1].cpu().unique()
    #         n_cls_preds = len(unique_labels)
    #         bbox_colors = random.sample(colors, n_cls_preds)
    #
    #         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    #             # if cls_pred != 8 and cls_pred != 9:
    #             if int(cls_pred)==0:
    #                 # polyp_num += 1
    #                 polyp_flag = 1
    #             # elif int(cls_pred) == 10:
    #             #     fake_polyp_num += 1
    #             else:
    #                 continue
    #             print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    #             box_w = x2 - x1
    #             box_h = y2 - y1
    #             print('x1: {}, y1: {}, x2: {}, y2: {}'.format(x1, y1, x2, y2))
    #
    #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #             # Create a Rectangle patch
    #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    #             # Add the bbox to the plot
    #             ax.add_patch(bbox)
    #             # Add label
    #             plt.text(
    #                 x1,
    #                 y1-30,
    #                 s = classes[int(cls_pred)],
    #                 color="white",
    #                 verticalalignment="top",
    #                 bbox={"color": color, "pad": 0},
    #             )
    #
    #     # Save generated image with detections
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #     filename = path.split("/")[-1].split(".")[0]
    #     id = path.split("/")[-4]
    #     if polyp_flag:
    #         dst_path = os.path.join(opt.image_folder, id, 'yolo', "p", "{}.jpg".format(filename))
    #         polyp_num += 1
    #     else:
    #         dst_path = os.path.join(opt.image_folder, id, 'yolo', "f", "{}.jpg".format(filename))
    #     os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #     plt.savefig(dst_path, bbox_inches="tight", pad_inches=0.0)
    #     plt.close()
    # print(polyp_num, fake_polyp_num)
    # end_time = time.time()
    # print('total time ', end_time - start_time)
    # print('prepare time ', detect_start - start_time)
    # print('detect time ', draw_start - detect_start)
    # print('draw time ', end_time - draw_start)
