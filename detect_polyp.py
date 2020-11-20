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

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/my.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    dataset = ImageFolder(opt.image_folder, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    polyp_path1 = r'../data/info/src/29.80-polyp-and-fake/polyp'
    polyp_path2 = r'../data/info/src/29.80-polyp-406'
    polyp_name_406 = []
    polyp_name_569 = []
    final_name = {}
    num = 0
    for root, dirs, files in os.walk(polyp_path2):
        for file in files:
            basename = os.path.basename(file)
            polyp_name_406.append(file)
    print('polyp_num', len(polyp_name_406))
    for root, dirs, files in os.walk(polyp_path1):
        for file in files:
            basename = os.path.basename(file)
            polyp_name_569.append(file)
    print('polyp_num', len(polyp_name_569))

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            time1 = time.time()
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            time2 = time.time()

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        print('nms time:', time2-time1)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    polyp_num_406 = 0
    polyp_num_569 = 0
    fake_polyp_num = 0
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        polyp_flag = 0
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        basename = os.path.basename(path)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # result_list.append({"image_id": id_num, "category_id": int(cls_pred), "bbox": [x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()], "score": float(cls_conf)})
                if int(cls_pred)==0 and cls_conf > 0.5 and conf > 0.8:
                    polyp_flag = 1
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    plt.text(
                        x1,
                        y1-30,
                        s = classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                else:
                    continue

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        if polyp_flag:
            if basename in polyp_name_406:
                polyp_num_406 += 1
            if basename in polyp_name_569:
                polyp_num_569 += 1
                # plt.savefig("output/polyp/{}.png".format(basename), bbox_inches="tight", pad_inches=0.0)
            else:
                fake_polyp_num += 1
        # else:
        #     plt.savefig("output/fake-polyp/{}.png".format(basename), bbox_inches="tight", pad_inches=0.0)

    print('total file:{}, polyp_num_406:{}, polyp_num_569:{}, fake_polyp_num:{}'.format(len(dataset), polyp_num_406, polyp_num_569, fake_polyp_num))
