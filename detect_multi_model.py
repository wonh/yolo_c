from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
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
import time
import torchvision.transforms as T

if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/my.name", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(init_method='tcp://127.0.0.1:23455', backend="nccl", world_size=1, rank=0,
                            group_name="pytorch_test")

    # Set up model
    def create_model(weights_path, model_def):
        model = Darknet(model_def, img_size=opt.img_size).to('cuda')
        model.load_state_dict(torch.load(weights_path))
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        model.eval()  # Set in evaluation mode
        return model
    model_config = [
        # (r'./checkpoints/csp_pin_map_0.5634.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_pin_map_0.6178.pth',r'./config/csp.cfg'),
        # (r'./checkpoints/csp_map_0.5163.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_99.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/spp-rand_map_0.6000921792692367.pth', r'./config/spp.cfg'),
        # (r'./checkpoints/spp_img_map_0.5354.pth', r'./config/spp.cfg'),
        # (r'./checkpoints/csp_rand_3w6+cds_270.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_rand_3w6+cds_95.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_random_3w6+pin_155.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_random_3w6+pin_170.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_rand+3w6_170.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_rand_3w6+pin+cds_105.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_7w2_0.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/yolov3_7w2_110.pth', r'./config/yolov3.cfg'),
        # (r'./checkpoints/spp_7w2_95.pth', r'./config/spp.cfg'),
        # (r'./checkpoints/tiny_7w2_110.pth', r'./config/yolov3-tiny.cfg'),
        # (r'./checkpoints/yolov3_3w6+pinp1+cds_85.pth', r'./config/yolov3.cfg'),#
        # (r'./checkpoints/csp_3w6+pinp1+cds_65.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/spp_3w6+pinp1+cds_130.pth', r'./config/spp.cfg'),#第一版
        # (r'./checkpoints/spp_3w6+pinp1+cds_135.pth', r'./config/spp.cfg'),#第二版
        # (r'./checkpoints/tiny_3w6+pin_max_map.pth', r'./config/yolov3-tiny.cfg'),
        # (r'./checkpoints/yolov3_3w6r+pinp1+cds_90.pth', r'./config/yolov3.cfg'),
        (r'./checkpoints/csp_3w6r+pinp1+cds_70.pth', r'./config/csp.cfg'),#或者55
        # (r'./checkpoints/spp_3w6r+pinp1+cds_60.pth', r'./config/spp.cfg'),
        # (r'./checkpoints/csp_3w6r+pinp1+cds_140_95.pth', r'./config/csp_140.cfg'),
    ]
    models = []
    models_detections = []
    result_list = []
    save_name = 'csp_3w6r+cds+pinp1'
    for i, (model_def, weights_path) in enumerate(model_config):
        model = create_model(model_def, weights_path)
        models.append(model)
        models_detections.append([])
        result_list.append([])
        print('create model {}'.format(i))
    # dataset = Netdataset3(opt.image_folder)
    normalizer = transforms.Normalize(mean=[0.265, 0.279, 0.423],
                                      std=[0.209, 0.217, 0.297])
    transformer = T.Compose([
        # transforms.ColorJitter(0.1, 0.1, 0.1),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.ToTensor(),
        normalizer,
    ])
    dataset_406 = ImageFolder(r'../data/info/src/polyp-bland-test/29.80-polyp-406', img_size=opt.img_size,transform=transformer)
    dataset_569 = ImageFolder(r'../data/info/src/29.80-polyp-and-fake/polyp', img_size=opt.img_size, transform=transformer)
    # dataset_569 = ImageFolder(r'../data/info/src/29.80-polyp-and-fake/polyp', img_size=opt.img_size, transform=None)
    dataset_3119 = ImageFolder(r'../data/info/src/29.80-polyp-and-fake/fake-polyp', img_size=opt.img_size, transform=transformer)
    dataset_2879 = ImageFolder(r'../data/info/src/polyp-bland-test/jam_polyp-1500', img_size=opt.img_size, transform=transformer)#polyp
    dataset_3285 = ImageFolder(r'../data/info/src/polyp-bland-test', img_size=opt.img_size, transform=transformer)  #polyp
    dataset_3600 = ImageFolder(r'../data/info/src/fake3600', img_size=opt.img_size)
    dataset_bland = ImageFolder2(r'../data/info/src/polyp-bland-test', img_size=opt.img_size,transform=transformer)  # polyp
    dataset = ImageFolder2(opt.image_folder, transform=transformer)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )

    dataloader_406 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_406,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_569 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_569,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_3119 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_3119,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_3600 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_3600,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_2879 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_2879,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_3285 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_3285,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )
    dataloader_bland = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_bland,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        # sampler=sampler,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    print('nums',len(dataloader_406))
    detect_start = time.time()
    prev_time = time.time()

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    polyp_num = 0

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

    # polyp_num_406 = 0
    # polyp_num_569 = 0
    # fake_polyp_num = 0
    model_predict = []
    # ============json================
    import json
    import pickle
    with open(r'json/test_Annotation_fake_polyp.json') as f:
        fp_dict = json.load(f)
    # with open(r'json/test_Annotation_3285.json') as f:
    # with open(r'json/test_Annotation_polyp.json') as f:
    with open(r'json/test_Annotation_polyp-bland-test2.json') as f:
        p_dict = json.load(f)
    fp_images = fp_dict['images']
    image_to_id = {}
    for d in fp_images:
        image_to_id.update({d['file_name']: d['id']})
    p_images = p_dict['images']
    for d in p_images:
        image_to_id.update({d['file_name']: d['id']})

    # ================================
    from tqdm import tqdm
    for batch_i, (img_paths, input_imgs) in enumerate(tqdm(dataloader_bland)):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            time0 = time.time()
            for i, model in enumerate(models):
                detections = models[i](input_imgs)
                time1 = time.time()
                detections = non_max_suppression2(detections, opt.conf_thres, opt.nms_thres)
                time2 = time.time()
                models_detections[i].extend(detections)
        print(time2-time1,time1-time0, time2 - time0)

        # Save image and detections
        imgs.extend(img_paths)

    for i, (path, _) in enumerate(zip(imgs,imgs)):
        plt.figure()
        fig, ax = plt.subplots(1)
        img = np.array(Image.open(path))
        # print(image.shape)
        # img = np.asarray(transforms.ToPILImage()(image.cpu()))
        ax.imshow(img)
        model_num = 0
        model_num_fake = 0
        basename = os.path.basename(path)
        try:
            id_num = image_to_id[basename]
        except:
            continue
        # print('test',len(models_detections),len(models_detections[0]))
        for j, model_detection in enumerate(models_detections):  #(model_num, len(file))
            polyp_flag = 0
            fake_polyp_flag = 0
            # for detections in model_detection:
            detections = model_detection[i]
            # Draw bounding boxes and labels of detection
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    result_list[j].append({"image_id": id_num, "category_id": int(cls_pred), "bbox": [x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()], "score": float(cls_conf)})
                    if int(cls_pred) == 0 and cls_conf > 0 and conf > 0.5:
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
                            y1 - 30,
                            s=classes[int(cls_pred)]+'{:.4f}'.format(conf.item()),
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )
                    if int(cls_pred) ==10:
                        fake_polyp_flag = 1
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
                        y1 - 30,
                        s=classes[int(cls_pred)] + '{:.4f}'.format(conf.item()),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            if polyp_flag:
                model_num += 1
            if fake_polyp_flag:
                model_num_fake += 1

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        model_predict.append(model_num)
        dst_path = r"./output/{}/{}".format(save_name,model_num)
        os.makedirs(dst_path, exist_ok=True)
        # filename = path.split("/")[-1].split(".")[0]
        filename = os.path.basename(path)
        print(filename)
        plt.savefig(os.path.join(dst_path, filename), bbox_inches="tight", pad_inches=0.0)
        plt.close()

    print(model_num_fake)
    # with open('results-polyp.pkl', 'wb') as f:
    with open('output/{}/results-polyp-bland.pkl'.format(save_name), 'wb') as f:
        pickle.dump(result_list, f)
    end_time = time.time()
    print('total time', end_time - start_time)
    print('prepare time', detect_start - start_time)
    print('detect time', end_time - detect_start)
    print(polyp_num)
    print('p0:{} p1:{} p2:{} p3:{} p4:{} p5:{} p6:{}'.format(model_predict.count(0),model_predict.count(1),model_predict.count(2),
                                                      model_predict.count(3),model_predict.count(4),model_predict.count(5),model_predict.count(6)))


