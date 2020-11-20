import argparse
import os
import torch.utils.data.distributed
import torch.distributed as dist
from torch import nn
from torchvision.models import Inception3
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from dataset import Net1_dataset, pf_dataset, Netdataset3
import shutil
from efficientnet_pytorch import EfficientNet
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='This is a script to training something!')
parser.add_argument('-i', '--input', help="测试集名称, 在../data/info/src/目录下查找",type=str, required=True)
# parser.add_argument('-d', '--dst', help="预测结果输出路径", required=True)
# parser.add_argument('-n', '--name', help="预测使用的模型文件",required=True)
parser.add_argument('--resume',type=str,default='')
parser.add_argument('-b','--batch_size',type=int,default=32,help="batch size")
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:33456')
parser.add_argument('--rank', type=int,default=0)
parser.add_argument('--world_size', type=int,default=1)
parser.add_argument('--limit',type=str,default='')
args = parser.parse_args()

dist.init_process_group(init_method=args.init_method, backend="nccl", world_size=args.world_size, rank=args.rank,
                            group_name="pytorch_test")
#================define path=================
src_path = os.path.join('../data/info/src/', args.input)
labelpath_polyp_fake = os.path.join('../data/train/','polyp-fake')
labelpath_Net1 = os.path.join('../data/train/', 'Net1')
labelpath_polyp = os.path.join('../data/train/', 'polyp')
dst_path = os.path.join('/home/data/info/dst/combine/', args.input)
print(' target path is %s'%dst_path)
print('source path', src_path)
#=================define output result=========
result = {'patientID':'', "pictures":[{"picId":'img1',
                                "yolo":{"category":[{"main_class":'', "sub_class":'',},
                                                    {"main_class":'', "sub_class":'',},
                                                    ],
                                        "detections":[{"model_id":'', "bbox":{'', '', '', ''}, "category":'', "confidence":'', "detail":''},
                                                      {"model_id":'', "bbox":{'x1', 'y1', 'x2', 'y2'}, "category":'', "confidence":'', "detail":''},
                                                      ],
                                        },
                                "Net-Jam":{"main_class":'', "sub_class":'','confidence':'',
                                           'classification':[{"model_id":'', "category":'', "confidence":'', "detail":''},
                                                             {"model_id":'', "category":'', "confidence":'', "detail":''},
                                                             ]
                                           },
                                }
                               ]
          }
result = {}
result_images_list = []
yolo_dict = {}
yolo_category = []
yolo_detections = []
id_result = {}
#===============build label===================
net1_test = True
fake_polyp_test = False
polyp_classification = False
yolo_test = True
#============datas load===============
normalizer = T.Normalize(mean=[0.265, 0.279, 0.423],
                             std=[0.209, 0.217, 0.297])
test_transformer = T.Compose([
        T.Resize((299, 299), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
#============model create============
classify = len(os.listdir(labelpath_polyp_fake))
def create_model(model,weights_name):
    # state_dict = torch.load(r'../bin/weight_iv3_ft_' + weights_name + '.pkl')
    state_dict =torch.load(weights_name)
    model.load_state_dict(state_dict, strict=False)
    print("load pretrained weights")
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    model.eval()
    return model

def create_ef_model(weights_name, arch='efficientnet-b6'):
    model_ef = EfficientNet.from_name(arch, override_params={'num_classes': 34})
    model_ef.load_state_dict(torch.load(weights_name))
    print("load pretrained weights")
    model_ef = model_ef.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_ef.eval()
    return model_ef

model_in0 = Inception3(num_classes=len(os.listdir(labelpath_Net1)),aux_logits=False)
model_in1 = Inception3(num_classes=classify,aux_logits=False)
model_in2 = Inception3(num_classes=classify,aux_logits=False)
model_in3 = Inception3(num_classes=classify,aux_logits=False)
model_in4 = Inception3(num_classes=len(os.listdir(labelpath_polyp)),aux_logits=False)
weights0 = r'../bin/weight_iv3_ft_' + 'Net1_sus-2' + '.pkl'
weights1 = r'../bin/weight_iv3_ft_' + 'polyp-fake-img-95-3' + '.pkl'
weights2 = r'../bin/weight_iv3_ft_' + 'polyp-fake-img-95' + '.pkl'
weights3 = r'../bin/weight_iv3_ft_' + 'polyp-fake-sus-95' + '.pkl'
weights4 = r'../bin/weight_iv3_ft_' + 'polyp-fake-sus-95' + '.pkl'
weights5 = r'../bin/weight_iv3_ft_' + 'polyp-fake-rd' + '.pkl'
weights6 = r'../bin/weight_iv3_ft_' + 'polyp-sus' + '.pkl'

#===========net1 test===============
# Net1 label
'''
对图片进行粗分，301类
'''
if net1_test:
    print('net1 test start')
    dataset_1 = ImageFolder(labelpath_Net1)
    class_to_id_dict_1 = dataset_1.class_to_idx
    labels_1 = list(class_to_id_dict_1.keys())
    ids_1 = list(class_to_id_dict_1.values())

    model_net1 = create_model(model_in0, weights0)
    net1_dataset = Net1_dataset(src_path, transform=test_transformer)
    net1_dataloader = torch.utils.data.DataLoader(
        net1_dataset,
        batch_size=args.batchsize * 3,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print('net1 test num: ', len(net1_dataset))
    net1_start = time.time()

    for ids, imgs, filenames in tqdm(net1_dataloader):
        data = imgs.cuda()
        with torch.no_grad():
            output0 = model_net1(data)
        values0, pred0 = output0.data.max(1, keepdim=False)
        for i in range(len(filenames)):
            net_jam_dict = {}
            net_jam_clssification = []
            # basename = os.path.basename(filenames[i])
            classname = labels_1[ids_1.index(pred0[i])]
            main_class = classname.split('_')[0]
            sub_class = classname.split('_')[1:]
            id = ids[i]
            net1_conf = values0[i]

            net_jam_clssification.append({"model_id": "net1", "category": classname, "confidence": values0[i], "detail":""})
            net_jam_dict.update({"main_class":main_class, "sub_class": sub_class, "confidence": values0[i], "classification": net_jam_clssification})
            # result_images_list.append()
            # id_result_dict.update({str(id):{"net_jam":ne_jam_dict}})
    net1_end = time.time()
    net1_consume = net1_end - net1_start
    print('net1_consume', net1_consume)

#===========fake polyp test==============
'''
过滤 一部分 fake polyp
'''
if fake_polyp_test:
    # polyp-fake label
    dataset_2 = ImageFolder(labelpath_polyp_fake)
    class_to_id_dict_2 = dataset_2.class_to_idx
    labels_2 = list(class_to_id_dict_2.keys())
    ids_2 = list(class_to_id_dict_2.values())

    model1 = create_model(model_in1, weights2)
    model3 = create_model(model_in2, weights4)
    model5 = create_model(model_in3, weights5)
    model_ef1 = create_ef_model(r'../bin/ef-sus-nr-2.pth')
    model_ef2 = create_ef_model(r'../bin/ef-rd-2.pth')
    model_ef3 = create_ef_model(r'../bin/ef-img-nr-2.pth')

    start_time = time.time()
    src_path_pf = os.path.join(dst_path, args.input)
    pf_dataset = pf_dataset(src_path_pf, transform=test_transformer)
    pf_sampler = torch.utils.data.distributed.DistributedSampler(pf_dataset)
    pf_dataloader = torch.utils.data.DataLoader(
        pf_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=pf_sampler,
    )
    print("polyp_fake num: ", len(pf_dataset))

    fake_start = time.time()

    for imgs, filenames in tqdm(pf_dataloader):
        time1 = time.time()
        data = imgs.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        with torch.no_grad():
            output1 = model1(data)
            output3 = model3(data)
            output5 = model5(data)
            output2 = model_ef1(data)
            output4 = model_ef2(data)
            output6 = model_ef3(data)

            # print(output.shape,output)
        values1, pred1 = output1.data.max(1, keepdim=False)  # get the index of the max log-probability
        values3, pred3 = output3.data.max(1, keepdim=False)  # get the index of the max log-probability
        values5, pred5 = output5.data.max(1, keepdim=False)  # get the index of the max log-probability
        values2, pred2 = output2.data.max(1, keepdim=False)  # get the index of the max log-probability
        values4, pred4 = output4.data.max(1, keepdim=False)  # get the index of the max log-probability
        values6, pred6 = output6.data.max(1, keepdim=False)  # get the index of the max log-probability

        pred1 = pred1 > 13  # prediction is p
        pred3 = pred3 > 13
        pred5 = pred5 > 13
        pred2 = pred2 > 13  # prediction is p
        pred4 = pred4 > 13
        pred6 = pred6 > 13

        pred = (pred1 + pred3 + pred5 + pred2 + pred4 + pred6).cpu().numpy()
        values = (values1 + values2 + values3 + values4 + values5 + values6).cpu().numpy()
        for i in range(len(filenames)):
            basename = os.path.basename(filenames[i])
            id = filenames[i].split('/')[-4]
            result.update({basename: pred[i]})
            dest_fpath = os.path.join(dst_path, id, 'fp/p{}'.format(pred[i]), basename)
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            try:
                shutil.copyfile(filenames[i], dest_fpath)
            except:
                continue
        time2 = time.time()
        print('time:', (time2 - time1))
    end_time = time.time()
    print('total_time: ', (end_time - start_time))
    co = list(result.values())
    pp0 = co.count(0)
    pp1 = co.count(1)
    pp2 = co.count(2)
    pp3 = co.count(3)
    pp4 = co.count(4)
    pp5 = co.count(5)
    pp6 = co.count(6)

    print('pp0:', pp0)
    print('pp1:', pp1)
    print('pp2:', pp2)
    print('pp3:', pp3)
    print('pp4:', pp4)
    print('pp5:', pp5)
    print('pp6:', pp6)

#==========polyp test============
# polyp label
if polyp_classification:
    dataset_3 = ImageFolder(labelpath_polyp)
    class_to_id_dict_3 = dataset_3.class_to_idx
    labels_3 = list(class_to_id_dict_3.keys())
    ids_3 = list(class_to_id_dict_3.values())
    polyp_start = time.time()
    src_path_pf = os.path.join(dst_path, args.input)
    p_dataset = Netdataset3(src_path_pf, transform=test_transformer)
    p_sampler = torch.utils.data.distributed.DistributedSampler(p_dataset)
    p_dataloader = torch.utils.data.DataLoader(
        p_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=p_sampler,
    )
    print("polyp num: ", len(p_dataset))
    model7 = create_model(model_in4, weights6)
    model7.eval()
    result = {}
    for imgs, filenames in tqdm(p_dataloader):
        data = imgs.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        with torch.no_grad():
            output7 = model7(data)
        pred7 = output7.data.max(1, keepdim=False)[1]  # get the index of the max log-probability

        for i in range(len(filenames)):
            classname7 = labels_3[ids_3.index(pred7[i])]
            basename = os.path.basename(filenames[i])
            # id = filenames[i].split('/')[-4]
            dest_fpath = os.path.join(dst_path, 'polyp_final2/{}'.format(classname7), basename)
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            try:
                shutil.copyfile(filenames[i], dest_fpath)
            except:
                # print()
                continue
    polyp_end = time.time()

    polyp_path = r'../data/info/src/29.80-polyp-and-fake/polyp'
    final_path = r'/mnt/disk/wzn/data/info/dst/combine/29.80/polyp_final2'
    polyp_name = []
    final_name = []
    num = 0
    for root, dirs, files in os.walk(polyp_path):
        for file in files:
            # basename = os.path.basename(files)
            polyp_name.append(file)
    print('', len(polyp_name))

    for root, dirs, files in os.walk(final_path):
        for file in files:
            # basename = os.path.basename(files)
            # polyp_name.append(file)
            if file in polyp_name:
                num += 1
    print(num)

#==========yolo test=============
if yolo_test:
    from models import *
    from utils.utils import *
    from utils.datasets import *
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import matplotlib.patches as patches
    from matplotlib.ticker import NullLocator
    def create_yolo_model(weights_path, model_def):
        model = Darknet(model_def, img_size=416).to(torch.device('cuda'))
        model.load_state_dict(torch.load(weights_path))
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
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
        # (r'./checkpoints/csp_random_3w6+pin_95.pth', r'./config/csp.cfg'),
        # (r'./checkpoints/csp_rand+3w6_170.pth', r'./config/csp.cfg'),
        (r'./checkpoints/csp_rand_3w6_95.pth', r'./config/csp.cfg'),
    ]
    models = []
    models_detections = []
    result_list = []
    for i, (model_def, weights_path) in enumerate(model_config):
        models.append(create_yolo_model(model_def,weights_path))
        models_detections.append([])
        result_list.append([])
    # dataset = Netdataset3(opt.image_folder)
    dataset_406 = ImageFolder(r'../data/info/src/29.80-polyp-406', img_size=416)
    dataset_569 = ImageFolder(r'../data/info/src/29.80-polyp-and-fake/polyp', img_size=416)
    dataset_3119 = ImageFolder(r'../data/info/src/29.80-polyp-and-fake/fake-polyp', img_size=416)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader_406 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_406,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        # sampler=sampler,
    )
    dataloader_569 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_569,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        # sampler=sampler,
    )
    dataloader_3119 = DataLoader(
        # ImageFolder(opt.image_folder, img_size=opt.img_size),
        dataset=dataset_3119,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        # sampler=sampler,
    )
    classes = load_classes(r'./data/my.name')  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs = []  # Stores image paths
    model_predict = []
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # ============json================
    import json
    import pickle
    with open(r'json/test_Annotation_fake_polyp.json') as f:
        fp_dict = json.load(f)
    with open(r'json/test_Annotation_polyp.json') as f:
        p_dict = json.load(f)
    fp_images = fp_dict['images']
    image_to_id = {}
    for d in fp_images:
        image_to_id.update({d['file_name']: d['id']})
    p_images = p_dict['images']
    for d in p_images:
        image_to_id.update({d['file_name']: d['id']})

    detect_start = time.time()
    #detection
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader_406):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            time0 = time.time()
            for i, model in enumerate(models):
                detections = models[i](input_imgs)
                time1 = time.time()
                detections = non_max_suppression2(detections, 0.4, 0.4)
                time2 = time.time()
                models_detections[i].extend(detections)
        print(time2-time1,time1-time0)
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
        basename = os.path.basename(path)
        id_num = image_to_id[basename]
        # print('test',len(models_detections),len(models_detections[0]))
        for j, model_detection in enumerate(models_detections):  #(model_num, len(file))
            polyp_flag = 0
            # for detections in model_detection:
            detections = model_detection[i]
            # Draw bounding boxes and labels of detection
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, 416, img.shape[:2])
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
                    else:
                        continue
            if polyp_flag:
                model_num += 1

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        model_predict.append(model_num)
        dst_path = r"./output/{}".format(model_num)
        os.makedirs(dst_path, exist_ok=True)
        filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(os.path.join(dst_path, filename+'.jpg'), bbox_inches="tight", pad_inches=0.0)
        plt.close()

    with open('results-polyp.pkl', 'wb') as f:
        pickle.dump(result_list, f)
    end_time = time.time()
    print('total time', end_time - start_time)
    print('prepare time', detect_start - start_time)
    print('detect time', end_time - detect_start)
    # print(polyp_num)
    print('p0:{} p1:{} p2:{} p3:{} p4:{} p5:{} p6:{}'.format(model_predict.count(0),model_predict.count(1),model_predict.count(2),
                                                      model_predict.count(3),model_predict.count(4),model_predict.count(5),model_predict.count(6)))


