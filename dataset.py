import os
import re
from torch.utils.data import dataset, sampler
from torchvision.datasets.folder import default_loader
from PIL import Image

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    # return sorted([os.path.join(root, f)
    #                for root, _, files in os.walk(directory) for f in files
    #                if re.match(r'([\w]+\.(?:' + ext + '))', f)])
    imgs = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('jpg') or f.endswith('Jpeg'):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)

class Net1_dataset(dataset.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [path for path in list_pictures(self.root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        # img = self.loader(path)
        img = Image.open(path)
        # name = os.path.split(path)
        return self.transform(img),path

    def __len__(self):
        return len(self.imgs)

def list_pictures2(directory, ext='jpg|jpeg|bmp|png|ppm'):
    imgs = []
    category = ["dirty_dense_thickband", "dirty_dense_thinband", "dirty_largeintestine", "dirty_largeintestine_linehole",
                "dirty_vein_thick", "dirty_vein_thin",
                "dirty_bighole_blocking", "dirty_bighole_hollow", "dirty_vein_leaf", "dirty_vein_bubble", "dirty_angle_middle",
                "focus_hw_big","focus_mucous_small_faltline",
                "suspect_dense", "suspect_largeintestine", "suspect_vein", "suspect_stone", "suspect_waterline",
                "suspect_flower", "suspect_watercover", "suspect_vein",
                ]
    for root, dirs, files in os.walk(directory):
        id1 = root.split('/')[-1]
        if ('polyp' in id1) or (id1 in category):
            for f in files:
                if f.endswith('jpg') or f.endswith('Jpeg'):
                    imgs.append(os.path.join(root, f))
    return sorted(imgs)

class pf_dataset(dataset.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [path for path in list_pictures2(self.root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        return self.transform(img), path

    def __len__(self):
        return len(self.imgs)


def list_pictures3(directory, ext='jpg|jpeg|bmp|png|ppm'):
    imgs = []
    category = ['p']
    # category = ['p6', 'p5', 'p4', 'p3', 'p2', 'p0', 'p1']
    for root, dirs, files in os.walk(directory):
        id1 = root.split('/')[-1]
        id2 = root.split('/')[-2]
        # print(id1)
        if id2 in ['yolo2', 'yolo1']:
            continue
        if not (id1 in category):
            continue
        # print(dirs)
        for f in files:
            if f.endswith('jpg') or f.endswith('Jpeg') or f.endswith('png'):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)

class Netdataset3(dataset.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [path for path in list_pictures3(self.root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        return self.transform(img), path

    def __len__(self):
        return len(self.imgs)