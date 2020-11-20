
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

class Netdataset(dataset.Dataset):
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
    category = ["dirty_dense_thickband", "dirty_dense_thinband", "dirty_largeintestine", "dirty_vein_thick", "dirty_vein_thin",
                "suspect_dense", "suspect_largeintestine", "suspect_vein",
                ]
    for root, dirs, files in os.walk(directory):
        id = root.split('/')[-1]
        if not (id in category or "polyp" in id):
            # print(dir)
            continue
        print(dirs)
        for f in files:
            if f.endswith('jpg') or f.endswith('Jpeg'):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)

class Netdataset2(dataset.Dataset):
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
    category = ["dirty_dense_thickband", "dirty_dense_thinband", "dirty_largeintestine",
                "dirty_largeintestine_linehole", "dirty_vein_thick", "dirty_vein_thin",
                "dirty_bighole_blocking", "dirty_bighole_hollow", "dirty_vein_leaf", "dirty_vein_bubble",
                "dirty_angel_middle"
                "focus_hw_big", "focus_nucous_small_faltline",
                "suspect_dense", "suspect_largeintestine", "suspect_vein", "suspect_stone", "suspect_waterline",
                "suspect_flover", "suspect_watercover",
                "suspect_vein",
                ]
    for root, dirs, files in os.walk(directory):
        id2 = root.split('/')[-2]
        id1 = root.split('/')[-1]
        if id2 != 'fp':
            continue
        if id1 in ['p0']:
            continue
        for f in files:
            if f.endswith('jpg') or f.endswith('Jpeg'):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class Netdataset3(dataset.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [path for path in list_pictures3(self.root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        # ori_img = Image.open(path)
        # return self.transform(img), path

        img = transforms.ToTensor()(Image.open(path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, 416)

        return path, img

    def __len__(self):
        return len(self.imgs)

def list_pictures_nj(directory, ext='jpg|jpeg|bmp|png|ppm'):
    imgs = []
    for root, dirs, files in os.walk(directory):
        id2 = root.split('/')[-2]
        id1 = root.split('/')[-1]
        if not(('suspect'in id1) or ('dirty'in id1) or ('clear'in id1)):
            continue
        for f in files:
            if f.endswith('jpg') or f.endswith('Jpeg'):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)
class NetJam_dataset(dataset.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [path for path in list_pictures_nj(self.root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        # ori_img = Image.open(path)
        # return self.transform(img), path

        img = transforms.ToTensor()(Image.open(path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, 416)

        return path, img

    def __len__(self):
        return len(self.imgs)
