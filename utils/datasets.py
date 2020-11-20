import glob
import random
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip, vertical_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    # image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear").squeeze(0)
    image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="bilinear")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        if self.transform is not None:
            img = self.transform(Image.open(img_path))
        else:
            img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class ImageFolder2(Dataset):
    def __init__(self, folder_path, img_size=416, transform=None):
        # self.files = sorted(glob.glob("%s/*.*" % folder_path))
        paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                # if '20190121_324' in file:
                #     continue
                # if file.endswith('.jpg') or file.endswith('.Jpeg'):
                paths.append(os.path.join(root,file))
        self.files = paths
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        if self.transform is not None:
            img = self.transform(Image.open(img_path))
        else:
            img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            # path.replace("jpg_all", "labels_revise").replace(".png", ".txt").replace(".jpg", ".txt")
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace(".Jpeg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        # normalizer = transforms.Normalize(mean=[0.265, 0.279, 0.423],
        #                          std=[0.209, 0.217, 0.297])
        normalizer = transforms.Normalize(mean=[0.423, 0.279, 0.265],
                                          std=[0.297, 0.217, 0.209])
        self.transformer = transforms.Compose([
            transforms.ColorJitter(0.1,0.1,0.1),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalizer,
        ])

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img = self.transformer(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # targets = None
        out = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            # label: (center_x, center_y, width, height)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            # print boxes
            # print w_factor, h_factor, padded_w, padded_h
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            # boxes[:, 3] *= w_factor / padded_w
            # boxes[:, 4] *= h_factor / padded_h
            boxes[:, 3] = boxes[:, 3] * w_factor / padded_w
            boxes[:, 4] = boxes[:, 4] * h_factor / padded_h

            # print boxes
            length = boxes.shape[0]
            targets = torch.zeros((length, 6))
            targets[:, 1:] = boxes
            # print('shape:', targets.shape, len(boxes))
            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)
                if np.random.random() < 0.5:
                    img, targets = vertical_flip(img, targets)

            targets = targets.view(1,-1)
            # print(targets.shape)

            out = torch.zeros((1500,))
            out[:length*6] = targets

        return img, out


    def collate_fn(self, batch):
        # paths, imgs, targets = list(zip(*batch))
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # # Add sample index to targets
        # for i, boxes in enumerate(targets):
        #     # boxes[:, 0] = i
        #     boxes = boxes.view(1,-1)
        # print('targets',len(targets),targets[0].shape,targets[1].shape)
        targets = torch.stack(targets, 0)
        # print("collate:",targets.shape)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        # return paths, imgs, targets
        return imgs, targets

    def __len__(self):
        return len(self.img_files)
