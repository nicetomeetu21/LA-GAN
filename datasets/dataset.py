# -*- coding:utf-8 -*-
# basic 2d datasets
import os
import cv2 as cv
from torch.utils.data.dataset import Dataset
import numpy as np
from natsort import natsorted
import torch
import torchvision
from torchvision import transforms
import albumentations as A


class AugmentedDataset(Dataset):
    def __init__(self, A_dir, B_dir, C_dir, D_dir, folds_id, transform = None):
        self.A_paths = []
        self.B_paths = []
        self.C_paths = []
        self.D_paths = []
        self.filenames = []
        self.transform = transform
        for fold in folds_id:
            names = natsorted(os.listdir(os.path.join(A_dir, str(fold))))
            self.filenames += names
            self.A_paths += map(lambda x: os.path.join(A_dir, str(fold), x), names)
            self.B_paths += map(lambda x: os.path.join(B_dir, str(fold), x), names)
            self.C_paths += map(lambda x: os.path.join(C_dir, str(fold), x), names)
            self.D_paths += map(lambda x: os.path.join(D_dir, str(fold), x), names)

    def __getitem__(self, index):
        real_A = cv.imread(os.path.join(self.A_paths[index]), cv.IMREAD_COLOR)
        real_B = cv.imread(os.path.join(self.B_paths[index]), cv.IMREAD_GRAYSCALE)
        real_C = cv.imread(os.path.join(self.C_paths[index]), cv.IMREAD_GRAYSCALE)
        real_D = cv.imread(os.path.join(self.D_paths[index]), cv.IMREAD_GRAYSCALE)
        # real_C = real_C // 255
        if self.transform is not None:
            transformed = self.transform(image=real_A, masks=[real_B,real_C, real_D])
            real_A = transformed['image']
            real_B = transformed['masks'][0]
            real_C = transformed['masks'][1]
            real_D = transformed['masks'][2]

        real_A = transforms.ToTensor()(real_A)
        real_B = transforms.ToTensor()(real_B)
        # real_C = torch.from_numpy(real_C).long()
        real_C = transforms.ToTensor()(real_C)
        real_D = transforms.ToTensor()(real_D)
        return real_A, real_B, real_C, real_D, self.filenames[index]

    def __len__(self):
        return len(self.filenames)

if __name__ =='__main__':
    A_dir = '/home/huangkun/PycharmProjects/Color2FFA/data/dataset3/1024_5_split/A'
    B_dir = '/home/huangkun/PycharmProjects/Color2FFA/data/dataset3/1024_5_split/B'
    CD_dir = '/home/huangkun/PycharmProjects/Color2FFA/data/dataset3/1024_5_split/CD_onehot'
    result_dir = '/home/huangkun/PycharmProjects/Color2FFA/data/dataset3/result/test'
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(p=0.5, interpolation=cv.INTER_NEAREST),
        A.OneOf([
            A.RandomResizedCrop(width=768, height=768, scale=(0.3, 1.0), interpolation=cv.INTER_NEAREST),
            A.RandomCrop(width=768, height=768)
        ], p=1.0),
    ])
    dataset = AugmentedDataset(A_dir, B_dir, CD_dir, [0, 1, 2, 3], transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (A, B, CD, name) in enumerate(dataloader):
        # cv.imwrite('test/A_'+name[0],A[0])
        print(name)
        torchvision.utils.save_image(A, result_dir+'/'+name[0][:-4]+'_A.png')
        torchvision.utils.save_image(B, result_dir+'/'+name[0][:-4]+'_B.png')
        torchvision.utils.save_image(CD, result_dir+ '/'+name[0][:-4]+'_CD.png')
        if i == 5:
            exit()
