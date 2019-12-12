# libraries
import numpy as np
import pandas as pd
import sys
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import albumentations 
from albumentations import pytorch as AT


class URESDataset(Dataset):
    def __init__(self, data_dir='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE', \
        split=None, csv=None, mode=None, augment=None, size=(1024, 1024)):

        self.data_dir = data_dir
        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment
        self.size = size

        self.uid = list(np.concatenate([np.load(self.data_dir + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(self.data_dir + '/%s'%f).fillna('') for f in csv])

        self.df = df
        self.num_image = len(self.uid)


    def __len__(self):
        return self.num_image
    
    def get_mask_overlay(self, image, mask, alpha=(0.20,1.20)):
        H,W,C = image.shape
        mask = mask.copy()
        neg = mask< 0.5
        pos = mask>=0.5
        mask[neg]=alpha[0]
        mask[pos]=alpha[1]

        mask    = mask.reshape(H,W,1)
        overlay = image.astype(np.float32)
        overlay = overlay*mask
        overlay = np.clip(overlay,0,255)
        overlay = overlay.astype(np.uint8)
        return overlay
    
    def draw_mask_overlay(self, height=15, width=2, augment=False, alpha=(0.20,1.20)):
        fig, axs = plt.subplots(height, width, figsize=(width * 10, height * 10))

        for n in range(0, min(height * width, self.__len__())):
            i = n #i = np.random.choice(len(dataset))

            h = n // width
            w = n % width

            image, mask, info = self.__getitem__(i, augment)
            overlay = self.get_mask_overlay(image, mask, alpha)

            axs[h, w].imshow(overlay) #plot the data
            axs[h, w].axis('off')
            axs[h, w].set_title('%05d : %s'%(i, "overlay_" + info))

        plt.show()
    
    def draw_mask(self, height=15, width=2, augment=False):
        fig, axs = plt.subplots(height, width, figsize=(width * 10, height * 10))

        for n in range(0, min(height * width, self.__len__())):
            i = n #i = np.random.choice(len(dataset))

            h = n // width
            w = n % width

            image, mask, info = self.__getitem__(i, augment)

            axs[h, w].imshow(mask) #plot the data
            axs[h, w].axis('off')
            axs[h, w].set_title('%05d : %s'%(i, "label_" + info))

        plt.show()
    
    def draw_image(self, height=15, width=2, augment=False):
        fig, axs = plt.subplots(height, width, figsize=(width * 10, height * 10))

        for n in range(0, min(height * width, self.__len__())):
            i = n #i = np.random.choice(len(dataset))

            h = n // width
            w = n % width

            image, mask, info = self.__getitem__(i, augment)

            axs[h, w].imshow(image) #plot the data
            axs[h, w].axis('off')
            axs[h, w].set_title('%05d : %s'%(i, "image_" + info))

        plt.show()


    def __getitem__(self, index, augment=True):
        image_id, folder = self.uid[index]
        image = cv2.imread(self.data_dir + '/%s/%s.png'%(folder,image_id), cv2.IMREAD_COLOR)

        if self.mode == 'train':
            # mask = cv2.imread(self.data_dir + '/labels/%s/%s.tiff'%(folder, image_id), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(self.data_dir + '/labels/%s/%s.tiff'%(folder, image_id), 0)
        else:
            mask = np.zeros((self.size[0], self.size[1], 1), np.uint8)

        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        # mask = np.where(mask > 1, 1)
        # mask = 1 - mask
        mask  = 1 - mask.astype(np.float32)/255
        # print(mask.shape)
        
        info = image_id
        
        if ((self.augment is None) or (not augment)):
            return image, mask, info
        else:
            return self.augment(image, mask, info)            
  
def transform_train(image, mask, info):
    
    if random.random() < 0.5:
        image = albumentations.VerticalFlip(p=1)(image=image)['image']
        mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        
        image = albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.1),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.1),
        ])(image=image)['image']
    
    if random.random() < 0.5:  
        image = albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5)(image=image)['image']

    if random.random() < 0.5:
        image = albumentations.Cutout(num_holes=2, max_h_size=8, max_w_size=8, p=1)(image=image)['image']
        mask = albumentations.Cutout(num_holes=2, max_h_size=8, max_w_size=8, p=1)(image=mask)['image']
        
    if random.random() < 0.5:
        image = albumentations.RandomRotate90(p=1)(image=image)['image']
        mask = albumentations.RandomRotate90(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.Transpose(p=1)(image=image)['image']
        mask = albumentations.Transpose(p=1)(image=mask)['image']
    
    if random.random() < 0.5:
        image = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=image)['image']
        mask = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=mask)['image']
    
    # vimage = albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)(image=image)['image']
    # image = albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0)(image=image)['image']
    
    return image, mask, info    

def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_mask  = []
    for b in range(batch_size):
        input.append(batch[b][0])
        # print(batch[b][1].shape)
        truth_mask.append(batch[b][1])

    input = np.stack(input)
    input = input[...,::-1].copy()
    input = input.transpose(0,3,1,2)
    
    truth_mask = np.stack(truth_mask)
    if (truth_mask.ndim < 4):   
        truth_mask = np.expand_dims(truth_mask, axis=3)
    truth_mask = truth_mask.transpose(0,3,1,2)

    #----
    input = torch.from_numpy(input).float()
    truth_mask = torch.from_numpy(truth_mask).float()

    return input, truth_mask
        
if __name__ == "__main__":
    
    dataset = URESDataset(data_dir='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE', \
        split=['train_fold_0_seed_42.npy',], \
        csv=['train.csv'], \
        mode='train', \
        augment=transform_train, \
        size=(1024, 1024)
    )
    
    # dataset.draw_image(height=15, width=2, augment=False)
    # dataset.draw_image(height=15, width=2, augment=True)
    # dataset.draw_mask(height=15, width=2, augment=False)
    # dataset.draw_mask(height=15, width=2, augment=True)
    # dataset.draw_mask_overlay(height=15, width=2, augment=False, alpha=(1.20, 0.20))
    dataset.draw_mask_overlay(height=15, width=2, augment=True, alpha=(1.20, 0.20))