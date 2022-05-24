# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

import albumentations as A

from ..utils.transforms import generate_target, transform_pixel


# from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class xRAY(data.Dataset):

    def __init__(self, cfg, is_train=True, if_trans=False, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET

        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.if_trans = if_trans
        # self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE #h*w
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.numb = cfg.MODEL.NUM_JOINTS  #new
        self.label_type = cfg.MODEL.TARGET_TYPE


        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.159, 0.158, 0.157], dtype=np.float32)
        self.std = np.array([0.165, 0.164, 0.164], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
                                  
        x,y = self.input_size[1], self.input_size[0]
        
        pts = self.landmarks_frame.iloc[idx, 1:].values
        pts = pts.astype('float').reshape(-1, 2)

        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        if img.shape[0]/img.shape[1] > 2:
            t = img.shape[0]//2
            padding = A.Compose(       
              [ 
              A.PadIfNeeded(min_width=t, border_mode=0),
              A.Resize(width=x, height=y)
              ], 
              keypoint_params=A.KeypointParams(format='xy')       
            )
        else:    
            t = img.shape[1]*2        
            padding = A.Compose(    
              [ 
              A.PadIfNeeded(min_height=t, border_mode=0),
              A.Resize(width=x, height=y)
              ],
              keypoint_params=A.KeypointParams(format='xy')
            )
        padded = padding(image=img, keypoints=pts)
        img = padded['image']
        pts = padded['keypoints']
             
        
   
        if self.if_trans:
            r = int(random.uniform(1,9))*(y/8)
            
            transform = A.Compose(
                [
                
                 A.Compose([
                    A.RandomCrop (width=x, height=r),
                    A.PadIfNeeded(min_height=y, min_width=x, border_mode=0)]),
                 A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.35, rotate_limit=45, border_mode=0, p=0.2),

                
                 A.Compose([
                     A.OneOf([
                         A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                     ], p=0.2),  # 应用选定变换的概率
                     A.OneOf([
                         A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                         A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                         A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                     ], p=0.2),
                 A.OneOf([
                     A.Sharpen(),
                     A.Emboss(),
                     A.RandomBrightnessContrast(),
                 ], p=0.3),
                 A.HueSaturationValue(p=0.3),
                 ], p=0.8),
                 ],
                 keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
             )
            
            transformed = transform(image=img, keypoints=pts)
            img = transformed['image']
            pts = transformed['keypoints'] 
     
                 
                 
        
        


        pts = np.array(pts)  # number is 4e+02,not float
        pts = pts.astype('float')
        target = np.zeros((self.numb, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):

            if 0<tpts[i, 0]<x and 0<tpts[i, 1]<y and img[int(tpts[i, 1]),int(tpts[i, 0])].any():
                j = i % self.numb
                
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, self.output_size)
                target[j] = generate_target(target[j], tpts[i] - 1, self.sigma,    #index x out of bounds for axis 0 with size 56
                                            label_type=self.label_type)
            else:
                tpts[i, 0:2] = (0, 0)

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)


        tpts = torch.Tensor(tpts)

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts, 'name': self.landmarks_frame.iloc[idx, 0]}


        return img, target, meta


if __name__ == '__main__':
    pass
