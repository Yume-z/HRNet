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


class us(data.Dataset):

    def __init__(self, cfg, is_train=True, if_trans=False, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET

        else:
            self.csv_file = cfg.DATASET.TESTSET

        if if_trans:
            self.transform = A.Compose(
                [
                    # A.PadIfNeeded(min_height=64, min_width=128, border_mode=0),
                    A.Resize(height=64, width=128),
                    # A.RandomCrop(width=512, height=1024), #point number changed
                    # # whether predict as the point sequence? And output size need to change?Or just change input size
                    #
                    #

                    # A.VerticalFlip(p=0.5),
                    # A.OneOf([
                    #     A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                    # ], p=0.2),  # 应用选定变换的概率
                    # A.OneOf([
                    #     A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                    #     A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                    #     A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                    # ], p=0.2),
                    # # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.01, rotate_limit=5, p=1),
                    # # 随机应用仿射变换：平移，缩放和旋转输入 will change num
                    #
                    A.Compose([
                        A.OneOf([
                            A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                        ], p=0.2),  # 应用选定变换的概率
                        A.OneOf([
                            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                        ], p=0.2),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.2),
                        # 随机应用仿射变换：平移，缩放和旋转输入 will change num
                        #
                        A.OneOf([
                            A.Sharpen(),
                            A.Emboss(),
                            A.RandomBrightnessContrast(),
                        ], p=0.3),
                        A.HueSaturationValue(p=0.3),
                    ], p=0.8),
                ],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )
        else:
            self.transform = A.Compose(
                # [A.PadIfNeeded(min_height=64, min_width=128, border_mode=0)
                [A.Resize(height=64, width=128),
                 ],
                keypoint_params=A.KeypointParams(format='xy')
            )

        self.is_train = is_train
        # self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.numb = cfg.MODEL.NUM_JOINTS  # new
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.083, 0.085, 0.086], dtype=np.float32)
        self.std = np.array([0.066, 0.067, 0.069], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])

        # scale = self.landmarks_frame.iloc[idx, 1]
        #
        # center_w = self.landmarks_frame.iloc[idx, 2]
        # center_h = self.landmarks_frame.iloc[idx, 3]
        # center = torch.Tensor([center_w, center_h])
        #
        # pts = self.landmarks_frame.iloc[idx, 4:].values

        pts = self.landmarks_frame.iloc[idx, 1:].values

        # print('pts', pts)
        pts = pts.astype('float').reshape(-1, 2)

        # scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        # imggg = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        # print('image', imggg.shape)

        # if img.shape[0] / img.shape[1] > 2:
        #
        #     t = img.shape[0] // 2
        #
        #     padding = A.Compose(
        #
        #         [A.PadIfNeeded(min_width=t, border_mode=0)],
        #
        #         keypoint_params=A.KeypointParams(format='xy')
        #
        #     )
        #
        # else:
        #
        #     t = img.shape[1] * 2
        #
        #     padding = A.Compose(
        #
        #         [A.PadIfNeeded(min_height=t, border_mode=0)],
        #
        #         keypoint_params=A.KeypointParams(format='xy')
        #
        #     )
        #
        # padded = padding(image=img, keypoints=pts)
        #
        # img = padded['image']
        #
        # pts = padded['keypoints']

        # data agumentation
        transform = self.transform
        transformed = transform(image=img, keypoints=pts)
        img = transformed['image']
        pts = transformed['keypoints']  # need transform back

        pts = np.array(pts)  # number is 4e+02,not float
        pts = pts.astype('float')

        # r = 0
        # if self.is_train:
        #     scale = scale * (random.uniform(1 - self.scale_factor,
        #                                     1 + self.scale_factor))
        #     r = random.uniform(-self.rot_factor, self.rot_factor) \
        #         if random.random() <= 0.6 else 0
        # if random.random() <= 0.5 and self.flip:
        #     img = np.fliplr(img)
        #     pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
        #     center[0] = img.shape[1] - center[0]

        # img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((self.numb, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()
        # print(tpts)
        # l = len(tpts)
        # print(l)
        for i in range(nparts):
            # print(tpts[i, 1])
            if tpts[i, 0] > 0 and tpts[i, 1] > 0:
                # print(tpts[i, 0:2])
                j = i % self.numb
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, self.output_size)
                # print('tpt')
                target[j] = generate_target(target[j], tpts[i] - 1, self.sigma,
                                            # index x out of bounds for axis 0 with size 56
                                            label_type=self.label_type)
            else:
                tpts[i, 0:2] = (0, 0)
                # print('target')
        # print("tpts", tpts)
        # print("target", target)
        img = img.astype(np.float32)
        # print('image', img)
        img = (img / 255.0 - self.mean) / self.std
        # img = img / 255.0
        img = img.transpose([2, 0, 1])
        # print('image', img)
        target = torch.Tensor(target)

        # target = target.permute([0, 2, 1])
        # print('target', target.shape)

        tpts = torch.Tensor(tpts)
        # print('tpts', tpts.shape)
        # center = torch.Tensor(center)
        # print('out', img.shape)

        # meta = {'index': idx, 'center': center, 'scale': scale,
        #         'pts': torch.Tensor(pts), 'tpts': tpts}
        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts, 'name': self.landmarks_frame.iloc[idx, 0]}
        # print("meta", meta)

        return img, target, meta


if __name__ == '__main__':
    pass