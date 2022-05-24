# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import cv2
import torch
import scipy
import skimage
import skimage.transform
import numpy as np
from PIL import Image


def get_transform(output_size):
    """
    General image processing functions
    """
    # Generate transformation matrix
    # transform input to output size(512->128)
    # h = 512
    # t: [[0.25 0.   0.  ]
    #  [0.   0.25 0.  ]
    #  [0.   0.   1.  ]]
    # print("outputsize", output_size)
    t = np.zeros((3, 3))
    t[0, 0] = 0.25
    t[1, 1] = 0.25
    t[0, 2] = 0
    t[1, 2] = 0
    t[2, 2] = 1
    # t[0, 0] = float(output_size[1]) / 512
    # t[1, 1] = float(output_size[0]) / 1024
    # t[0, 2] = output_size[1] * (-float(center[0]) / 512 + .5)
    # t[1, 2] = output_size[0] * (-float(center[1]) / 1024 + .5)
    # t[2, 2] = 1
    # if not rot == 0:
    #     rot = -rot  # To match direction of rotation from cropping
    #     rot_mat = np.zeros((3, 3))
    #     rot_rad = rot * np.pi / 180
    #     sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    #     rot_mat[0, :2] = [cs, -sn]
    #     rot_mat[1, :2] = [sn, cs]
    #     rot_mat[2, 2] = 1
    #     # Need to rotate around center
    #     t_mat = np.eye(3)
    #     t_mat[0, 2] = -output_size[1]/2
    #     t_mat[1, 2] = -output_size[0]/2
    #     t_inv = t_mat.copy()
    #     t_inv[:2, 2] *= -1
    #     t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_pixel(pt, output_size, invert=0):
    # Transform pixel location to different reference
    t = get_transform(output_size)
    # print("t:", t)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    # print("new_pt:", new_pt)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, output_size):
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], output_size, 1))
    return coords


# def crop(img, center, scale, output_size, rot=0):
#     center_new = center.clone()
#
#     # Preprocessing for efficient cropping
#     ht, wd = img.shape[0], img.shape[1]
#     sf = scale * 200.0 / output_size[0]
#     if sf < 2:
#         sf = 1
#     else:
#         new_size = int(np.math.floor(max(ht, wd) / sf))
#         new_ht = int(np.math.floor(ht / sf))
#         new_wd = int(np.math.floor(wd / sf))
#         if new_size < 2:
#             return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
#                         if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
#         else:
#             # img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)
#             img = np.array(Image.fromarray(img.astype(np.uint8)).resize([new_wd, new_ht]))  # (0-1)-->(0-255)
#             center_new[0] = center_new[0] * 1.0 / sf
#             center_new[1] = center_new[1] * 1.0 / sf
#             scale = scale / sf
#
#     # Upper left point
#     ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))
#     # Bottom right point
#     br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))
#
#     # Padding so that when rotated proper amount of context is included
#     pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
#     if not rot == 0:
#         ul -= pad
#         br += pad
#
#     new_shape = [br[1] - ul[1], br[0] - ul[0]]
#     if len(img.shape) > 2:
#         new_shape += [img.shape[2]]
#
#     new_img = np.zeros(new_shape, dtype=np.float32)
#
#     # Range to fill new array
#     new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
#     new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
#     # Range to sample from original image
#     old_x = max(0, ul[0]), min(len(img[0]), br[0])
#     old_y = max(0, ul[1]), min(len(img), br[1])
#     new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
#
#     if not rot == 0:
#         # Remove padding
#         # new_img = scipy.misc.imrotate(new_img, rot)
#         new_img = skimage.transform.rotate(new_img, rot)
#         # new_img = new_img[pad:-pad, pad:-pad]
#     # new_img = scipy.misc.imresize(new_img, output_size)
#     new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(output_size))
#     return new_img


def generate_target(img, pt, sigma, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
