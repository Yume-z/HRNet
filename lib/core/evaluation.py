# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds
import torch.nn as nn


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    kernel = 3
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        scores, (kernel, kernel), stride=1, padding=pad)  # max pooling 找到邻域内的最大值
    keep = (hmax == scores).float()
    scores = scores * keep

    B = scores.size(0)  # 4
    C = scores.size(1)  # 6
    N = scores.size(2) * scores.size(3)  # 256*128
    val1, idx1 = torch.sort(scores.view(B, C, N), descending=True)  # descending为False，升序，为True，降序

    k = 16
    # maxval = torch.zeros((B, C, k))
    idx2 = torch.zeros((B, C, k))
    for i in range(B):
        for j in range(C):
            idx2[i, j, ] = idx1[i, j, :k]

    idx, _ = torch.sort(idx2, descending=False)  # resort idx to get correct order
    idx = idx.view(B, 96, 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    trans = preds.clone()
    for i in range(B):
        for j in range(preds.size(1)):
            m = j // C
            n = j % C
            trans[i, j,] = preds[i, m + 16 * n,]
            
    return trans


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    a = np.zeros((5, N))


    for i in range(N):
        # size = meta['size'][i][0]/1024
        pts_pred, pts_gt = preds[i, ], target[i, ]

        d = np.linalg.norm(pts_pred - pts_gt, axis=1)
        # print(size, d)
        m = d.mean()
        j1,j2,j3,j4,j5 = 0,0,0,0,0
        for item in d:
            # if item/10*size<= 1:
            if item/10<= 1:
                j1 += 1
            if item/7<= 1:
                j2 += 1
            if item/5<= 1:
                j3 += 1
            if item/3<= 1:
                j4 += 1
            if item/1<= 1:
                j5 += 1
        
                
        a[0,i] = j1/L
        a[1,i] = j2/L
        a[2,i] = j3/L
        a[3,i] = j4/L
        a[4,i] = j5/L

        rmse[i] = np.sum(np.power(np.linalg.norm(pts_pred - pts_gt, axis=1), 2)) / L  # mse
        #accuracy

    # return rmse
    return a, rmse



def decode_preds(output, res):
    coords = get_preds(output)  # float type
    # print("coords", coords)
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            v = p % 6
            hm = output[n][v]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())
    # print("coord_preds", preds)

    return preds
