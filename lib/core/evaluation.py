# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    a = np.zeros(N)
    # SMAPE = np.zeros(N)

    # visualize
    # for b in preds.tolist():
    #     p = []
    #     for item in b:
    #         p.append( ( int(item[0]), int(item[1]) ) )
    #

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        d = np.linalg.norm(pts_pred - pts_gt, axis=1)
        m = d.mean()
        j = 0
        for item in d:
            if item/5 <= 1:
                j += 1
        a[i] = j/L

        print(f"m:{m},a:{a[i]}.")
        if a[i] <= 0.6:
            print(meta['name'])

        # rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / L
        rmse[i] = np.sum(np.power(np.linalg.norm(pts_pred - pts_gt, axis=1), 2)) / L / 10 # mse
        #accuracy
        # SMAPE[i] = (np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1) / (np.linalg.norm(pts_pred, axis=1) + (np.linalg.norm(pts_gt, axis=1))))) * 2 * 100 / L  evaluate angle
        # print(f"pts_pred:{pts_pred},pts_gt:{pts_gt},loss{rmse[i]}.")


    # return rmse
    return a, rmse
    # return SMAPE


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type
    # print("coords", coords)
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())
    # print("coord_preds", preds)

    return preds
