# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

import cv2
import os

from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0
    a_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # compute the output
        # print("input", inp)
        output = model(inp)
        target = target.cuda(non_blocking=True)
        # print("out", output.size(), target.size())
        loss = critertion(output, target)
        # print("loss", loss)
        # NME
        score_map = output.data.cpu()
        # print("score_map", score_map)
        preds = decode_preds(score_map, [128, 256])
        # print("preds",preds)
        # if epoch > 90:
        #     print(f"train_pred:{preds}.")

        a_temp, nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        a_batch_sum += np.sum(a_temp)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time() - end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=inp.size(0) / batch_time.val,
                data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    a = a_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} a:{:.4f}' \
        .format(epoch, batch_time.avg, losses.avg, a)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    a_batch_sum = 0

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, [128, 256])
            # if epoch > 30:
            # compute_nme(preds, meta)
            #     print(f"val_pred:{preds}.")

            # NME
            a_temp, nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            a_batch_sum += np.sum(a_temp)
            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    a = a_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    # msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
    #       '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
    #                             failure_008_rate, failure_010_rate)
    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} a:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, a,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    # if writer_dict:
    #     writer = writer_dict['writer']
    #     global_steps = writer_dict['valid_global_steps']
    #     writer.add_scalar('valid_loss', losses.avg, global_steps)
    #     writer.add_scalar('valid_nme', nme, global_steps)
    #     writer_dict['valid_global_steps'] = global_steps + 1
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_a', a, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    # return nme, predictions
    return a, nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    tp = {}
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, [128, 256])

            # # visualize if don't need visulize, comment
            # for j, b in enumerate(preds.tolist()):
            #     p = []
            #     for item in b:
            #         p.append((int(item[0]), int(item[1])))
            #     n = meta['name'][j]
            #     tp[n] = p
            #     j += 1


            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # # visualize if don't need visulize, comment
    # for file in tp:
    #     path1 = '/public/home/zhaojh1/git_main/HRNet/visual_data/'
    #     path2 = '/public/home/zhaojh1/git_main/HRNet/visual/'
    #     image = cv2.imread(os.path.join(path1, file))
    #     point_size = 1
    #     point_color = (0, 0, 255)  # BGR
    #     thickness = 4  # 可以为 0 、4、8
    #     lp = tp[file]
    #     for point in lp:
    #         cv2.circle(image, point, point_size, point_color, thickness)
    #         cv2.imwrite(f"{os.path.join(path2, file[0:5])}.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


    # for root, dirs, files in os.walk('/public/home/zhaojh1/git_main/HRNet/visual_data/', True):
    #     for file in files:
    #         image = cv2.imread(os.path.join(root, file))  # 读取文件名对应的图片
    #         point_size = 1
    #         point_color = (0, 0, 255)  # BGR
    #         thickness = 4  # 可以为 0 、4、8
    #         j = 0
    #         for point in tp[i]:
    #             cv2.circle(image, point, point_size, point_color, thickness)
    #             cv2.imwrite(f"/public/home/zhaojh1/git_main/HRNet/visual/{file}.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    #         j += 1

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions
