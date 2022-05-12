# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
from sklearn.model_selection import KFold


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED



    dataset_type = get_dataset(config)

    # KFOLD
    dataset = dataset_type(config, is_train=True)
    kf = KFold(n_splits=config.TRAIN.KFOLD_NUM)
    accuracy = []
    MSE = []
    epoch_num = []
    

    for fold, (t, v) in enumerate(kf.split(dataset)):

        torch.set_num_threads(4)
        
        model = models.get_net(config)
        
        gpus = list(config.GPUS)

        model = nn.DataParallel(model, device_ids=gpus).cuda()
      
        criterion = torch.nn.MSELoss(reduction='mean').cuda()
        
        optimizer = utils.get_optimizer(config, model)


        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        


        train_set = torch.utils.data.dataset.Subset(dataset_type(config, is_train=True, if_trans=True), t)
        val_set = torch.utils.data.dataset.Subset(dataset, v)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            prefetch_factor=8,
            pin_memory=config.PIN_MEMORY)

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            prefetch_factor=8,
            pin_memory=config.PIN_MEMORY
        )
        
        
        

        best_a = 0
        best_m = 0
        best_epoch = 0
        last_epoch = config.TRAIN.BEGIN_EPOCH
        if isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, last_epoch - 1
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, last_epoch - 1
            )

        for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
            # lr_scheduler.step()

            function.train(config, train_loader, model, criterion,
                           optimizer, epoch, writer_dict)

            # a, nme, predictions = function.train(config, train_loader, model, criterion,
            #                                      optimizer, epoch, writer_dict, lr_scheduler)

            lr_scheduler.step()

            # evaluate
            # nme, predictions = function.validate(config, val_loader, model,
            #                                      criterion, epoch, writer_dict)

            a, nme, predictions = function.validate(config, val_loader, model,
                                                    criterion, epoch, writer_dict)

            # is_best = nme < best_nme
            # best_nme = min(nme, best_nme)
            is_best = a > best_a
            best_a = max(a, best_a)
            
            
            if is_best == True:
                best_epoch = epoch
                best_m = nme
                best_model_state_file = os.path.join(final_output_dir,
                                                      f'{fold}best_state.pth')
                torch.save(model.module.state_dict(), best_model_state_file)

            # logger.info('=> saving checkpoint to {}'.format(final_output_dir))

            if epoch == config.TRAIN.END_EPOCH - 1:
                accuracy.append(best_a)
                MSE.append(best_m)
                epoch_num.append(best_epoch)
                print(fold, best_epoch, a, best_a, nme, best_m)
            # utils.save_checkpoint(
            #     {"state_dict": model,
            #      "epoch": epoch + 1,
            #      # "best_nme": best_nme,
            #      "best_a": best_a,
            #      "optimizer": optimizer.state_dict(),
            #      }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

        print(f"{fold}th fold is finished")
        final_model_state_file = os.path.join(final_output_dir,
                                              f'{fold}final_state.pth')
        logger.info(f'saving {fold}th model state to {final_model_state_file}')

        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

        # debug and test
        

    print(accuracy)
    print(MSE)
    print(epoch_num)


if __name__ == '__main__':
    main()
