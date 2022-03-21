# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------


from .Xray import xRAY

__all__ = ['xRAY']


def get_dataset(config):

    if config.DATASET.DATASET == 'Xray':
        return xRAY

    else:
        raise NotImplemented()

