# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------


from .Xray import xRAY
from .US import us

__all__ = ['xRAY', 'us']


def get_dataset(config):

    if config.DATASET.DATASET == 'Xray':
        return xRAY
    if config.DATASET.DATASET == 'US':
        return us

    else:
        raise NotImplemented()

