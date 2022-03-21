# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------


from .face300w import Face300W

__all__ = ['Face300W']


def get_dataset(config):

    if config.DATASET.DATASET == '300W':
        return Face300W

    else:
        raise NotImplemented()

