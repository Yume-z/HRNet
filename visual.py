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



model = models.get_face_alignment_net(config)
# model code by self is not effective

inputs = torch.randn(2, 3, 1024, 512)  # 生成输入的张量
with SummaryWriter(comment='constantModel') as w:
    w.add_graph(model, inputs, True)