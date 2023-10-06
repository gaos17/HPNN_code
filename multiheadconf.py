"""

作者： gaos
日期：2021/12/15
"""
import torch
import torch.nn as nn
import config
from flowconf import Flowconf

class Multiheadconf(nn.Module):
    def __init__(self, method = config.flowconfmethod):
        super(Multiheadconf, self).__init__()
        self.line1 = config.clones(Flowconf(), config.conv1d_out_channel)


    def forward(self, inputline):
        output = torch.zeros(config.batch_size, config.conv1d_out_channel).to(config.device)
        for i in range(config.conv1d_out_channel):
            output[:,i] = self.line1[i](inputline[:, i, :])
        return output.sum(dim=-1)

