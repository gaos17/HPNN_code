"""

作者： gaoshuai
日期： 2021年12月03日
"""

import torch.nn as nn
import torch
from flowgene import Flowgene
from multiheadconf import Multiheadconf
import config

class Attenmodel(nn.Module):
    def __init__(self):
        super(Attenmodel, self).__init__()
        self.line1 = config.clones(Flowgene(), config.site)
        self.line2 = config.clones(Multiheadconf(), config.site)

    def forward(self, input, flowbase, area):
        output = torch.zeros(config.batch_size, config.site).to(config.device)
        for i in range(config.site):
            inputline = input[:,:, i]
            trans = self.line1[i](inputline)
            output[:,i] = self.line2[i](trans)
        output = output.squeeze(1) * area + flowbase
        return output

if __name__ =="__main__":
    net = Attenmodel()
    print(net)
    param = net.named_parameters()
    for i in param:
        print(i)