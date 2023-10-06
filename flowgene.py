"""
径流生成部分也需要分站点进行，与汇流部分相同
2021/12/14
去掉gru方法, gru方法更适合在汇流中应用
使用1d卷积进行产流计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Flowgene(nn.Module):
    def __init__(self, method = "convrelu"):
        super(Flowgene, self).__init__()
        self.method = method
        self.relu = F.relu
        self.conv = nn.Conv1d(in_channels=1,out_channels=config.conv1d_out_channel, kernel_size=1, stride=1, padding=0,bias=config.conv_bias)
        nn.init.uniform_(self.conv.weight, a=0.1, b=0.9)


    def forward(self, input):

        return self.convrelu_score(input)

    def convrelu_score(self, input):
        """
        input: [batch-size, timesteps]
        output: [batch-size,out-channel, timesteps]
        """
        input1 = self.conv(input.unsqueeze(1))
        output = self.relu(input1)
        return output




