'''
2021/12/14
增加gru方法
2021/12/16
将序列-残差-注意力整合到一起
'''


import torch.nn as nn
import torch
import torch.nn.functional as F
import config






class Flowconf(nn.Module):
    """
    此为单站点的径流聚合函数
    """
    def __init__(self,):
        super(Flowconf, self).__init__()
        self.relu = F.relu
        self.softmax = F.softmax
        self.Wa = nn.Parameter(torch.rand(1, config.timesteps))



    def forward(self, inputline):

        return self.unitattention_score(inputline)



    def unitattention_score(self, inputline):
        '''

        :param inputline:[batch-size,out-channel, timesteps]
        :return:
        '''

        # onetensor = torch.ones(config.timesteps)
        attn = self.softmax(self.Wa, dim=-1)
        # print(1)
        # print(np.array(attn.cpu()))

        output = inputline * attn

        return output.sum(dim=-1)








