'''
准备数据集, 导入数据, 并且只对输入数据进行归一化
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
import pandas as pd

class NumDataset(Dataset,):
    def __init__(self, dataset1 = 'train',lead_time=0):
        if dataset1 == 'train':
            raindistrain = pd.read_csv('../渔潭数据/raindistrain' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                index_col=0)
            rd = raindistrain.values
        if dataset1 == 'test':
            raindistest = pd.read_csv('../渔潭数据/raindistest' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                index_col=0)
            rd = raindistest.values
        if dataset1 == 'val':
            raindisval = pd.read_csv('../渔潭数据/raindisval' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                index_col=0)
            rd = raindisval.values
        self.data = rd.reshape((rd.shape[0] // config.timesteps, config.timesteps, rd.shape[1]))
        self.lead_time = lead_time

    def __getitem__(self, item ):
        #不同模型的lable不完全一样,需要根据模型做具体的修改
        #time只取了timesteps最后一个维度的
        time = self.data[item,-1,:2 ]
        #实现预测降雨用前期降雨均值替代
        input = self.data[item,:,2:3 ]
        input[(len(input)-self.lead_time):len(input),:] = 0

        flowbase = self.data[item,-1, 2+config.site ]
        area = self.data[item,-1, 3+config.site ]
        lable = self.data[item,-1, 4+config.site]
        return time, input, flowbase, area,lable,

    def __len__(self):
        return self.data.shape[0]
def collate_fn(batch):
    time, input, flowbase, area, lable = zip(*batch)
    input = np.array(input).astype(float)
    input = torch.FloatTensor(input)
    flowbase = torch.FloatTensor(flowbase)
    area = torch.FloatTensor(area)
    lable = torch.FloatTensor(lable)
    return time, input, flowbase,area,  lable

def get_dataloader(dataset1 = 'train',lead_time = 0):
    batch_size = config.batch_size
    return DataLoader(NumDataset(dataset1,lead_time),batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

if __name__ == '__main__':
    for j, (time, input, flowbase,area, target) in enumerate(get_dataloader(dataset1='val',lead_time=6)):
        # input: [batch_size, timesteps, feature]
        # target: [batch_size, timesteps, leadtimes]
        if j == 0:
            print(j)
            print(input.shape)
            print(input)
            print(flowbase)
            # print(target)
            print(target)
            break
