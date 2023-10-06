"""

作者： gaoshuai
日期： 2021年12月30日
"""

import config
import os
import datadeal
import pandas as pd
import re

timesteps = config.timesteps
leadtimes = config.leadtimes
time = config.time
class GeneData():
    def __init__(self):
        path = '../渔潭面均雨量数据'
        files = os.listdir(path)
        print(files)
        for i, file in enumerate(files):
            event = pd.read_csv(os.path.join(path, file), header=0, index_col=0)
            reshaped = datadeal.event_chuli(event, config.timesteps)
            if i == 0:
                raindistrain = reshaped
            elif i == 33:
                raindisval = reshaped
            elif i == 44:
                raindistest = reshaped
            elif i > 33 and i < 44:
                raindisval = pd.concat([raindisval, reshaped])
            elif i > 44 and i < 55:
                raindistest = pd.concat([raindistest, reshaped])
            else:
                raindistrain = pd.concat([raindistrain, reshaped])
        raindisval = raindisval.round(2)
        raindistest = raindistest.round(2)
        raindistrain = raindistrain.round(2)
        raindisval.to_csv('../渔潭数据/raindisval' + str(timesteps) + '_' + str(leadtimes) + '.csv')
        raindistest.to_csv('../渔潭数据/raindistest'+ str(timesteps) + '_' + str(leadtimes) + '.csv')
        raindistrain.to_csv('../渔潭数据/raindistrain'+ str(timesteps) + '_' + str(leadtimes) + '.csv')

if __name__ == "__main__":
    gene = GeneData()
    gene
