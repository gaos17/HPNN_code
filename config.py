'''
参数文件
'''

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import copy
#每次运行前必改参数


time = 10
flowconfmethod = "unitattention"
# lead_time = 0


#通用的一些参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
timesteps =30
leadtimes = 1
site = 1
# area = torch.tensor([372/3.6,137/3.6,180/3.6,80/3.6]).to(device)
target_dim = 1
hidden_size = 16
num_layer =1
conv1d_out_channel = 1
attn_method ='general'
conv_bias = True


#保存路径
datestring = datetime.strftime(datetime.now(), "%Y%m%d")
model_save_path = "./model/seq2seq_"+str(time)+"attention.model"
optimizer_save_path = "./model/optimizer"+str(time)+"_attention_model"
resultpicture_save_path = "./hydrograph/result"
patience = 30

# dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#结果处理相关参数---test.py

plot_leadtime = 1

save_nse_path = "./result/nse_rmse"+datestring+str(time)+".xlsx"
save_result_path = "./result/result"+datestring+str(time)+".xlsx"
save_plotloss_path = './picture/loss_plot'+datestring+str(time)+'.png'

def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

