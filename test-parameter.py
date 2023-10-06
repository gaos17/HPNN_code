'''
模型评估
'''

import torch
from attenmodel import Attenmodel
from dataset import get_dataloader
import config
import plotresults as pr
import pandas as pd
import numpy as np

def eval(lead_time,NSE):
    model = Attenmodel().to(config.device)
    model.load_state_dict(torch.load(config.model_save_path))
    # 下面代码用于输出模型参数
    jlxs = []
    hlxs = pd.DataFrame()
    for i in model.parameters():
        print(np.array(i.data.cpu()).reshape(-1))
        it = np.array(i.data.cpu()).reshape(-1)
        if len(it) == 1:
            jlxs.append(it)
        else:
            if len(hlxs) == 0:
                hlxs = pd.Series(it)
            else:
                hlxs = pd.concat([hlxs, pd.Series(it)], axis=0)
    hlxs.to_excel("./parameter/concentration"+str(config.time)+".xlsx")
    jlxs = pd.DataFrame(jlxs)
    jlxs.to_excel("./parameter/generation"+str(config.time)+".xlsx")

    with torch.no_grad():
        data_loader = get_dataloader(dataset1='test',lead_time=lead_time)
        for idx,(time, input, flowbase, area,  target) in enumerate(data_loader):
            input = input.to(config.device)
            flowbase = flowbase.to(config.device)
            area = area.to(config.device)
            decoder_outputs = model(input, flowbase, area)
            decoder_outputs = decoder_outputs.cpu()
            target = target
            results = pd.concat(
                [pd.DataFrame(time), pd.DataFrame(target.numpy()), pd.DataFrame(decoder_outputs.numpy())], axis=1)
            if idx == 0:
                results_all = results
            else:
                results_all = pd.concat([results_all, results], axis=0)
        results_all.columns = ['times', 'rainmean'] + ['real_' + str(i + 1) for i in range(config.leadtimes)] + ['pred_' + str(i + 1) for i in range(config.leadtimes)]
        results_all.sort_values(by='times', inplace=True)
        results_all.to_excel(config.save_result_path)
        # print(results_all.head())
        nse, rmse, rr, pte = pr.evaluate(results_all)
        nse_rmse = pd.DataFrame([nse, rmse, rr, pte])
        NSE = pd.concat([NSE,nse_rmse],axis=1)

        print(nse_rmse)
        # savepath = config.resultpicture_save_path
        # pr.plotrealpred(results_all, path=savepath, leadtime=config.plot_leadtime)
        return NSE


if __name__ == '__main__':
    NSE = pd.DataFrame()
    NSE = eval(lead_time=0,NSE=NSE)
    # NSE.to_excel(config.save_nse_path)


