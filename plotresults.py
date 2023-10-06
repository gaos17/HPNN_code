'''
评价指标的计算以及结果的展示
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import pandas as pd
import datetime
import config

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def calculate_rmse(x, y):
    var = np.power(x-y, 2).sum()
    z = np.power(var/len(x), 0.5)
    return z

def calculate_nse(x, y):
    z = 1 - np.power((x - y), 2).sum() / np.power((x.mean() - x), 2).sum()
    return z

def calculate_tpe(x, y):
    x_sort = np.sort(x)[::-1]
    x_argsort = np.argsort(-x)
    y_sort = np.array(y)[x_argsort]
    z = abs(x_sort[0:len(x)//50]-y_sort[0:len(x)//50]).sum()/x_sort[0:len(x)//50].sum()
    return z

def calculate_kge(x, y):
    w = y.mean()/x.mean()
    var_real = np.power(x - x.mean(), 2).sum()
    var_pred = np.power(y - y.mean(), 2).sum()
    v = np.power(var_pred/var_real, 0.5)
    var1 = ((x - x.mean()) * (y - y.mean())).sum()
    r = var1/(np.power(var_real, 0.5)*np.power(var_pred, 0.5))
    z = 1-np.power(np.power(w-1, 2)+np.power(v-1, 2)+np.power(r-1,2), 0.5)
    return z


def evaluate(results_all):
    nse_t = np.array([])
    rmse_t = np.array([])
    rr_t = np.array([])
    tpe_t = np.array([])
    for i in range(config.leadtimes):
        real, pred = results_all.iloc[:, i+2], results_all.iloc[:, i+2+config.leadtimes]
        rmse = calculate_rmse(real, pred)
        nse = calculate_nse(real, pred)
        rr = calculate_kge(real, pred)
        tpe = calculate_tpe(real, pred)
        nse_t = np.append(nse_t, nse)
        rmse_t = np.append(rmse_t, rmse)
        rr_t = np.append(rr_t, rr)
        tpe_t = np.append(tpe_t, tpe)

    return nse_t, rmse_t, rr_t, tpe_t


def plotrealpred(pfa, path,leadtime =1, ):
    timese = [pfa.loc[0,'times']]
    index = [0]
    for i in np.arange(len(pfa)-1):
        if pd.to_datetime(pfa.iat[i+1,0]) - pd.to_datetime(pfa.iat[i,0]) > 24*datetime.timedelta(seconds=3600):
            timese.append(pfa.iat[i,0])
            index.append(i)
            timese.append(pfa.iat[i+1,0])
            index.append(i+1)
    timese.append(pfa.iat[-1,0])
    index.append(len(pfa)-1)
    for j in np.arange(len(index)//2):   
        pf1 = pfa.iloc[index[2*j]:index[2*j+1],:]
        x = [i for i in np.arange(len(pf1))]
        yreal = pf1.iloc[:,leadtime+1]
        ypre = pf1.iloc[:,leadtime+1+config.leadtimes]
        yrain = pf1.iloc[:,1]
        fig = plt.figure(figsize = (20,10))  
        ax1 = host_subplot(111)
        ax1.plot(x, yreal,'r--',label=u'real');
        ax1.plot(x, ypre,'b-',label=u'annpre');
        ax1.legend(loc=1)
        ax2 = ax1.twinx() # this is the important function 
        #ax2.set_xticks(x)
        #ax2.set_xticklabels(x)
        ax2.bar(x,yrain,alpha=0.3,color='blue',label=u'rain')
        ax2.set_ylim([0,60])
        ax2.invert_yaxis()
        ax2.legend(loc=2)
        plt.savefig(path+str(leadtime)+str(j)+'.jpg')
        #plt.show()



def plotloss(train_loss, val_loss):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = val_loss.index(min(val_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(config.save_plotloss_path, bbox_inches='tight')
    plt.show()