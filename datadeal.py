'''
对数据进行滑动处理, 构造'30timestep+12leadtime'的格式,然后将处理好的数据传给dataset.py
'''


import numpy as np
import pandas as pd
import config
import datetime



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def hua_chuang(data, timesteps):
    df = data
    df1 = pd.DataFrame(columns = df.columns)
    for i in range(len(df)-timesteps+1):
        if pd.to_datetime(df.iloc[i+timesteps-1,0]) == pd.to_datetime(df.iloc[i,0]) + (timesteps-1)*datetime.timedelta(seconds=3600):
            df1 = pd.concat([df1, df.iloc[i:i+timesteps]])
    return df1

def event_chuli(event,timesteps):
    #将流量数据dis换到最后一列
    reframed = series_to_supervised(pd.DataFrame(event.dis), 0, 1+config.leadtimes)
    event = pd.merge(event,reframed,left_index=True,right_index=True,how='right')
    meanrain = event.rain
    event.drop(['dis'],axis=1,inplace= True)
    event.insert(1, 'rainmean', meanrain)
    #如果要改leadtimes, 下面这一行必须要改
    event.columns = ['times','rainmean','rain','baseflow','area','dis']+['dis'+str(i+1) for i in range(config.leadtimes)]#,'dis7','dis8','dis9','dis10','dis11','dis12']
    event.index = np.arange(len(event))
    reshaped = hua_chuang(event,timesteps)
    # reshaped.drop(['times'], axis=1,inplace= True)
    return reshaped

if __name__ == '__main__':

    event = pd.read_csv('../渔潭面均雨量数据/event0.csv', index_col=0)
    print(event.head())
    event_reshaped = event_chuli(event, config.timesteps)
    #event_reshped:['times','rainmean','shuixi','quanshang','hucun', 'yutan','dis', 'dis1','dis2','dis3','dis4','dis5','dis6']
    print(event_reshaped.head(100))