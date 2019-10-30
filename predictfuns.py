# this is all funs for predict model
# this is all the test algorithm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error

def predict(u,i):
    return u*i

def data2mx(data,unum,pnum):
    mx = np.zeros([unum, pnum])
    for i in range(data.shape[0]):
        u = data[i,0]
        p = data[i,1]
        r = data[int(i),2]
        mx[int(u),int(p)] = r
    return mx


def usercf(trainmx, testmx):
    print('begin user cf')
    sim = cosine_similarity(trainmx)
    mean = np.average(trainmx,1)
    df = trainmx - mean[:,np.newaxis]
    pred = np.dot(sim, df)
    pred = pred / np.array([np.abs(sim).sum(axis=1)]).T
    pred += mean[:,np.newaxis]
    pred[np.isnan(pred)] = 0
    return rmse(pred, testmx)

def itemcf(trainmx, testmx):
    print('begin item cf')
    sim = cosine_similarity(trainmx.T)
    pred = np.dot(trainmx, sim)
    pred = pred / np.array([np.abs(sim).sum(axis=1)])
    pred[np.isnan(pred)] = 0
    return rmse(pred, testmx)


def rmse(pred, test):
    p = pred[test.nonzero()].flatten()
    t = test[test.nonzero()].flatten()
    return mean_squared_error (p,t)


