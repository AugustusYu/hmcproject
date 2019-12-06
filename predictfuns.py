# this is all funs for predict model
# this is all the test algorithm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import surprise
from surprise.prediction_algorithms.matrix_factorization import NMF as nmf
import lightfm
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.datasets import fetch_movielens
from scipy.sparse import coo_matrix


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
    #return rmse(pred, testmx)
    return pred

def itemcf(trainmx, testmx):
    print('begin item cf')
    sim = cosine_similarity(trainmx.T)
    pred = np.dot(trainmx, sim)
    pred = pred / np.array([np.abs(sim).sum(axis=1)])
    pred[np.isnan(pred)] = 0
    #return rmse(pred, testmx)
    return pred


def mysvd(trainmx,f):
    print('begin svd algo')
    u, s, vt = svds(trainmx, k=f)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    return X_pred


def mynmf(trainmx,testmx, f):
    print ('begin nmf algo')
    A = nmf(n_factors = f)
    A.fit(trainmx)
    return A.test(testmx)


def mylightfm(trainmx, testmx, f ):
    print('begin light mf model')
    model = lightfm.LightFM(no_components = f,loss='warp')
    coo_trainmx = coo_matrix(trainmx)
    model.fit(coo_trainmx)
    pred = np.zeros(trainmx.shape)
    item = np.arange(0,trainmx.shape[1])
    for i in range (trainmx.shape[0]):
        #for j in range(trainmx.shape[1]):
        pred[i,:] = model.predict(i,item)
    print('finish light fm ')
    return pred


def trainmodel(trainmx, testmx, f,mode):
    if mode == 0:
        return usercf(trainmx, testmx)
    elif mode == 1 :
        return itemcf(trainmx, testmx)
    elif mode == 2 :
        return mysvd(trainmx, f)
    elif mode == 3 :
        return mynmf(trainmx, testmx,f)
    elif mode == 4:
        return mylightfm(trainmx, testmx, f)
    else:
        print('check for model parameter')

def rmse(pred, test):
    p = pred[test.nonzero()].flatten()
    t = test[test.nonzero()].flatten()
    return mean_squared_error (p,t)

def calhr(pred,testmx,t):
    hr = np.zeros(pred.shape[0],dtype = np.float)
    n = 0
    for i in range(pred.shape[0]):
        if np.sum(testmx[i,:]) == 0:
            hr[i] = 0
        else:
            n += 1
            a = np.argsort(-pred[0,:])
            for k in range(pred.shape[1]):
                if testmx[i,k] != 0 and k in a[:t]:
                    hr[i] += 1.0
            hr[i] = hr[i] / np.sum(testmx[i,:]!=0)
    return np.sum(hr) / n

def calhrlist(pred,testmx,tlist):
    hr = np.zeros([pred.shape[0], np.size(tlist)],dtype = np.float)
    n = 0
    for i in range(pred.shape[0]):
        if np.sum(testmx[i,:]) == 0:
            hr[i] = 0
        else:
            n += 1
            a = np.argsort(-pred[0,:])
            for k in range(pred.shape[1]):
                if testmx[i,k] != 0 :
                    for f in range(len(tlist)):
                        if k in a[:tlist[f]]:
                            hr[i,f:] += 1.0
                            break
            hr[i,:] = hr[i,:] / np.sum(testmx[i,:]!=0)
    return np.sum(hr,axis = 0) / n

def calhit(pred,testmx, t,hitlist):
    y = hitlist.copy()
    for i in range(len(hitlist)):
        if hitlist[i] == 0 and np.sum(testmx[i,:]) != 0:
            a = np.argsort(-pred[0,:])
            for k in range(pred.shape[1]):
                if testmx[i,k] != 0 :
                    if k in a[:t]:
                        y[i] = 1
                        break

    return y

def gethitmx(trainmx, alltrainmx, hitlist):
    y = trainmx.copy()
    for i in range(len(hitlist)):
        if hitlist[i] !=0:
            y[i,:] = alltrainmx[i,:]
    return y