# this is test for recommender performance

import numpy as np
import scipy.io as scio
from predictfuns import *
import time

def trainsplit(p, outp):
    # this is generate train data and add noise to this
    data = np.load(p)
    data = data.astype(np.int32)

    num = data.shape[0]
    unum = np.max(data[:,0]) + 1
    pnum = np.max(data[:,1]) + 1
    idx = np.arange(num)
    np.random.shuffle(idx)
    trainnum = np.int(0.8*num)
    testnum = num - trainnum
    train = data[idx[:trainnum],0:3]
    test = data[idx[trainnum:],0:3]
    print(train.shape)
    trainmerge = np.zeros([10, train.shape[0], train.shape[1]],dtype = np.int32)
    for i in range(10):
        p = i/10.0+0.1
        trainmerge[i,:,:] = trainmergef(train, p)
    np.save(outp, [train, test, trainmerge, unum, pnum])

def trainmergef(data, p ):
    # use 4 to merge
    num = data.shape[0]
    idx = np.arange(num)
    y = data.copy()
    np.random.shuffle(idx)
    l = np.int(p*num)
    for i in range(l):
        y[idx[i],2] = 3
    return y.astype(np.int32)


def mainmodel(p):
    data = np.load(p,allow_pickle=True)
    train = data[0]
    test = data[1]
    trainmerge = data[2]
    # print(train.shape)
    # print(trainmerge.shape)
    unum = data[3]
    pnum = data[4]

    testmx = data2mx(test,unum,pnum)
    trainmx = data2mx(train,unum,pnum)
    tstart = time.time()

    outd = np.zeros(11)
    outd[0] = itemcf(trainmx, testmx)
    print('rmse = %f in %f seconds in i = 0'%(outd[0], time.time()-tstart))
    for i in range(10):
        a = trainmerge[i,:,:].reshape([train.shape[0], train.shape[1]])
        trainmx = data2mx(a, unum, pnum)
        outd[i+1] = itemcf(trainmx, testmx)
        print('rmse = %f in %f seconds in i = %d' % (outd[i+1], time.time() - tstart, i+1))
    outp = 'itemcf-3.mat'
    scio.savemat(outp, {'usercf':outd})



if __name__ == '__main__':
    print('begin train and recom prosess ')
    inp = 'data/yelppre_u200_p200_re.npy'
    outp = 'data/yelptest-3.npy'
    #trainsplit(inp, outp)
    mainmodel(outp)