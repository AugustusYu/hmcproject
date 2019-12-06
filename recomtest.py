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
    # for i in range(10):
    #     p = i/10.0+0.1
    #     trainmerge[i,:,:] = trainmergef(train, p)
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)
    for i in range(10):
        trainmerge[i,:,:] = train.copy()
        trainmerge[i,idx[:np.int(train.shape[0]* (i/10.0+0.1))],2] = 4
    np.save(outp, [train, test, trainmerge, unum, pnum])


def trainmergef(data, p ):
    # use 4 to merge
    num = data.shape[0]
    idx = np.arange(num)
    y = data.copy()
    np.random.shuffle(idx)
    l = np.int(p*num)
    for i in range(l):
        y[idx[i],2] = 4
    return y.astype(np.int32)




def mainmodel(p):
    data = np.load(p,allow_pickle=True)
    train = data[0]
    test = data[1]
    trainmerge = data[2]
    unum = data[3]
    pnum = data[4]
    testmx = data2mx(test,unum,pnum)
    trainmx = data2mx(train,unum,pnum)
    tstart = time.time()
    print('train mx is ',np.sum(trainmx))
    print('test mx is ',np.sum(testmx))
    algo = 2
    f = 10
    hrlist = np.array([10,30,100,300])
    hr = np.zeros([11, np.size(hrlist)])
    pred = trainmodel(trainmx, testmx, f, algo)
    print('user is ',pred.shape[0])
    print('item is ',pred.shape[1])

    hr[0,:] = calhrlist(pred,testmx,hrlist)
    print('%f seconds in i = 0' % (time.time() - tstart), hr[0,:])
    for i in range(10):
        a = trainmerge[i,:,:].reshape([train.shape[0], train.shape[1]])
        trainmx = data2mx(a, unum, pnum)
        pred = trainmodel(trainmx, testmx, f, algo)
        hr[i+1, :] = calhrlist(pred, testmx, hrlist)
        print('%f seconds in i = %d' % (time.time() - tstart,i+1), hr[i+1, :])
    outp = 'test1130-1.mat'
    scio.savemat(outp, {'hr':hr,'hrlist':hrlist})

def maxutest(p):
    data = np.load(p,allow_pickle=True)
    train = data[0]
    test = data[1]
    trainmerge = data[2]
    unum = data[3]
    pnum = data[4]
    testmx = data2mx(test,unum,pnum)
    trainmx = data2mx(train,unum,pnum)
    tstart = time.time()
    print('train mx is ',np.sum(trainmx))
    print('test mx is ',np.sum(testmx))
    algo = 1
    f = 10
    hrlist = np.arange(1,pnum,20)
    #hrlist = np.arange(1,21)
    hr = np.zeros([2, np.size(hrlist)])

    # zero
    pred = trainmodel(trainmx, testmx, f, algo)
    hr[0, :] = calhrlist(pred, testmx, hrlist)

    #full
    a = trainmerge[9, :, :].reshape([train.shape[0], train.shape[1]])
    trainmx = data2mx(a, unum, pnum)
    pred = trainmodel(trainmx, testmx, f, algo)
    hr[1, :] = calhrlist(pred, testmx, hrlist)
    print(np.mean(trainmx[trainmx !=0]))
    outp = 'test1130-itemcf.mat'
    scio.savemat(outp, {'hr':hr,'hrlist':hrlist})
    print('finish write')


def fulltest(p):
    data = np.load(p,allow_pickle=True)
    train = data[0]
    test = data[1]
    trainmerge = data[2]
    unum = data[3]
    pnum = data[4]
    testmx = data2mx(test,unum,pnum)
    trainmx = data2mx(train,unum,pnum)
    tstart = time.time()
    print('train mx is ',np.sum(trainmx))
    print('test mx is ',np.sum(testmx))
    algo = 4
    f = 10
    hrlist = np.array([1,5,10,100,300,500,800,1000])
    hr = np.zeros([11, np.size(hrlist)])

    pred = trainmodel(trainmx, testmx, f, algo)
    hr[0, :] = calhrlist(pred, testmx, hrlist)

    print('%f seconds in i = 0' % (time.time() - tstart), hr[0,:])
    for i in range(10):
        a = trainmerge[i,:,:].reshape([train.shape[0], train.shape[1]])
        trainmx = data2mx(a, unum, pnum)
        pred = trainmodel(trainmx, testmx, f, algo)
        hr[i+1, :] = calhrlist(pred, testmx, hrlist)
        print('%f seconds in i = %d' % (time.time() - tstart,i+1), hr[i+1, :])
    outp = 'test1130-3-warp.mat'
    scio.savemat(outp, {'hr':hr,'hrlist':hrlist})
    return hr


def fulltest10time():
    hr = np.zeros([10,11,8])
    inp = 'data/yelppre_u200_p200_re.npy'
    outp = 'tmp2/yelp-3'
    for i in range(10):
        trainsplit(inp, outp+'_'+str(i))
        hr[i,:,:] = fulltest(outp+'_'+str(i)+'.npy')
        print('finish big %d test'%i)
    outp = 'test1130-4all-warp.mat'
    scio.savemat(outp, {'hr': hr})
    return hr


def simutest1201(p):
    # this is test for simu work
    data = np.load(p,allow_pickle=True)
    train = data[0]
    test = data[1]
    trainmerge = data[2]
    unum = data[3]
    pnum = data[4]
    testmx = data2mx(test,unum,pnum)
    alltrainmx = data2mx(train,unum,pnum) # explicit
    a = trainmerge[9, :, :].reshape([train.shape[0], train.shape[1]])
    trainmx = data2mx(a, unum, pnum)  # implicit
    tstart = time.time()
    print('train mx is ',np.sum(trainmx))
    print('test mx is ',np.sum(testmx))
    algo = 4
    f = 10
    hrlist = np.array([1,5,10,100,300,500,800,1000])

    t = 300
    iter = 10
    hr = np.zeros([iter, np.size(hrlist)])
    rate = np.zeros([iter] )
    hitlist = np.zeros([unum])
    for i in range(iter):
        rate[i] = np.sum(hitlist) / unum
        trainmx = gethitmx(trainmx, alltrainmx, hitlist)
        pred = trainmodel(trainmx, testmx, f, algo)
        hr[i, :] = calhrlist(pred, testmx, hrlist)
        hitlist = calhit(pred,testmx,t,hitlist)
        print('finish %d test'%i)
        print(rate[i], hr[i,:])
    outp = 'test1202-simu.mat'
    scio.savemat(outp, {'hr': hr,'rate':rate,'t':t})




if __name__ == '__main__':
    print('begin train and recom prosess ')
    inp = 'data/yelppre_u200_p200_re.npy'
    outp = 'data/yelptest-3.npy'
    #trainsplit(inp, outp)
    #mainmodel(outp)
    #maxutest(outp)
    #fulltest10time()
    simutest1201(outp)
