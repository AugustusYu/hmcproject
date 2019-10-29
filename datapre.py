import json
import numpy as np


def readyelpreview():
    # read yelp review data
    # output is: user item star time
    p = 'data/yelp/review.json'
    alldata = []
    with open(p,'r') as inp:
        for line in inp:
            alldata.append(json.loads(line))
    print('finish json load')
    userdict = {}
    poidict = {}
    rate = np.zeros([len(alldata), 4])
    for i in range(len(alldata)):
        d = alldata[i]
        u = d['user_id']
        p = d['business_id']
        if u not in userdict:
            userdict[u] = len(userdict)
        if p not in poidict:
            poidict[p] = len(poidict)
        rate[i,:] = np.array([userdict[u], poidict[p],
                              np.int(d['stars']), date2num(d['date'])])
    outp = 'data/yelpreview.npy'
    np.save(outp, [rate])
    print('finish all')

def yelpreviewpre():
    p = 'data/yelpreview.npy'
    print('begin pre process data')
    data = np.load(p)
    data = data[0].astype(np.int32)
    unum = max(data[:,0])+1
    pnum = max(data[:,1]) + 1
    ut = 250
    pt = 200
    udict = np.zeros([unum])
    pdict = np.zeros([pnum])
    for i in range(data.shape[0]):
        udict[data[i,0]] +=1
        pdict[data[i, 1]] += 1
    print('user is %d'%(np.sum(udict >= ut)))
    print('poi is %d' % (np.sum(pdict >= pt)))
    uidx = udict[data[:,0]] >=ut
    pidx = pdict[data[:, 1]] >= pt
    outdata = data[uidx & pidx,:]
    outp = 'data/yelppre_u'+str(ut)+'_p'+str(pt)+'.npy'
    print('r =%d'%(np.sum(uidx & pidx)))
    np.save(outp, outdata)

def reshapedata(data):
    y = data.copy()
    return y


def yelpreviewstats():
    p = 'data/yelpreview.npy'
    data = np.load(p)
    data = data[0]
    print(data.shape)
    print(data[0:10,:])
    print(max(data[:,0]))
    print(max(data[:,1]))
    print(min(data[:,2]), max(data[:,2]))
    print(min(data[:, 3]), max(data[:, 3]))

def date2num(date):
    y = date.split('-')
    a = ''.join(y)
    return np.int(a)

def yelpcheckin():
    p = 'data/yelp/checkin.json'
    alldata = []
    with open(p,'r') as inp:
        for line in inp:
            alldata.append(json.loads(line))
    print('finish json load')
    print('length is %d'%(len(alldata)))
    print(alldata[1])

if __name__ == '__main__':
    print('begin data pre prosess ')
    #readyelpreview()
    #yelpcheckin()
    #yelpreviewstats()
    yelpreviewpre()