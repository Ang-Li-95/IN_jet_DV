#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import pickle
import uproot
#import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import numpy as np

No = 10
Ds = 4
Nr = No*(No-1)
Dp = 150
Dr = 1
De = 300
lambda_dcorr = 0.05
lambda_param = 0.0005

normalize_factors = {}
# orders: ['vtx_x', 'vtx_y', 'vtx_z', 'vtx_ntrack', 'vtx_dBV', 'vtx_err_dBV', 'vtx_px', 'vtx_py', 'vtx_pz', 'vtx_E']


# In[2]:

fndir = '/uscms/home/ali/nobackup/LLP/crabdir/JetTreeV36METm/'

def GetDataAndLabel(fns, split, isSignal, lumi=8): # lumi: luminosity/10000
    data_train = None
    data_val = None
    data_test = None
    ntk_train = None
    ntk_val = None
    ntk_test = None
    for fn in fns:
        print(fn)
        f = uproot.open(fndir+fn+'.root')
        f = f["mfvJetTreer/tree_DV"]
        variables = ['met_pt', 'max_SV_ntracks', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
        jetvar = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
        matrix = f.arrays(variables, library='pd')
        matrix = matrix.dropna()
        matrix = matrix[matrix['met_pt']>=150]
        # get events with SVs
        matrix = matrix[matrix['max_SV_ntracks']>2]
        #matrix = matrix[jetvar]
        idx = np.sort(np.array(list(set(matrix.index.droplevel(['subentry'])))))
        
        # define the max number of events to pick and the number of train/val/test events to use
        train_idx = -1
        val_idx = -1
        nevt = 0
        if len(idx)==0:
            print(" no events!")
            continue
        if fns[fn]>=0:
            nevt = fns[fn]*lumi
            if nevt>idx[-1]:
              nevt = idx[-1]
        else:
            nevt = idx[-1]
        train_idx = int(nevt*split[0])
        val_idx = int(nevt*(split[0]+split[1]))
        
        # train
        mdf = matrix.loc[:train_idx]
        m = zeropadding(mdf[jetvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        if len(m)>0:
            if data_train is not None:
                data_train = np.concatenate([data_train,m])
                ntk_train = np.concatenate([ntk_train, ntk_np])
            else:
                data_train = m
                ntk_train = ntk_np
        
        # val
        mdf = matrix.loc[train_idx+1:val_idx]
        m = zeropadding(mdf[jetvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        if len(m)>0:
            if data_val is not None:
                data_val = np.concatenate([data_val,m])
                ntk_val = np.concatenate([ntk_val, ntk_np])
            else:
                data_val = m
                ntk_val = ntk_np
        
        # test
        mdf = matrix.loc[val_idx+1:nevt]
        m = zeropadding(mdf[jetvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        if len(m)>0:
            if data_test is not None:
                data_test = np.concatenate([data_test,m])
                ntk_test = np.concatenate((ntk_test, ntk_np))
            else:
                data_test = m
                ntk_test = ntk_np
    if not isSignal:
        label_train = np.zeros((len(ntk_train), 1))
        label_val = np.zeros((len(ntk_val), 1))
        label_test = np.zeros((len(ntk_test), 1))
    elif isSignal:
        label_train = np.ones((len(ntk_train), 1))
        label_val = np.ones((len(ntk_val), 1))
        label_test = np.ones((len(ntk_test), 1))
        
    return (data_train, ntk_train, label_train), (data_val, ntk_val, label_val), (data_test, ntk_test, label_test)


def GetDataAndLabel_old(fns, split, isSignal):
    data_train = None
    data_val = None
    data_test = None
    ntk_train = None
    ntk_val = None
    ntk_test = None
    for fn in fns:
        print(fn)
        f = uproot.open(fndir+fn+'.root')
        f = f["mfvJetTreer/tree_DV"]
        variables = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
        matrix = f.arrays(variables, library='pd')
        matrix = matrix.dropna()
        idx = np.sort(np.array(list(set(matrix.index.droplevel(['subentry'])))))
        ntk = np.array(f['max_SV_ntracks'].array())
        ntk = ntk.reshape((-1,1))
        train_idx = -1
        val_idx = -1
        nevt = 0
        if fns[fn]>=0:
            nevt = fns[fn]
        else:
            nevt = len(idx)
        train_idx = int(nevt*split[0])
        val_idx = int(nevt*(split[0]+split[1]))
        
        # train
        mdf = matrix.loc[:train_idx]
        m = zeropadding(mdf, No)
        m_idx = np.sort(np.array(list(set(mdf.index.droplevel(['subentry'])))))
        if len(m)>0:
            if data_train is not None:
                #print(m.shape)
                data_train = np.concatenate([data_train,m])
                #data_train = data_train.append(m, ignore_index=False)
                ntk_train = np.concatenate([ntk_train, ntk[m_idx]])
            else:
                data_train = m
                ntk_train = ntk[m_idx]
        
        # val
        mdf = matrix.loc[train_idx+1:val_idx]
        m = zeropadding(mdf, No)
        m_idx = np.sort(np.array(list(set(mdf.index.droplevel(['subentry'])))))
        if len(m)>0:
            if data_val is not None:
                #m = matrix.loc[train_idx+1:val_idx]
                #data_val = data_val.append(m, ignore_index=False)
                data_val = np.concatenate([data_val,m])
                ntk_val = np.concatenate([ntk_val, ntk[m_idx]])
            else:
                #m = matrix.loc[train_idx+1:val_idx]
                data_val = m
                ntk_val = ntk[m_idx]
        
        # test
        mdf = matrix.loc[val_idx+1:idx[nevt-1]]
        m = zeropadding(mdf, No)
        #print(len(m))
        #print(np.array(list(set(mdf.index.droplevel(['subentry'])))))
        #print(ntk.shape)
        if len(m)>0:
            if data_test is not None:
                #m = matrix.loc[val_idx+1:]
                #data_test = data_test.append(m, ignore_index=False)
                data_test = np.concatenate([data_test,m])
                ntk_test = np.concatenate((ntk_test, ntk[np.array(list(set(mdf.index.droplevel(['subentry']))))]))
            else:
                #m = matrix.loc[val_idx+1:]
                data_test = m
                ntk_test = ntk[np.array(list(set(mdf.index.droplevel(['subentry']))))]
    if not isSignal:
        label_train = np.zeros((len(ntk_train), 1))
        label_val = np.zeros((len(ntk_val), 1))
        label_test = np.zeros((len(ntk_test), 1))
    elif isSignal:
        label_train = np.ones((len(ntk_train), 1))
        label_val = np.ones((len(ntk_val), 1))
        label_test = np.ones((len(ntk_test), 1))
        
    return (data_train, ntk_train, label_train), (data_val, ntk_val, label_val), (data_test, ntk_test, label_test)


def importData(split, normalize=True,shuffle=True):
    '''
    import training/val/testing data from root file normalize, padding and shuffle if needed
    split: [train, val, test] fraction
    '''

    fns_bkg = {
        "qcdht0200_2017": -1,
        "qcdht0300_2017": -1,
        "qcdht0500_2017": -1,
        "qcdht0700_2017": -1,
        "qcdht1000_2017": -1,
        "qcdht1500_2017": 3705,
        "qcdht2000_2017": 2017,
        "wjetstolnu_2017": -1,
        "wjetstolnuext_2017": -1,
        "zjetstonunuht0100_2017": 1620,
        "zjetstonunuht0200_2017": 2902,
        "zjetstonunuht0400_2017": 87,
        "zjetstonunuht0600_2017": 306,
        "zjetstonunuht0800_2017": 190,
        "zjetstonunuht1200_2017": 68,
        "zjetstonunuht2500_2017": 2,
        "ttbar_2017": 51315,
    }
    fns_signal = {
        "mfv_splitSUSY_tau000000000um_M2000_1800_2017": -1,
        "mfv_splitSUSY_tau000000000um_M2000_1900_2017": -1,
        "mfv_splitSUSY_tau000000300um_M2000_1800_2017": -1,
        "mfv_splitSUSY_tau000000300um_M2000_1900_2017": -1,
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017": -1,
        "mfv_splitSUSY_tau000001000um_M2000_1900_2017": -1,
        "mfv_splitSUSY_tau000010000um_M2000_1800_2017": -1,
        "mfv_splitSUSY_tau000010000um_M2000_1900_2017": -1,
    }
    '''
    fns_bkg = {
        #"qcdht0200_2017": -1,
        #"qcdht0300_2017": -1,
        #"qcdht0500_2017": -1,
        #"qcdht0700_2017": -1,
        #"qcdht1000_2017": -1,
        "qcdht1500_2017": 3705,
        #"qcdht2000_2017": 2017,
        #"wjetstolnu_2017": -1,
        #"wjetstolnuext_2017": -1,
        #"zjetstonunuht0100_2017": 1620,
        #"zjetstonunuht0200_2017": 2902,
        #"zjetstonunuht0400_2017": 87,
        #"zjetstonunuht0600_2017": 306,
        #"zjetstonunuht0800_2017": 190,
        #"zjetstonunuht1200_2017": 68,
        #"zjetstonunuht2500_2017": 2,
        #"ttbar_2017": 51315,
    }
    fns_signal = {
        #"mfv_splitSUSY_tau000000000um_M2000_1800_2017": -1,
        #"mfv_splitSUSY_tau000000000um_M2000_1900_2017": -1,
        #"mfv_splitSUSY_tau000000300um_M2000_1800_2017": -1,
        #"mfv_splitSUSY_tau000000300um_M2000_1900_2017": -1,
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017": -1,
        #"mfv_splitSUSY_tau000001000um_M2000_1900_2017": -1,
        #"mfv_splitSUSY_tau000010000um_M2000_1800_2017": -1,
        #"mfv_splitSUSY_tau000010000um_M2000_1900_2017": -1,
    }
    '''
    train_sig, val_sig, test_sig = GetDataAndLabel(fns_signal, split, True)
    train_bkg, val_bkg, test_bkg = GetDataAndLabel(fns_bkg, split, False)
    sig_bkg_weight = float(len(train_bkg[0]))/len(train_sig[0])
    print("Training data: {0} signals {1} backgrounds".format(len(train_sig[0]), len(train_bkg[0])))
    data_train = [None]*3
    data_val = [None]*3
    data_test = [None]*3
    for i in range(3):
        data_train[i] = np.concatenate([train_sig[i], train_bkg[i]])
        data_val[i] = np.concatenate([val_sig[i], val_bkg[i]])
        data_test[i] = np.concatenate([test_sig[i], test_bkg[i]])
    
    
    if shuffle:
        shuffler = np.random.permutation(len(data_train[0]))
        for i in range(3):
            data_train[i] = data_train[i][shuffler]
        
        shuffler = np.random.permutation(len(data_val[0]))
        for i in range(3):
            data_val[i] = data_val[i][shuffler]
        
        shuffler = np.random.permutation(len(data_test[0]))
        for i in range(3):
            data_test[i] = data_test[i][shuffler]
            
    if normalize:
        data_train[0] = normalizedata(data_train[0])
        data_val[0] = normalizedata(data_val[0])
        data_test[0] = normalizedata(data_test[0])
        #data_n, label_n = normalizedata(data.copy(),label.copy())
    #print("data  imported: shape {0}".format(data.shape))
    #print("label imported: shape {0}".format(label.shape))
    
    return data_train, data_test, data_val, sig_bkg_weight

def zeropadding(df, l):
    '''
    make the number of object the same for every event, zero padding those
    df: pandas dataframe of data
    l: expected length of each event (# objects)
    '''
    m_mod = []
    idx = np.sort(np.array(list(set(df.index.droplevel(['subentry'])))))
    #print(idx)
    for i in idx:
        # transfer df to matrix for each event
        m = np.array(df.loc[i].T)
        if m.shape[1]<l:
            idx_mod = l-m.shape[1]
            pad = np.zeros((m.shape[0],idx_mod))
            m_mod.append(np.concatenate((m,pad), axis=1))
            #print(np.concatenate((m[i],pad), axis=1))
        else:
            m_mod.append(m[:,0:l])
        #print (m_mod[i].shape)
    return np.array(m_mod)

def normalizedata(data):
    n_features_data = Ds
    
    for i in range(n_features_data):
        l = np.sort(np.reshape(data[:,i,:],[1,-1])[0])
        l = l[l!=0]
        median = l[int(len(l)*0.5)]
        l_min = l[int(len(l)*0.05)]
        l_max = l[int(len(l)*0.95)]
        normalize_factors[i] = [median,l_min,l_max]
        data[:,i,:][data[:,i,:]!=0] = (data[:,i,:][data[:,i,:]!=0]-median)*(2.0/(l_max-l_min))
    return data

def splitData(m, train, val, test):
    m_train = m[0:train]
    m_val = m[train:train+val]
    m_test = m[train+val:]
    return m_train, m_val, m_test


# In[3]:


def variable_summaries(var,idx):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+str(idx)):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# In[4]:


def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
 
    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)
   
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
 
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
    epsilon = 1e-08
    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon)
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon)
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon))**power
  
    return dCorr


# In[5]:


def getRmatrix(mini_batch_num=100):
    # Set Rr_data, Rs_data, Ra_data and X_data
    Rr_data=np.zeros((mini_batch_num,No,Nr),dtype=float)
    Rs_data=np.zeros((mini_batch_num,No,Nr),dtype=float)
    Ra_data=np.ones((mini_batch_num,Dr,Nr),dtype=float)
    cnt=0
    for i in range(No):
        for j in range(No):
            if(i!=j):
                Rr_data[:,i,cnt]=1.0
                Rs_data[:,j,cnt]=1.0
                cnt+=1
    return Rr_data, Rs_data, Ra_data
                
def m(O,Rr,Rs,Ra):
    '''
    The marshalling function that rearranges the object and relations into interacting terms
    In the code, ORr-ORs is used instead of ORr, ORs seperately
    '''
    return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)), Ra],1)

def phi_R(B):
    '''
    The phi_R function that predict the effect of each interaction by applying f_R to each column of B
    '''
    h_size = 300
    B_trans = tf.transpose(B,[0,2,1])
    B_trans = tf.reshape(B_trans, [-1,(Ds+Dr)]) #FIXME: need to make those params more consistent
    w1 = tf.Variable(tf.random.truncated_normal([(Ds+Dr),h_size], stddev=0.1), name="r_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size]), name="r_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(B_trans, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size,h_size], stddev=0.1), name="r_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h_size]), name="r_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([h_size,h_size], stddev=0.1), name="r_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([h_size]), name="r_b3", dtype=tf.float32)
    h3 = tf.nn.relu(tf.matmul(h2, w3)+b3)
    w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4", dtype=tf.float32)
    b4 = tf.Variable(tf.zeros([h_size]), name="r_b4", dtype=tf.float32)
    h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)
    w5 = tf.Variable(tf.truncated_normal([h_size, De], stddev=0.1), name="r_w5", dtype=tf.float32)
    b5 = tf.Variable(tf.zeros([De]), name="r_b5", dtype=tf.float32)
    h5 = tf.matmul(h4, w5) + b5
    h5_trans=tf.reshape(h5,[-1,Nr,De])
    h5_trans=tf.transpose(h5_trans,[0,2,1])
    return h5_trans

def a(O,Rr,E):
    '''
    sum all effect applied on given receiver and then combine it with all other components
    '''
    E_bar = tf.matmul(E,tf.transpose(Rr,[0,2,1]))
    return tf.concat([O,E_bar],1)

def phi_O(C):
    '''
    the phi_O function that predict the final result by applying f_O on each object
    '''
    h_size = 200
    C_trans = tf.transpose(C,[0,2,1])
    C_trans = tf.reshape(C_trans,[-1,Ds+De])
    w1 = tf.Variable(tf.random.truncated_normal([Ds+De, h_size], stddev=0.1), name="o_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(C_trans,w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size,Dp], stddev=0.1), name="o_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([Dp]), name="o_b2", dtype=tf.float32)
    h2 = tf.matmul(h1, w2)+b2
    h2_trans = tf.reshape(h2, [-1, No, Dp])
    h2_trans = tf.transpose(h2_trans, [0,2,1])
    return h2_trans

def sumrows_O(P):
    '''
    sums rows of input P, take input with shape (None, Dp, No)
    output shape (None, Dp, 1)
    '''
    return tf.reduce_sum(P, axis=2, keepdims=True)

def phi_output_sum(P):
    '''
    phi_output: NN that output the score of classifier
    '''
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([h_size, Dp], stddev=0.1), name="out_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size,1]), name="out_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(w1, P)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size, h_size], stddev=0.1), name="out_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h_size,1]), name="out_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(w2, h1)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([1, h_size], stddev=0.1), name="out_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="out_b3", dtype=tf.float32)
    h3 = tf.matmul(w3, h2)+b3
    #h3 = tf.nn.relu(tf.matmul(w2, h1)+b2)
    #h1 = tf.math.sigmoid(tf.matmul(w1, P)+b1)
    h3 = tf.reshape(h3, [-1,1])
    return h3

def phi_output(P):
    '''
    phi_output: NN that output the score of classifier
    '''
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([No, h_size], stddev=0.1), name="out_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([Dp, h_size]), name="out_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(P, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size, 1], stddev=0.1), name="out_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([Dp,1]), name="out_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([1, Dp], stddev=0.1), name="out_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="out_b3", dtype=tf.float32)
    h3 = tf.matmul(w3, h2)+b3
    #h3 = tf.nn.relu(tf.matmul(w2, h1)+b2)
    #h1 = tf.math.sigmoid(tf.matmul(w1, P)+b1)
    h3 = tf.reshape(h3, [-1,1])
    #print(P.shape)
    #print(h1.shape)
    #print(h2.shape)
    #print(h3.shape)
    return h3

# In[6]:


train, val, test, pos_weight = importData([0.7,0.15,0.15], True, True)


# In[ ]:


#def train():

O = tf.placeholder(tf.float32, [None, Ds, No], name='O')
Rr = tf.placeholder(tf.float32, [None, No, Nr], name="Rr")
Rs = tf.placeholder(tf.float32, [None, No, Nr], name="Rs")
Ra = tf.placeholder(tf.float32, [None, Dr, Nr], name="Ra")

label = tf.placeholder(tf.float32, [None, 1], name="label")
ntk_max = tf.placeholder(tf.float32, [None, 1], name="ntk_max")
evtweight = tf.placeholder(tf.float32, [None, 1], name="evtweight")

B = m(O,Rr,Rs,Ra)

E = phi_R(B)

C = a(O,Rr,E)

P = phi_O(C)

#P = tf.reduce_sum(P, axis=2, keepdims=True)
out = phi_output(P)
out_sigmoid = tf.math.sigmoid(out, name="INscore")

params_list = tf.global_variables()
for i in range(len(params_list)):
    variable_summaries(params_list[i],i)
    
#mse = tf.reduce_mean(tf.reduce_min(tf.concat([D12,D21],axis=1),axis=1))
#bce = tf.keras.losses.BinaryCrossentropy()
#loss_bce = bce(out, label)
loss_bce = tf.nn.weighted_cross_entropy_with_logits(labels=label,logits=out,pos_weight=pos_weight)
loss_bce = tf.reduce_mean(loss_bce)
#mse = tf.reduce_mean(D12,axis=None)
#mse=tf.reduce_mean(tf.reduce_mean(tf.square(out-label),[1,2]))
loss_param = 0.3*tf.nn.l2_loss(E)
#loss = 0
for i in params_list:
    loss_param+=tf.nn.l2_loss(i)
dcorr = distance_corr(ntk_max, out_sigmoid, evtweight)
loss = loss_bce+lambda_param*loss_param+lambda_dcorr*dcorr
optimizer = tf.train.AdamOptimizer(0.001)
trainer=optimizer.minimize(loss)

# tensorboard
tf.summary.scalar('loss_bce',loss_bce)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./')

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_num = 512
Rr_data, Rs_data, Ra_data = getRmatrix(batch_num)


# training
num_epochs=300
history = []
history_val = []
h_bce = []
h_bce_val = []
h_dcorr = []
h_dcorr_val = []
min_loss = 100
for i in range(num_epochs):
    loss_train = 0
    l_bce_train = 0
    l_dcorr_train = 0
    for j in range(int(len(train[0])/batch_num)):
        batch_data = train[0][j*batch_num:(j+1)*batch_num]
        batch_label = train[2][j*batch_num:(j+1)*batch_num]
        batch_ntk = train[1][j*batch_num:(j+1)*batch_num]
        batch_weight = (batch_label-1)*(-1)
        #batch_weight = np.ones(batch_label.shape)
        l_train,_,bce_train,dcorr_train=sess.run([loss,trainer,loss_bce,dcorr],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight})
        loss_train+=l_train
        l_bce_train+=bce_train
        l_dcorr_train+=dcorr_train
    history.append(loss_train)
    h_bce.append(l_bce_train)
    h_dcorr.append(l_dcorr_train)
    #shuffle data after each epoch
    train_idx = np.array(range(len(train[0])))
    np.random.shuffle(train_idx)
    train[0] = train[0][train_idx]
    train[1] = train[1][train_idx]
    train[2] = train[2][train_idx]
    
    # validation after each epoch
    loss_val = 0
    l_bce_val = 0
    l_dcorr_val = 0
    for j in range(int(len(val[0])/batch_num)):
        batch_data = val[0][j*batch_num:(j+1)*batch_num]
        batch_label = val[2][j*batch_num:(j+1)*batch_num]
        batch_ntk = val[1][j*batch_num:(j+1)*batch_num]
        batch_weight = (batch_label-1)*(-1)
        #batch_weight = np.ones(batch_label.shape)
        l_val,_,bce_val,dcorr_val=sess.run([loss,out_sigmoid,loss_bce,dcorr],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight})
        #print("V12: {}".format(_))
        #print("P: {}".format(batch_label))
        loss_val+=l_val
        l_bce_val+=bce_val
        l_dcorr_val+=dcorr_val
        
    if loss_val < min_loss:
        min_loss = loss_val
        saver = tf.train.Saver()
        saver.save(sess,"test_model")
    history_val.append(loss_val)
    h_bce_val.append(l_bce_val)
    h_dcorr_val.append(l_dcorr_val)
    val_idx = np.array(range(len(val[0])))
    np.random.shuffle(val_idx)
    val[0] = val[0][val_idx]
    val[1] = val[1][val_idx]
    val[2] = val[2][val_idx]
    
    print("Epoch {0} Training loss: {1}, BCE: {2}, dCorr: {3} Val loss: {4}, BCE: {5}, dCorr: {6} "
          .format(i,loss_train/float(int(len(train[0])/batch_num)), l_bce_train/float(int(len(train[0])/batch_num)), l_dcorr_train/float(int(len(train[0])/batch_num)), 
                  loss_val/float(int(len(val[0])/batch_num)), l_bce_val/float(int(len(val[0])/batch_num)), l_dcorr_val/float(int(len(val[0])/batch_num)) ))


# In[ ]:

outputs = ["INscore"]
#constant_graph = tf.graph_util.convert_variables_to_constants(
#    sess, sess.graph.as_graph_def(), outputs)
#tf.train.write_graph(constant_graph, "./", "constantgraph.pb", as_text=False)


#builder = tf.saved_model.builder.SavedModelBuilder("savedModel")
#builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#builder.save()

saver = tf.train.Saver(max_to_keep=20)
saver.save(sess,"test_model")
with tf.Session() as newsess:
    newsaver = tf.train.import_meta_graph("test_model.meta")
    newsaver.restore(newsess, tf.train.latest_checkpoint('./'))
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), outputs)
    tf.train.write_graph(constant_graph, "./", "constantgraph.pb", as_text=False)
    Rr_test, Rs_test, Ra_test = getRmatrix(len(test[0]))
    #a,b = newsess.run([loss,out_sigmoid],feed_dict={O:test[0],Rr:Rr_test,Rs:Rs_test,Ra:Ra_test})#,label:test[2],ntk_max:test[1],evtweight:np.ones(test[2].shape)})
    b = newsess.run([out_sigmoid],feed_dict={O:test[0],Rr:Rr_test,Rs:Rs_test,Ra:Ra_test})#,label:test[2],ntk_max:test[1],evtweight:np.ones(test[2].shape)})


# In[ ]:

b = b[0]
plt.hist(b[test[2]==1], bins=50, alpha=0.5, density=True, stacked=True, label="signal")
plt.hist(b[test[2]==0], bins=50, alpha=0.5, density=True, stacked=True, label="background")
plt.legend(loc="best")
plt.title("IN score")
plt.xlabel('score')
plt.ylabel('A.U.')
plt.savefig("INscore.png")
plt.close()


t_A = test[2][(b>0.4) & (test[1]>=5)]
t_B = test[2][(b<0.4) & (test[1]>=5)]
t_C = test[2][(b>0.4) & (test[1]<5) & (test[1]>2)]
t_D = test[2][(b<0.4) & (test[1]<5) & (test[1]>2)]

print("region A: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_A), np.sqrt(np.count_nonzero(t_A)), len(t_A)-np.count_nonzero(t_A), np.sqrt(len(t_A)-np.count_nonzero(t_A))))
print("region B: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_B), np.sqrt(np.count_nonzero(t_B)), len(t_B)-np.count_nonzero(t_B), np.sqrt(len(t_B)-np.count_nonzero(t_B))))
print("region C: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_C), np.sqrt(np.count_nonzero(t_C)), len(t_C)-np.count_nonzero(t_C), np.sqrt(len(t_C)-np.count_nonzero(t_C))))
print("region D: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_D), np.sqrt(np.count_nonzero(t_D)), len(t_D)-np.count_nonzero(t_D), np.sqrt(len(t_D)-np.count_nonzero(t_D))))
