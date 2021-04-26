#!/usr/bin/env python
# coding: utf-8

# In[66]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import numpy as np
import awkward as ak

No = 50
Ds = 7
Nr = No*(No-1)
Dp = 20
Dr = 1
De = 20
lambda_dcorr = 0.01
lambda_param = 0.0005
lambda_dcorr_met = 0
lr = 0.0005
use_dR = False

normalize_factors = {}

fndir = 'root://cmseos.fnal.gov//store/user/ali/JetTreehighVkeeptk_v1METm/'


# In[2]:


def GetXsec(sample):
    xsecs = {
        "qcdht0200_2017": 1.547e+06,
        "qcdht0300_2017": 3.226E+05,
        "qcdht0500_2017": 2.998E+04,
        "qcdht0700_2017": 6.351E+03,
        "qcdht1000_2017": 1.096E+03,
        "qcdht1500_2017": 99.0,
        "qcdht2000_2017": 20.2,
        "wjetstolnu_2017": 5.28E+04,
        "wjetstolnuext_2017": 5.28E+04,
        "zjetstonunuht0100_2017": 302.8,
        "zjetstonunuht0200_2017": 92.59,
        "zjetstonunuht0400_2017": 13.18,
        "zjetstonunuht0600_2017": 3.257,
        "zjetstonunuht0800_2017": 1.49,
        "zjetstonunuht1200_2017": 0.3419,
        "zjetstonunuht2500_2017": 0.005146,
        "ttbar_2017": 832,
        "mfv_splitSUSY_tau000000000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000000um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1900_2017": 1e-03,
    }
    if sample not in xsecs:
        raise ValueError("Sample {} not available!!!".format(sample))
    return xsecs[sample]

def GetNormWeight(fns, int_lumi=1):
    xsecs = GetXsec(fns)
    nevents = GetNevts(fns)
    assert(len(xsecs)==len(nevents))
    normweights = []
    for i in range(len(xsecs)):
        normweights.append((xsecs[i]*int_lumi)/nevents[i])
    return normweights

def GetNevts(f):
    nevt = f['mfvWeight/h_sums'].values()[f['mfvWeight/h_sums'].axis().labels().index('sum_nevents_total')]
    return nevt

def GetLoadFactor(fn,f,lumi):
    '''
    To make the fraction of background similar actual case (xsec normalization), 
    calculate the factor so that (Number_selected_events)*LoadFactor 
    represent the number of selected events from given sample at given luminosity
    '''
    nevt = GetNevts(f) # total number of events before selection
    xsec = GetXsec(fn)
    return xsec*lumi/nevt
    

def GetDataAndLabel(fns, split, isSignal, cut="(met_pt >= 150) & (max_SV_ntracks > 2)", lumi=100000):
    data_train = None
    data_val = None
    data_test = None
    ntk_train = None
    ntk_val = None
    ntk_test = None
    met_train = None
    met_val = None
    met_test = None
    for fn in fns:
        print("Loading sample {}...".format(fn))
        f = uproot.open(fndir+fn+'.root')
        loadfactor = GetLoadFactor(fn, f, lumi)
        f = f["mfvJetTreer/tree_DV"]
        if len(f['evt'].array())==0:
          print( "ne events!!!")
          continue
        variables = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig','met_pt','max_SV_ntracks']
        mlvar = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig']
        matrix = f.arrays(variables, library='pd')
        matrix = matrix.dropna()
        # apply cuts
        matrix = matrix[(matrix['met_pt']>=150) & (matrix['max_SV_ntracks']>2)]
        #matrix = matrix[matrix['max_SV_ntracks']>2]
        idx = np.sort(np.array(list(set(matrix.index.droplevel(['subentry'])))))
        
        
        # define the max number of events to pick and the number of train/val/test events to use
        if len(idx)==0:
            print(" no events!")
            continue
        train_idx = -1
        val_idx = -1
        nevt_total = idx[-1]
        nevt = int(loadfactor*nevt_total)
        if nevt>nevt_total:
            nevt = nevt_total
        if isSignal:
            nevt = nevt_total
            
        train_idx = int(nevt*split[0])
        val_idx = int(nevt*(split[0]+split[1]))
        
        # train
        mdf = matrix.loc[:train_idx]
        m = zeropadding(mdf[mlvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        met_np = mdf['met_pt'].groupby(level='entry').first().to_numpy()
        met_np = met_np.reshape((-1,1))
        if len(m)>0:
            if data_train is not None:
                data_train = np.concatenate([data_train,m])
                ntk_train = np.concatenate([ntk_train, ntk_np])
                met_train = np.concatenate([met_train, met_np])
            else:
                data_train = m
                ntk_train = ntk_np
                met_train = met_np
        
        # val
        mdf = matrix.loc[train_idx+1:val_idx]
        m = zeropadding(mdf[mlvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        met_np = mdf['met_pt'].groupby(level='entry').first().to_numpy()
        met_np = met_np.reshape((-1,1))
        if len(m)>0:
            if data_val is not None:
                data_val = np.concatenate([data_val,m])
                ntk_val = np.concatenate([ntk_val, ntk_np])
                met_val = np.concatenate([met_val, met_np])
            else:
                data_val = m
                ntk_val = ntk_np
                met_val = met_np
        
        # test
        mdf = matrix.loc[val_idx+1:nevt]
        m = zeropadding(mdf[mlvar], No)
        ntk_np = mdf['max_SV_ntracks'].groupby(level='entry').first().to_numpy()
        ntk_np = ntk_np.reshape((-1,1))
        met_np = mdf['met_pt'].groupby(level='entry').first().to_numpy()
        met_np = met_np.reshape((-1,1))
        if len(m)>0:
            if data_test is not None:
                data_test = np.concatenate([data_test,m])
                ntk_test = np.concatenate((ntk_test, ntk_np))
                met_test = np.concatenate((met_test, met_np))
            else:
                data_test = m
                ntk_test = ntk_np
                met_test = met_np
    if not isSignal:
        label_train = np.zeros((len(ntk_train), 1))
        label_val = np.zeros((len(ntk_val), 1))
        label_test = np.zeros((len(ntk_test), 1))
    elif isSignal:
        label_train = np.ones((len(ntk_train), 1))
        label_val = np.ones((len(ntk_val), 1))
        label_test = np.ones((len(ntk_test), 1))
        
    return (data_train, ntk_train, label_train, met_train), (data_val, ntk_val, label_val, met_val), (data_test, ntk_test, label_test, met_test)

def importData(split, normalize=True,shuffle=True):
    '''
    import training/val/testing data from root file normalize, padding and shuffle if needed
    split: [train, val, test] fraction
    returns data_train/val/test, which are tuples, structure:
      (data, ntk, label, met, data_no_normalized)
    '''

    fns_bkg = [
        "qcdht0200_2017",
        "qcdht0300_2017",
        "qcdht0500_2017",
        "qcdht0700_2017",
        "qcdht1000_2017",
        "qcdht1500_2017",
        "qcdht2000_2017",
        "wjetstolnu_2017",
        "wjetstolnuext_2017",
        "zjetstonunuht0100_2017",
        "zjetstonunuht0200_2017",
        "zjetstonunuht0400_2017",
        "zjetstonunuht0600_2017",
        "zjetstonunuht0800_2017",
        "zjetstonunuht1200_2017",
        "zjetstonunuht2500_2017",
        "ttbar_2017",
    ]
    fns_signal = [
        "mfv_splitSUSY_tau000000000um_M2000_1800_2017",
        "mfv_splitSUSY_tau000000000um_M2000_1900_2017",
        "mfv_splitSUSY_tau000000300um_M2000_1800_2017",
        "mfv_splitSUSY_tau000000300um_M2000_1900_2017",
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017",
        "mfv_splitSUSY_tau000001000um_M2000_1900_2017",
        "mfv_splitSUSY_tau000010000um_M2000_1800_2017",
        "mfv_splitSUSY_tau000010000um_M2000_1900_2017",
    ]
    train_sig, val_sig, test_sig = GetDataAndLabel(fns_signal, split, True)
    train_bkg, val_bkg, test_bkg = GetDataAndLabel(fns_bkg, split, False)
    sig_bkg_weight = float(len(train_bkg[0]))/len(train_sig[0])
    print("Training data: {0} signals {1} backgrounds".format(len(train_sig[0]), len(train_bkg[0])))
    nitems = len(train_sig)
    data_train = [None]*(nitems+1)
    data_val = [None]*(nitems+1)
    data_test = [None]*(nitems+1)
    for i in range(nitems):
        data_train[i] = np.concatenate([train_sig[i], train_bkg[i]])
        data_val[i] = np.concatenate([val_sig[i], val_bkg[i]])
        data_test[i] = np.concatenate([test_sig[i], test_bkg[i]])
    
    
    if shuffle:
        shuffler = np.random.permutation(len(data_train[0]))
        for i in range(nitems):
            data_train[i] = data_train[i][shuffler]
        
        shuffler = np.random.permutation(len(data_val[0]))
        for i in range(nitems):
            data_val[i] = data_val[i][shuffler]
        
        shuffler = np.random.permutation(len(data_test[0]))
        for i in range(nitems):
            data_test[i] = data_test[i][shuffler]
            
    data_train[4] = data_train[0]
    data_val[4] = data_val[0]
    data_test[4] = data_test[0]

    if normalize:
        data_train[0] = normalizedata(data_train[0])
        data_val[0] = normalizedata(data_val[0])
        data_test[0] = normalizedata(data_test[0])
    
    return data_train, data_test, data_val, sig_bkg_weight

def zeropadding(df, l):
    '''
    make the number of object the same for every event, zero padding those
    df: pandas dataframe of data
    l: expected length of each event (# objects)
    '''
    m_mod = []
    idx = np.sort(np.array(list(set(df.index.droplevel(['subentry'])))))
    for i in idx:
        # transfer df to matrix for each event
        m = np.array(df.loc[i].T)
        sortedidx = np.argsort(m[0,:])[::-1]
        m = m[:,sortedidx]
        if m.shape[1]<l:
            idx_mod = l-m.shape[1]
            pad = np.zeros((m.shape[0],idx_mod))
            m_mod.append(np.concatenate((m,pad), axis=1))
        else:
            m_mod.append(m[:,0:l])
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
        print(normalize_factors[i])
        data[:,i,:][data[:,i,:]!=0] = (data[:,i,:][data[:,i,:]!=0]-median)*(2.0/(l_max-l_min))
    return data


# In[3]:


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

def getRmatrix_dR2(jets):
    n_evt = len(jets)
    # Set Rr_data, Rs_data, Ra_data and X_data
    Rr_data=np.zeros((n_evt,No,Nr),dtype=float)
    Rs_data=np.zeros((n_evt,No,Nr),dtype=float)
    Ra_data=np.ones((n_evt,Dr,Nr),dtype=float)
    cnt=0
    for i in range(No):
        for j in range(No):
            if(i!=j):
                # use mask to get rid the padded non-existing jets
                mask = np.multiply(jets[:,0,i],jets[:,0,j])==0
                dR2 = np.sum(np.square(jets[:,0:2,i]-jets[:,0:2,j]),axis=1)
                dR2[mask] = -1
                dR2_inverse = (1e-03)/dR2
                dR2_inverse[mask] = 0
                Rr_data[:,i,cnt]=dR2_inverse
                Rs_data[:,j,cnt]=dR2_inverse
                cnt+=1
    R_sum = np.sum(Rr_data,axis=(1,2))
    #Rr_data = Rr_data/R_sum
    #Rs_data = Rs_data/R_sum
    for i in range(len(R_sum)):
        if R_sum[i]==0:
            continue
        Rr_data[i] = Rr_data[i]/R_sum[i]
        Rs_data[i] = Rs_data[i]/R_sum[i]
    return Rr_data, Rs_data, Ra_data


# In[4]:


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
    h_size = 50
    B_trans = tf.transpose(B,[0,2,1])
    B_trans = tf.reshape(B_trans, [-1,(Ds+Dr)]) 
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
    h_size = 50
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
    #h1 = tf.math.sigmoid(tf.matmul(w1, P)+b1)
    h3 = tf.reshape(h3, [-1,1])
    return h3


# In[5]:


train, val, test, pos_weight = importData([0.7,0.15,0.15], True, True)


# In[6]:




# In[7]:


#def train():

O = tf.placeholder(tf.float32, [None, Ds, No], name='O')
Rr = tf.placeholder(tf.float32, [None, No, Nr], name="Rr")
Rs = tf.placeholder(tf.float32, [None, No, Nr], name="Rs")
Ra = tf.placeholder(tf.float32, [None, Dr, Nr], name="Ra")

label = tf.placeholder(tf.float32, [None, 1], name="label")
ntk_max = tf.placeholder(tf.float32, [None, 1], name="ntk_max")
met = tf.placeholder(tf.float32, [None, 1], name="met")
evtweight = tf.placeholder(tf.float32, [None, 1], name="evtweight")

B = m(O,Rr,Rs,Ra)

E = phi_R(B)

C = a(O,Rr,E)

P = phi_O(C)

#P = tf.reduce_sum(P, axis=2, keepdims=True)
out = phi_output(P)
out_sigmoid = tf.math.sigmoid(out, name="INscore")

params_list = tf.global_variables()
#for i in range(len(params_list)):
    #variable_summaries(params_list[i],i)
    
loss_bce = tf.nn.weighted_cross_entropy_with_logits(labels=label,logits=out,pos_weight=pos_weight)
loss_bce = tf.reduce_mean(loss_bce)
loss_param = tf.nn.l2_loss(E)
#loss = 0
for i in params_list:
    loss_param+=tf.nn.l2_loss(i)
dcorr = distance_corr(ntk_max, out_sigmoid, evtweight)
dcorr_met = distance_corr(met, out_sigmoid, evtweight)
loss = loss_bce+lambda_param*loss_param+lambda_dcorr*dcorr+lambda_dcorr_met*dcorr_met
optimizer = tf.train.AdamOptimizer(lr)
trainer=optimizer.minimize(loss)

# tensorboard
tf.summary.scalar('loss_bce',loss_bce)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./')

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_num = 1024
if use_dR:
    Rr_train, Rs_train, Ra_train = getRmatrix_dR2(train[4][:,1:3,:])
    Rr_val, Rs_val, Ra_val = getRmatrix_dR2(val[4][:,1:3,:])
else:
    Rr_train, Rs_train, Ra_train = getRmatrix(batch_num)
    Rr_val, Rs_val, Ra_val = getRmatrix(batch_num)

# training
num_epochs=100
history = []
history_val = []
h_bce = []
h_bce_val = []
h_dcorr = []
h_dcorr_val = []
h_dcorr_met = []
h_dcorr_met_val = []
min_loss = 100
for i in range(num_epochs):
    loss_train = 0
    l_bce_train = 0
    l_dcorr_train = 0
    l_dcorr_met_train = 0
    for j in range(int(len(train[0])/batch_num)):
        batch_data = train[0][j*batch_num:(j+1)*batch_num]
        batch_label = train[2][j*batch_num:(j+1)*batch_num]
        batch_ntk = train[1][j*batch_num:(j+1)*batch_num]
        batch_met = train[3][j*batch_num:(j+1)*batch_num]
        if use_dR:
          batch_Rr = Rr_train[j*batch_num:(j+1)*batch_num]
          batch_Rs = Rs_train[j*batch_num:(j+1)*batch_num]
          batch_Ra = Ra_train[j*batch_num:(j+1)*batch_num]
        else:
          batch_Rr = Rr_train
          batch_Rs = Rs_train
          batch_Ra = Ra_train

        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        l_train,_,bce_train,dcorr_train,dcorr_met_train=sess.run([loss,trainer,loss_bce,dcorr,dcorr_met],feed_dict={O:batch_data,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,label:batch_label,ntk_max:batch_ntk,met:batch_met,evtweight:batch_weight})
        loss_train+=l_train
        l_bce_train+=bce_train
        l_dcorr_train+=dcorr_train
        l_dcorr_met_train+=dcorr_met_train
    history.append(loss_train)
    h_bce.append(l_bce_train)
    h_dcorr.append(l_dcorr_train)
    h_dcorr_met.append(l_dcorr_met_train)

    #shuffle data after each epoch
    train_idx = np.array(range(len(train[0])))
    np.random.shuffle(train_idx)
    for ite in range(len(train)):
      train[ite] = train[ite][train_idx]
    if use_dR:
      Rr_train = Rr_train[train_idx]
      Rs_train = Rs_train[train_idx]
      Ra_train = Ra_train[train_idx]
    
    # validation after each epoch
    loss_val = 0
    l_bce_val = 0
    l_dcorr_val = 0
    l_dcorr_met_val = 0
    for j in range(int(len(val[0])/batch_num)):
        batch_data = val[0][j*batch_num:(j+1)*batch_num]
        batch_label = val[2][j*batch_num:(j+1)*batch_num]
        batch_ntk = val[1][j*batch_num:(j+1)*batch_num]
        batch_met = val[3][j*batch_num:(j+1)*batch_num]
        if use_dR:
          batch_Rr = Rr_val[j*batch_num:(j+1)*batch_num]
          batch_Rs = Rs_val[j*batch_num:(j+1)*batch_num]
          batch_Ra = Ra_val[j*batch_num:(j+1)*batch_num]
        else:
          batch_Rr = Rr_val
          batch_Rs = Rs_val
          batch_Ra = Ra_val

        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        l_val,_,bce_val,dcorr_val,dcorr_met_val=sess.run([loss,out_sigmoid,loss_bce,dcorr,dcorr_met],feed_dict={O:batch_data,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,label:batch_label,ntk_max:batch_ntk,met:batch_met,evtweight:batch_weight})
        loss_val+=l_val
        l_bce_val+=bce_val
        l_dcorr_val+=dcorr_val
        l_dcorr_met_val+=dcorr_met_val
        
    if loss_val < min_loss:
        min_loss = loss_val
        saver = tf.train.Saver()
        saver.save(sess,"test_model")
    history_val.append(loss_val)
    h_bce_val.append(l_bce_val)
    h_dcorr_val.append(l_dcorr_val)
    h_dcorr_met_val.append(l_dcorr_met_val)
    val_idx = np.array(range(len(val[0])))
    np.random.shuffle(val_idx)

    for ite in range(len(val)):
      val[ite] = val[ite][val_idx]

    if use_dR:
      Rr_val = Rr_val[val_idx]
      Rs_val = Rs_val[val_idx]
      Ra_val = Ra_val[val_idx]
    
    print("Epoch {}:".format(i))
    print("Training loss: {0}, BCE: {1}, dCorr: {2}, dCorrMET: {3} "
          .format(loss_train/float(int(len(train[0])/batch_num)), l_bce_train/float(int(len(train[0])/batch_num)), l_dcorr_train/float(int(len(train[0])/batch_num)), l_dcorr_met_train/float(int(len(train[0])/batch_num)) ))
    print("Validation loss: {0}, BCE: {1}, dCorr: {2}, dCorrMET: {3} "
          .format(loss_val/float(int(len(val[0])/batch_num)), l_bce_val/float(int(len(val[0])/batch_num)), l_dcorr_val/float(int(len(val[0])/batch_num)) , l_dcorr_met_val/float(int(len(val[0])/batch_num)) ))


# In[8]:


outputs = ["INscore"]

saver = tf.train.Saver(max_to_keep=20)
saver.save(sess,"test_model")
pred = []
truth = []
ntk = []
with tf.Session() as newsess:
    newsaver = tf.train.import_meta_graph("test_model.meta")
    newsaver.restore(newsess, tf.train.latest_checkpoint('./'))

    if use_dR:
      Rr_test, Rs_test, Ra_test = getRmatrix_dR2(test[4][:,1:3,:])
    else:
      Rr_test, Rs_test, Ra_test = getRmatrix(batch_num)
    for j in range(int(len(test[0])/batch_num)+1):
        if j==int(len(test[0])/batch_num):
          next_idx = len(test[0])
          if not use_dR:
            Rr_test, Rs_test, Ra_test = getRmatrix(next_idx-j*batch_num)
        else:
          next_idx = (j+1)*batch_num
        batch_data = test[0][j*batch_num:next_idx]
        batch_label = test[2][j*batch_num:next_idx]
        batch_ntk = test[1][j*batch_num:next_idx]
        batch_met = test[3][j*batch_num:next_idx]
        if use_dR:
          batch_Rr = Rr_test[j*batch_num:next_idx]
          batch_Rs = Rs_test[j*batch_num:next_idx]
          batch_Ra = Ra_test[j*batch_num:next_idx]
        else:
          batch_Rr = Rr_test
          batch_Rs = Rs_test
          batch_Ra = Ra_test

        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        b = newsess.run([out_sigmoid],feed_dict={O:batch_data,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra})
        pred.append(b[0])
        truth.append(batch_label)
        ntk.append(batch_ntk)

pred = np.concatenate(pred,axis=None)
truth = np.concatenate(truth,axis=None)
ntk = np.concatenate(ntk,axis=None)
#print("number of test: {}".format(len(test[0])))
#print(pred.shape)
#print(truth.shape)

# In[9]:



#b = b[0]
#plt.hist(b[test[2]==1], bins=50, alpha=0.5, density=True, stacked=True, label="signal")
#plt.hist(b[test[2]==0], bins=50, alpha=0.5, density=True, stacked=True, label="background")
plt.hist(pred[truth==1], bins=50, alpha=0.5, density=True, stacked=True, label="signal")
plt.hist(pred[truth==0], bins=50, alpha=0.5, density=True, stacked=True, label="background")
plt.legend(loc="best")
plt.title("IN score")
plt.xlabel('score')
plt.ylabel('A.U.')
plt.savefig("INscore.png")
plt.close()


#t_A = test[2][(b>0.4) & (test[1]>=5)]
#t_B = test[2][(b<0.4) & (test[1]>=5)]
#t_C = test[2][(b>0.4) & (test[1]<5) & (test[1]>2)]
#t_D = test[2][(b<0.4) & (test[1]<5) & (test[1]>2)]
t_A = truth[(pred>0.4) & (ntk>=5)]
t_B = truth[(pred<0.4) & (ntk>=5)]
t_C = truth[(pred>0.4) & (ntk<5) & (ntk>2)]
t_D = truth[(pred<0.4) & (ntk<5) & (ntk>2)]

print("region A: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_A), np.sqrt(np.count_nonzero(t_A)), len(t_A)-np.count_nonzero(t_A), np.sqrt(len(t_A)-np.count_nonzero(t_A))))
print("region B: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_B), np.sqrt(np.count_nonzero(t_B)), len(t_B)-np.count_nonzero(t_B), np.sqrt(len(t_B)-np.count_nonzero(t_B))))
print("region C: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_C), np.sqrt(np.count_nonzero(t_C)), len(t_C)-np.count_nonzero(t_C), np.sqrt(len(t_C)-np.count_nonzero(t_C))))
print("region D: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_D), np.sqrt(np.count_nonzero(t_D)), len(t_D)-np.count_nonzero(t_D), np.sqrt(len(t_D)-np.count_nonzero(t_D))))

