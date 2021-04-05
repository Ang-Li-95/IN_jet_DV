#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import uproot
#import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
No=10
Ds=4
Dr=1
import numpy as np
fn_dir = '/uscms/home/ali/nobackup/LLP/crabdir/JetTreeVmetthresv1METm/'
m_path = './'
save_plot_path='./'
normalize_factors = {}
# orders: ['vtx_x', 'vtx_y', 'vtx_z', 'vtx_ntrack', 'vtx_dBV', 'vtx_err_dBV', 'vtx_px', 'vtx_py', 'vtx_pz', 'vtx_E']


# In[2]:


def GetXsec(fns):
    xsecs = []
    fns_xsec = {
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
        "ttbar_2017": 51315,
        "mfv_splitSUSY_tau000000000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000000um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1900_2017": 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1900_2017": 1e-03,
    }
    for fn in fns:
        assert(fn in fns_xsec)
        xsecs.append(fns_xsec[fn])
    return xsecs

def GetNormWeight(fns, int_lumi=1):
    xsecs = GetXsec(fns)
    nevents = GetNevts(fns)
    assert(len(xsecs)==len(nevents))
    normweights = []
    for i in range(len(xsecs)):
        normweights.append((xsecs[i]*int_lumi)/nevents[i])
    return normweights

def GetNevts(fns):
    nevents = []
    for fn in fns:
        f = uproot.open(fn_dir+fn+'.root')
        nevt = f['mfvWeight/h_sums'].values()[f['mfvWeight/h_sums'].axis().labels().index('sum_nevents_total')]
        nevents.append(nevt)
    return nevents

def GetData(fns, cut="(met_pt < 150) & (max_SV_ntracks > 0)"):
    ML_inputs = []
    phys_variables = []
    ML_variables = ['evt', 'max_SV_ntracks', 'met_pt', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
    jet_variables = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
    variables = ['evt', 'weight', 'max_SV_ntracks', 'met_pt', 'met_phi', 
                 'nsv', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 
                 'vtx_ntk', 'vtx_dBV', 'vtx_dBVerr', 'vtx_tk_pt', 'vtx_tk_eta', 'vtx_tk_nsigmadxy']
    for fn in fns:
        print(fn)
        f = uproot.open(fn_dir+fn+'.root')
        f = f["mfvJetTreer/tree_DV"]
        phys = f.arrays(variables, cut, library='np')
        matrix = np.array([phys['jet_pt'], phys['jet_eta'], phys['jet_phi'], phys['jet_energy']])
        m = zeropadding(matrix, No)
        m = normalizedata(m)
        ML_inputs.append(m)
        phys_variables.append(phys)
        
    return ML_inputs, phys_variables

def calcMLscore(ML_inputs, model_path='./', model_name="test_model.meta"):
    batch_size=4096
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        MLscores = []
        for ML_input in ML_inputs:
            evt=0
            outputscore = []
            while evt<len(ML_input):
                if evt+4096 <= len(ML_input):
                    batch_input = ML_input[evt:evt+4096]
                    
                else:
                    batch_input = ML_input[evt:]
                evt += 4096
                Rr, Rs, Ra = getRmatrix(len(batch_input))
                ML_output = sess.run(['INscore:0'],feed_dict={'O:0':batch_input,'Rr:0':Rr,'Rs:0':Rs,'Ra:0':Ra})#,label:test[2],ntk_max:test[1],evtweight:(test[2]-1)*(-1)})
                outputscore.append(ML_output[0])
                #print(ML_output[0].shape)
            outputscore = np.concatenate(outputscore)
            MLscores.append(outputscore)
    return MLscores


# In[3]:


def zeropadding(matrix, l):
    '''
    make the number of object the same for every event, zero padding those
    matrix: np.array of data
    l: expected length of each event (# objects)
    '''
    m_mod = []
    #idx = np.sort(np.array(list(set(df.index.droplevel(['subentry'])))))
    #print(idx)
    for i in range(matrix.shape[1]):
        # transfer df to matrix for each event
        m = np.array([matrix[:,i][0],matrix[:,i][1], matrix[:,i][2], matrix[:,i][3]])
        #m = np.array(df.loc[i].T)
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
    normalize_factors = [
        [80.74742126464844, 25.610614776611328, 402.87548828125],
        [0.0019912796560674906, -1.7208560705184937, 1.7145205736160278],
        [-0.02133125066757202, -2.826939582824707, 2.815577507019043],
        [117.2960205078125, 34.48314666748047, 679.9258422851562],
    ]
    for i in range(n_features_data):
        #l = np.sort(np.reshape(data[:,i,:],[1,-1])[0])
        #l = l[l!=0]
        #median = l[int(len(l)*0.5)]
        #l_min = l[int(len(l)*0.05)]
        #l_max = l[int(len(l)*0.95)]
        #normalize_factors[i] = [median,l_min,l_max]
        #print(normalize_factors[i])
        median = normalize_factors[i][0]
        l_min = normalize_factors[i][1]
        l_max = normalize_factors[i][2]
        data[:,i,:][data[:,i,:]!=0] = (data[:,i,:][data[:,i,:]!=0]-median)*(2.0/(l_max-l_min))
    return data

def getRmatrix(mini_batch_num=100):
    # Set Rr_data, Rs_data, Ra_data and X_data
    Nr = No*(No-1)
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


# In[7]:


def makehist(data, weight, label, name, **kwargs):
    plt.hist(data,weights=weight, **kwargs)
    plt.yscale('log')
    plt.title(label[0]+'_'+name)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.savefig(label[0]+name+'.png')
    plt.close()
    return

def comparehists(datas, weight, legends, label, name, **kwargs):
    for i in range(len(datas)):
        plt.hist(datas[i],weights=weight[i], label=legends[i], alpha=0.5, density=True, **kwargs)
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.legend(loc='best')
    plt.savefig(label[0]+name+'.png')
    plt.close()
    return 

def MLoutput(signals, sig_fns, backgrounds, bkg_fns):
    weights = GetNormWeight(bkg_fns, int_lumi=41521.0)
    MLoutput_bkg = []
    w_bkg = []
    for i in range(len(bkg_fns)):
        # w includes event-level weight and xsec normalizetion
        w_bkg.append(backgrounds[i]['weight']*weights[i])
        #w = backgrounds[i]['weight']*weights[i]
        MLoutput_bkg.append(backgrounds[i]['MLScore'])
    MLoutput_bkg = np.concatenate(MLoutput_bkg, axis=None)
    w_bkg = np.concatenate(w_bkg, axis=None)
    
    MLoutput_sig = []
    for i in range(len(sig_fns)):
        MLoutput_sig.append(signals[i]['MLScore'])
    MLoutput_sig = np.concatenate(MLoutput_sig, axis=None)
    w_sig = np.ones(MLoutput_sig.shape)
    comparehists([MLoutput_sig, MLoutput_bkg], [w_sig, w_bkg], ['signal', 'background'], 
                 ['MLscore', 'MLscore', 'fraction of events'],'_sig_bkg_compare', bins=100, range=(0,1))
    #compare.show()
    #return compare

def plotWithIdx(phys_vars, idx, name, fns):
    
    # variables that each event has only one value
    plot_vars_single = {
        'met_pt':['MET','MET (GeV)','# events'], 
        'nsv':['nSV','nSV','# events'], 
        'max_SV_ntracks':['max_ntracks_SV','max(ntracks/SV)','# events'],
        'MLScore':['MLScore','MLScore','# events'],
    }
    
    # variables that each event has possibly more than one value
    plot_vars_multi = {
        'vtx_ntk':['vtx_ntk','nTracks/SV','# events'],
        'vtx_dBV':['vtx_dBV','dist2d(SV, beamspot) (cm)','# events'],
        'vtx_dBVerr':['vtx_dBVerr','error dist2d(SV, beamspot) (cm)','# events']
    }
    
    # variables that each event has multiple arrays
    plot_vars_nestedarray = {
        'vtx_tk_pt':['vtx_tk_pt','all SV tracks pT (GeV)','# events'], 
        'vtx_tk_eta':['vtx_tk_eta','all SV tracks eta','# events'], 
        'vtx_tk_nsigmadxy':['vtx_tk_nsigmadxy','all SV tracks nsigmadxy','# events'],
    }
    
    plot_setting = {
        'met_pt': {'range':(0,150), 'bins':60},
        'nsv': {'range':(0,10), 'bins':10},
        'max_SV_ntracks': {'range':(0,40), 'bins':40},
        'MLScore': {'range':(0,1), 'bins':100},
        'vtx_ntk': {'range':(0,40), 'bins':40},
        'vtx_dBV': {'range':(0,0.4), 'bins':100},
        'vtx_dBVerr': {'range':(0,0.005), 'bins':50},
        'vtx_tk_pt': {'range':(0,200), 'bins':100},
        'vtx_tk_eta': {'range':(-4,4), 'bins':50},
        'vtx_tk_nsigmadxy': {'range':(0,40), 'bins':100},
        
    }
    
    weights = GetNormWeight(fns, int_lumi=41521.0)
    plot_w = {}
    plot_data = {}
    
    for i in range(len(fns)):
        # w includes event-level weight and xsec normalizetion
        #w.append(phys_vars[i]['weight'][idx]*weights[i])
        #print("sample {0} w length {1} idx length {2}".format(fns[i], len(phys_vars[i]['weight']), len(idx[i])))
        if len(phys_vars[i]['weight'][idx[i]])==0:
            continue
        w = phys_vars[i]['weight'][idx[i]]*weights[i]
        for v in plot_vars_single:
            if v in plot_data:
                plot_data[v].append(phys_vars[i][v][idx[i]])
                plot_w[v].append(w)
            else:
                phys_vars[i][v][idx[i]].shape
                plot_data[v] = [phys_vars[i][v][idx[i]]]
                plot_w[v] = [w]
        
        for v in plot_vars_multi:
            var = phys_vars[i][v][idx[i]]
            # make w the same dimension as variables
            w_extended = []
            for ievt in range(len(w)):
                w_extended.append([w[ievt]]*len(var[ievt]))
            var_flattern = np.concatenate(var)
            w_extended = np.concatenate(w_extended)
            #print(w_extended.shape)
            #print()
            if v in plot_data:
                plot_data[v].append(var_flattern)
                plot_w[v].append(w_extended)
            else:
                plot_data[v] = [var_flattern]
                plot_w[v] = [w_extended]
           
        for v in plot_vars_nestedarray:
            var = phys_vars[i][v][idx[i]]
            # flattern variable data and make w the same dimensions
            w_extended = []
            var_flattern = []
            for ievt in range(len(w)):
                var_ievt_array = np.concatenate(var[ievt].tolist(), axis=None)
                w_extended.append([w[ievt]]*len(var_ievt_array))
                var_flattern.append(var_ievt_array)
            w_extended = np.concatenate(w_extended)
            var_flattern = np.concatenate(var_flattern, axis=None)
            if v in plot_data:
                plot_data[v].append(var_flattern)
                plot_w[v].append(w_extended)
            else:
                plot_data[v] = [var_flattern]
                plot_w[v] = [w_extended]
            
    for v_dict in [plot_vars_single, plot_vars_multi, plot_vars_nestedarray]:
        for v in v_dict:
            #print(plot_data[v])
            data_processed = np.concatenate(plot_data[v], axis=None)
            w_processed = np.concatenate(plot_w[v], axis=None)
            if v in plot_setting:
                makehist(data_processed, w_processed, v_dict[v], name, **plot_setting[v])
            else:
                makehist(data_processed, w_processed, v_dict[v], name)
            #p.show()
    

def main():
    fns = [
      'qcdht0200_2017',
      'qcdht0300_2017',
      'qcdht0500_2017',
      'qcdht0700_2017', 
      'qcdht1000_2017', 
      'qcdht1500_2017', 
      'qcdht2000_2017', 
      'wjetstolnu_2017', 
      'wjetstolnuext_2017', 
      'zjetstonunuht0100_2017', 
      'zjetstonunuht0200_2017', 
      'zjetstonunuht0400_2017', 
      'zjetstonunuht0600_2017', 
      'zjetstonunuht0800_2017', 
      'zjetstonunuht1200_2017', 
      'zjetstonunuht2500_2017', 
      'ttbar_2017'
    ]
    MLscore_threshold = 0.4
    ML_inputs, phys_vars = GetData(fns)
    ML_outputs = calcMLscore(ML_inputs)
    for i in range(len(fns)):
        phys_vars[i]['MLScore'] = ML_outputs[i]
    idx_highML = []
    idx_lowML = []
    for out in ML_outputs:
        highML = out>MLscore_threshold
        idx_highML.append(np.reshape(highML, len(highML)))
        lowML = out<=MLscore_threshold
        idx_lowML.append(np.reshape(lowML, len(lowML)))
    plotWithIdx(phys_vars, idx_highML, 'highML', fns)
    plotWithIdx(phys_vars, idx_lowML, 'lowML', fns)
    #sig_fns = ['mfv_splitSUSY_tau000001000um_M1400_1200_2017']
    sig_fns = ['mfv_splitSUSY_tau000001000um_M2000_1800_2017']
    MLinputs_sig, phys_vars_sig = GetData(sig_fns)
    ML_outputs_sig = calcMLscore(MLinputs_sig)
    for i in range(len(sig_fns)):
        phys_vars_sig[i]['MLScore'] = ML_outputs_sig[i]
    MLoutput(phys_vars_sig, sig_fns, phys_vars, fns)


# In[6]:


main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




