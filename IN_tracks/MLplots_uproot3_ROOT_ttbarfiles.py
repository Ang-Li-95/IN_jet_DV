#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
import sys
import copy
import uproot
import ROOT
#import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np

doSignal = False
doBackground = True
No=50
Nr = No*(No-1)
Ds=7
Dr=1
use_dR = False
#fn_dir = 'root://cmseos.fnal.gov//store/user/ali/JetTreelowMETothogonalVTXVkeeptk_v1METm/'
fn_dir = 'root://cmseos.fnal.gov//store/user/ali/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/JetTreelowMETothogonalVTXVkeeptk_v1METm/210427_221010/0000/'
fn_idx = ''
if len(sys.argv)>1:
  fn_idx = sys.argv[1]
m_path = './20210415_0/'
#save_plot_path='./20210411_0/'
save_plot_path='./'
normalize_factors = {}
# orders: ['vtx_x', 'vtx_y', 'vtx_z', 'vtx_ntrack', 'vtx_dBV', 'vtx_err_dBV', 'vtx_px', 'vtx_py', 'vtx_pz', 'vtx_E']

mlvar = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig']

# In[2]:

plot_vars_titles = {
    'met_pt':['MET','MET (GeV)','# events'], 
    'nsv':['nSV','nSV','# events'], 
    'max_SV_ntracks':['max_ntracks_SV','max(ntracks/SV)','# events'],
    'MLScore':['MLScore','MLScore','# events'],
    'vtx_ntk':['vtx_ntk','nTracks/SV','# events'],
    'vtx_dBV':['vtx_dBV','dist2d(SV, beamspot) (cm)','# events'],
    'vtx_dBVerr':['vtx_dBVerr','error dist2d(SV, beamspot) (cm)','# events'],
    'vtx_tk_pt':['vtx_tk_pt','all SV tracks pT (GeV)','# events'], 
    'vtx_tk_eta':['vtx_tk_eta','all SV tracks eta','# events'], 
    'vtx_tk_phi':['vtx_tk_phi','all SV tracks phi','# events'],
    'vtx_tk_dxy':['vtx_tk_dxy','all SV tracks dxybs (cm)','# events'],
    'vtx_tk_dxy_err':['vtx_tk_dxy_err','all SV tracks err(dxybs) (cm)','# events'],
    'vtx_tk_nsigmadxy':['vtx_tk_nsigmadxy','all SV tracks nsigma(dxybs)','# events'],
    'vtx_tk_dz':['vtx_tk_dz','all SV tracks dz (cm)','# events'],
    'vtx_tk_dz_err':['vtx_tk_dz_err','all SV tracks err(dz) (cm)','# events'],
    'vtx_tk_nsigmadz':['vtx_tk_nsigmadz','all SV tracks nsigma(dz)','# events'],
    'jet_pt':['jet_pt','jet pT (GeV)','# events'],
    'jet_eta':['jet_eta','jet eta','# events'],
    'jet_phi':['jet_phi','jet phi','# events'],
    'tk_pt':['tk_pt','track pT (GeV)','# events'],
    'tk_eta':['tk_eta','track eta','# events'],
    'tk_phi':['tk_phi','track phi','# events'],
    'tk_dxybs':['tk_dxybs','track dxybs (cm)','# events'],
    'tk_dxybs_sig':['tk_dxybs_sig','track nsigma(dxybs)','# events'],
    'tk_dxybs_err':['tk_dxybs_err','track err(dxybs) (cm)','# events'],
    'tk_dz':['tk_dz','track dz (cm)','# events'],
    'tk_dz_sig':['tk_dz_sig','track nsigma(dz)','# events'],
    'tk_dz_err':['tk_dz_err','track err(dz) (cm)','# events'],

}

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
    'vtx_dBVerr':['vtx_dBVerr','error dist2d(SV, beamspot) (cm)','# events'],
    'jet_pt':['jet_pt','jet pT (GeV)','# events'],
    'jet_eta':['jet_eta','jet eta','# events'],
    'jet_phi':['jet_phi','jet phi','# events'],
    'tk_pt':['tk_pt','track pT (GeV)','# events'],
    'tk_eta':['tk_eta','track eta','# events'],
    'tk_phi':['tk_phi','track phi','# events'],
    'tk_dxybs':['tk_dxybs','track dxybs (cm)','# events'],
    'tk_dxybs_sig':['tk_dxybs_sig','track nsigma(dxybs)','# events'],
    'tk_dxybs_err':['tk_dxybs_err','track err(dxybs) (cm)','# events'],
    'tk_dz':['tk_dz','track dz (cm)','# events'],
    'tk_dz_sig':['tk_dz_sig','track nsigma(dz)','# events'],
    'tk_dz_err':['tk_dz_err','track err(dz) (cm)','# events'],
}

# variables that each event has multiple arrays
plot_vars_nestedarray = {
    'vtx_tk_pt':['vtx_tk_pt','all SV tracks pT (GeV)','# events'], 
    'vtx_tk_eta':['vtx_tk_eta','all SV tracks eta','# events'], 
    'vtx_tk_phi':['vtx_tk_phi','all SV tracks phi','# events'],
    'vtx_tk_dxy':['vtx_tk_dxy','all SV tracks dxybs (cm)','# events'],
    'vtx_tk_dxy_err':['vtx_tk_dxy_err','all SV tracks err(dxybs) (cm)','# events'],
    'vtx_tk_nsigmadxy':['vtx_tk_nsigmadxy','all SV tracks nsigma(dxybs)','# events'],
    'vtx_tk_dz':['vtx_tk_dz','all SV tracks dz (cm)','# events'],
    'vtx_tk_dz_err':['vtx_tk_dz_err','all SV tracks err(dz) (cm)','# events'],
    'vtx_tk_nsigmadz':['vtx_tk_nsigmadz','all SV tracks nsigma(dz)','# events'],
}

plot_setting = {
    'met_pt': {'range':(0,150), 'bins':60},
    'nsv': {'range':(0,10), 'bins':10},
    'max_SV_ntracks': {'range':(0,40), 'bins':40},
    'MLScore': {'range':(0,1), 'bins':100},
    'vtx_ntk': {'range':(0,40), 'bins':40},
    'vtx_dBV': {'range':(0,0.4), 'bins':100},
    'vtx_dBVerr': {'range':(0,0.05), 'bins':100},
    'vtx_tk_pt': {'range':(0,200), 'bins':100},
    'vtx_tk_eta': {'range':(-4,4), 'bins':50},
    'vtx_tk_phi': {'range':(-3.2,3.2), 'bins':64},
    'vtx_tk_dxy': {'range':(0,0.5), 'bins':50},
    'vtx_tk_dxy_err': {'range':(0,0.025), 'bins':50},
    'vtx_tk_nsigmadxy': {'range':(0,40), 'bins':100},
    'vtx_tk_dz': {'range':(0,20), 'bins':50},
    'vtx_tk_dz_err': {'range':(0,0.15), 'bins':100},
    'vtx_tk_nsigmadz': {'range':(0,3000), 'bins':100},
    'jet_pt': {'range':(0,500), 'bins':50},
    'jet_eta': {'range':(-4,4), 'bins':50},
    'jet_phi': {'range':(-3.2,3.2), 'bins':64},
    'tk_pt': {'range':(0,500), 'bins':200},
    'tk_eta': {'range':(-4,4), 'bins':50},
    'tk_phi': {'range':(-3.2,3.2), 'bins':64},
    'tk_dxybs': {'range':(-0.5,0.5), 'bins':100},
    'tk_dxybs_sig': {'range':(-40,40), 'bins':100},
    'tk_dxybs_err': {'range':(0,0.06), 'bins':50},
    'tk_dz': {'range':(-15,15), 'bins':50},
    'tk_dz_sig': {'range':(-3000,3000), 'bins':100},
    'tk_dz_err': {'range':(0,0.1), 'bins':50},
}

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
        "ttbar_2017": 832,
        "jettree": 832,
        "mfv_splitSUSY_tau000000000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000000um_M2000_1900_2017": 1e-03,
        'mfv_splitSUSY_tau000000000um_M2400_2300_2017': 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000000300um_M2000_1900_2017": 1e-03,
        'mfv_splitSUSY_tau000000300um_M2400_2300_2017': 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000001000um_M2000_1900_2017": 1e-03,
        'mfv_splitSUSY_tau000001000um_M2400_2300_2017': 1e-03,
        'mfv_splitSUSY_tau000001000um_M1200_1100_2017': 1e-03,
        'mfv_splitSUSY_tau000001000um_M1400_1200_2017': 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1800_2017": 1e-03,
        "mfv_splitSUSY_tau000010000um_M2000_1900_2017": 1e-03,
        'mfv_splitSUSY_tau000010000um_M2400_2300_2017': 1e-03,
        'mfv_splitSUSY_tau000010000um_M1200_1100_2017': 1e-03,
        'mfv_splitSUSY_tau000010000um_M1400_1200_2017': 1e-03,
    }
    for fn in fns:
        assert(fn in fns_xsec)
        xsecs.append(fns_xsec[fn])
    return xsecs

def GetNormWeight(fns, int_lumi=1):
    xsecs = GetXsec(fns)
    nevents = [1.5392681e+08]*len(xsecs)
    #nevents = GetNevts(fns)
    assert(len(xsecs)==len(nevents))
    normweights = []
    for i in range(len(xsecs)):
        normweights.append((xsecs[i]*int_lumi)/nevents[i])
    return normweights

def GetNevts(fns):
    nevents = []
    for fn in fns:
        f = uproot.open(fn_dir+fn+fn_idx+'.root')
        nevt = f['mfvWeight/h_sums'].values[f['mfvWeight/h_sums'].xlabels.index('sum_nevents_total')]
        nevents.append(nevt)
        del f
    return nevents

def GetData(fns, cut="(met_pt < 150) & (max_SV_ntracks > 0)"):
    ML_inputs = []
    ML_inputs_original = []
    phys_variables = []
    variables = ['evt', 'weight', 'max_SV_ntracks', 'met_pt', 'met_phi', 'nsv', 
                 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 
                 'tk_pt', 'tk_eta', 'tk_phi', 
                 'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err', 
                 'vtx_ntk', 'vtx_dBV', 'vtx_dBVerr', 
                 'vtx_tk_pt', 'vtx_tk_eta', 'vtx_tk_phi', 
                 'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz']
    for fn in fns:
        #print(fn)
        print("opening {}...".format(fn_dir+fn+fn_idx+'.root'))
        f = uproot.open(fn_dir+fn+fn_idx+'.root')
        f = f["mfvJetTreer/tree_DV"]
        if len(f['evt'].array())==0:
          print( "no events!!!")
          continue
        phys = f.arrays(variables, namedecode="utf-8")
        del f
        evt_select = (phys['met_pt']<150) & (phys['max_SV_ntracks']>0) & (phys['vtx_dBV'].max()>0.005) & (phys['vtx_dBVerr'].min()<0.01)
        for v in phys:
          phys[v] = np.array(phys[v][evt_select])
        if len(phys['evt'])==0:
            print("no events after selection!")
            continue
        matrix = np.array([phys[v] for v in mlvar])
        #matrix = np.array([phys['jet_pt'], phys['jet_eta'], phys['jet_phi'], phys['jet_energy']])
        m = zeropadding(matrix, No)
        ML_inputs_original.append(m.copy())
        m = normalizedata(m)
        ML_inputs.append(m)
        phys_variables.append(phys)
        
    return ML_inputs, ML_inputs_original, phys_variables

def calcMLscore(ML_inputs, ML_inputs_ori, model_path='./', model_name="test_model.meta"):
    batch_size=4096
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        MLscores = []
        for iML in range(len(ML_inputs)):
            ML_input = ML_inputs[iML]
            ML_input_ori = ML_inputs_ori[iML]
            evt=0
            outputscore = []
            while evt<len(ML_input):
                if evt+batch_size <= len(ML_input):
                    batch_input = ML_input[evt:evt+batch_size]
                    batch_input_ori = ML_input_ori[evt:evt+batch_size]
                else:
                    batch_input = ML_input[evt:]
                    batch_input_ori = ML_input_ori[evt:]
                evt += batch_size
                if use_dR:
                  Rr, Rs, Ra = getRmatrix_dR2(batch_input_ori[:,1:3,:])
                else:
                  Rr, Rs, Ra = getRmatrix(len(batch_input))
                ML_output = sess.run(['INscore:0'],feed_dict={'O:0':batch_input,'Rr:0':Rr,'Rs:0':Rs,'Ra:0':Ra})#,label:test[2],ntk_max:test[1],evtweight:(test[2]-1)*(-1)})
                outputscore.append(ML_output[0])
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
        #m = np.array([matrix[:,i][0],matrix[:,i][1], matrix[:,i][2], matrix[:,i][3]])
        m = np.array([matrix[:,i][v] for v in range(len(mlvar))])
        sortedidx = np.argsort(m[0,:])[::-1]
        m = m[:,sortedidx]
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
        #[2.1816406679450657, 1.062500040644502, 20.078124592116605],
        #[0.12451552155432945, -1.8609576341644036, 2.053407424463079],
        #[-0.06035980748269992, -2.8188307302253506, 2.801946409279582],
        #[-1.5050249861509261e-05, -0.026793859606046376, 0.026827086804321543],
        #[-0.0039861770169507685, -9.171536302834037, 9.322075933251009],
        #[-0.1705471082009359, -5.720800580412514, 5.219436095637275],
        #[-22.98692569571753, -1463.7623074782282, 1192.6415322048178],
        [2.9902343476623363, 1.6103515050761894, 21.671875749412926],
        [-0.0043946650753236205, -2.049745121231547, 2.04150510191022],
        [-0.05507905809333474, -2.817760236900055, 2.8013536897606315],
        [-1.5896082453516604e-05, -0.028481722307173097, 0.028575759352729244],
        [-0.005081050368458236, -11.53984885429798, 11.746383776779878],
        [-0.14610174776891854, -5.729244305990972, 5.3753020975522325],
        [-20.675411128530353, -1595.7082149035527, 1344.5222105467956],
    ]
    for i in range(n_features_data):
        #l = np.sort(np.reshape(data[:,i,:],[1,-1])[0])
        #l = l[l!=0]
        #median = l[int(len(l)*0.5)]
        #l_min = l[int(len(l)*0.05)]
        #l_max = l[int(len(l)*0.95)]
        median = normalize_factors[i][0]
        l_min = normalize_factors[i][1]
        l_max = normalize_factors[i][2]
        data[:,i,:][data[:,i,:]!=0] = (data[:,i,:][data[:,i,:]!=0]-median)*(2.0/(l_max-l_min))
    return data

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

# In[7]:

def makehist(data,weight,var):
    assert(var in plot_setting)
    assert(var in plot_vars_titles)
    label = plot_vars_titles[var]
    setting = plot_setting[var]
    h = ROOT.TH1F(label[0],";".join(label),setting['bins'],setting['range'][0],setting['range'][1])
    for i in range(len(data)):
      h.Fill(data[i],weight[i])
    return h

def make2dhist(datax,datay,weight,varx,vary):
    assert(len(datax)==len(datay))
    assert(varx in plot_setting)
    assert(varx in plot_vars_titles)
    assert(vary in plot_setting)
    assert(vary in plot_vars_titles)
    labelx = plot_vars_titles[varx][1]
    labely = plot_vars_titles[vary][1]
    settingx = plot_setting[varx]
    settingy = plot_setting[vary]
    h = ROOT.TH2F("h_"+varx+"_"+vary,";"+labelx+";"+labely,settingx['bins'],settingx['range'][0],settingx['range'][1],settingy['bins'],settingy['range'][0],settingy['range'][1])
    for i in range(len(datax)):
      h.Fill(datax[i],datay[i],weight[i])
    return h

def plotcategory(f,dirname,vars_name,data,weight):
    f.cd()
    f.mkdir(dirname)
    f.cd(dirname)
    #hists = []
    for v in vars_name:
      hist = makehist(data[v], weight[v], v)
      hist.Write()
    hist2d = make2dhist(data['vtx_dBV'],data['vtx_dBVerr'],weight['vtx_dBV'],'vtx_dBV','vtx_dBVerr')
    hist2d.Write()
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
        #MLoutput_sig.append(signals[i]['MLScore'])
        #MLoutput_sig = np.concatenate(MLoutput_sig, axis=None)
        #w_sig = np.ones(MLoutput_sig.shape)
        MLoutput_sig = signals[i]['MLScore']
        w_sig = signals[i]['weight']
        comparehists([MLoutput_sig, MLoutput_bkg], [w_sig, w_bkg], ['signal', 'background'], 
                     [sig_fns[i], 'MLscore', 'fraction of events'],True, '_sig_bkg_compare'+sig_fns[i], bins=100, range=(0,1))
    #compare.show()
    #return compare


def getPlotData(phys_vars, vars_name, idx, fns):
    '''
    this function produced 1d arrays with weights for pyplot hist
    used for combine different source of background samples with different weight (can be event level)
    phys_vars: data of different variables
    vars_name: variables that to be combined
    idx: indices of events that is going to be used
    fns: root filenames of all those samples 
    '''

    weights = GetNormWeight(fns, int_lumi=41521.0)
    plot_w = {}
    plot_data = {}

    single_vars = []
    multi_vars = []
    nested_vars = []
    for v in vars_name:
      if v in plot_vars_single:
        single_vars.append(v)
      elif v in plot_vars_multi:
        multi_vars.append(v)
      elif v in plot_vars_nestedarray:
        nested_vars.append(v)
      else:
        raise ValueError("variable {} doesn't belong to any variable type!".format(v))
    
    for i in range(len(fns)):
        # w includes event-level weight and xsec normalizetion
        if len(phys_vars[i]['weight'][idx[i]])==0:
            continue
        w = phys_vars[i]['weight'][idx[i]]*weights[i]
        for v in single_vars:
            if v in plot_data:
                plot_data[v].append(phys_vars[i][v][idx[i]])
                plot_w[v].append(w)
            else:
                phys_vars[i][v][idx[i]].shape
                plot_data[v] = [phys_vars[i][v][idx[i]]]
                plot_w[v] = [w]
        
        for v in multi_vars:
            var = phys_vars[i][v][idx[i]]
            # make w the same dimension as variables
            w_extended = []
            for ievt in range(len(w)):
                w_extended.append([w[ievt]]*len(var[ievt]))
            var_flattern = np.concatenate(var)
            w_extended = np.concatenate(w_extended)
            if v in plot_data:
                plot_data[v].append(var_flattern)
                plot_w[v].append(w_extended)
            else:
                plot_data[v] = [var_flattern]
                plot_w[v] = [w_extended]
           
        for v in nested_vars:
            var = phys_vars[i][v][idx[i]]
            # flattern variable data and make w the same dimensions
            w_extended = []
            var_flattern = []
            for ievt in range(len(w)):
                var_ievt_array = np.concatenate(var[ievt], axis=None)
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

    for v in vars_name:
        if v in plot_data:
          plot_data[v] = np.concatenate(plot_data[v], axis=None)
          plot_w[v] = np.concatenate(plot_w[v], axis=None)
        else:
          plot_data[v] = np.array([])
          plot_w[v] = np.array([])
    
    return plot_data, plot_w

def makeplotfile(fns,newfn,isSignal):
    fnew = ROOT.TFile(newfn+".root","RECREATE")
    MLscore_threshold = 0.4
    ML_inputs, ML_inputs_ori, phys_vars = GetData(fns)
    assert(len(fns)==len(ML_inputs))
    assert(len(fns)==len(ML_inputs_ori))
    assert(len(fns)==len(phys_vars))
    ML_outputs = calcMLscore(ML_inputs, ML_inputs_ori, model_path=m_path)
    for i in range(len(fns)):
        phys_vars[i]['MLScore'] = ML_outputs[i]
    idx_highML = []
    idx_lowML = []
    idx_all = []
    ntk_idx = {
      '3trk':[],
      '4trk':[],
      '5trk':[]
    } # 3-trk, 4-trk, >=5-trk
    for out in ML_outputs:
        highML = out>MLscore_threshold
        idx_highML.append(np.reshape(highML, len(highML)))
        lowML = out<=MLscore_threshold
        idx_lowML.append(np.reshape(lowML, len(lowML)))
        allML = np.array([True]*len(lowML))
        idx_all.append(allML)

    for i in range(len(fns)):
        max_ntk = phys_vars[i]['max_SV_ntracks']
        ntk_3 = max_ntk==3
        ntk_4 = max_ntk==4
        ntk_5 = max_ntk>=5
        ntk_idx['3trk'].append(np.reshape(ntk_3, len(ntk_3)))
        ntk_idx['4trk'].append(np.reshape(ntk_4, len(ntk_5)))
        ntk_idx['5trk'].append(np.reshape(ntk_5, len(ntk_5)))

    vars_name = [
      'met_pt','nsv','max_SV_ntracks','MLScore',
      'jet_pt', 'jet_eta', 'jet_phi',
      'vtx_ntk','vtx_dBV','vtx_dBVerr',
      'tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err',
      'vtx_tk_pt','vtx_tk_eta','vtx_tk_phi', 'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz',
                ]
    data_highML, weight_highML = getPlotData(phys_vars, vars_name, idx_highML, fns)
    data_lowML, weight_lowML = getPlotData(phys_vars, vars_name, idx_lowML, fns)
    data_all, weight_all = getPlotData(phys_vars, vars_name, idx_all, fns)
    plotcategory(fnew,"highML_inclusive",vars_name,data_highML,weight_highML)
    plotcategory(fnew,"lowML_inclusive",vars_name,data_lowML,weight_lowML)
    plotcategory(fnew,"allML_inclusive",vars_name,data_all,weight_all)

    for intk in ntk_idx:
      pick_idx_high = []
      pick_idx_low = []
      for iidx in range(len(idx_highML)):
        pick_idx_high.append(idx_highML[iidx] & ntk_idx[intk][iidx])
        pick_idx_low.append(idx_lowML[iidx] & ntk_idx[intk][iidx])
      data_highML, weight_highML = getPlotData(phys_vars, vars_name, pick_idx_high, fns)
      data_lowML, weight_lowML = getPlotData(phys_vars, vars_name, pick_idx_low, fns)
      plotcategory(fnew,"highML_"+intk,vars_name,data_highML,weight_highML)
      plotcategory(fnew,"lowML_"+intk,vars_name,data_lowML,weight_lowML)

    fnew.Close()

    # print number of events in each region
    weights = GetNormWeight(fns, int_lumi=41521.0)
    cut_var = 'max_SV_ntracks'
    cut_val = 5
    # total_sum/var = [A,B,C,D] representing regions
    region_names = ['A', 'B', 'C', 'D']
    total_sum = [0,0,0,0]
    total_var = [0,0,0,0]
    for i in range(len(fns)):
        w = phys_vars[i]['weight']
        cut_var_array = phys_vars[i][cut_var]
        cut_region = [
            (idx_highML[i]) & (cut_var_array>=cut_val), # A
            (idx_lowML[i]) & (cut_var_array>=cut_val),  # B
            (idx_highML[i]) & (cut_var_array<cut_val),  # C
            (idx_lowML[i]) & (cut_var_array<cut_val),   # D
        ]
        for iregion in range(len(cut_region)):
            w_region = w[cut_region[iregion]]
            nevt_region = np.sum(w_region)*weights[i]
            nevt_variance_region = nevt_region*weights[i]
            total_sum[iregion] += nevt_region
            total_var[iregion] += nevt_variance_region
            print("sample {} in region {} : {} +- {}".format(fns[i],region_names[iregion],nevt_region,np.sqrt(nevt_variance_region)))
            
    if not isSignal:
      print("Summing together: ")
      for iregion in range(len(region_names)):
          print("Region {}: {} +- {}".format(region_names[iregion],total_sum[iregion],np.sqrt(total_var[iregion])))
    


def main():
    fns = [
      #'qcdht0200_2017',
      #'qcdht0300_2017',
      #'qcdht0500_2017',
      #'qcdht0700_2017', 
      #'qcdht1000_2017', 
      #'qcdht1500_2017', 
      #'qcdht2000_2017', 
      #'wjetstolnu_2017', 
      #'wjetstolnuext_2017', 
      #'zjetstonunuht0100_2017', 
      #'zjetstonunuht0200_2017', 
      #'zjetstonunuht0400_2017', 
      #'zjetstonunuht0600_2017', 
      #'zjetstonunuht0800_2017', 
      #'zjetstonunuht1200_2017', 
      #'zjetstonunuht2500_2017', 
      #'ttbar_2017',
      'jettree', # temproraily for ttbar
    ]
    if doBackground:
      #makeplotfile(fns,"background_lowMET",False)
      makeplotfile(fns,"ttbar"+fn_idx,False)

    sig_fns = ['mfv_splitSUSY_tau000000000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000000000um_M2400_2300_2017',
               'mfv_splitSUSY_tau000000300um_M2000_1800_2017',
               'mfv_splitSUSY_tau000000300um_M2400_2300_2017',
               'mfv_splitSUSY_tau000001000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000001000um_M2400_2300_2017',
               'mfv_splitSUSY_tau000001000um_M1200_1100_2017',
               'mfv_splitSUSY_tau000001000um_M1400_1200_2017',
               'mfv_splitSUSY_tau000010000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000010000um_M2400_2300_2017',
               'mfv_splitSUSY_tau000010000um_M1200_1100_2017',
               'mfv_splitSUSY_tau000010000um_M1400_1200_2017',
              ]
    if doSignal:
      for sig_fn in sig_fns:
        makeplotfile([sig_fn],sig_fn+"lowMET",True)


# In[6]:


main()


