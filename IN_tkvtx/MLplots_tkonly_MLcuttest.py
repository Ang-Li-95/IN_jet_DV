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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
from array import array

mlvar_tk = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig']
mlvar_vtx = ['vtx_ntk', 'vtx_dBV', 'vtx_dBVerr']

METNoMu_avai = False
doSignal = False
doBackground = True
No=50
Nr = No*(No-1)
Dp = 20
Ds = len(mlvar_tk)
Dv = len(mlvar_vtx)
Dr=1
fn_dir = '/uscms/home/ali/nobackup/LLP/crabdir/MLTreeAllVtxVkeeptk_v1METm/'
#fn_dir = '/uscms/home/ali/nobackup/LLP/crabdir/MLTreeAllVtxVkeeptk_ttbarht_v1METm/'
#fn_dir = '/uscms/home/ali/nobackup/LLP/crabdir/MLTreeHighMETAllVtxVkeeptk_v2_metMETm/'
m_path = './model_0618/'
save_plot_path='./'



# In[2]:

plot_vars_titles = {
    'met_pt':['MET','MET (GeV)','# events'], 
    'metnomu_pt':['METNoMu','METNoMu (GeV)','# events'], 
    'nsv':['nSV','nSV','# events'], 
    'MLScore':['MLScore','MLScore','# events'],
    'vtx_ntk':['vtx_ntk','nTracks/SV','# events'],
    'vtx_dBV':['vtx_dBV','dist2d(SV, beamspot) (cm)','# events'],
    'vtx_dBVerr':['vtx_dBVerr','error dist2d(SV, beamspot) (cm)','# events'],
    'vtx_mass_track':['vtx_mass_track','SV tracks-only mass (GeV)','# events'],
    'vtx_mass_jet':['vtx_mass_jet','SV jets-by-ntracks -only mass (GeV)','# events'],
    'vtx_mass_trackjet':['vtx_mass_trackjet','SV tracks-plus-jets-by-ntracks mass (GeV)','# events'],
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
    'metnomu_pt':['METNoMu','METNoMu (GeV)','# events'], 
    'nsv':['nSV','nSV','# events'], 
    'MLScore':['MLScore','MLScore','# events'],
    'vtx_ntk':['vtx_ntk','nTracks/SV','# events'],
    'vtx_dBV':['vtx_dBV','dist2d(SV, beamspot) (cm)','# events'],
    'vtx_dBVerr':['vtx_dBVerr','error dist2d(SV, beamspot) (cm)','# events'],
    'vtx_mass_track':['vtx_mass_track','SV tracks-only mass (GeV)','# events'],
    'vtx_mass_jet':['vtx_mass_jet','SV jets-by-ntracks -only mass (GeV)','# events'],
    'vtx_mass_trackjet':['vtx_mass_trackjet','SV tracks-plus-jets-by-ntracks mass (GeV)','# events'],
}

# variables that each event has possibly more than one value
plot_vars_multi = {
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

# variables that each event has multiple arrays
plot_vars_nestedarray = {
}
bins_dBV = [0,0.01,0.02,0.03]
bins_dBV += [i*0.04 for i in range(1,26)]
plot_setting = {
    'met_pt': {'range':(0,2000), 'bins':100},
    'metnomu_pt': {'range':(0,2000), 'bins':100},
    'nsv': {'range':(0,10), 'bins':10},
    'MLScore': {'range':(0,1), 'bins':25},
    'vtx_ntk': {'range':(0,40), 'bins':40},
    #'vtx_dBV': {'range':(0,1), 'bins':25},
    'vtx_dBV': {'binlist':array('d', bins_dBV), 'range':(0,1), 'bins':25},
    'vtx_dBVerr': {'range':(0,0.005), 'bins':25},
    'vtx_mass_track': {'range':(0,200), 'bins':25},
    'vtx_mass_jet': {'range':(0,500), 'bins':50},
    'vtx_mass_trackjet': {'range':(0,500), 'bins':50},
    'vtx_tk_pt': {'range':(0,200), 'bins':50},
    'vtx_tk_eta': {'range':(-4,4), 'bins':50},
    'vtx_tk_phi': {'range':(-3.2,3.2), 'bins':64},
    'vtx_tk_dxy': {'range':(0,0.5), 'bins':25},
    'vtx_tk_dxy_err': {'range':(0,0.025), 'bins':25},
    'vtx_tk_nsigmadxy': {'range':(0,40), 'bins':25},
    'vtx_tk_dz': {'range':(0,20), 'bins':50},
    'vtx_tk_dz_err': {'range':(0,0.15), 'bins':50},
    'vtx_tk_nsigmadz': {'range':(0,3000), 'bins':50},
    'jet_pt': {'range':(0,500), 'bins':50},
    'jet_eta': {'range':(-4,4), 'bins':50},
    'jet_phi': {'range':(-3.2,3.2), 'bins':64},
    'tk_pt': {'range':(0,500), 'bins':100},
    'tk_eta': {'range':(-4,4), 'bins':50},
    'tk_phi': {'range':(-3.2,3.2), 'bins':64},
    'tk_dxybs': {'range':(-0.5,0.5), 'bins':50},
    'tk_dxybs_sig': {'range':(-40,40), 'bins':50},
    'tk_dxybs_err': {'range':(0,0.06), 'bins':50},
    'tk_dz': {'range':(-15,15), 'bins':50},
    'tk_dz_sig': {'range':(-3000,3000), 'bins':50},
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
        "wjetstolnusum_2017": 5.28E+04,
        #"wjetstolnu_2017": 5.28E+04,
        #"wjetstolnuext_2017": 5.28E+04,
        "zjetstonunuht0100_2017": 302.8,
        "zjetstonunuht0200_2017": 92.59,
        "zjetstonunuht0400_2017": 13.18,
        "zjetstonunuht0600_2017": 3.257,
        "zjetstonunuht0800_2017": 1.49,
        "zjetstonunuht1200_2017": 0.3419,
        "zjetstonunuht2500_2017": 0.005146,
        "ttbar_2017": 832,
        "ttbarht0600_2017": 1.821,
        "ttbarht0800_2017": 0.7532,
        "ttbarht1200_2017": 0.1316,
        "ttbarht2500_2017": 0.001407,
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
        nevt = f['mfvWeight/h_sums'].values[f['mfvWeight/h_sums'].xlabels.index('sum_nevents_total')]
        nevents.append(nevt)
        del f
    return nevents

def GetData(fns, cut="(met_pt < 150) & (vtx_ntk > 0)"):
    ML_inputs_tk = []
    ML_inputs_vtx = []
    phys_variables = []
    #variables = ['evt', 'weight', 'met_pt', 'met_phi', 'metnomu_pt', 'metnomu_phi', 'nsv', 
    variables = ['evt', 'weight', 'met_pt', 'met_phi', 'nsv', 
                 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 
                 'tk_pt', 'tk_eta', 'tk_phi', 
                 'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err', 
                 'vtx_ntk', 'vtx_dBV', 'vtx_dBVerr', 'vtx_mass_track', 'vtx_mass_jet', 'vtx_mass_trackjet',
                 'vtx_tk_pt', 'vtx_tk_eta', 'vtx_tk_phi', 
                 'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz']
    if METNoMu_avai:
      variables += ['metnomu_pt', 'metnomu_phi']
    for fn in fns:
        #print(fn)
        print("opening {}...".format(fn_dir+fn+'.root'))
        f = uproot.open(fn_dir+fn+'.root')
        f = f["mfvJetTreer/tree_DV"]
        if len(f['evt'].array())==0:
          print( "no events!!!")
          continue
        phys = f.arrays(variables, namedecode="utf-8")
        del f
        evt_select = (phys['met_pt']>=80) & (phys['met_pt']<150) & (phys['vtx_ntk']>0) & (phys['vtx_dBVerr']<0.0025)
        #evt_select = (phys['vtx_ntk']>0) & (phys['vtx_dBVerr']<0.0025)
        for v in phys:
          phys[v] = np.array(phys[v][evt_select])
        if len(phys['evt'])==0:
            print("no events after selection!")
            continue
        m_tk = np.array([phys[v] for v in mlvar_tk])
        m_vtx = np.array([phys[v] for v in mlvar_vtx]).T
        m_tk = zeropadding(m_tk, No)
        m_tk = normalizedata(m_tk)
        m_vtx = normalizedata(m_vtx)
        ML_inputs_tk.append(m_tk)
        ML_inputs_vtx.append(m_vtx)
        phys_variables.append(phys)
        
    return ML_inputs_tk, ML_inputs_vtx, phys_variables

def calcMLscore(ML_inputs_tk, ML_inputs_vtx, model_path='./', model_name="test_model.meta"):
    batch_size=4096
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        MLscores = []
        for iML in range(len(ML_inputs_tk)):
            ML_input_tk = ML_inputs_tk[iML]
            ML_input_vtx = ML_inputs_vtx[iML]
            evt=0
            outputscore = []
            while evt<len(ML_input_tk):
                if evt+batch_size <= len(ML_input_tk):
                    batch_input_tk = ML_input_tk[evt:evt+batch_size]
                    #batch_input_vtx = ML_input_vtx[evt:evt+batch_size]
                else:
                    batch_input_tk = ML_input_tk[evt:]
                    #batch_input_vtx = ML_input_vtx[evt:]
                evt += batch_size
                Rr, Rs, Ra = getRmatrix(len(batch_input_tk))
                ML_output = sess.run(['INscore:0'],feed_dict={'O:0':batch_input_tk,'Rr:0':Rr,'Rs:0':Rs,'Ra:0':Ra})
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
    for i in range(matrix.shape[1]):
        # transfer df to matrix for each event
        m = np.array([matrix[:,i][v] for v in range(len(mlvar_tk))])
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
  if len(data.shape)==3: # track information
      n_features_data = Ds
      normalize_factors = [
        [3.0058594415751156, 1.6191405668484613, 22.183593957456264],
        [-0.0034791098279665987, -2.0468154272796326, 2.042786889451172],
        [-0.0543001882707337, -2.815811409048992, 2.8012590283634102],
        [-1.235642339086932e-05, -0.025959662283924146, 0.026071622642012487],
        [-0.003970001134868386, -10.319389411895663, 10.576279478554758],
        [-0.14118215968696157, -5.734189675229984, 5.397960310096741],
        [-18.854126507212367, -1587.1630172014382, 1357.0385916506366],
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
  elif len(data.shape)==2:
      normalize_factors = [
        [4.0, 3.0, 12.0],
        [0.032811831682920456, 0.011056187562644482, 0.5899604558944702],
        [0.001431089243851602, 0.0007026850944384933, 0.0023525492288172245],
      ]
      for i in range(Dv):
        #l = np.sort(np.reshape(data[:,i],[1,-1])[0])
        #median = l[int(len(l)*0.5)]
        #l_min = l[int(len(l)*0.05)]
        #l_max = l[int(len(l)*0.95)]
        median = normalize_factors[i][0]
        l_min = normalize_factors[i][1]
        l_max = normalize_factors[i][2]
        data[:,i] = (data[:,i]-median)*(2.0/(l_max-l_min))

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

def makehist(data,weight,var):
    assert(var in plot_setting)
    assert(var in plot_vars_titles)
    label = plot_vars_titles[var]
    setting = plot_setting[var]
    if 'binlist' in setting:
      h = ROOT.TH1F(label[0],";".join(label), len(setting['binlist'])-1, setting['binlist'])
    else:
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
    ntk_ML = make2dhist(data['vtx_ntk'],data['MLScore'],weight['vtx_ntk'],'vtx_ntk','MLScore')
    dbv_ML = make2dhist(data['vtx_dBV'],data['MLScore'],weight['vtx_dBV'],'vtx_dBV','MLScore')
    hist2d.Write()
    ntk_ML.Write()
    dbv_ML.Write()
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
                     [sig_fns[i], 'MLscore', 'fraction of events'],True, '_sig_bkg_compare'+sig_fns[i], bins=50, range=(0,1))
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
    #MLscore_threshold = 0.4
    MLscore_threshold_high = 0.4
    MLscore_threshold_low = 0.3
    ML_inputs_tk, ML_inputs_vtx, phys_vars = GetData(fns)
    assert(len(fns)==len(ML_inputs_tk))
    assert(len(fns)==len(ML_inputs_vtx))
    assert(len(fns)==len(phys_vars))
    ML_outputs = calcMLscore(ML_inputs_tk, ML_inputs_vtx, model_path=m_path)
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
        highML = out>MLscore_threshold_high
        idx_highML.append(np.reshape(highML, len(highML)))
        lowML = out<=MLscore_threshold_low
        idx_lowML.append(np.reshape(lowML, len(lowML)))
        allML = np.array([True]*len(lowML))
        idx_all.append(allML)

    for i in range(len(fns)):
        max_ntk = phys_vars[i]['vtx_ntk']
        ntk_3 = max_ntk==3
        ntk_4 = max_ntk==4
        ntk_5 = max_ntk>=5
        ntk_idx['3trk'].append(np.reshape(ntk_3, len(ntk_3)))
        ntk_idx['4trk'].append(np.reshape(ntk_4, len(ntk_5)))
        ntk_idx['5trk'].append(np.reshape(ntk_5, len(ntk_5)))

    vars_name = [
      #'met_pt','metnomu_pt','nsv','MLScore',
      'met_pt','nsv','MLScore',
      'jet_pt', 'jet_eta', 'jet_phi',
      'vtx_ntk','vtx_dBV','vtx_dBVerr', 'vtx_mass_track', 'vtx_mass_jet', 'vtx_mass_trackjet',
      'tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err',
      'vtx_tk_pt','vtx_tk_eta','vtx_tk_phi', 'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz',
                ]
    if METNoMu_avai:
      vars_name.append('metnomu_pt')
    data_highML, weight_highML = getPlotData(phys_vars, vars_name, idx_highML, fns)
    data_lowML, weight_lowML = getPlotData(phys_vars, vars_name, idx_lowML, fns)
    data_all, weight_all = getPlotData(phys_vars, vars_name, idx_all, fns)
    plotcategory(fnew,"highML_inclusive",vars_name,data_highML,weight_highML)
    plotcategory(fnew,"lowML_inclusive",vars_name,data_lowML,weight_lowML)
    plotcategory(fnew,"allML_inclusive",vars_name,data_all,weight_all)

    for intk in ntk_idx:
      pick_idx_incl = []
      pick_idx_high = []
      pick_idx_low = []
      for iidx in range(len(idx_highML)):
        pick_idx_incl.append(ntk_idx[intk][iidx])
        pick_idx_high.append(idx_highML[iidx] & ntk_idx[intk][iidx])
        pick_idx_low.append(idx_lowML[iidx] & ntk_idx[intk][iidx])
      data_ntk_incl, weight_ntk_incl = getPlotData(phys_vars, vars_name, pick_idx_incl, fns)
      data_highML, weight_highML = getPlotData(phys_vars, vars_name, pick_idx_high, fns)
      data_lowML, weight_lowML = getPlotData(phys_vars, vars_name, pick_idx_low, fns)
      plotcategory(fnew,"inclusive_"+intk,vars_name,data_ntk_incl,weight_ntk_incl)
      plotcategory(fnew,"highML_"+intk,vars_name,data_highML,weight_highML)
      plotcategory(fnew,"lowML_"+intk,vars_name,data_lowML,weight_lowML)

    fnew.Close()

    # print number of events in each region
    weights = GetNormWeight(fns, int_lumi=41521.0)
    cut_var = 'vtx_ntk'
    cut_val_high = 5
    cut_val_val = 4
    # total_sum/var = [A,B,C,D] representing regions
    region_names = ['5tk high', '5tk low', '4tk high', '4tk low', '3tk high', '3tk low']
    total_sum = [0,0,0,0,0,0]
    total_var = [0,0,0,0,0,0]
    for i in range(len(fns)):
        w = phys_vars[i]['weight']
        cut_var_array = phys_vars[i][cut_var]
        cut_region = [
            (idx_highML[i]) & (cut_var_array>=cut_val_high), # A
            (idx_lowML[i]) & (cut_var_array>=cut_val_high),  # B
            (idx_highML[i]) & (cut_var_array>=cut_val_val) & (cut_var_array<cut_val_high),
            (idx_lowML[i]) & (cut_var_array>=cut_val_val) & (cut_var_array<cut_val_high),
            (idx_highML[i]) & (cut_var_array<cut_val_val),  # C
            (idx_lowML[i]) & (cut_var_array<cut_val_val),   # D
        ]
        for iregion in range(len(cut_region)):
            w_region = w[cut_region[iregion]]
            nevt_region = np.sum(w_region)*weights[i]
            nevt_raw = len(w_region)
            nevt_variance_region = nevt_region*weights[i]
            total_sum[iregion] += nevt_region
            total_var[iregion] += nevt_variance_region
            print("sample {} in region {} : {} +- {} raw: {}".format(fns[i],region_names[iregion],nevt_region,np.sqrt(nevt_variance_region),nevt_raw))
            
    if not isSignal:
      print("Summing together: ")
      for iregion in range(len(region_names)):
          print("Region {}: {} +- {}".format(region_names[iregion],total_sum[iregion],np.sqrt(total_var[iregion])))
    


def main():
    fns = [
      #'qcdht0200_2017',
      'qcdht0300_2017',
      'qcdht0500_2017',
      'qcdht0700_2017', 
      'qcdht1000_2017', 
      'qcdht1500_2017', 
      'qcdht2000_2017', 
      'wjetstolnusum_2017',
      'zjetstonunuht0100_2017', 
      'zjetstonunuht0200_2017', 
      'zjetstonunuht0400_2017', 
      'zjetstonunuht0600_2017', 
      'zjetstonunuht0800_2017', 
      'zjetstonunuht1200_2017', 
      'zjetstonunuht2500_2017', 
      'ttbar_2017',
      #'ttbarht0600_2017',
      #'ttbarht0800_2017',
      #'ttbarht1200_2017',
      #'ttbarht2500_2017',
    ]
    if doBackground:
      #makeplotfile(fns,"background_METtrigger",False)
      makeplotfile(fns,"background_lowMET",False)
      #makeplotfile(fns,"ttbarHT_highMET",False)

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
        #makeplotfile([sig_fn],sig_fn+"_METtrigger",True)
        makeplotfile([sig_fn],sig_fn+"_lowMET",True)
        #makeplotfile([sig_fn],sig_fn+"_highMET",True)


# In[6]:


main()


