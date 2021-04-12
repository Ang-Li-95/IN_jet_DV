from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import numpy as np
import awkward as ak

fndir = 'root://cmseos.fnal.gov//store/user/ali/JetTreehighVkeeptk_v1METm/'


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

ntk_bkg = []
ntk_sig = []
for fn in fns_bkg:
  f = uproot.open(fndir+fn+'.root')
  f = f["mfvJetTreer/tree_DV"]
  if len(f['evt'].array())==0:
    print( "no events!!!")
    continue
  ntk = np.array(ak.count(f['tk_pt'].array(), axis=1))
  ntk_bkg.append(ntk)

for fn in fns_signal:
  f = uproot.open(fndir+fn+'.root')
  f = f["mfvJetTreer/tree_DV"]
  if len(f['evt'].array())==0:
    print( "no events!!!")
    continue
  ntk = np.array(ak.count(f['tk_pt'].array(), axis=1))
  ntk_sig.append(ntk)

ntk_bkg = np.concatenate(ntk_bkg, axis = None)
ntk_sig = np.concatenate(ntk_sig, axis = None)

plt.hist(ntk_bkg,label='background',bins=350, range=(0,350),alpha=0.5,density=True)
plt.hist(ntk_sig,label='signal',bins=350, range=(0,350),alpha=0.5,density=True)
plt.xlabel("number of tracks")
plt.ylabel("A.U.")
plt.legend(loc='best')
plt.savefig("ntk_quality.png")

