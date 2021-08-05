import ROOT
import math
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(ROOT.kTRUE)
fn_bkg = "background_METtrigger_cumu.root"
fns_sig = [
  "mfv_splitSUSY_tau000000000um_M2000_1800_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000000000um_M2400_2300_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000000300um_M2000_1800_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000000300um_M2400_2300_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000001000um_M1200_1100_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000001000um_M1400_1200_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000001000um_M2000_1800_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000001000um_M2400_2300_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000010000um_M1200_1100_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000010000um_M1400_1200_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000010000um_M2000_1800_2017_METtrigger_cumu.root",
  "mfv_splitSUSY_tau000010000um_M2400_2300_2017_METtrigger_cumu.root",
]
legends = [
  "00100um_M2000_1800",
  "00100um_M2400_2300",
  "00300um_M2000_1800",
  "00300um_M2400_2300",
  "01000um_M1200_1100",
  "01000um_M1400_1200",
  "01000um_M2000_1800",
  "01000um_M2400_2300",
  "10000um_M1200_1100",
  "10000um_M1400_1200",
  "10000um_M2000_1800",
  "10000um_M2400_2300",
]
color = [2,3,4,5,6,7,8,9,40,41,42,43]
hs = []
n = "inclusive_5trk/MET_cumulative"
l = ROOT.TLegend(0.7,0.6,0.9,0.9)
c = ROOT.TCanvas("ct","ct",800,800)
c.cd()
f_bkg = ROOT.TFile(fn_bkg)
h_bkg = f_bkg.Get(n)
for ibin in range(h_bkg.GetNcells()):
  h_bkg.SetBinContent(ibin,math.sqrt(h_bkg.GetBinContent(ibin)))
for i in range(len(fns_sig)):
  f = ROOT.TFile(fns_sig[i])
  h = f.Get(n)
  #h.SetLineColor(i+1)
  h.SetDirectory(0)
  hs.append(h)

for i in range(len(fns_sig)):
  #hs[i].SetTitle(legends[i])
  hs[i].GetYaxis().SetTitle("signal significance")
  hs[i].SetLineColor(color[i])
  for ibin in range(hs[i].GetNcells()):
    ns = hs[i].GetBinContent(ibin)
    nb = h_bkg.GetBinContent(ibin)
    if nb==0: continue
    sig = math.sqrt(2*(((ns+nb)*math.log(1+(ns/nb)))-ns))
    hs[i].SetBinContent(ibin, sig)

  hs[i].Scale(1.0/hs[i].Integral())

  if i==0:
    hs[i].GetYaxis().SetRangeUser(0,0.12)
    hs[i].Draw()
  else:
    hs[i].Draw("same")
  l.AddEntry(hs[i],legends[i])

l.Draw()
c.SaveAs("MET_sig_total.png")




