import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(ROOT.kTRUE)
fns = [
  "background_METtrigger_cumu.root",
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
  "background",
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
color = [1,2,3,4,5,6,7,8,9,40,41,42,43]
hs = []
n = "inclusive_5trk/MET_cumulative"
l = ROOT.TLegend(0.7,0.6,0.9,0.9)
c = ROOT.TCanvas("c","c",800,800)
c.cd()
for i in range(len(fns)):
  f = ROOT.TFile(fns[i])
  h = f.Get(n)
  h.SetLineColor(color[i])
  h.SetDirectory(0)
  hs.append(h)
for i in range(len(fns)):
  nevts = hs[i].GetBinContent(1)
  hs[i].Scale(1.0/nevts)
  if i==0:
    hs[i].GetXaxis().SetRangeUser(0,500)
    hs[i].GetYaxis().SetTitle("fraction of events after cut")
    hs[i].Draw()
  else:
    hs[i].Draw("same")
  l.AddEntry(hs[i],legends[i])

l.Draw()
c.SaveAs("MET_cumu.png")




