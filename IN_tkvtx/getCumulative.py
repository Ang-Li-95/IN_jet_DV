import ROOT
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', 
                    help='name of input file')
parser.add_argument('--dir', 
                    help='directory to plot')

args = parser.parse_args()

fin = ROOT.TFile(args.input+'.root')
hin_met = fin.Get(args.dir+'/MET')
hin_metnomu = fin.Get(args.dir+'/METNoMu')
n_met = hin_met.Integral()
n_metnomu = hin_metnomu.Integral()
hc_met = hin_met.GetCumulative(False)
#hc_met.SetTitle("Fraction of events cut off;MET(GeV);fraction of events")
#hc_met_cut = hc_met.Clone()
#hc_met_cut.SetTitle("Number of events after cut;MET(GeV);# events")
#hc_met_sqrt = hc_met.Clone()
#hc_met_sqrt.SetTitle("sqrt number of events after cut;MET(GeV);")
#hc_met.Scale(1.0/hin_met.Integral())
hc_metnomu = hin_metnomu.GetCumulative(False)
#hc_metnomu.SetTitle("Fraction of events cut off;METNoMu(GeV);fraction of events")
#hc_metnomu_cut = hc_metnomu.Clone()
#hc_metnomu_cut.SetTitle("Number of events after cut;METNoMu(GeV);# events")
#hc_metnomu_sqrt = hc_metnomu.Clone()
#hc_metnomu_sqrt.SetTitle("sqrt number of events after cut;METNoMu(GeV);")
#hc_metnomu.Scale(1.0/hin_metnomu.Integral())

#for ibin in range(hc_met_cut.GetNcells()):
#  hc_met_cut.SetBinContent(ibin,n_met-hc_met_cut.GetBinContent(ibin))
#  #hc_met_sqrt.SetBinContent(ibin,math.sqrt(n_met-hc_met_sqrt.GetBinContent(ibin)))
#
#for ibin in range(hc_metnomu_cut.GetNcells()):
#  hc_metnomu_cut.SetBinContent(ibin,n_metnomu-hc_metnomu_cut.GetBinContent(ibin))
#  #hc_metnomu_sqrt.SetBinContent(ibin,math.sqrt(n_metnomu-hc_metnomu_sqrt.GetBinContent(ibin)))

fout = ROOT.TFile(args.input+'_cumu.root','RECREATE')
fout.mkdir(args.dir)
fout.cd(args.dir+'/')
hc_met.Write()
hc_metnomu.Write()
hin_met.Write()
hin_metnomu.Write()
#hc_met_cut.Write()
#hc_met_sqrt.Write()
#hc_metnomu_cut.Write()
#hc_metnomu_sqrt.Write()
fout.Close()
fin.Close()
