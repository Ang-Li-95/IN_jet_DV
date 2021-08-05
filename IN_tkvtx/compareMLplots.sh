
METup=200
METdown=0
comparehists.py background_lowMET_nobquark.root "highML_inclusive" "lowML_inclusive" /publicweb/a/ali/MLscore_comp_${1} --nice "high ML" "low ML"  --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "highML_3trk" "lowML_3trk" /publicweb/a/ali/MLscore_comp_3tk_${1} --nice "3tk high ML" "3tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "highML_4trk" "lowML_4trk" /publicweb/a/ali/MLscore_comp_4tk_${1} --nice "4tk high ML" "4tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "highML_5trk" "lowML_5trk" /publicweb/a/ali/MLscore_comp_5tk_${1} --nice "5tk high ML" "5tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "inclusive_3trk" "inclusive_4trk" "inclusive_5trk" /publicweb/a/ali/MLtrk_comp_${1} --nice "3tk" "4tk" "5tk" --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "highML_3trk" "highML_4trk" "highML_5trk" /publicweb/a/ali/MLtrk_comp_high_${1} --nice "3tk highML" "4tk highML" "5tk highML" --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"
comparehists.py background_lowMET_nobquark.root "lowML_3trk" "lowML_4trk" "lowML_5trk" /publicweb/a/ali/MLtrk_comp_low_${1} --nice "3tk lowML" "4tk lowML" "5tk lowML" --x-range "[$METdown,$METup] if 'MET' in name else [0,1.2] if name=='vtx_dBV' else [0,1.4] if 'ML' in name else None"

