
METup=200
METdown=0
comparehists.py background_lowMET_MLrange_BInfo.root "highML_inclusive" "midML_inclusive" "lowML_inclusive" /publicweb/a/ali/MLscore_comp_${1} --nice "high ML" "mid ML" "low ML"  --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "highML_3trk" "midML_3trk" "lowML_3trk" /publicweb/a/ali/MLscore_comp_3tk_${1} --nice "3tk high ML" "3tk mid ML" "3tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "highML_4trk" "midML_4trk" "lowML_4trk" /publicweb/a/ali/MLscore_comp_4tk_${1} --nice "4tk high ML" "4tk mid ML" "4tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "highML_5trk" "midML_5trk" "lowML_5trk" /publicweb/a/ali/MLscore_comp_5tk_${1} --nice "5tk high ML" "5tk mid ML" "5tk low ML"  --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "inclusive_3trk" "inclusive_4trk" "inclusive_5trk" /publicweb/a/ali/MLtrk_comp_${1} --nice "3tk" "4tk" "5tk" --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "highML_3trk" "highML_4trk" "highML_5trk" /publicweb/a/ali/MLtrk_comp_high_${1} --nice "3tk highML" "4tk highML" "5tk highML" --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "midML_3trk" "midML_4trk" "midML_5trk" /publicweb/a/ali/MLtrk_comp_low_${1} --nice "3tk midML" "4tk midML" "5tk midML" --x-range "[$METdown,$METup] if 'MET' in name else None"
comparehists.py background_lowMET_MLrange_BInfo.root "lowML_3trk" "lowML_4trk" "lowML_5trk" /publicweb/a/ali/MLtrk_comp_low_${1} --nice "3tk lowML" "4tk lowML" "5tk lowML" --x-range "[$METdown,$METup] if 'MET' in name else None"

