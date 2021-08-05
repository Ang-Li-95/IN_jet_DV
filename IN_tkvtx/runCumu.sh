name=(
  background_METtrigger
  mfv_splitSUSY_tau000000000um_M2000_1800_2017_METtrigger  
  mfv_splitSUSY_tau000000000um_M2400_2300_2017_METtrigger  
  mfv_splitSUSY_tau000000300um_M2000_1800_2017_METtrigger  
  mfv_splitSUSY_tau000000300um_M2400_2300_2017_METtrigger  
  mfv_splitSUSY_tau000001000um_M2400_2300_2017_METtrigger
  mfv_splitSUSY_tau000001000um_M1200_1100_2017_METtrigger  
  mfv_splitSUSY_tau000001000um_M1400_1200_2017_METtrigger  
  mfv_splitSUSY_tau000001000um_M2000_1800_2017_METtrigger
  mfv_splitSUSY_tau000010000um_M1200_1100_2017_METtrigger
  mfv_splitSUSY_tau000010000um_M1400_1200_2017_METtrigger
  mfv_splitSUSY_tau000010000um_M2000_1800_2017_METtrigger
  mfv_splitSUSY_tau000010000um_M2400_2300_2017_METtrigger
  )
for n in "${name[@]}"
do
  echo $n
  python getCumulative.py --input $n --dir inclusive_5trk
done
