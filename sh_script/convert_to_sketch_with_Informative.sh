#!/bin/bash

#conda activate drawings
#cd ../informative-drawings
#export PROJECT_ROOT=$(pwd)

style1="anime_style"
style2="opensketch_style"
target_folder="03001627_2d_projection"

total_amount=40668
amount=10000

i=1
while [ $i -le 5 ]
do
  new_folder=${target_folder}/${i}
  echo ">>>>>>>>>>> Processing ${new_folder}"
  python test.py --name ${style1} --dataroot ${new_folder} --how_many ${amount} --n_cpu 12
  python test.py --name ${style2} --dataroot ${new_folder} --how_many ${amount} --n_cpu 12
   i=$((i + 1))
done
echo "Done！"