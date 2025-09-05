#!/bin/bash

elements=("00198" "00282" "00298" "00704" "01200" "01303" "02483" "04745")

for i in {0..7}; do
    element=${elements[$i]}
    screen -dmS hpo_$i bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        CUDA_VISIBLE_DEVICES=$i python hpo_cl.py -d reninja_n_dwd/$element -m convlstm
    "
    echo "Started screen session hpo_$i on GPU $i with dataset reninja_n_dwd/$element"
done