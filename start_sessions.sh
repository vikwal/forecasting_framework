#!/bin/bash

for i in {0..3}; do
    screen -dmS hpo_$i bash -c "
        cd ~/Work/forecasting_framework
        source frcst/bin/activate
        CUDA_VISIBLE_DEVICES=$i python hpo_cl.py -m tft -c config_wind_100 -i $i
    "
    echo "Started screen session hpo_$i on GPU $i"
done