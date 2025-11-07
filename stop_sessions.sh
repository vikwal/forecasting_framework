#!/bin/bash

for i in {0..3}; do
    screen -X -S tft_$i quit
    echo "Closed screen sessions"
done