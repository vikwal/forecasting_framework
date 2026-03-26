#!/bin/bash

for i in {1..8}; do
    screen -X -S tft_$i quit
    echo "Closed screen sessions"
done