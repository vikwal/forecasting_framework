#!/bin/bash

for i in {1..6}; do
    screen -X -S hpo_$i quit
    echo "Closed screen sessions"
done