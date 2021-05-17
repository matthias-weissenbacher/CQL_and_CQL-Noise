#!/bin/bash

# Script to reproduce results

num_devices=2 #  N=2 number of available GPUS 

#Environment List
declare -a EnvArray=("antmaze-umaze-v0" "antmaze-umaze-diverse-v0" "antmaze-medium-play-v0" "antmaze-medium-diverse-v0" "antmaze-large-play-v0" "antmaze-large-diverse-v0")

#Run baseline CQL experiments
for env in ${EnvArray[@]}; do
        rand=$(( ( RANDOM % $num_devices )  ))
        python3 CQL/main.py \
                --env  $env  \
                --policy "CQL" \
                --cuda_device $rand \
done

#Run baseline CQL with state noise added (S4RL) experiments
for env in ${EnvArray[@]}; do
        rand=$(( ( RANDOM % $num_devices )  ))
        python3 CQL/main.py \
                --env  $env  \
                --policy "CQL_Noise" \
                --cuda_device $rand \
done

