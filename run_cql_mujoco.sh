#!/bin/bash

# Script to reproduce results

num_devices=2 #  N=2 number of available GPUS 

#  Envvirpnment_List
declare -a EnvArray=("halfcheetah-medium-expert-v0" "hopper-medium-expert-v0" "walker2d-medium-expert-v0" "halfcheetah-medium-v0" "hopper-medium-v0" "walker2d-medium-v0" "halfcheetah-medium-replay-v0" "hopper-medium-replay-v0" "walker2d-medium-replay-v0" "halfcheetah-random-v0" "hopper-random-v0" "walker2d-random-v0")

#Algorithm
 
# Iterate the string array using for loop

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

