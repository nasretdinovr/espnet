#!/usr/bin/env bash

# Check if the required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <path_to_audio> <path_to_espnet> <path_to_exp> <model_name>"
    exit 1
fi

# Input arguments
test_set="$1"
path_to_espnet="$2"
path_to_exp="$3"
inference_model="$4"

# run inference in docker
NV_GPU='0' nvidia-docker run -i --rm --name espnet_gpu0_$(date +%Y%m%dT%H%M) \
        -v "${path_to_espnet}/egs:/espnet/egs" \
        -v "${path_to_espnet}/espnet:/espnet/espnet" \
        -v "${path_to_espnet}/test:/espnet/test" \
        -v "${path_to_espnet}/utils:/espnet/utils" \
        -v "${path_to_espnet}/egs2:/espnet/egs2" \
        -v "${path_to_espnet}/espnet2:/espnet/espnet2" \
	-v "/data/SE/urgent2025_challenge/:/data/SE/urgent2025_challenge" \
        -v /dev/shm:/dev/shm urgent_2025_rn:se_mamba \
         /bin/bash -c "cd /espnet/egs2/urgent25/enh1/ && ./run.sh --stage 7 --stop-stage 7 --enh_exp '${path_to_exp}' --gpu_inference true --inference_model '${inference_model}' --inference_nj 1 --valid_set '${test_set}' --test_sets '${test_set}'"
