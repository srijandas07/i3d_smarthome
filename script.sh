#!/usr/bin/env bash
source ~/.bashrc

exp_name=$1
protocol=$2
data_path=$3

mkdir -p weights_$exp_name

conda activate torch

python train_i3d.py --root $data_path --save_model weights_${exp_name}/ --protocol $protocol

