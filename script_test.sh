#!/usr/bin/env bash
source ~/.bashrc

conda activate torch

weights_path=$1
data_path=$2

python makecsv.py
python test_i3d.py --path $weights_path --root $data_path
python test_temporal.py
