#!/bin/bash

python train.py -s ./data/penguin -r 1 -m output/penguin --config_file config/gaussian_dataset/train.json --n_spectra 10

python train.py -s ./data/penguin_ir -r 1 -m output/penguin_ir --config_file config/gaussian_dataset/train.json --n_spectra 1
