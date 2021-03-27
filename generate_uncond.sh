#!/bin/bash

python3 ./generate.py -B science --uncond --length 100 --gamma 1.5 --num_iterations 3 --num_samples 2  --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --sample --verbosity quiet --pretrained_model "/home/pw90/code/abstracttemplate/data/9/gpt2-theses/checkpoint-423200" 
