#!/bin/bash

python3 -W ignore ./generate.py  --model_type=gpt2 --model_name_or_path=./data/9/gpt2-theses/checkpoint-423200 --num_return_sequences=5
