#/bin/bash

START_TIME=$SECONDS
python3 train.py --epochs 50 --batch_size 32 --data_path=./data/5/ --output_path=./data/9/
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " train.py took " $ELAPSED_TIME >> train_log.txt

