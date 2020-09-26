#/bin/bash

START_TIME=$SECONDS
python3 train.py --epochs 2 --batch_size 32 --data_path=./data/3/
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " train.py took " $ELAPSED_TIME >> train_log.txt

