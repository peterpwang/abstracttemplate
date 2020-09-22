#/bin/bash

START_TIME=$SECONDS
python3 train_run.py --epochs 2 --batch_size 32 --data_path=./data/text/
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " train.py took " $ELAPSED_TIME >> train_log.txt

