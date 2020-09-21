#/bin/bash

START_TIME=$SECONDS
python3 train_run.py --epochs $1 --batch_size $2
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " train.py took " $ELAPSED_TIME >> train_log.txt

