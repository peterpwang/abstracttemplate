#/bin/bash

START_TIME=$SECONDS
python3 generate.py --data_path=./data/text/ --model_path=./data/results/save.mdl
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " generate.py took " $ELAPSED_TIME >> generate_log.txt

