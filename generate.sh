#/bin/bash

START_TIME=$SECONDS
python3 generate.py --data_path=./data/3/ --model_path=./data/9/save.mdl
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " generate.py took " $ELAPSED_TIME >> generate_log.txt

