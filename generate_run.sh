#/bin/bash

START_TIME=$SECONDS
python3 generate_run.py
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " generate.py took " $ELAPSED_TIME >> generate_log.txt

