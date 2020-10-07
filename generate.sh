#/bin/bash

START_TIME=$SECONDS
python3 generate.py --data_path=./data/3/ --model_file=./data/9/save.mdl --output_path=./data/9/ --num_sentences=10 --first_word=This
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " generate.py took " $ELAPSED_TIME >> generate_log.txt

