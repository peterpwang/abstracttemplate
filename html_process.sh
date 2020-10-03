#/bin/bash

START_TIME=$SECONDS
python3 html_process.py --debug=0 --input_path=./data/0/ --text_path=./data/1/ --upos_path=./data/2 --tfidf_path=./data/3/ --skip_extraction=1 --skip_upos=1 --skip_tfidf=1
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " html_process.py took " $ELAPSED_TIME >> process_log.txt


