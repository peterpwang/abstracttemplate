#/bin/bash

START_TIME=$SECONDS
python3 html_process.py --debug=0 --input_path=./data/0/ --text_path=./data/1/ --tfidf_path=./data/2/
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " html_process.py took " $ELAPSED_TIME >> process_log.txt


