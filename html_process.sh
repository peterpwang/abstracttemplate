#/bin/bash

START_TIME=$SECONDS
# Test for RNN: HTML(0)->TEXT(1)->POS TEXT(2)->POS&TFIDF TEXT(3)->TEXT filtered by Common words(4)->First sentences(5)->Model(9)
#python3 html_process.py --debug=0 --input_path=./data/0/ --text_path=./data/1/ --upos_path=./data/2/ --tfidf_path=./data/3/ --input_common_word_path=./data/8/ --common_word_path=./data/4/ --first_sentence_path=./data/5/ --skip_extraction=1 --skip_upos=1 --skip_tfidf=1 --skip_common_word=1 --skip_first_sentence=1
# Test for GPT2: HTML(0)->TEXT(1)->Model(9)
python3 html_process.py --debug=0 --input_path=./data/0/ --text_path=./data/1/ --skip_extraction=1 --skip_upos=1 --skip_tfidf=1 --skip_common_word=1 --skip_first_sentence=1
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo " html_process.py took " $ELAPSED_TIME >> process_log.txt


