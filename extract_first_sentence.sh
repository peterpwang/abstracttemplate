#/bin/bash

awk -F '.' {'print $1"."'} data/text_whole/train.txt > data/text/train.txt
awk -F '.' {'print $1"."'} data/text_whole/validation.txt > data/text/validation.txt
awk -F '.' {'print $1"."'} data/text_whole/test.txt > data/text/test.txt

