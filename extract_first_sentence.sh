#/bin/bash

awk -F '.' {'print $1"."'} data/2/train.txt > data/3/train.txt
awk -F '.' {'print $1"."'} data/2/validation.txt > data/3/validation.txt
awk -F '.' {'print $1"."'} data/2/test.txt > data/3/test.txt

