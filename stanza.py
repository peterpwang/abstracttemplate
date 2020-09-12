import stanza

import csv
import argparse
import sys
import os

# public variables
debug = 0;


def read_text(text_path):
    data = []
    with open(text_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            data.append(row[0])
    return data


def create_upos(corpus):

    stanza.download('en')
    nlp = stanza.Pipeline('en')

    doc = 0
    while doc < 5:
        text = corpus[doc]
        upos = nlp(text)

        print(upos)
        print(upos.entities)
        print('\n')
        doc = doc + 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, type=int, metavar='N',
                        help='Debug?') 
    parser.add_argument('--input', default='./data/text/data.cvs', metavar='N',
                        help='CVS data file path') 
    parser.add_argument('--output', default='./data/text/tfidf.cvs', metavar='N',
                        help='TFIDF data file path') 
    args = parser.parse_args()

    debug = args.debug

    corpus = read_text(args.input)
    create_upos(corpus)
    #print(features)
