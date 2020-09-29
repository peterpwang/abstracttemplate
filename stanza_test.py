import csv
import argparse
import sys
import os
import stanza

from util import read_text


def create_upos(corpus):

    stanza.download('en')
    nlp = stanza.Pipeline('en')

    doc = 0
    while doc < 5:
        text = corpus[doc]
        entry = nlp(text)

        for i, sent in enumerate(entry.sentences):
            print(*[f'{word.text}[{word.upos}]' for word in sent.words], sep=' ')
        #print(upos.entities)
        print('\n')
        doc = doc + 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/1/data.txt', metavar='N',
                        help='Text data file path') 
    args = parser.parse_args()

    corpus = read_text(args.input)
    create_upos(corpus)
    #print(features)
