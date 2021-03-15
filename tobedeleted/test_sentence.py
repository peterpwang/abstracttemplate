#from bs4 import BeautifulSoup

import argparse
import sys
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza

from util import read_text

def read_original_data(html_data_dir, text_data_dir, upos_data_dir, tfidf_data_dir, first_sentence_dir):
    print("Reading...", flush=True)
    lines = read_text(text_data_dir + "/data.txt")
    number_htmls = len(lines)
    print(str(number_htmls) +  " read from dataset.", flush=True)

    print("Tagging...", flush=True)
    lines = create_upos(lines, upos_data_dir + "/data_upos.txt", upos_data_dir + "/data_upos.html")
    print("UPOS tags created.", flush=True)


def create_upos(lines, upos_text_file, upos_html_file):
    global debug

    #stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    docs = []
    i = 0
    for line in lines:
        doc = nlp(line)
        docs.append(doc)
        i = i + 1
        if debug == 1 and i%1000 == 0:
            print(".", end = '', flush=True)

    # Output 5 documents into sample HTML file
    f = open(upos_html_file, 'w')
    f.write("<!DOCTYPE html>\n")
    f.write("<html lang=\"en\">\n")
    f.write("  <head>\n")
    f.write("    <meta charset=\"utf-8\">\n")
    f.write("    <style>\n")
    f.write("      .entity { color: green; }\n")
    f.write("    </style>\n")
    f.write("  </head>\n")
    f.write("  <body>\n")

    i = 0
    for doc in docs:
        if i >= 5:
            break

        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.ner != "O":
                    f.write('<span class="entity">' + token.text + '</span> ')
                else:
                    f.write(token.text + ' ')

        f.write("<br>\n");
        i = i + 1

    f.write("  </body>\n")
    f.write("</html>\n")
    f.write("\n");
    f.close()

    # Export documents into plain text
    f = open(upos_text_file, 'w')
    for doc in docs:
        for sentence in doc.sentences:
            previous_ner = False
            for token in sentence.tokens:
                if token.ner != "O":
                    if not previous_ner:
                        f.write('NNNN ')
                        previous_ner = True
                else:
                    f.write(token.text + ' ')
                    previous_ner = False
        f.write("\n");
    f.close()

    # return in list
    ner_list = []
    for doc in docs:
        s = ""
        for sentence in doc.sentences:
            previous_ner = False
            for token in sentence.tokens:
                if token.ner != "O":
                    if not previous_ner:
                        s += 'NNNN '
                        previous_ner = True
                else:
                    s += token.text + ' '
                    previous_ner = False
        ner_list.append(s)

    return ner_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', default='./data/1/', metavar='N',
                        help='Output path of text data') 
    parser.add_argument('--upos_path', default='./data/2/', metavar='N',
                        help='Output path of UPOS tagged text data') 
    args = parser.parse_args()

    read_original_data(args.text_path, args.upos_path)

